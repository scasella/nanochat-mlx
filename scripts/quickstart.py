"""
MLX Quickstart GUI — step-by-step wizard for the full nanochat training pipeline.

Serves a web UI that walks through: data download → tokenizer → training → SFT → chat.
Each stage runs as a subprocess with live SSE streaming of stdout/stderr.

Usage:
    python -m scripts.quickstart
    python -m scripts.quickstart --port 8080
"""

import argparse
import asyncio
import gc
import importlib.util
import json
import os
import re
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from nanochat_mlx.common import SetupError
from nanochat_mlx.preflight import (
    count_downloaded_shards,
    require_checkpoint,
    require_tokenizer,
    require_training_data,
)


def build_parser():
    parser = argparse.ArgumentParser(description="NanoChat MLX Quickstart")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=8.0,
        help="MLX memory limit in GB (default 8, conservative for shared use)",
    )
    return parser


args = argparse.Namespace(port=8000, host="127.0.0.1", memory_limit_gb=8.0)

# --- Globals ---

running_process: Optional[asyncio.subprocess.Process] = None
loaded_engine = None
loaded_tokenizer = None
loaded_model = None  # keep ref for explicit cleanup
loaded_depth = None
loaded_step = None
loaded_source = None

METRIC_RE = re.compile(
    r"step\s+(\d+)/(\d+).*?loss:\s*([\d.]+).*?tok/s:\s*([\d,]+)"
)


def get_base_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    return base


def check_status():
    """Check which pipeline stages are complete by inspecting the filesystem."""
    base = get_base_dir()
    tok_path = os.path.join(base, "tokenizer", "tokenizer.pkl")
    ckpt_base = os.path.join(base, "mlx_checkpoints")
    shard_count = count_downloaded_shards(base)

    data_ready = shard_count >= 2

    tok_ready = os.path.isfile(tok_path)

    # Find all trained depths (base models)
    trained = {}
    sft_trained = {}
    if os.path.isdir(ckpt_base):
        for d in sorted(os.listdir(ckpt_base)):
            if not d.startswith("d"):
                continue
            is_sft = d.endswith("_sft")
            depth_str = d[1:].replace("_sft", "") if is_sft else d[1:]
            if not depth_str.isdigit():
                continue
            depth = int(depth_str)
            dpath = os.path.join(ckpt_base, d)
            safetensors = [
                f for f in os.listdir(dpath)
                if f.endswith(".safetensors") and not f.endswith("_optim.safetensors")
            ]
            if safetensors:
                if is_sft:
                    sft_trained[depth] = len(safetensors)
                else:
                    trained[depth] = len(safetensors)

    chat_ready = loaded_engine is not None

    return {
        "data": data_ready,
        "data_shards": shard_count,
        "tokenizer": tok_ready,
        "train": trained,
        "sft": sft_trained,
        "chat": chat_ready,
        "chat_model": {"depth": loaded_depth, "step": loaded_step, "source": loaded_source} if chat_ready else None,
        "running": running_process is not None and running_process.returncode is None,
    }


# --- FastAPI ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_error_response(message, code=400):
    """Return a one-shot SSE error response that the UI can render."""
    async def stream():
        yield f"data: {json.dumps({'type': 'error', 'text': message, 'code': code})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
async def root():
    ui_path = os.path.join(os.path.dirname(__file__), "..", "nanochat_mlx", "quickstart_ui.html")
    ui_path = os.path.normpath(ui_path)
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/status")
async def status():
    return check_status()


def preflight_stage(stage: str, depth: int, step: int):
    """Validate stage prerequisites before launching subprocesses."""
    if stage == "tokenizer":
        require_training_data()
    elif stage == "train":
        require_training_data()
        require_tokenizer()
    elif stage == "sft":
        require_checkpoint(depth=depth, source="base", step=step if step > 0 else None)
        require_tokenizer()
    elif stage == "import" and importlib.util.find_spec("torch") is None:
        raise SetupError(
            "HuggingFace import requires the optional convert dependencies. "
            "Install them first with: uv sync --extra convert"
        )


@app.get("/run/{stage}")
async def run_stage(stage: str, n_shards: int = 4, depth: int = 4,
                    step: int = -1,
                    num_iterations: int = -1, use_simple_adamw: bool = False,
                    window_pattern: str = "L", max_seq_len: int = 512,
                    device_batch_size: int = 1, save_every: int = -1,
                    eval_every: int = 100, memory_limit_gb: float = 0,
                    repo: str = "nanochat-students/base-d20",
                    force_tokenizer: bool = False,
                    skip_verify: bool = False):
    """Run a pipeline stage as a subprocess, streaming output via SSE."""
    global running_process

    if running_process is not None and running_process.returncode is None:
        raise HTTPException(status_code=409, detail="A process is already running")

    # Default to server's memory limit if not overridden
    if memory_limit_gb <= 0:
        memory_limit_gb = args.memory_limit_gb

    python = sys.executable

    try:
        if stage == "data":
            cmd = [python, "-m", "nanochat_mlx.dataset", "-n", str(n_shards)]
        elif stage == "tokenizer":
            preflight_stage(stage, depth, step)
            cmd = [python, "-m", "scripts.tok_train"]
        elif stage == "train":
            preflight_stage(stage, depth, step)
            cmd = [python, "-m", "scripts.train",
                   f"--depth={depth}",
                   f"--max-seq-len={max_seq_len}",
                   f"--window-pattern={window_pattern}",
                   f"--device-batch-size={device_batch_size}",
                   f"--memory-limit-gb={memory_limit_gb}",
                   f"--eval-every={eval_every}"]
            if num_iterations > 0:
                cmd.append(f"--num-iterations={num_iterations}")
            effective_save_every = save_every if save_every > 0 else 500
            cmd.append(f"--save-every={effective_save_every}")
            if use_simple_adamw:
                cmd.append("--use-simple-adamw")
        elif stage == "sft":
            preflight_stage(stage, depth, step)
            cmd = [python, "-m", "scripts.sft",
                   f"--depth={depth}",
                   f"--device-batch-size={device_batch_size}",
                   f"--memory-limit-gb={memory_limit_gb}",
                   f"--eval-every={eval_every}"]
            if step > 0:
                cmd.append(f"--step={step}")
            if num_iterations > 0:
                cmd.append(f"--num-iterations={num_iterations}")
            effective_save_every = save_every if save_every > 0 else 500
            cmd.append(f"--save-every={effective_save_every}")
        elif stage == "import":
            preflight_stage(stage, depth, step)
            cmd = [python, "-m", "scripts.convert_from_hf",
                   f"--repo={repo}",
                   f"--memory-limit-gb={memory_limit_gb}"]
            if force_tokenizer:
                cmd.append("--force")
            if skip_verify:
                cmd.append("--skip-verify")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")
    except SetupError as exc:
        return sse_error_response(str(exc))

    def _low_priority():
        """Set subprocess to low CPU/IO priority so it doesn't starve other apps."""
        try:
            os.nice(10)  # lower priority (higher nice value)
        except OSError:
            pass

    async def stream():
        global running_process
        try:
            running_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                preexec_fn=_low_priority,
            )

            async for line_bytes in running_process.stdout:
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")

                # Try to parse training metrics
                m = METRIC_RE.search(line)
                if m:
                    yield f"data: {json.dumps({'type': 'metric', 'step': int(m.group(1)), 'total': int(m.group(2)), 'loss': float(m.group(3)), 'tok_per_sec': int(m.group(4).replace(',', ''))})}\n\n"

                yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"

            await running_process.wait()
            code = running_process.returncode
            if code == 0:
                yield f"data: {json.dumps({'type': 'done', 'code': 0})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Process exited with code {code}', 'code': code})}\n\n"

        except asyncio.CancelledError:
            if running_process and running_process.returncode is None:
                running_process.terminate()
            yield f"data: {json.dumps({'type': 'error', 'text': 'Cancelled'})}\n\n"
        finally:
            running_process = None

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/stop")
async def stop():
    global running_process
    if running_process is None or running_process.returncode is not None:
        return {"status": "no_process"}
    running_process.terminate()
    try:
        await asyncio.wait_for(running_process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        running_process.kill()
    running_process = None
    return {"status": "stopped"}


@app.get("/checkpoints")
async def list_checkpoints():
    base = get_base_dir()
    ckpt_base = os.path.join(base, "mlx_checkpoints")
    results = []
    if not os.path.isdir(ckpt_base):
        return results
    for d in sorted(os.listdir(ckpt_base)):
        if not d.startswith("d"):
            continue
        is_sft = d.endswith("_sft")
        depth_str = d[1:].replace("_sft", "") if is_sft else d[1:]
        if not depth_str.isdigit():
            continue
        depth = int(depth_str)
        source = "sft" if is_sft else "base"
        dpath = os.path.join(ckpt_base, d)
        for f in sorted(os.listdir(dpath)):
            if f.endswith("_meta.json"):
                meta_path = os.path.join(dpath, f)
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    mtime = os.path.getmtime(meta_path)
                    results.append({
                        "depth": depth,
                        "step": meta.get("step", 0),
                        "n_embd": meta.get("n_embd", 0),
                        "n_head": meta.get("n_head", 0),
                        "sequence_len": meta.get("sequence_len", 0),
                        "window_pattern": meta.get("window_pattern", "L"),
                        "source": source,
                        "date": mtime,
                    })
                except Exception:
                    pass
    return results


class LoadRequest(BaseModel):
    depth: int = 12
    step: Optional[int] = None
    source: str = "base"


def _unload_model():
    """Free the currently loaded chat model and reclaim memory."""
    global loaded_engine, loaded_tokenizer, loaded_model, loaded_depth, loaded_step, loaded_source
    loaded_engine = None
    loaded_tokenizer = None
    loaded_model = None
    loaded_depth = None
    loaded_step = None
    loaded_source = None
    gc.collect()


@app.post("/chat/load")
async def chat_load(req: LoadRequest):
    global loaded_engine, loaded_tokenizer, loaded_model, loaded_depth, loaded_step, loaded_source

    # Free previous model first to avoid double memory usage
    if loaded_model is not None:
        _unload_model()

    from nanochat_mlx.common import set_memory_limit
    set_memory_limit(args.memory_limit_gb)

    from scripts.chat import load_model
    from nanochat_mlx.tokenizer import get_tokenizer
    from nanochat_mlx.engine import Engine

    try:
        model = load_model(depth=req.depth, step=req.step, source=req.source)
        tokenizer = get_tokenizer()
        loaded_engine = Engine(model, tokenizer)
        loaded_tokenizer = tokenizer
        loaded_model = model
        loaded_depth = req.depth
        loaded_step = req.step
        loaded_source = req.source
        return {"status": "loaded", "depth": req.depth, "source": req.source}
    except SetupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat/unload")
async def chat_unload():
    """Unload the chat model to free memory."""
    if loaded_model is None:
        return {"status": "no_model"}
    _unload_model()
    return {"status": "unloaded"}


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.8
    max_tokens: int = 256
    top_k: int = 50
    repetition_penalty: float = 1.0


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if loaded_engine is None or loaded_tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. POST /chat/load first.")

    tokenizer = loaded_tokenizer
    engine = loaded_engine
    bos_id = tokenizer.get_bos_token_id()

    # Build conversation tokens
    try:
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        has_special = True
    except Exception:
        has_special = False

    tokens = [bos_id]
    if has_special:
        for msg in request.messages:
            if msg.role == "user":
                tokens.append(user_start)
                tokens.extend(tokenizer.encode(msg.content))
                tokens.append(user_end)
            elif msg.role == "assistant":
                tokens.append(assistant_start)
                tokens.extend(tokenizer.encode(msg.content))
                tokens.append(assistant_end)
        tokens.append(assistant_start)
    else:
        # Fallback: plain text
        for msg in request.messages:
            tokens.extend(tokenizer.encode(msg.content))

    async def stream():
        import random
        accumulated = []
        last_clean = ""
        for token_column, token_masks in engine.generate(
            tokens, num_samples=1,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            seed=random.randint(0, 2**31 - 1),
        ):
            tok = token_column[0]
            if has_special and (tok == assistant_end or tok == bos_id):
                break
            accumulated.append(tok)
            text = tokenizer.decode(accumulated)
            if not text.endswith("\ufffd"):
                new = text[len(last_clean):]
                if new:
                    yield f"data: {json.dumps({'token': new}, ensure_ascii=False)}\n\n"
                    last_clean = text
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def main(argv=None):
    global args
    args = build_parser().parse_args(argv)
    import uvicorn
    print(f"NanoChat MLX Quickstart → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
