"""
Download a pretrained nanochat model from HuggingFace and convert it for MLX.

Downloads the PyTorch checkpoint, converts weights to MLX safetensors format,
installs the bundled tokenizer, and verifies the result loads correctly.

Usage:
    python -m scripts.convert_from_hf                                    # Default: d20
    python -m scripts.convert_from_hf --repo karpathy/nanochat-d34       # d34
    python -m scripts.convert_from_hf --force                            # Overwrite existing tokenizer
    python -m scripts.convert_from_hf --skip-verify                      # Skip verification
"""

import argparse
import importlib
import json
import os
import re
import shutil

import numpy as np
import mlx.core as mx
from huggingface_hub import hf_hub_download, list_repo_files

from nanochat_mlx.common import SetupError, get_base_dir, print_setup_error


def require_torch():
    """Import torch only when conversion is actually requested."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise SetupError(
            "HuggingFace import requires the optional convert dependencies. "
            "Install them first with: uv sync --extra convert"
        ) from exc


def resolve_files(repo):
    """Find model, meta, and tokenizer files in the HF repo."""
    files = list_repo_files(repo)
    model_file = None
    meta_file = None
    step = None
    for filename in files:
        model_match = re.match(r"model_(\d+)\.pt$", filename)
        if model_match:
            model_file = filename
            step = int(model_match.group(1))
        meta_match = re.match(r"meta_(\d+)\.json$", filename)
        if meta_match:
            meta_file = filename
    assert model_file, f"No model_*.pt found in {repo}. Files: {files}"
    assert meta_file, f"No meta_*.json found in {repo}. Files: {files}"
    has_tokenizer = "tokenizer.pkl" in files and "token_bytes.pt" in files
    return model_file, meta_file, step, has_tokenizer


def install_tokenizer(repo, force=False, torch_module=None):
    """Download and install the HF model's tokenizer to ~/.cache/nanochat/tokenizer/."""
    base_dir = get_base_dir()
    tok_dir = os.path.join(base_dir, "tokenizer")
    pkl_dst = os.path.join(tok_dir, "tokenizer.pkl")

    if os.path.exists(pkl_dst) and not force:
        print(f"Tokenizer already exists at {tok_dir}, skipping (use --force to overwrite)")
        return

    print("Downloading tokenizer from HuggingFace...")
    pkl_src = hf_hub_download(repo, "tokenizer.pkl")
    tb_src = hf_hub_download(repo, "token_bytes.pt")

    os.makedirs(tok_dir, exist_ok=True)
    shutil.copy2(pkl_src, pkl_dst)
    shutil.copy2(tb_src, os.path.join(tok_dir, "token_bytes.pt"))

    if torch_module is not None:
        token_bytes = torch_module.load(tb_src, map_location="cpu", weights_only=True)
        np.save(os.path.join(tok_dir, "token_bytes.npy"), token_bytes.numpy().astype(np.int32))

    print(f"Installed tokenizer to {tok_dir}")


def convert_state_dict(pt_path, torch_module):
    """Load a PyTorch checkpoint and convert to MLX-compatible dict."""
    print(f"Loading PyTorch checkpoint ({os.path.getsize(pt_path) / 1e9:.2f} GB)...")
    state = torch_module.load(pt_path, map_location="cpu", weights_only=True)

    mlx_weights = {}
    for key, tensor in state.items():
        key = key.removeprefix("_orig_mod.")
        key = key.replace("transformer.wte.", "wte.")
        key = key.replace("transformer.h.", "blocks.")
        if key.endswith(".cos") or key.endswith(".sin"):
            continue
        arr = tensor.float().numpy().astype(np.float16)
        mlx_weights[key] = mx.array(arr)

    print(f"Converted {len(mlx_weights)} weight tensors")
    return mlx_weights


def save_mlx_checkpoint(weights, meta, depth, step):
    """Save converted weights and metadata in MLX checkpoint format."""
    base_dir = get_base_dir()
    ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{depth}")
    os.makedirs(ckpt_dir, exist_ok=True)

    weights_path = os.path.join(ckpt_dir, f"step_{step:06d}.safetensors")
    meta_path = os.path.join(ckpt_dir, f"step_{step:06d}_meta.json")

    mx.save_safetensors(weights_path, weights)
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"Saved MLX checkpoint to {ckpt_dir}")
    print(f"  weights: {weights_path} ({size_mb:.1f} MB)")
    print(f"  metadata: {meta_path}")
    return ckpt_dir


def verify(depth, step, memory_limit_gb):
    """Verify the converted checkpoint loads and generates text."""
    from nanochat_mlx.common import set_memory_limit

    set_memory_limit(memory_limit_gb)

    print("\n--- Verification ---")
    from nanochat_mlx.sft import _find_latest_checkpoint
    from nanochat_mlx.tokenizer import get_tokenizer
    from scripts.chat import load_model

    model = load_model(depth=depth, step=step)
    print("  [OK] Model loaded successfully")

    counts = model.num_scaling_params()
    print(f"  Parameters: {counts['total']:,}")

    tokenizer = get_tokenizer()
    prompt = "The capital of France is"
    bos_id = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(prompt, prepend=bos_id)
    ids = mx.array([tokens], dtype=mx.int32)
    for _ in range(16):
        logits = model(ids)
        next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        ids = mx.concatenate([ids, next_id], axis=1)
        mx.eval(ids)
    output = tokenizer.decode(ids[0].tolist())
    print(f"  Greedy generation: {output}")

    base_dir = get_base_dir()
    ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{depth}")
    _, _, meta = _find_latest_checkpoint(ckpt_dir)
    assert meta is not None, "SFT cannot find checkpoint"
    print(f"  [OK] SFT can discover checkpoint (step={meta['step']})")
    print("--- Verification passed ---\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Download & convert HuggingFace nanochat model to MLX")
    parser.add_argument("--repo", type=str, default="nanochat-students/base-d20",
                        help="HuggingFace repo (default: nanochat-students/base-d20)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Model depth (auto-detected from metadata if not set)")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (auto-detected from filename if not set)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing tokenizer")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification after conversion")
    parser.add_argument("--memory-limit-gb", type=float, default=16.0,
                        help="MLX memory limit for verification (default: 16)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch_module = require_torch()

    print(f"Converting {args.repo} to MLX format\n")

    model_file, meta_file, auto_step, has_tokenizer = resolve_files(args.repo)
    step = args.step if args.step is not None else auto_step
    print(f"Found: {model_file}, {meta_file} (step={step})")

    print("\nDownloading from HuggingFace...")
    model_path = hf_hub_download(args.repo, model_file)
    meta_path = hf_hub_download(args.repo, meta_file)
    print(f"  model: {model_path}")
    print(f"  meta:  {meta_path}")

    if has_tokenizer:
        install_tokenizer(args.repo, force=args.force, torch_module=torch_module)
    else:
        print("No tokenizer files in repo, using local tokenizer")

    with open(meta_path) as handle:
        pt_meta = json.load(handle)
    model_config = pt_meta["model_config"]
    depth = args.depth if args.depth is not None else model_config["n_layer"]
    window_pattern = model_config.get("window_pattern", "L")

    print(
        f"\nModel config: depth={depth}, n_embd={model_config['n_embd']}, "
        f"vocab_size={model_config['vocab_size']}, window={window_pattern}"
    )

    mlx_weights = convert_state_dict(model_path, torch_module)

    mlx_meta = {
        "step": step,
        "depth": depth,
        "n_embd": model_config["n_embd"],
        "n_head": model_config["n_head"],
        "n_kv_head": model_config["n_kv_head"],
        "vocab_size": model_config["vocab_size"],
        "sequence_len": model_config["sequence_len"],
        "window_pattern": window_pattern,
    }

    save_mlx_checkpoint(mlx_weights, mlx_meta, depth, step)

    if not args.skip_verify:
        verify(depth, step, args.memory_limit_gb)
    else:
        print("\nSkipping verification (--skip-verify)")

    print("Done!")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as exc:
        print_setup_error(exc)
        raise SystemExit(2)
