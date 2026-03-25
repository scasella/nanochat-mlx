"""
Chat CLI using MLX inference engine.

Usage:
    python -m scripts.chat -p "Why is the sky blue?"
    python -m scripts.chat --interactive
"""

import os
import argparse

import mlx.core as mx

from nanochat_mlx.common import SetupError, print_setup_error
from nanochat_mlx.gpt import GPT, GPTConfig
from nanochat_mlx.preflight import require_checkpoint
from nanochat_mlx.train import _load_weights_into_model
from nanochat_mlx.engine import Engine
from nanochat_mlx.common import print0, get_base_dir, set_memory_limit


def load_model(depth=12, step=None, source="base"):
    """Load a trained MLX model from checkpoint.

    Args:
        depth: model depth
        step: checkpoint step (None = latest)
        source: "base" for pretrained, "sft" for fine-tuned
    """
    base_dir = get_base_dir()
    if source == "sft":
        ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{depth}_sft")
    else:
        ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{depth}")

    weights_path, meta_path = require_checkpoint(depth=depth, source=source, step=step)

    # Load metadata
    import json
    with open(meta_path) as f:
        meta = json.load(f)

    print0(f"Loading checkpoint: {weights_path}")
    print0(f"  depth={meta['depth']}, n_embd={meta['n_embd']}, step={meta['step']}")

    # Build model
    config = GPTConfig(
        sequence_len=meta["sequence_len"],
        vocab_size=meta["vocab_size"],
        n_layer=meta["depth"],
        n_head=meta["n_head"],
        n_kv_head=meta["n_kv_head"],
        n_embd=meta["n_embd"],
        window_pattern=meta.get("window_pattern", "L"),
    )
    model = GPT(config)
    _load_weights_into_model(model, weights_path)
    mx.eval(model.parameters())
    return model


def main():
    parser = argparse.ArgumentParser(description="Chat with MLX model")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="single prompt")
    parser.add_argument("--interactive", action="store_true", help="interactive chat mode")
    parser.add_argument("--depth", type=int, default=12, help="model depth to load")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    parser.add_argument("--source", type=str, default="base", choices=["base", "sft"],
                        help="checkpoint source: base or sft (default: base)")
    parser.add_argument("--max-tokens", type=int, default=256, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
    parser.add_argument("--memory-limit-gb", type=float, default=16.0, help="MLX memory limit")
    args = parser.parse_args()

    set_memory_limit(args.memory_limit_gb)

    # Load model and tokenizer
    from nanochat_mlx.tokenizer import get_tokenizer
    model = load_model(depth=args.depth, step=args.step, source=args.source)
    tokenizer = get_tokenizer()
    engine = Engine(model, tokenizer)
    bos_id = tokenizer.get_bos_token_id()

    def generate_response(prompt):
        tokens = tokenizer.encode(prompt, prepend=bos_id)
        for token_column, token_masks in engine.generate(
            tokens, num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            token = token_column[0]
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
        print()

    if args.prompt:
        generate_response(args.prompt)
    elif args.interactive:
        print0("Interactive chat mode. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue
                print("Assistant: ", end="")
                generate_response(prompt)
                print()
            except (KeyboardInterrupt, EOFError):
                print()
                break
    else:
        # Default prompt
        generate_response("The capital of France is")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as exc:
        print_setup_error(exc)
        raise SystemExit(2)
