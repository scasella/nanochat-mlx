"""
Train a GPT model using MLX on Apple Silicon.

Usage:
    python -m scripts.train --depth=4 --num-iterations=50
    python -m scripts.train --depth=12 --num-iterations=200
    python -m scripts.train --resume-from-step=100 --depth=4

Prerequisites:
    python -m nanochat_mlx.dataset -n 8    # Download data shards
    python -m scripts.tok_train            # Train tokenizer
"""

import argparse

from nanochat_mlx.common import SetupError, print_setup_error
from nanochat_mlx.train import train


def build_parser():
    parser = argparse.ArgumentParser(description="Train GPT model with MLX")

    parser.add_argument("--depth", type=int, default=4, help="Transformer depth (4=tiny, 12=medium, 26=full)")
    parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    parser.add_argument("--head-dim", type=int, default=128, help="target head dimension")
    parser.add_argument("--max-seq-len", type=int, default=512, help="max context length")
    parser.add_argument("--window-pattern", type=str, default="L", help="attention window pattern (L=full, S=half, e.g. SSSL)")

    parser.add_argument("--num-iterations", type=int, default=-1, help="number of training steps (-1=auto)")
    parser.add_argument("--target-param-data-ratio", type=float, default=12.0, help="data:param ratio for auto steps")

    parser.add_argument("--device-batch-size", type=int, default=1, help="per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=512, help="total batch size in tokens (-1=auto)")
    parser.add_argument("--use-simple-adamw", action="store_true", help="use simple AdamW instead of Muon+AdamW")
    parser.add_argument("--simple-lr", type=float, default=3e-4, help="learning rate for simple AdamW")

    parser.add_argument("--embedding-lr", type=float, default=0.3, help="embedding LR (AdamW)")
    parser.add_argument("--unembedding-lr", type=float, default=0.004, help="unembedding LR (AdamW)")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="matrix LR (Muon)")
    parser.add_argument("--scalar-lr", type=float, default=0.5, help="scalar LR (AdamW)")
    parser.add_argument("--weight-decay", type=float, default=0.2, help="weight decay for Muon")
    parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2")

    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="warmup ratio")
    parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="warmdown ratio")
    parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR fraction")

    parser.add_argument("--eval-every", type=int, default=100, help="eval val loss every N steps (-1=disable)")
    parser.add_argument("--eval-steps", type=int, default=5, help="number of eval steps (for simple loss eval)")
    parser.add_argument("--eval-bpb", action="store_true", help="use BPB evaluation instead of simple loss")
    parser.add_argument("--eval-tokens", type=int, default=0, help="tokens for BPB eval (0=auto from eval-steps)")
    parser.add_argument("--sample-every", type=int, default=200, help="sample text every N steps (-1=disable)")

    parser.add_argument("--save-every", type=int, default=-1, help="save checkpoint every N steps (-1=only at end)")
    parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1=disable)")

    parser.add_argument("--memory-limit-gb", type=float, default=16.0, help="MLX memory limit in GB")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.num_iterations == -1 and args.depth <= 4:
        args.num_iterations = 50
        print(f"Defaulting to {args.num_iterations} iterations for depth={args.depth}")

    try:
        train(args)
        return 0
    except SetupError as exc:
        print_setup_error(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
