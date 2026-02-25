"""
SFT (Supervised Fine-Tuning) for MLX.

Fine-tunes a pretrained MLX base model on a mixture of conversation tasks
to produce an instruction-following chat model on Apple Silicon.

Usage:
    python -m scripts.sft --depth=12
    python -m scripts.sft --depth=12 --num-iterations=2000
    python -m scripts.sft --depth=4 --memory-limit-gb=8

Prerequisites:
    python -m scripts.train --depth=12    # Pretrain base model first
"""

import argparse
from nanochat_mlx.sft import sft


parser = argparse.ArgumentParser(description="SFT fine-tuning for MLX")

# Model loading
parser.add_argument("--depth", type=int, required=True, help="Base model depth to load and fine-tune")
parser.add_argument("--step", type=int, default=None, help="Base checkpoint step (default: latest)")
parser.add_argument("--load-optimizer", type=int, default=1, help="Warm-start optimizer from pretrained (0=no, 1=yes)")

# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="Number of steps (-1 = full epoch)")

# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=1, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="Total batch size in tokens (-1 = device_batch_size * max_seq_len)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="Max context length")

# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.3, help="Embedding LR (AdamW)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="Unembedding LR (AdamW)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="Matrix LR (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="Scalar LR (AdamW)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="Initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Warmup ratio")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="Warmdown ratio")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="Final LR fraction")

# Evaluation
parser.add_argument("--eval-every", type=int, default=100, help="Eval every N steps (-1=disable)")
parser.add_argument("--eval-steps", type=int, default=5, help="Number of eval steps")

# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="MMLU epochs in training mixture")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="GSM8K epochs in training mixture")

# Checkpointing
parser.add_argument("--save-every", type=int, default=-1, help="Save every N steps (-1=only at end)")

# System
parser.add_argument("--memory-limit-gb", type=float, default=8.0, help="MLX memory limit in GB")
parser.add_argument("--window-pattern", type=str, default="", help="Attention window pattern (default: from checkpoint)")

args = parser.parse_args()
sft(args)
