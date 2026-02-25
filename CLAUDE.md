# CLAUDE.md

## Project Overview

nanochat-mlx is a self-contained MLX port of Karpathy's nanochat for Apple Silicon. Full pipeline from data download to chat, no PyTorch required.

## Setup

```bash
uv sync                       # Install dependencies
source .venv/bin/activate
```

Python 3.10+. Package manager is `uv` (not pip).

## Common Commands

```bash
# Quick start (web GUI wizard)
python -m scripts.quickstart

# Data
python -m nanochat_mlx.dataset -n 8

# Tokenizer
python -m scripts.tok_train

# Training
python -m scripts.train --depth=4
python -m scripts.train --depth=12 --num-iterations=200

# SFT
python -m scripts.sft --depth=12

# Chat
python -m scripts.chat --depth=12 --source=sft --interactive

# Evaluation
python -m scripts.chat_eval --depth=12

# Import from HuggingFace (requires: uv sync --extra convert)
python -m scripts.convert_from_hf --repo nanochat-students/base-d20
```

## Tests

```bash
python -m pytest tests/ -v
```

## Architecture

Single complexity dial: `--depth` controls everything. Width, heads, batch size, LR, training tokens all auto-computed.

### Key Modules (nanochat_mlx/)

- **gpt.py**: GPT transformer (RoPE, QK-norm, ReLU², GQA, sliding window, logit softcap, value embeddings, per-layer residual scaling)
- **optim.py**: Muon+AdamW multi-optimizer
- **engine.py**: Inference with KV cache, calculator tool use
- **dataloader.py**: BOS-aligned best-fit packing
- **sft_dataloader.py**: SFT conversation packing
- **train.py**: Training loop with gradient accumulation, checkpointing, resume
- **sft.py**: SFT pipeline
- **eval.py**: BPB evaluation
- **common.py**: Memory management, utilities
- **dataset.py**: Data download and iteration
- **tokenizer.py**: BPE tokenizer (RustBPE + tiktoken)

### Key Patterns

- **Explicit `mx.eval()` after every micro-batch**: Prevents computation graph buildup
- **No meta device**: MLX parameters are lazy by default
- **Manual weight loading**: `_load_weights_into_model()` walks model tree via getattr/setattr
- **Memory management**: `set_memory_limit(gb)` caps Metal memory

## Artifacts

Stored under `~/.cache/nanochat/`: data shards, tokenizer, MLX checkpoints (`mlx_checkpoints/d{depth}/`).
