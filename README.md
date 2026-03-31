# nanochat-mlx

Train your own ChatGPT on Apple Silicon. Minimal MLX port of [Karpathy's nanochat](https://github.com/karpathy/nanochat).

## Why I Built This

I wanted to train a chatbot from scratch on my MacBook without touching PyTorch or a cloud GPU. This is a full MLX port of Karpathy's nanochat — one `--depth` dial controls everything from model size to training duration. The whole pipeline runs on Apple Silicon: data download, tokenizer training, pretraining, fine-tuning, and chat.

## What is this?

A self-contained MLX port of nanochat that runs entirely on Apple Silicon. One complexity dial (`--depth`) controls everything: model size, learning rate, batch size, and training duration. The full pipeline goes from raw data download to a working chatbot -- no PyTorch required.

- Single complexity dial: `--depth` sets all hyperparameters automatically
- Full pipeline: data download, tokenizer training, pretraining, SFT, chat, evaluation
- Web GUI wizard: `python -m scripts.quickstart` walks you through everything
- No PyTorch dependency (unless importing pretrained checkpoints)

## Quick Start

```bash
git clone https://github.com/scasella/nanochat-mlx.git
cd nanochat-mlx
uv sync
source .venv/bin/activate
python -m scripts.quickstart
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The wizard walks you through downloading data, training a tokenizer, training a model, and chatting with it.

## Import a Pretrained Model

Skip training entirely by importing a pretrained model from HuggingFace:

```bash
uv sync --extra convert   # Adds torch dependency for checkpoint conversion
python -m scripts.convert_from_hf --repo nanochat-students/base-d20
```

Or use the GUI: run `python -m scripts.quickstart`, then click "Import from HuggingFace" in the training step.

## Full Pipeline (CLI)

Run each step manually for full control:

```bash
# 1. Download data (minimum 2 shards so train/val split exists)
python -m nanochat_mlx.dataset -n 2

# 2. Train BPE tokenizer (vocab size 32768)
python -m scripts.tok_train

# 3. Train base model (depth=4 for a quick test)
python -m scripts.train --depth=4

# 4. Supervised fine-tuning
python -m scripts.sft --depth=4

# 5. Chat with your model
python -m scripts.chat --depth=4 --source=sft --interactive

# 6. Evaluate
python -m scripts.chat_eval --depth=4
```

Or run everything at once with the quickstart script:

```bash
bash runs/quickstart.sh
```

## The Depth Dial

The `--depth` parameter is the single complexity dial. All other hyperparameters (width, heads, batch size, learning rate, training tokens) are auto-computed from depth via scaling laws.

| Depth | Params | Time (M3 Pro) | Use case |
|-------|--------|---------------|----------|
| 4 | ~5M | ~1 min | Quick test, debugging |
| 12 | ~125M | ~1 hour | Reasonable quality |
| 20 | ~350M | ~8 hours | Good quality |
| 26 | ~600M | ~24 hours | GPT-2 reproduction |

The "miniseries principle" requires any architectural change to work across all depths.

## Hardware Requirements

Apple Silicon is required (M1, M2, M3, M4 -- any variant).

Recommended RAM by depth:

- **8 GB** -- depth 4 (quick tests and debugging)
- **16 GB** -- depth 12 (reasonable quality training)
- **32 GB+** -- depth 20 and above (good to full quality)

For a 24 GB Apple Silicon laptop, the recommended smoke-test path is depth 4 with low memory caps and 2-4 shards.

## Project Structure

```
nanochat_mlx/          Core MLX modules
  gpt.py               GPT transformer model
  optim.py             Muon+AdamW optimizer
  engine.py            Inference with KV cache
  train.py             Training loop
  sft.py               SFT pipeline
  eval.py              BPB evaluation
  dataloader.py        BOS-aligned best-fit packing
  sft_dataloader.py    SFT conversation packing
  dataset.py           Data download and iteration
  tokenizer.py         BPE tokenizer
  common.py            Memory management, utilities
scripts/               Entry points
  quickstart.py        Web GUI wizard
  train.py             Training CLI
  sft.py               SFT CLI
  chat.py              Chat CLI
  chat_eval.py         Evaluation CLI
  tok_train.py         Tokenizer training
  convert_from_hf.py   HuggingFace checkpoint import
tasks/                 Eval tasks (ARC, MMLU, GSM8K, etc.)
tests/                 Test suite
runs/                  Shell scripts
```

## Tests

```bash
python -m pytest tests/ -v                    # All tests
python -m pytest tests/ -v -m "not slow"      # Skip slow tests
```

Tests use mock classes to avoid loading real models. MLX-specific tests are skipped when `mlx` is not installed.

## Attribution

This is a community MLX port of [Karpathy's nanochat](https://github.com/karpathy/nanochat), focused on making the MLX pipeline standalone and easy to use on Apple Silicon. All credit for the original architecture, training recipes, and scaling law insights goes to the nanochat project and its contributors.
