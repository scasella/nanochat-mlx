# nanochat-mlx

Train your own ChatGPT on Apple Silicon. Minimal MLX port of [Karpathy's nanochat](https://github.com/karpathy/nanochat).

## Why I Built This

I wanted to train a chatbot from scratch on my MacBook without touching PyTorch or a cloud GPU. This is a full MLX port of Karpathy's nanochat — one `--depth` dial controls everything from model size to training duration. The whole pipeline runs on Apple Silicon: data download, tokenizer training, pretraining, fine-tuning, and chat.

Check out my [personal site](https://casella.dev) for more projects and research.

## What is this?

A self-contained MLX port of nanochat that runs entirely on Apple Silicon. One complexity dial (`--depth`) controls everything: model size, learning rate, batch size, and training duration. The full pipeline goes from raw data download to a working chatbot -- no PyTorch required.

- Single complexity dial: `--depth` sets all hyperparameters automatically
- Full pipeline: data download, tokenizer training, pretraining, SFT, chat, evaluation
- Web GUI wizard: `uv run python -m scripts.quickstart` walks you through everything
- No PyTorch dependency (unless importing pretrained checkpoints)

## Quick Start

Use either `uv run` for each command, or activate the virtual environment once.

### Option 1: `uv run` (no shell activation)

```bash
git clone https://github.com/scasella/nanochat-mlx.git
cd nanochat-mlx
uv sync
uv run python -m scripts.quickstart
```

### Option 2: activate `.venv`

```bash
git clone https://github.com/scasella/nanochat-mlx.git
cd nanochat-mlx
uv sync
source .venv/bin/activate
python -m scripts.quickstart
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The wizard walks you through downloading data, training a tokenizer, training a model, and chatting with it.

For the rest of the README, use the command style that matches your setup: keep `uv run python -m ...` without shell activation, or drop `uv run` after activating `.venv`.

## Import a Pretrained Model

Skip training entirely by importing a pretrained model from HuggingFace:

```bash
uv sync --extra convert   # Adds torch dependency for checkpoint conversion
uv run python -m scripts.convert_from_hf --repo nanochat-students/base-d20
```

If `.venv` is activated, you can run the same command as `python -m scripts.convert_from_hf --repo nanochat-students/base-d20`.

Or use the GUI: run `uv run python -m scripts.quickstart` or `python -m scripts.quickstart` from an activated `.venv`, then click "Import from HuggingFace" in the training step.

## Full Pipeline (CLI)

Run each step manually for full control:

The examples in this section assume `.venv` is activated. If you are not activating the environment, prefix each `python -m ...` command with `uv run`.

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

Without activating `.venv`, run that script as `uv run bash runs/quickstart.sh`.

## The Depth Dial

The `--depth` parameter is the single complexity dial. All other hyperparameters (width, heads, batch size, learning rate, training tokens) are auto-computed from depth via scaling laws.

| Depth | Params | Time (M3 Pro) | Use case |
|-------|--------|---------------|----------|
| 4 | ~5M | ~1 min | Quick test, debugging |
| 12 | ~125M | ~1 hour | Reasonable quality |
| 20 | ~350M | ~8 hours | Good quality |
| 26 | ~600M | ~24 hours | GPT-2 reproduction |

The "miniseries principle" requires any architectural change to work across all depths.

## Which Mac should I use?

These are practical starting points for people arriving with hardware-compatibility questions first. Actual runtime depends on thermals, data shards, background memory pressure, and whether you are training from scratch or importing a checkpoint.

| Mac | Expected depth | Expected time | Known caveats |
|-----|----------------|---------------|---------------|
| M1 with 8 GB unified memory | Depth 4 | Minutes | Best for smoke tests, debugging, and proving the pipeline works. Keep shard counts low. |
| M1/M2 with 16 GB unified memory | Depth 12 | About 1-2 hours | Good starter path for a real run. Close memory-heavy apps before training. |
| M3/M4 with 24 GB unified memory | Depth 12 comfortably; depth 20 as an overnight run | About 1 hour at depth 12; many hours at depth 20 | Use depth 4 first, then scale. Depth 20 may need smaller batches or fewer competing apps. |
| 32 GB+ Apple Silicon | Depth 20+ | About 8 hours at depth 20; roughly a day at depth 26 | Best fit for full-quality local experiments. Depth 26 is still a long run. |

## Hardware Requirements

Apple Silicon is required (M1, M2, M3, M4 -- any variant).

Recommended RAM by depth:

- **8 GB** -- depth 4 (quick tests and debugging)
- **16 GB** -- depth 12 (reasonable quality training)
- **32 GB+** -- depth 20 and above (good to full quality)

For a 24 GB Apple Silicon laptop, the recommended smoke-test path is depth 4 with low memory caps and 2-4 shards.

## Related Apple Silicon ML projects

- [gemma4-m4-pro](https://github.com/scasella/gemma4-m4-pro) — Gemma 4 on a 24GB MacBook: measured recipes, runtimes, and fallback paths.
- [train-gemma4-sudoku-on-your-macbook](https://github.com/scasella/train-gemma4-sudoku-on-your-macbook) — One-notebook Gemma 4 RL on Apple Silicon.
- [ttt-discover-autoresearch-mlx](https://github.com/scasella/ttt-discover-autoresearch-mlx) — Run RL-driven autoresearch on Apple Silicon.
- [autoresearch-evo](https://github.com/scasella/autoresearch-evo) — Novelty-search-inspired autonomous research loops, run memory, and agentic review.

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

With `.venv` activated:

```bash
python -m pytest tests/ -v                    # All tests
python -m pytest tests/ -v -m "not slow"      # Skip slow tests
```

Without activating `.venv`, prefix those commands with `uv run`.

Tests use mock classes to avoid loading real models. MLX-specific tests are skipped when `mlx` is not installed.

## Attribution

This is a community MLX port of [Karpathy's nanochat](https://github.com/karpathy/nanochat), focused on making the MLX pipeline standalone and easy to use on Apple Silicon. All credit for the original architecture, training recipes, and scaling law insights goes to the nanochat project and its contributors.
