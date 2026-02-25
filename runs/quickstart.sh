#!/bin/bash
# nanochat-mlx quickstart — full pipeline in one script
set -e

echo "=== nanochat-mlx quickstart ==="
echo ""

# Step 1: Download data (8 shards, ~800MB)
echo "Step 1/5: Downloading data..."
python -m nanochat_mlx.dataset -n 8

# Step 2: Train tokenizer
echo "Step 2/5: Training tokenizer..."
python -m scripts.tok_train

# Step 3: Train base model (depth=4 for quick test)
echo "Step 3/5: Training base model (depth=4)..."
python -m scripts.train --depth=4

# Step 4: SFT
echo "Step 4/5: Fine-tuning (SFT)..."
python -m scripts.sft --depth=4

# Step 5: Chat
echo "Step 5/5: Starting chat..."
python -m scripts.chat --depth=4 --source=sft --interactive
