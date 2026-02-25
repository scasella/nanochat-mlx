"""
Evaluation utilities for MLX models.
Port of nanochat/loss_eval.py — bits per byte (BPB) evaluation.
"""

import math
import mlx.core as mx
import mlx.nn as nn


def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Evaluate bits per byte (BPB) — tokenization-independent loss metric.

    Args:
        model: GPT model
        batches: iterator yielding (inputs, targets) as mx.array
        steps: number of eval steps
        token_bytes: mx.array of shape (vocab_size,), bytes per token id (0 for special tokens)
    """
    total_nats = 0.0
    total_bytes = 0

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)

        # Forward pass to get logits
        logits = model(x)  # (B, T, vocab_size)

        # Compute per-token cross-entropy
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        y_flat = y.reshape(-1)

        # Handle ignore_index=-1
        valid = y_flat >= 0
        y_safe = mx.where(valid, y_flat, mx.zeros_like(y_flat))
        ce = nn.losses.cross_entropy(logits_flat, y_safe, reduction="none")  # (B*T,)

        # Map targets to byte lengths
        num_bytes = mx.take(token_bytes, y_safe, axis=0)  # (B*T,)
        num_bytes = mx.where(valid, num_bytes, mx.zeros_like(num_bytes))

        # Only count tokens that have positive byte length
        count_mask = num_bytes > 0
        step_nats = mx.sum(ce * count_mask).item()
        step_bytes = mx.sum(num_bytes).item()

        total_nats += step_nats
        total_bytes += int(step_bytes)

    if total_bytes == 0:
        return float("inf")

    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
