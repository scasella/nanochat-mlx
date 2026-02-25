"""
Supervised fine-tuning (SFT) for MLX.

Full fine-tuning (no LoRA, no frozen layers) on a mixture of conversation tasks.
Port of scripts/chat_sft.py training loop, adapted for MLX single-device.
"""

import os
import json
import time
import urllib.request

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from nanochat_mlx.gpt import GPT, GPTConfig, loss_fn
from nanochat_mlx.common import print0, get_base_dir, get_active_memory_mb, get_peak_memory_mb
from nanochat_mlx.optim import (
    MultiOptimizer, OptimizerConfig, setup_optimizer,
    get_muon_momentum,
)
from nanochat_mlx.train import (
    _load_weights_into_model, _save_optimizer_state, _load_optimizer_state,
)
from nanochat_mlx.sft_dataloader import sft_dataloader_bos_bestfit

IDENTITY_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"


def _ensure_identity_conversations(filepath):
    """Download identity conversations if not present."""
    if os.path.exists(filepath):
        return
    print0(f"Downloading identity conversations to {filepath}...")
    urllib.request.urlretrieve(IDENTITY_URL, filepath)
    print0("Done.")


def _find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint step in a directory."""
    if not os.path.isdir(ckpt_dir):
        return None, None, None
    safetensor_files = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".safetensors") and not f.endswith("_optim.safetensors")
    ])
    if not safetensor_files:
        return None, None, None
    latest = safetensor_files[-1]
    weights_path = os.path.join(ckpt_dir, latest)
    meta_path = weights_path.replace(".safetensors", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return weights_path, meta_path, meta


def get_sft_lr_multiplier(progress, warmup_ratio=0.0, warmdown_ratio=0.5, final_lr_frac=0.0):
    """Progress-based LR schedule: linear warmup -> constant -> linear warmdown."""
    if progress < warmup_ratio:
        return (progress + 1e-8) / warmup_ratio
    elif progress <= 1.0 - warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
        return (1 - decay) * 1.0 + decay * final_lr_frac


def sft(args):
    """Run SFT training on a pretrained MLX base model."""
    from nanochat_mlx.tokenizer import get_tokenizer
    from nanochat_mlx.common import set_memory_limit

    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk
    from tasks.customjson import CustomJSON
    from tasks.spellingbee import SimpleSpelling, SpellingBee

    # Memory limit
    set_memory_limit(args.memory_limit_gb)

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # --- Load pretrained base checkpoint ---
    base_dir = get_base_dir()
    base_ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{args.depth}")

    if args.step is not None:
        weights_path = os.path.join(base_ckpt_dir, f"step_{args.step:06d}.safetensors")
        meta_path = os.path.join(base_ckpt_dir, f"step_{args.step:06d}_meta.json")
        assert os.path.exists(weights_path), f"Checkpoint not found: {weights_path}"
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        weights_path, meta_path, meta = _find_latest_checkpoint(base_ckpt_dir)
        assert meta is not None, f"No base checkpoint found in {base_ckpt_dir}"

    print0(f"Loading base checkpoint: {weights_path}")
    print0(f"  depth={meta['depth']}, n_embd={meta['n_embd']}, step={meta['step']}")

    # Build model from checkpoint metadata
    window_pattern = args.window_pattern if args.window_pattern else meta.get("window_pattern", "L")
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=meta["vocab_size"],
        n_layer=meta["depth"],
        n_head=meta["n_head"],
        n_kv_head=meta["n_kv_head"],
        n_embd=meta["n_embd"],
        window_pattern=window_pattern,
    )
    model = GPT(config)
    _load_weights_into_model(model, weights_path)
    mx.eval(model.parameters())
    print0(f"Model loaded: depth={config.n_layer}, n_embd={config.n_embd}, window={config.window_pattern}")

    # --- SFT data mixture ---
    identity_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    _ensure_identity_conversations(identity_filepath)

    train_tasks = [
        SmolTalk(split="train"),
        CustomJSON(filepath=identity_filepath),
        CustomJSON(filepath=identity_filepath),  # 2 epochs of identity
        *[MMLU(subset="auxiliary_train", split="train") for _ in range(args.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ]
    train_dataset = TaskMixture(train_tasks)
    print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")

    val_dataset = TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ])
    print0(f"Validation mixture: {len(val_dataset):,} rows")

    # --- Batch sizes ---
    B = args.device_batch_size
    T = args.max_seq_len
    tokens_per_step = B * T
    grad_accum_steps = max(1, args.total_batch_size // tokens_per_step) if args.total_batch_size > 0 else 1
    total_batch_size = grad_accum_steps * tokens_per_step
    print0(f"Batch: {B}x{T} = {tokens_per_step:,} tok/step, grad_accum={grad_accum_steps}, total={total_batch_size:,}")

    # --- Optimizer (weight_decay=0.0 for SFT) ---
    n_embd = config.n_embd
    opt_config = OptimizerConfig(
        n_embd=n_embd,
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=0.0,  # pretraining decayed to zero, SFT continues with zero
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        scalar_lr=args.scalar_lr,
    )
    optimizer = setup_optimizer(model, opt_config)

    # --- Warm-start optimizer from pretrained checkpoint ---
    if args.load_optimizer:
        opt_path = weights_path.replace(".safetensors", "_optim.safetensors")
        if os.path.exists(opt_path):
            _load_optimizer_state(optimizer, opt_path)
            print0("Loaded optimizer state from pretrained (momentum buffers, LRs preserved)")
        else:
            print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer")

    # Apply init_lr_frac scaling
    for path in optimizer.initial_lrs:
        optimizer.initial_lrs[path] *= args.init_lr_frac
        optimizer.param_config[path]["lr"] = optimizer.initial_lrs[path]
    print0(f"Initial LR fraction: {args.init_lr_frac}")

    # --- Dataloader ---
    train_loader = sft_dataloader_bos_bestfit(
        train_dataset, tokenizer, B, T,
        num_iterations=args.num_iterations,
    )

    # --- Loss + grad function ---
    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    # --- Training loop ---
    sft_ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{args.depth}_sft")
    os.makedirs(sft_ckpt_dir, exist_ok=True)

    print0(f"\nStarting SFT training...")
    step = 0
    last_step = False
    progress = 0.0
    smooth_loss = 0.0
    total_training_time = 0.0
    ema_beta = 0.9

    while True:
        # --- Evaluation ---
        if last_step or (args.eval_every > 0 and step % args.eval_every == 0 and step > 0):
            val_loader = sft_dataloader_bos_bestfit(val_dataset, tokenizer, B, T)
            val_loss = 0.0
            eval_steps = min(args.eval_steps, 20)
            for eval_step in range(eval_steps):
                try:
                    vx, vy, _, _ = next(val_loader)
                except StopIteration:
                    break
                vl = model(vx, targets=vy)
                mx.eval(vl)
                val_loss += vl.item()
            if eval_steps > 0:
                val_loss /= eval_steps
                print0(f"Step {step:05d} | Val loss: {val_loss:.4f}")

        # --- Save checkpoint ---
        should_save = last_step and step > 0
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            should_save = True

        if should_save:
            weights_save_path = os.path.join(sft_ckpt_dir, f"step_{step:06d}.safetensors")
            model.save_weights(weights_save_path)

            opt_save_path = os.path.join(sft_ckpt_dir, f"step_{step:06d}_optim.safetensors")
            _save_optimizer_state(optimizer, opt_save_path)

            sft_meta = {
                "step": step,
                "depth": args.depth,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "n_kv_head": config.n_kv_head,
                "vocab_size": config.vocab_size,
                "sequence_len": config.sequence_len,
                "window_pattern": config.window_pattern,
                "source": "sft",
                "base_checkpoint": weights_path,
            }
            sft_meta_path = os.path.join(sft_ckpt_dir, f"step_{step:06d}_meta.json")
            with open(sft_meta_path, "w") as f:
                json.dump(sft_meta, f, indent=2)
            print0(f"Saved SFT checkpoint to {sft_ckpt_dir}")

        if last_step:
            break

        # --- Training step ---
        t0 = time.time()
        accum_loss = 0.0
        accum_grads = None

        for micro in range(grad_accum_steps):
            try:
                inputs, targets, is_last, data_progress = next(train_loader)
            except StopIteration:
                last_step = True
                break
            progress = max(progress, data_progress)
            if is_last:
                last_step = True

            loss, grads = loss_grad_fn(model, inputs, targets)

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

            mx.eval(loss, accum_grads)
            accum_loss += loss.item()

        if accum_grads is None:
            # No data was consumed this step (dataset exhausted before first micro-batch)
            last_step = True
            continue

        # Average gradients
        if grad_accum_steps > 1:
            accum_grads = tree_map(lambda g: g * (1.0 / grad_accum_steps), accum_grads)

        # LR schedule (progress-based)
        lrm = get_sft_lr_multiplier(progress, args.warmup_ratio, args.warmdown_ratio, args.final_lr_frac)
        optimizer.set_lr_multiplier(lrm)
        optimizer.set_muon_momentum(get_muon_momentum(step))
        optimizer.update(model, accum_grads)
        mx.eval(model.parameters(), *optimizer.state)

        t1 = time.time()
        dt = t1 - t0

        # Logging
        step += 1
        micro_count = min(grad_accum_steps, micro + 1) if not last_step else micro + 1
        train_loss = accum_loss / max(micro_count, 1)
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_loss / (1 - ema_beta ** step)

        if step > 5:
            total_training_time += dt

        tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
        mem_mb = get_active_memory_mb()
        pct = 100 * progress

        # Estimate total steps for progress display
        estimated_total = max(int(step / progress), step) if progress > 0.01 else 0

        log_every = max(1, min(100, estimated_total // 50)) if estimated_total > 0 else 1
        if step % log_every == 0 or step <= 5:
            print0(
                f"step {step:05d}/{estimated_total:05d} ({pct:.1f}%) | "
                f"loss: {debiased_loss:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | "
                f"tok/s: {tok_per_sec:,} | mem: {mem_mb:.0f}MB"
            )

    print0(f"\nSFT complete. Total time: {total_training_time/60:.1f}m")
    print0(f"Peak memory: {get_peak_memory_mb():.0f}MB")
    return model
