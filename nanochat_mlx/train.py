"""
Training loop for MLX.

Supports both simple AdamW and full Muon+AdamW multi-optimizer.
Features: BPB evaluation, checkpoint resume, periodic saving.
"""

import os
import json
import time
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

from nanochat_mlx.gpt import GPT, GPTConfig, loss_fn
from nanochat_mlx.common import print0, get_base_dir, get_active_memory_mb, get_peak_memory_mb
from nanochat_mlx.optim import (
    MultiOptimizer, OptimizerConfig, setup_optimizer,
    get_lr_multiplier, get_muon_momentum, get_weight_decay,
)


def build_model(depth, vocab_size, aspect_ratio=64, head_dim=128, max_seq_len=2048, window_pattern="SSSL"):
    """Build a GPT model for the given depth."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    model = GPT(config)
    model.init_weights()
    mx.eval(model.parameters())
    return model


def _load_weights_into_model(model, weights_path):
    """Load weights from safetensors into model by walking the tree directly."""
    weights = mx.load(weights_path)
    for path, value in weights.items():
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)


def _save_optimizer_state(optimizer, path):
    """Save multi-optimizer state to a safetensors file."""
    state = {}
    # Adam states
    for param_path, s in optimizer.adam_state.items():
        state[f"adam.{param_path}.m"] = s["m"]
        state[f"adam.{param_path}.v"] = s["v"]
        state[f"adam.{param_path}.t"] = mx.array(s["t"], dtype=mx.int32)
    # Muon momentum buffers
    for param_path, buf in optimizer.muon_state.items():
        state[f"muon.{param_path}"] = buf
    mx.save_safetensors(path, state)


def _load_optimizer_state(optimizer, path):
    """Load multi-optimizer state from a safetensors file."""
    state = mx.load(path)
    for key, value in state.items():
        if key.startswith("adam."):
            rest = key[5:]  # strip "adam."
            if rest.endswith(".m"):
                param_path = rest[:-2]
                if param_path not in optimizer.adam_state:
                    optimizer.adam_state[param_path] = {"m": None, "v": None, "t": 0}
                optimizer.adam_state[param_path]["m"] = value
            elif rest.endswith(".v"):
                param_path = rest[:-2]
                if param_path not in optimizer.adam_state:
                    optimizer.adam_state[param_path] = {"m": None, "v": None, "t": 0}
                optimizer.adam_state[param_path]["v"] = value
            elif rest.endswith(".t"):
                param_path = rest[:-2]
                if param_path not in optimizer.adam_state:
                    optimizer.adam_state[param_path] = {"m": None, "v": None, "t": 0}
                optimizer.adam_state[param_path]["t"] = int(value.item())
        elif key.startswith("muon."):
            param_path = key[5:]  # strip "muon."
            optimizer.muon_state[param_path] = value


def train(args):
    """Main training function."""
    from nanochat_mlx.tokenizer import get_tokenizer
    from nanochat_mlx.dataloader import dataloader_bos_bestfit
    from nanochat_mlx.common import set_memory_limit

    # Memory limit
    set_memory_limit(args.memory_limit_gb)

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # Build model
    model = build_model(
        depth=args.depth,
        vocab_size=vocab_size,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        window_pattern=args.window_pattern,
    )
    config = model.config
    print0(f"Model config: depth={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}, window={config.window_pattern}")

    # Parameter counts
    param_counts = model.num_scaling_params()
    print0(f"Parameter counts:")
    for key, value in param_counts.items():
        print0(f"  {key:24s}: {value:,}")

    # --- Scaling laws (ported from base_train.py) ---
    num_scaling_params = param_counts["transformer_matrices"] + param_counts["lm_head"]
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)

    # Reference model for batch size scaling
    ref_depth = 12
    ref_base_dim = ref_depth * args.aspect_ratio
    ref_model_dim = ((ref_base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    ref_num_heads = ref_model_dim // args.head_dim
    ref_config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=ref_depth, n_head=ref_num_heads, n_kv_head=ref_num_heads, n_embd=ref_model_dim,
    )
    ref_model = GPT(ref_config)
    ref_counts = ref_model.num_scaling_params()
    D_REF = args.target_param_data_ratio * (ref_counts["transformer_matrices"] + ref_counts["lm_head"])
    B_REF = 2**19
    del ref_model

    # Auto-compute batch size
    total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = B_REF * batch_size_ratio ** 0.383
        total_batch_size = 2 ** round(math.log2(max(predicted_batch_size, 1)))
        print0(f"Auto-computed batch size: {total_batch_size:,} tokens")

    # Batch LR scaling
    batch_lr_scale = (total_batch_size / B_REF) ** 0.5 if total_batch_size != B_REF else 1.0
    if batch_lr_scale != 1.0:
        print0(f"Batch LR scale: {batch_lr_scale:.4f}")

    # Weight decay scaling
    weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / max(target_tokens, 1))

    # Training iterations
    tokens_per_step = args.device_batch_size * args.max_seq_len
    # Clamp batch size up to at least one step's worth of tokens
    if total_batch_size < tokens_per_step:
        print0(f"Clamping total_batch_size from {total_batch_size} to {tokens_per_step} (device_batch_size * max_seq_len)")
        total_batch_size = tokens_per_step
    # Align to multiple of tokens_per_step
    if total_batch_size % tokens_per_step != 0:
        aligned = ((total_batch_size + tokens_per_step - 1) // tokens_per_step) * tokens_per_step
        print0(f"Aligning total_batch_size from {total_batch_size} to {aligned}")
        total_batch_size = aligned
    grad_accum_steps = total_batch_size // tokens_per_step

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    else:
        num_iterations = target_tokens // total_batch_size
        # Cap for single-device MLX — the scaling law targets multi-GPU throughput
        MAX_AUTO_ITERS = 50000
        if num_iterations > MAX_AUTO_ITERS:
            print0(f"Auto-computed {num_iterations:,} steps (multi-GPU target), capping to {MAX_AUTO_ITERS} for single-device MLX")
            num_iterations = MAX_AUTO_ITERS

    print0(f"Training: {num_iterations} steps, batch={total_batch_size}, grad_accum={grad_accum_steps}")

    # --- Checkpoint directory ---
    base_dir = get_base_dir()
    ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{args.depth}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Resume from checkpoint ---
    resume_step = getattr(args, 'resume_from_step', -1)
    resuming = resume_step > 0
    dataloader_resume_state = None
    loop_state = None

    if resuming:
        meta_path = os.path.join(ckpt_dir, f"step_{resume_step:06d}_meta.json")
        weights_path = os.path.join(ckpt_dir, f"step_{resume_step:06d}.safetensors")
        assert os.path.exists(meta_path), f"Resume meta not found: {meta_path}"
        assert os.path.exists(weights_path), f"Resume weights not found: {weights_path}"

        with open(meta_path) as f:
            resume_meta = json.load(f)
        print0(f"Resuming from step {resume_step}")
        _load_weights_into_model(model, weights_path)
        mx.eval(model.parameters())

        dataloader_resume_state = resume_meta.get("dataloader_state")
        loop_state = resume_meta.get("loop_state")

    # --- Optimizer ---
    if args.use_simple_adamw:
        optimizer = optim.AdamW(learning_rate=args.simple_lr)
        use_multi = False
        print0(f"Using simple AdamW, lr={args.simple_lr}")
    else:
        opt_config = OptimizerConfig(
            n_embd=config.n_embd,
            unembedding_lr=args.unembedding_lr * batch_lr_scale,
            embedding_lr=args.embedding_lr * batch_lr_scale,
            matrix_lr=args.matrix_lr * batch_lr_scale,
            weight_decay=weight_decay_scaled,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            scalar_lr=args.scalar_lr * batch_lr_scale,
        )
        optimizer = setup_optimizer(model, opt_config)
        use_multi = True
        print0("Using Muon + AdamW multi-optimizer")

        # Resume optimizer state
        if resuming:
            opt_path = os.path.join(ckpt_dir, f"step_{resume_step:06d}_optim.safetensors")
            if os.path.exists(opt_path):
                _load_optimizer_state(optimizer, opt_path)
                print0("Loaded optimizer state")

    # --- BPB evaluation setup ---
    token_bytes = None
    eval_bpb = getattr(args, 'eval_bpb', False)
    if eval_bpb:
        token_bytes_path = os.path.join(base_dir, "tokenizer", "token_bytes.npy")
        pt_path = os.path.join(base_dir, "tokenizer", "token_bytes.pt")
        if os.path.exists(token_bytes_path):
            import numpy as np
            tb_np = np.load(token_bytes_path)
            token_bytes = mx.array(tb_np, dtype=mx.float32)
            print0(f"Loaded token_bytes for BPB evaluation ({token_bytes.shape[0]} tokens)")
        elif os.path.exists(pt_path):
            import torch
            tb_torch = torch.load(pt_path, map_location="cpu")
            token_bytes = mx.array(tb_torch.numpy(), dtype=mx.float32)
            print0(f"Loaded token_bytes for BPB evaluation ({token_bytes.shape[0]} tokens) [from .pt]")
        else:
            print0(f"Warning: token_bytes not found, BPB eval disabled")
            eval_bpb = False

    # --- Dataloader ---
    train_loader = dataloader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train",
        resume_state_dict=dataloader_resume_state,
    )

    # --- Loss + grad function ---
    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    # --- Training loop ---
    print0(f"\nStarting training...")
    start_step = resume_step if resuming else 0

    dataloader_state = dataloader_resume_state  # track for checkpoint saves

    if loop_state:
        smooth_loss = loop_state.get("smooth_loss", 0.0)
        total_training_time = loop_state.get("total_training_time", 0.0)
        min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
    else:
        smooth_loss = 0.0
        total_training_time = 0.0
        min_val_bpb = float("inf")

    save_every = getattr(args, 'save_every', -1)

    for step in range(start_step, num_iterations + 1):
        last_step = step == num_iterations

        # --- Evaluation ---
        if last_step or (args.eval_every > 0 and step % args.eval_every == 0 and step > start_step):
            if eval_bpb and token_bytes is not None:
                from nanochat_mlx.dataloader import dataloader_bos_bestfit_no_state
                from nanochat_mlx.eval import evaluate_bpb
                eval_tokens = getattr(args, 'eval_tokens', args.eval_steps * args.device_batch_size * args.max_seq_len)
                eval_steps_bpb = eval_tokens // (args.device_batch_size * args.max_seq_len)
                val_loader = dataloader_bos_bestfit_no_state(
                    tokenizer, args.device_batch_size, args.max_seq_len, split="val"
                )
                bpb = evaluate_bpb(model, val_loader, eval_steps_bpb, token_bytes)
                if bpb < min_val_bpb:
                    min_val_bpb = bpb
                print0(f"Step {step:05d} | Val BPB: {bpb:.6f} (min: {min_val_bpb:.6f})")
            elif args.eval_steps > 0:
                from nanochat_mlx.dataloader import dataloader_bos_bestfit_no_state
                val_loader = dataloader_bos_bestfit_no_state(
                    tokenizer, args.device_batch_size, args.max_seq_len, split="val"
                )
                val_loss = 0.0
                for eval_step in range(args.eval_steps):
                    vx, vy = next(val_loader)
                    vl = model(vx, targets=vy)
                    mx.eval(vl)
                    val_loss += vl.item()
                val_loss /= args.eval_steps
                print0(f"Step {step:05d} | Val loss: {val_loss:.4f}")

        # Sample text
        if args.sample_every > 0 and step > start_step and (last_step or step % args.sample_every == 0):
            prompts = ["The capital of France is", "If 5*x + 3 = 13, then x is"]
            bos_id = tokenizer.get_bos_token_id()
            for prompt in prompts:
                tokens = tokenizer.encode(prompt, prepend=bos_id)
                ids = mx.array([tokens], dtype=mx.int32)
                for _ in range(32):
                    logits = model(ids)
                    next_logit = logits[:, -1, :]
                    next_id = mx.argmax(next_logit, axis=-1, keepdims=True)
                    ids = mx.concatenate([ids, next_id], axis=1)
                    mx.eval(ids)
                print0(tokenizer.decode(ids[0].tolist()))

        # Save checkpoint
        should_save = last_step and step > start_step
        if save_every > 0 and step > start_step and step != start_step and step % save_every == 0:
            should_save = True

        if should_save:
            # Remove previous checkpoint to save disk (keep only latest)
            keep_only_latest = getattr(args, 'keep_only_latest', True)
            if keep_only_latest and not last_step:
                for old_f in os.listdir(ckpt_dir):
                    if old_f.endswith((".safetensors", "_meta.json")):
                        os.remove(os.path.join(ckpt_dir, old_f))

            weights_path = os.path.join(ckpt_dir, f"step_{step:06d}.safetensors")
            model.save_weights(weights_path)

            # Save optimizer state for resume
            if use_multi:
                opt_path = os.path.join(ckpt_dir, f"step_{step:06d}_optim.safetensors")
                _save_optimizer_state(optimizer, opt_path)

            meta_path = os.path.join(ckpt_dir, f"step_{step:06d}_meta.json")
            meta = {
                "step": step,
                "depth": args.depth,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "n_kv_head": config.n_kv_head,
                "vocab_size": config.vocab_size,
                "sequence_len": config.sequence_len,
                "window_pattern": config.window_pattern,
                "dataloader_state": dataloader_state,
                "loop_state": {
                    "smooth_loss": smooth_loss,
                    "total_training_time": total_training_time,
                    "min_val_bpb": min_val_bpb,
                },
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print0(f"Saved checkpoint to {ckpt_dir}")

        if last_step:
            break

        # --- Training step ---
        t0 = time.time()

        # Gradient accumulation
        accum_loss = 0.0
        accum_grads = None

        for micro in range(grad_accum_steps):
            inputs, targets, dataloader_state = next(train_loader)

            loss, grads = loss_grad_fn(model, inputs, targets)

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

            # Evaluate to prevent graph buildup
            mx.eval(loss, accum_grads)
            accum_loss += loss.item()

        # Average gradients
        if grad_accum_steps > 1:
            accum_grads = tree_map(lambda g: g * (1.0 / grad_accum_steps), accum_grads)

        # LR schedule
        if use_multi:
            lrm = get_lr_multiplier(step, num_iterations,
                                     args.warmup_ratio, args.warmdown_ratio, args.final_lr_frac)
            optimizer.set_lr_multiplier(lrm)
            optimizer.set_muon_momentum(get_muon_momentum(step))
            optimizer.set_muon_weight_decay(get_weight_decay(step, num_iterations, weight_decay_scaled))
            optimizer.update(model, accum_grads)
            mx.eval(model.parameters(), *optimizer.state)
        else:
            optimizer.update(model, accum_grads)
            mx.eval(model.parameters(), optimizer.state)

        t1 = time.time()
        dt = t1 - t0

        # Logging
        train_loss = accum_loss / grad_accum_steps
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

        if step > start_step + 5:
            total_training_time += dt

        tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
        mem_mb = get_active_memory_mb()
        pct = 100 * step / num_iterations

        log_every = min(max(1, num_iterations // 50), 100)  # at least every 100 steps
        if step % log_every == 0 or step < start_step + 5:
            print0(
                f"step {step:05d}/{num_iterations:05d} ({pct:.1f}%) | "
                f"loss: {debiased_loss:.4f} | dt: {dt*1000:.0f}ms | "
                f"tok/s: {tok_per_sec:,} | mem: {mem_mb:.0f}MB"
            )

    print0(f"\nTraining complete. Total time: {total_training_time/60:.1f}m")
    print0(f"Peak memory: {get_peak_memory_mb():.0f}MB")
    if min_val_bpb < float("inf"):
        print0(f"Min val BPB: {min_val_bpb:.6f}")
    return model
