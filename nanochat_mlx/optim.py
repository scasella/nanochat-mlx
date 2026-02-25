"""
Optimizer setup for MLX training.

Muon (Newton-Schulz orthogonalization) for 2D matrix params,
AdamW for embeddings and scalars. Mirrors nanochat/gpt.py:setup_optimizer.

All updates are computed manually and applied via model.update() to avoid
MLX tree structure mismatches when using partial gradient trees.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def newton_schulz(G, steps=5):
    """
    Newton-Schulz iteration to compute the matrix sign function (orthogonalization).
    Matches Keller Jordan's Muon implementation.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (mx.sqrt(mx.sum(G * G)) + 1e-7)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)  # polynomial in A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MultiOptimizer:
    """
    Combined Muon + AdamW optimizer.
    Applies the appropriate update rule per parameter based on its path.
    All updates are computed via flat param/grad lists and applied atomically.
    """

    def __init__(self, model, config):
        self.config = config
        n_embd = config.n_embd

        # muP LR scaling
        dmodel_lr_scale = (n_embd / 768) ** -0.5
        print(f"Scaling AdamW LRs by (n_embd/768)^(-0.5) = {dmodel_lr_scale:.6f}")

        # Classify parameters and set per-param LR/config
        self.param_config = {}  # path -> dict of optimizer config
        self.adam_state = {}    # path -> dict of {m, v, t}
        self.muon_state = {}   # path -> momentum buffer

        flat_params = tree_flatten(model.parameters())
        muon_count = 0
        adamw_counts = {"embed": 0, "ve": 0, "lm_head": 0, "resid": 0, "x0": 0}

        for path, p in flat_params:
            if "blocks" in path and p.ndim == 2:
                self.param_config[path] = {
                    "kind": "muon",
                    "lr": config.matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": 5,
                    "weight_decay": config.weight_decay,
                }
                muon_count += 1
            elif "wte" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": config.embedding_lr * dmodel_lr_scale,
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "eps": 1e-10, "weight_decay": 0.0,
                }
                adamw_counts["embed"] += 1
            elif "value_embeds" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": config.embedding_lr * dmodel_lr_scale,
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "eps": 1e-10, "weight_decay": 0.0,
                }
                adamw_counts["ve"] += 1
            elif "lm_head" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": config.unembedding_lr * dmodel_lr_scale,
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "eps": 1e-10, "weight_decay": 0.0,
                }
                adamw_counts["lm_head"] += 1
            elif "resid_lambdas" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": config.scalar_lr * 0.01,
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "eps": 1e-10, "weight_decay": 0.0,
                }
                adamw_counts["resid"] += 1
            elif "x0_lambdas" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": config.scalar_lr,
                    "betas": (0.96, 0.95),
                    "eps": 1e-10, "weight_decay": 0.0,
                }
                adamw_counts["x0"] += 1
            else:
                # Fallback: AdamW with base LR
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": config.embedding_lr * dmodel_lr_scale,
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "eps": 1e-10, "weight_decay": 0.0,
                }

        # Store initial LRs for scheduling
        self.initial_lrs = {path: cfg["lr"] for path, cfg in self.param_config.items()}

        print(f"Muon params: {muon_count}, AdamW groups: {adamw_counts}")

    def update(self, model, grads):
        """Apply optimizer updates to all parameters."""
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))

        updates = []
        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            cfg = self.param_config[path]
            param = flat_params[path]

            if cfg["kind"] == "muon":
                new_param = self._muon_step(path, grad, param, cfg)
            else:
                new_param = self._adamw_step(path, grad, param, cfg)

            updates.append((path, new_param))

        # Apply updates by walking the model tree directly
        for path, new_param in updates:
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
                obj[last] = new_param
            else:
                setattr(obj, last, new_param)

    def _adamw_step(self, path, grad, param, cfg):
        """Single AdamW step."""
        lr = cfg["lr"]
        beta1, beta2 = cfg["betas"]
        eps = cfg["eps"]
        wd = cfg["weight_decay"]

        if path not in self.adam_state:
            self.adam_state[path] = {
                "m": mx.zeros_like(grad),
                "v": mx.zeros_like(grad),
                "t": 0,
            }

        state = self.adam_state[path]
        state["t"] += 1
        t = state["t"]

        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad * grad)

        m_hat = state["m"] / (1 - beta1 ** t)
        v_hat = state["v"] / (1 - beta2 ** t)

        param = param - lr * (m_hat / (mx.sqrt(v_hat) + eps) + wd * param)
        return param

    def _muon_step(self, path, grad, param, cfg):
        """Single Muon step: Nesterov-style momentum + Newton-Schulz orthogonalization."""
        lr = cfg["lr"]
        momentum = cfg["momentum"]
        ns_steps = cfg["ns_steps"]
        wd = cfg["weight_decay"]

        if path not in self.muon_state:
            self.muon_state[path] = mx.zeros_like(grad)

        # Momentum update: buf = momentum * buf + grad
        buf = momentum * self.muon_state[path] + grad
        self.muon_state[path] = buf

        # Nesterov look-ahead: g = (1 + momentum) * buf (matches original Muon)
        g = (1 + momentum) * buf

        # Newton-Schulz orthogonalization
        update = newton_schulz(g, steps=ns_steps)

        # Scale by sqrt(max(rows, cols))
        rows, cols = param.shape
        update = update * math.sqrt(max(rows, cols))

        # Weight decay (decoupled)
        if wd > 0:
            update = update + wd * param

        return param - lr * update

    def set_lr_multiplier(self, multiplier):
        """Scale all learning rates by multiplier."""
        for path in self.param_config:
            self.param_config[path]["lr"] = self.initial_lrs[path] * multiplier

    def set_muon_momentum(self, momentum):
        for path, cfg in self.param_config.items():
            if cfg["kind"] == "muon":
                cfg["momentum"] = momentum

    def set_muon_weight_decay(self, wd):
        for path, cfg in self.param_config.items():
            if cfg["kind"] == "muon":
                cfg["weight_decay"] = wd

    @property
    def state(self):
        """All optimizer state arrays for mx.eval."""
        arrays = []
        for s in self.adam_state.values():
            arrays.extend([s["m"], s["v"]])
        for buf in self.muon_state.values():
            arrays.append(buf)
        return arrays


class OptimizerConfig:
    """Configuration for the multi-optimizer setup."""

    def __init__(self, n_embd,
                 unembedding_lr=0.004, embedding_lr=0.3, matrix_lr=0.02,
                 weight_decay=0.2, adam_beta1=0.8, adam_beta2=0.95,
                 scalar_lr=0.5):
        self.n_embd = n_embd
        self.unembedding_lr = unembedding_lr
        self.embedding_lr = embedding_lr
        self.matrix_lr = matrix_lr
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.scalar_lr = scalar_lr


def setup_optimizer(model, opt_config):
    """Create MultiOptimizer for the model."""
    return MultiOptimizer(model, opt_config)


# --- LR schedules (ported from base_train.py) ---

def get_lr_multiplier(step, num_iterations, warmup_ratio=0.0, warmdown_ratio=0.5, final_lr_frac=0.0):
    """Linear warmup -> constant -> linear warmdown."""
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def get_muon_momentum(step):
    """Momentum warmup: 0.85 -> 0.95 over first 300 steps."""
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(step, num_iterations, base_weight_decay):
    """Linear decay to zero."""
    return base_weight_decay * (1 - step / num_iterations)
