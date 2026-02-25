"""
GPT model for MLX (Apple Silicon).

Port of nanochat/gpt.py — same architecture, MLX-native implementation.
Notable features: RoPE, QK-norm, ReLU², GQA, value embeddings, logit softcap,
per-layer residual scaling. No DDP, no FP8, no torch.compile.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    """Parameterless RMSNorm."""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(T, dtype=mx.float32):
    """Create (T, T) additive causal mask: 0 for allowed, -inf for masked."""
    indices = mx.arange(T)
    mask = indices[None, :] > indices[:, None]  # upper triangular = True
    return mx.where(mask, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(T, window_size, dtype=mx.float32):
    """Create (T, T) additive mask with causal + sliding window.

    Each position can attend to at most `window_size` previous positions (inclusive).
    """
    indices = mx.arange(T)
    # Causal: can't attend to future
    causal = indices[None, :] > indices[:, None]
    # Sliding window: can't attend too far back
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = 32
        if has_ve(layer_idx, config.n_layer):
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

        # RoPE: traditional=True matches nanochat's split-half rotation pattern
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve=None, offset=0, kv_cache=None, mask=None):
        B, T, C = x.shape

        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer)
        if ve is not None and hasattr(self, "ve_gate"):
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head)
            v = v + mx.expand_dims(gate, axis=-1) * ve

        # Transpose to (B, H, T, D) for RoPE and SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE then QK-norm
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        q = norm(q)
        k = norm(k)

        # KV cache for inference
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)
            # After sliding window eviction, T_kv may be shorter than mask columns
            if mask is not None and mask.shape[-1] != k.shape[2]:
                mask = mask[:, -k.shape[2]:]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        # (B, H, T, D) -> (B, T, C)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2  # ReLU²
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve=None, offset=0, kv_cache=None, mask=None):
        x = x + self.attn(norm(x), ve, offset, kv_cache, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        # Pad vocab for efficiency
        self.padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if self.padded_vocab_size != config.vocab_size:
            print(f"Padding vocab_size from {config.vocab_size} to {self.padded_vocab_size}")

        self.wte = nn.Embedding(self.padded_vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, self.padded_vocab_size, bias=False)

        # Per-layer learnable scalars
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))

        # Value embeddings on alternating layers
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(self.padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }

        # Per-layer window sizes for sliding window attention
        self.window_sizes = self._compute_window_sizes(config)

    def init_weights(self):
        """Initialize weights matching nanochat/gpt.py."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embedding and unembedding
        self.wte.weight = mx.random.normal(self.wte.weight.shape) * 1.0
        self.lm_head.weight = mx.random.normal(self.lm_head.weight.shape) * 0.001

        # Transformer blocks
        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(-s, s, block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(-s, s, block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(-s, s, block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
            block.mlp.c_fc.weight = mx.random.uniform(-s, s, block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            if hasattr(block.attn, "ve_gate"):
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight)

        # Per-layer scalars
        self.resid_lambdas = mx.ones((self.config.n_layer,))
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1)

        # Value embeddings (same init as c_v: uniform)
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, ve.weight.shape)

    def num_scaling_params(self):
        """Parameter counts for scaling law analysis (mirrors nanochat/gpt.py:319-346)."""
        wte = self.wte.weight.size
        value_embeds = sum(ve.weight.size for ve in self.value_embeds.values())
        lm_head = self.lm_head.weight.size

        transformer_matrices = 0
        for block in self.blocks:
            flat = nn.utils.tree_flatten(block.parameters())
            for _, p in flat:
                transformer_matrices += p.size

        scalars = self.resid_lambdas.size + self.x0_lambdas.size
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def _compute_window_sizes(self, config):
        """Compute per-layer window sizes for sliding window attention.

        Pattern string is tiled across layers. Final layer always gets L (full context).
        L = full context (sequence_len), S = half context (sequence_len // 2).
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, T):
        """Get per-layer attention masks, caching by sequence length."""
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        # Deduplicate: only create unique masks
        unique_windows = set(self.window_sizes)
        for w in unique_windows:
            key = (T, w)
            if key not in self._mask_cache:
                if w >= T:
                    self._mask_cache[key] = create_additive_causal_mask(T)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(T, w)
        return [self._mask_cache[(T, w)] for w in self.window_sizes]

    def __call__(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
        offset = 0 if kv_cache is None else kv_cache.offset

        # Per-layer masks for training / prefill
        if kv_cache is None or T > 1:
            masks = self._get_masks(T)
        else:
            masks = [None] * self.config.n_layer  # single-token decode

        # Forward the trunk
        x = self.wte(idx)
        x = norm(x)
        x0 = x  # save for x0 residual

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, offset, kv_cache, masks[i])

        x = norm(x)

        # Logits with softcap
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)

        if targets is not None:
            # Cross-entropy loss with ignore_index=-1
            mask = targets != -1
            targets_safe = mx.where(mask, targets, mx.zeros_like(targets))
            ce = nn.losses.cross_entropy(logits, targets_safe, reduction="none")
            ce = ce * mask
            loss = mx.sum(ce) / mx.maximum(mx.sum(mask), 1)
            return loss

        return logits


def loss_fn(model, inputs, targets):
    """Standalone loss function for nn.value_and_grad."""
    return model(inputs, targets=targets)
