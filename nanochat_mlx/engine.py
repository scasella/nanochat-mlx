"""
Inference engine for MLX models with KV cache.

Port of nanochat/engine.py — efficient autoregressive generation.
"""

import signal
import warnings
from collections import deque
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn


# --- Calculator tool helpers (reused from nanochat/engine.py) ---

@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None


def use_calculator(expr):
    """Evaluate a Python expression safely."""
    expr = expr.replace(",", "")
    if all(x in "0123456789*+-/.() " for x in expr):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all(x in allowed_chars for x in expr):
        return None
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    if any(p in expr.lower() for p in dangerous_patterns):
        return None
    if '.count(' not in expr:
        return None
    return eval_with_timeout(expr)


# --- KV Cache ---

class KVCache:
    """KV cache for efficient autoregressive generation in MLX."""

    def __init__(self, n_layers, window_sizes=None):
        self.n_layers = n_layers
        self.keys = [None] * n_layers
        self.values = [None] * n_layers
        self.offset = 0
        # Per-layer window sizes for sliding window eviction (None = no limit)
        self.window_sizes = window_sizes

    def update(self, layer_idx, k, v):
        """
        Update cache for a layer.
        k, v: (B, H, T_new, D) format.
        Returns (k_full, v_full) including cached entries.
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = mx.concatenate([self.keys[layer_idx], k], axis=2)
            self.values[layer_idx] = mx.concatenate([self.values[layer_idx], v], axis=2)

        # Evict entries outside sliding window for this layer
        if self.window_sizes is not None:
            w = self.window_sizes[layer_idx]
            seq_len = self.keys[layer_idx].shape[2]
            if seq_len > w:
                self.keys[layer_idx] = self.keys[layer_idx][:, :, -w:, :]
                self.values[layer_idx] = self.values[layer_idx][:, :, -w:, :]

        # Advance offset after last layer processes
        if layer_idx == self.n_layers - 1:
            self.offset += k.shape[2]

        return self.keys[layer_idx], self.values[layer_idx]

    def reset(self):
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers
        self.offset = 0


# --- Sampling ---

def apply_repetition_penalty(logits, generated_tokens, penalty=1.0):
    """Apply repetition penalty to logits based on previously generated tokens.

    For each token that appears in generated_tokens, divide its logit by penalty
    if positive, or multiply by penalty if negative. penalty=1.0 means no effect.
    """
    if penalty == 1.0 or not generated_tokens:
        return logits
    vocab_size = logits.shape[-1]
    seen = mx.zeros((vocab_size,))
    unique_ids = mx.array(list(set(generated_tokens)), dtype=mx.int32)
    seen[unique_ids] = 1.0
    seen = seen.reshape(1, -1) > 0  # (1, vocab_size)
    penalized = mx.where(logits > 0, logits / penalty, logits * penalty)
    return mx.where(seen, penalized, logits)


def sample_next_token(logits, temperature=1.0, top_k=None, key=None,
                      generated_tokens=None, repetition_penalty=1.0):
    """Sample a single next token from logits of shape (B, vocab_size)."""
    if generated_tokens and repetition_penalty != 1.0:
        # Only look at recent tokens to keep penalty focused and efficient
        recent = generated_tokens[-256:] if len(generated_tokens) > 256 else generated_tokens
        logits = apply_repetition_penalty(logits, recent, repetition_penalty)

    if temperature == 0.0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        vals = mx.topk(logits, k=k, axis=-1)
        # Create mask for top-k
        threshold = vals[:, -1:]
        logits = mx.where(logits >= threshold, logits, mx.array(float("-inf")))

    logits = logits / temperature
    return mx.random.categorical(logits, axis=-1)[:, None]  # (B, 1)


# --- Inference Engine ---

class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42, repetition_penalty=1.0):
        """
        Autoregressive generation with KV cache.
        Yields (token_column, token_masks) per step.
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
        mx.random.seed(seed)

        # Special tokens for tool use
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # 1) Prefill with the prompt
        window_sizes = getattr(self.model, 'window_sizes', None)
        kv_cache = KVCache(self.model.config.n_layer, window_sizes=window_sizes)
        ids = mx.array([tokens], dtype=mx.int32)  # (1, T_prompt)
        logits = self.model(ids, kv_cache=kv_cache)  # (1, T_prompt, vocab)
        logits = logits[:, -1, :]  # (1, vocab)
        mx.eval(logits)

        # 2) Expand for num_samples (contiguous copies to avoid shared-memory corruption)
        if num_samples > 1:
            logits = mx.repeat(logits, num_samples, axis=0)
            expanded_cache = KVCache(self.model.config.n_layer, window_sizes=window_sizes)
            for i in range(self.model.config.n_layer):
                expanded_cache.keys[i] = mx.repeat(kv_cache.keys[i], num_samples, axis=0)
                expanded_cache.values[i] = mx.repeat(kv_cache.values[i], num_samples, axis=0)
            expanded_cache.offset = kv_cache.offset
            kv_cache = expanded_cache

        # 3) Init row states
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Generation loop
        num_generated = 0
        generated_so_far = list(tokens)  # track for repetition penalty
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(s.completed for s in row_states):
                break

            next_ids = sample_next_token(logits, temperature, top_k,
                                         generated_tokens=generated_so_far,
                                         repetition_penalty=repetition_penalty)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                generated_so_far.append(next_token)

                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            # Next step: feed token column through model with KV cache
            ids = mx.array(token_column, dtype=mx.int32).reshape(-1, 1)  # (B, 1)
            logits = self.model(ids, kv_cache=kv_cache)[:, -1, :]
            mx.eval(logits)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """Non-streaming batch generation. Returns (results, masks)."""
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks
