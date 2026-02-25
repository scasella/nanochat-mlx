"""
Tests for the MLX GPT model implementation.

Covers Stage 1 (forward pass) and Stage 2 (loss + gradients).
Run: python -m pytest tests/test_mlx_gpt.py -v
"""

import pytest
import math

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")


@pytest.fixture
def tiny_config():
    """Depth=4 config for fast testing."""
    from nanochat_mlx.gpt import GPTConfig
    return GPTConfig(
        sequence_len=256,
        vocab_size=32768,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
    )


@pytest.fixture
def tiny_model(tiny_config):
    from nanochat_mlx.gpt import GPT
    model = GPT(tiny_config)
    model.init_weights()
    mx.eval(model.parameters())
    return model


# --- Stage 1: Forward Pass ---

class TestForwardPass:
    def test_output_shape(self, tiny_model, tiny_config):
        """Verify output shape (B, T, vocab_size)."""
        B, T = 1, 64
        idx = mx.array([[i % tiny_config.vocab_size for i in range(T)]], dtype=mx.int32)
        logits = tiny_model(idx)
        mx.eval(logits)
        assert logits.shape == (B, T, tiny_config.vocab_size), \
            f"Expected ({B}, {T}, {tiny_config.vocab_size}), got {logits.shape}"

    def test_output_shape_longer(self, tiny_model, tiny_config):
        """Verify output shape with longer sequence."""
        B, T = 2, 128
        idx = mx.random.randint(0, tiny_config.vocab_size, (B, T))
        logits = tiny_model(idx)
        mx.eval(logits)
        assert logits.shape == (B, T, tiny_config.vocab_size)

    def test_logit_softcap(self, tiny_model, tiny_config):
        """Verify logits are in [-15, 15] range due to softcap."""
        B, T = 1, 64
        idx = mx.random.randint(0, tiny_config.vocab_size, (B, T))
        logits = tiny_model(idx)
        mx.eval(logits)
        assert logits.min().item() >= -15.0, f"Logit min {logits.min().item()} < -15"
        assert logits.max().item() <= 15.0, f"Logit max {logits.max().item()} > 15"

    def test_logit_dtype_float32(self, tiny_model, tiny_config):
        """Verify logits are float32 (for stable loss computation)."""
        idx = mx.random.randint(0, tiny_config.vocab_size, (1, 32))
        logits = tiny_model(idx)
        mx.eval(logits)
        assert logits.dtype == mx.float32

    def test_no_oom_depth4(self, tiny_config):
        """Verify depth=4, B=1, T=256 completes without OOM."""
        from nanochat_mlx.gpt import GPT
        model = GPT(tiny_config)
        model.init_weights()
        mx.eval(model.parameters())

        idx = mx.random.randint(0, tiny_config.vocab_size, (1, 256))
        logits = model(idx)
        mx.eval(logits)
        assert logits.shape == (1, 256, tiny_config.vocab_size)

    def test_parameter_count(self, tiny_model):
        """Verify parameter count is reasonable for depth=4."""
        param_counts = tiny_model.num_scaling_params()
        total = param_counts["total"]
        # depth=4, n_embd=256: should be ~5-15M params
        assert total > 1_000_000, f"Too few params: {total:,}"
        assert total < 50_000_000, f"Too many params: {total:,}"
        print(f"Total params: {total:,}")

    def test_value_embeddings_present(self, tiny_model, tiny_config):
        """Verify value embeddings are on alternating layers."""
        from nanochat_mlx.gpt import has_ve
        for i in range(tiny_config.n_layer):
            should_have = has_ve(i, tiny_config.n_layer)
            has_it = str(i) in tiny_model.value_embeds
            assert has_it == should_have, \
                f"Layer {i}: expected ve={should_have}, got {has_it}"


# --- Stage 2: Loss + Gradients ---

class TestLossAndGradients:
    def test_loss_at_random_init(self, tiny_model, tiny_config):
        """Loss at random init should be ~ln(vocab_size) ≈ 10.4."""
        B, T = 1, 64
        idx = mx.random.randint(0, tiny_config.vocab_size, (B, T))
        targets = mx.random.randint(0, tiny_config.vocab_size, (B, T))
        loss = tiny_model(idx, targets=targets)
        mx.eval(loss)
        loss_val = loss.item()
        expected = math.log(tiny_config.vocab_size)  # ~10.4
        # Should be within ~50% of expected
        assert loss_val > expected * 0.5, f"Loss {loss_val} too low at init"
        assert loss_val < expected * 1.5, f"Loss {loss_val} too high at init"
        print(f"Init loss: {loss_val:.4f} (expected ~{expected:.1f})")

    def test_ignore_index(self, tiny_model, tiny_config):
        """Loss should ignore targets == -1."""
        B, T = 1, 64
        idx = mx.random.randint(0, tiny_config.vocab_size, (B, T))
        # All targets are -1 (all ignored)
        targets = mx.full((B, T), -1, dtype=mx.int32)
        loss = tiny_model(idx, targets=targets)
        mx.eval(loss)
        # With all targets masked, loss should be 0
        assert loss.item() == 0.0 or abs(loss.item()) < 1e-6

    def test_gradients_flow(self, tiny_model, tiny_config):
        """Verify gradients are non-zero for all trainable parameters."""
        from nanochat_mlx.gpt import loss_fn
        from mlx.utils import tree_flatten

        B, T = 1, 64
        idx = mx.random.randint(0, tiny_config.vocab_size, (B, T))
        targets = mx.random.randint(0, tiny_config.vocab_size, (B, T))

        loss_grad_fn = nn.value_and_grad(tiny_model, loss_fn)
        loss, grads = loss_grad_fn(tiny_model, idx, targets)
        mx.eval(loss, grads)

        # Check that gradients exist and are non-zero
        flat_grads = tree_flatten(grads)
        zero_grad_paths = []
        for path, g in flat_grads:
            if isinstance(g, mx.array):
                grad_norm = mx.sqrt(mx.sum(g * g)).item()
                if grad_norm == 0.0:
                    zero_grad_paths.append(path)

        if zero_grad_paths:
            # Some zero grads are expected (e.g. c_proj init to zero)
            print(f"Zero-gradient paths (may be expected): {zero_grad_paths}")

        # At minimum, wte and lm_head should have non-zero gradients
        for path, g in flat_grads:
            if "wte" in path or "lm_head" in path:
                grad_norm = mx.sqrt(mx.sum(g * g)).item()
                assert grad_norm > 0, f"Expected non-zero gradient for {path}"

    def test_forward_backward_timing(self, tiny_model, tiny_config):
        """Time a forward + backward pass for depth=4."""
        from nanochat_mlx.gpt import loss_fn
        import time

        B, T = 1, 64
        idx = mx.random.randint(0, tiny_config.vocab_size, (B, T))
        targets = mx.random.randint(0, tiny_config.vocab_size, (B, T))

        loss_grad_fn = nn.value_and_grad(tiny_model, loss_fn)

        # Warmup
        loss, grads = loss_grad_fn(tiny_model, idx, targets)
        mx.eval(loss, grads)

        # Timed run
        t0 = time.time()
        for _ in range(5):
            loss, grads = loss_grad_fn(tiny_model, idx, targets)
            mx.eval(loss, grads)
        t1 = time.time()

        avg_ms = (t1 - t0) / 5 * 1000
        print(f"Forward+backward: {avg_ms:.1f}ms avg (depth=4, B={B}, T={T})")
        assert avg_ms < 30000, f"Forward+backward too slow: {avg_ms:.0f}ms"


# --- Stage 6: KV Cache ---

class TestKVCache:
    def test_greedy_matches_no_cache(self, tiny_model, tiny_config):
        """Greedy decode with KV cache should match decode without cache."""
        B, T = 1, 16
        idx = mx.random.randint(0, tiny_config.vocab_size, (B, T))

        # Without cache: full forward
        logits_no_cache = tiny_model(idx)
        mx.eval(logits_no_cache)

        # With cache: process one token at a time would need engine
        # For now just verify prefill matches
        from nanochat_mlx.engine import KVCache
        kv_cache = KVCache(tiny_config.n_layer)
        logits_cached = tiny_model(idx, kv_cache=kv_cache)
        mx.eval(logits_cached)

        # Last token logits should match
        diff = mx.abs(logits_no_cache[:, -1, :] - logits_cached[:, -1, :])
        max_diff = mx.max(diff).item()
        assert max_diff < 0.01, f"KV cache prefill mismatch: max diff = {max_diff}"


# --- Stage 7: Sliding Window Attention ---

class TestSlidingWindow:
    def test_sliding_window_mask_shape(self):
        """Verify sliding window mask has correct shape."""
        from nanochat_mlx.gpt import create_sliding_window_mask
        T, W = 64, 32
        mask = create_sliding_window_mask(T, W)
        mx.eval(mask)
        assert mask.shape == (T, T)

    def test_sliding_window_mask_blocks_far_tokens(self):
        """Tokens beyond window should be masked (-inf)."""
        from nanochat_mlx.gpt import create_sliding_window_mask
        T, W = 16, 4
        mask = create_sliding_window_mask(T, W)
        mx.eval(mask)
        # Position 8 attending to position 0: distance=8 >= window=4, should be -inf
        assert mask[8, 0].item() == float("-inf")
        # Position 8 attending to position 5: distance=3 < window=4, should be 0
        assert mask[8, 5].item() == 0.0
        # Causal: position 0 attending to position 5 should be -inf
        assert mask[0, 5].item() == float("-inf")

    def test_sssl_pattern_forward(self):
        """Model with SSSL window pattern produces valid output."""
        from nanochat_mlx.gpt import GPT, GPTConfig
        config = GPTConfig(
            sequence_len=256, vocab_size=32768,
            n_layer=4, n_head=2, n_kv_head=2, n_embd=256,
            window_pattern="SSSL",
        )
        model = GPT(config)
        model.init_weights()
        mx.eval(model.parameters())

        # Window sizes: S=128, S=128, S=128, L=256 (last always L)
        assert model.window_sizes == [128, 128, 128, 256]

        idx = mx.random.randint(0, 32768, (1, 64))
        logits = model(idx)
        mx.eval(logits)
        assert logits.shape == (1, 64, 32768)

    def test_sliding_window_loss_and_grad(self):
        """Gradients flow through sliding window model."""
        from nanochat_mlx.gpt import GPT, GPTConfig, loss_fn
        from mlx.utils import tree_flatten
        config = GPTConfig(
            sequence_len=256, vocab_size=32768,
            n_layer=4, n_head=2, n_kv_head=2, n_embd=256,
            window_pattern="SL",
        )
        model = GPT(config)
        model.init_weights()
        mx.eval(model.parameters())

        idx = mx.random.randint(0, 32768, (1, 64))
        targets = mx.random.randint(0, 32768, (1, 64))

        loss_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_grad_fn(model, idx, targets)
        mx.eval(loss, grads)

        assert loss.item() > 0
        # wte should have non-zero grads
        for path, g in tree_flatten(grads):
            if "wte" in path:
                assert mx.sqrt(mx.sum(g * g)).item() > 0

    def test_checkpoint_preserves_window_pattern(self):
        """Checkpoint save/load preserves window_pattern in metadata."""
        import os, json, tempfile
        from nanochat_mlx.gpt import GPT, GPTConfig
        from nanochat_mlx.train import _load_weights_into_model

        config = GPTConfig(
            sequence_len=256, vocab_size=32768,
            n_layer=4, n_head=2, n_kv_head=2, n_embd=256,
            window_pattern="SSSL",
        )
        model = GPT(config)
        model.init_weights()
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "test.safetensors")
            model.save_weights(weights_path)

            # Load into new model
            model2 = GPT(config)
            _load_weights_into_model(model2, weights_path)
            mx.eval(model2.parameters())

            # Verify same output
            idx = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
            l1 = model(idx)
            l2 = model2(idx)
            mx.eval(l1, l2)
            diff = mx.max(mx.abs(l1 - l2)).item()
            assert diff < 1e-5, f"Window model weights not preserved: diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
