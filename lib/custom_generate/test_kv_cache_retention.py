"""Tests for KV cache retention and RoPE correction after pruning.

These tests verify the mathematical correctness of the RoPE position-shift
correction and the surgical KV cache manipulation that replaces the old
discard-and-reprefill strategy.
"""

import math
import torch
import pytest
from transformers import DynamicCache

from .generate import (
    _rotate_half,
    _build_rope_correction,
    _apply_rope_correction,
    _retain_and_prune_kv_cache,
    _prune_model_inputs,
)


# ---------------------------------------------------------------------------
# Test parameters matching Qwen2.5-Math-1.5B
# ---------------------------------------------------------------------------
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
NUM_HEADS = 2       # num_key_value_heads for Qwen2.5-Math-1.5B
NUM_LAYERS = 2      # reduced for test speed (real model has 28)
BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_num_layers(cache: DynamicCache) -> int:
    if hasattr(cache, 'layers'):
        return len(cache.layers)
    return len(cache.key_cache)  # type: ignore[attr-defined]


def _cache_keys(cache: DynamicCache, layer_idx: int) -> torch.Tensor:
    if hasattr(cache, 'layers') and len(cache.layers) > layer_idx:
        return cache.layers[layer_idx].keys
    return cache.key_cache[layer_idx]  # type: ignore[attr-defined]


def _cache_values(cache: DynamicCache, layer_idx: int) -> torch.Tensor:
    if hasattr(cache, 'layers') and len(cache.layers) > layer_idx:
        return cache.layers[layer_idx].values
    return cache.value_cache[layer_idx]  # type: ignore[attr-defined]


def _make_cache(batch_size: int, num_layers: int, num_heads: int, seq_len: int, head_dim: int):
    """Create a DynamicCache with random KV entries (compatible with transformers ≥5.x)."""
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(k, v, layer_idx=layer_idx)
    return cache


def _apply_rope_at_position(keys: torch.Tensor, pos: int, head_dim: int, rope_theta: float):
    """Apply standard HF-style RoPE to keys at a single position.

    keys: shape (..., head_dim)
    Returns rotated keys of same shape.
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = pos * inv_freq
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    cos_full = torch.cat([cos_vals, cos_vals])
    sin_full = torch.cat([sin_vals, sin_vals])

    return keys * cos_full + _rotate_half(keys) * sin_full


def _make_mock_model(head_dim=HEAD_DIM, rope_theta=ROPE_THETA, num_heads=12, pad_token_id=0):
    """Create a minimal mock model with the config attributes needed by _prune_model_inputs."""
    from unittest.mock import MagicMock
    model = MagicMock()
    model.config.hidden_size = head_dim * num_heads
    model.config.num_attention_heads = num_heads
    model.config.rope_theta = rope_theta
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = 2
    return model


# ---------------------------------------------------------------------------
# _rotate_half
# ---------------------------------------------------------------------------

class TestRotateHalf:
    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = _rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_inverse(self):
        """rotate_half applied twice should negate the input."""
        x = torch.randn(4, 8)
        result = _rotate_half(_rotate_half(x))
        assert torch.allclose(result, -x, atol=1e-6)

    def test_batched(self):
        x = torch.randn(2, 3, 4, 128)
        result = _rotate_half(x)
        assert result.shape == x.shape
        half = 64
        assert torch.allclose(result[..., :half], -x[..., half:])
        assert torch.allclose(result[..., half:], x[..., :half])


# ---------------------------------------------------------------------------
# _build_rope_correction
# ---------------------------------------------------------------------------

class TestBuildRopeCorrection:
    def test_zero_shift(self):
        """Zero shift should give cos=1, sin=0 (identity rotation)."""
        cos_corr, sin_corr = _build_rope_correction(0, HEAD_DIM, ROPE_THETA, torch.device('cpu'), torch.float32)
        assert torch.allclose(cos_corr, torch.ones(HEAD_DIM), atol=1e-6)
        assert torch.allclose(sin_corr, torch.zeros(HEAD_DIM), atol=1e-6)

    def test_shape(self):
        cos_corr, sin_corr = _build_rope_correction(5, HEAD_DIM, ROPE_THETA, torch.device('cpu'), torch.float32)
        assert cos_corr.shape == (HEAD_DIM,)
        assert sin_corr.shape == (HEAD_DIM,)

    def test_symmetry(self):
        """cos should be duplicated across halves, sin should be negated and duplicated."""
        cos_corr, sin_corr = _build_rope_correction(3, 8, 10000.0, torch.device('cpu'), torch.float32)
        half = 4
        assert torch.allclose(cos_corr[:half], cos_corr[half:])
        assert torch.allclose(sin_corr[:half], sin_corr[half:])

    def test_sin_is_negative(self):
        """For positive shift, sin_corr should be -sin(shift * theta)."""
        shift = 5
        cos_corr, sin_corr = _build_rope_correction(shift, 8, 10000.0, torch.device('cpu'), torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
        expected_sin = -torch.sin(shift * inv_freq)
        assert torch.allclose(sin_corr[:4], expected_sin, atol=1e-6)


# ---------------------------------------------------------------------------
# _apply_rope_correction — mathematical correctness
# ---------------------------------------------------------------------------

class TestApplyRopeCorrection:
    def test_identity_with_zero_shift(self):
        """Zero-shift correction should not change keys."""
        keys = torch.randn(1, NUM_HEADS, 10, HEAD_DIM)
        cos_corr, sin_corr = _build_rope_correction(0, HEAD_DIM, ROPE_THETA, torch.device('cpu'), torch.float32)
        result = _apply_rope_correction(keys, cos_corr, sin_corr)
        assert torch.allclose(result, keys, atol=1e-5)

    def test_rope_shift_equals_reapply(self):
        """Core correctness test: correcting RoPE from position p to p-δ should
        equal applying RoPE at p-δ from scratch.

        Given:  k_cached = RoPE(k_orig, p)
        We want: k_new = RoPE(k_orig, p - δ)
        Using:  k_new = apply_correction(k_cached, shift=δ)

        Verify: k_new ≈ RoPE(k_orig, p - δ)
        """
        k_orig = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        old_pos = 42
        shift = 7
        new_pos = old_pos - shift

        # Apply RoPE at old position
        k_at_old = _apply_rope_at_position(k_orig, old_pos, HEAD_DIM, ROPE_THETA)

        # Apply correction to shift backward
        cos_corr, sin_corr = _build_rope_correction(shift, HEAD_DIM, ROPE_THETA, torch.device('cpu'), torch.float32)
        k_corrected = _apply_rope_correction(k_at_old, cos_corr, sin_corr)

        # Apply RoPE at new position directly
        k_at_new = _apply_rope_at_position(k_orig, new_pos, HEAD_DIM, ROPE_THETA)

        assert torch.allclose(k_corrected, k_at_new, atol=1e-5), (
            f"RoPE correction failed: max diff = {(k_corrected - k_at_new).abs().max().item()}"
        )

    def test_shift_multiple_positions(self):
        """Test correction across a batch of sequence positions."""
        seq_len = 20
        k_orig = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM)
        shift = 5

        # Apply RoPE at original positions 10..29
        k_cached = torch.cat([
            _apply_rope_at_position(k_orig[:, :, i:i+1, :], 10 + i, HEAD_DIM, ROPE_THETA)
            for i in range(seq_len)
        ], dim=2)

        # Apply correction
        cos_corr, sin_corr = _build_rope_correction(shift, HEAD_DIM, ROPE_THETA, torch.device('cpu'), torch.float32)
        k_corrected = _apply_rope_correction(k_cached, cos_corr, sin_corr)

        # Expected: RoPE at positions 5..24
        k_expected = torch.cat([
            _apply_rope_at_position(k_orig[:, :, i:i+1, :], 5 + i, HEAD_DIM, ROPE_THETA)
            for i in range(seq_len)
        ], dim=2)

        assert torch.allclose(k_corrected, k_expected, atol=1e-4), (
            f"Multi-position correction failed: max diff = {(k_corrected - k_expected).abs().max().item()}"
        )

    def test_large_shift(self):
        """Even large shifts should produce correct results."""
        k_orig = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        old_pos = 500
        shift = 200

        k_at_old = _apply_rope_at_position(k_orig, old_pos, HEAD_DIM, ROPE_THETA)
        cos_corr, sin_corr = _build_rope_correction(shift, HEAD_DIM, ROPE_THETA, torch.device('cpu'), torch.float32)
        k_corrected = _apply_rope_correction(k_at_old, cos_corr, sin_corr)
        k_at_new = _apply_rope_at_position(k_orig, old_pos - shift, HEAD_DIM, ROPE_THETA)

        assert torch.allclose(k_corrected, k_at_new, atol=1e-4)


# ---------------------------------------------------------------------------
# _retain_and_prune_kv_cache
# ---------------------------------------------------------------------------

class TestRetainAndPruneKvCache:
    def test_basic_prune(self):
        """Prune positions [thought+1..solution] from cache, verify shape."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        new_seq = _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len, HEAD_DIM, ROPE_THETA, prune_aware=True,
        )

        # Expected: keep [0..5] ∪ [11..18] = 6 + 8 = 14 entries (19 excluded as last)
        assert new_seq == 14
        assert _cache_keys(cache, 0).shape == (BATCH_SIZE, NUM_HEADS, 14, HEAD_DIM)
        assert _cache_values(cache, 0).shape == (BATCH_SIZE, NUM_HEADS, 14, HEAD_DIM)

    def test_values_unchanged_before_prune(self):
        """Values (not keys) and keys before the pruned block should be unchanged."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        # Save originals
        orig_v = _cache_values(cache, 0).clone()
        orig_k_before = _cache_keys(cache, 0)[:, :, :thought_pos + 1, :].clone()

        _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len, HEAD_DIM, ROPE_THETA, prune_aware=True,
        )

        # Values before block should be unchanged
        new_v_before = _cache_values(cache, 0)[:, :, :thought_pos + 1, :]
        assert torch.allclose(new_v_before, orig_v[:, :, :thought_pos + 1, :])

        # Keys before block should be unchanged (no RoPE correction needed)
        new_k_before = _cache_keys(cache, 0)[:, :, :thought_pos + 1, :]
        assert torch.allclose(new_k_before, orig_k_before)

    def test_values_unchanged_after_prune(self):
        """Values after the pruned block should be moved but not transformed."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        orig_v_after = _cache_values(cache, 0)[:, :, solution_pos + 1:seq_len - 1, :].clone()

        _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len, HEAD_DIM, ROPE_THETA, prune_aware=True,
        )

        # Values after block moved to new positions but content unchanged
        split_at = thought_pos + 1  # 6
        num_after = seq_len - 1 - (solution_pos + 1)  # 8 entries
        new_v_after = _cache_values(cache, 0)[:, :, split_at:split_at + num_after, :]
        assert torch.allclose(new_v_after, orig_v_after)

    def test_keys_after_prune_are_corrected(self):
        """Keys after the pruned block should have RoPE correction applied."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        shift = solution_pos - thought_pos
        prune_map = {0: (thought_pos, solution_pos)}

        orig_k_after = _cache_keys(cache, 0)[:, :, solution_pos + 1:seq_len - 1, :].clone()

        _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len, HEAD_DIM, ROPE_THETA, prune_aware=True,
        )

        split_at = thought_pos + 1
        num_after = seq_len - 1 - (solution_pos + 1)
        new_k_after = _cache_keys(cache, 0)[:, :, split_at:split_at + num_after, :]

        # Keys should NOT match original (they've been corrected)
        assert not torch.allclose(new_k_after, orig_k_after, atol=1e-6), (
            "Keys after pruned block should have RoPE correction applied"
        )

        # Verify correction matches expected rotation
        cos_corr, sin_corr = _build_rope_correction(
            shift, HEAD_DIM, ROPE_THETA, torch.device('cpu'), torch.float32,
        )
        expected = _apply_rope_correction(orig_k_after, cos_corr, sin_corr)
        assert torch.allclose(new_k_after, expected, atol=1e-5)

    def test_prune_agnostic_no_rope_correction(self):
        """In prune-agnostic mode, keys should NOT be RoPE-corrected."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        orig_k_after = _cache_keys(cache, 0)[:, :, solution_pos + 1:seq_len - 1, :].clone()

        _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len, HEAD_DIM, ROPE_THETA, prune_aware=False,
        )

        split_at = thought_pos + 1
        num_after = seq_len - 1 - (solution_pos + 1)
        new_k_after = _cache_keys(cache, 0)[:, :, split_at:split_at + num_after, :]

        # In agnostic mode, keys should be moved but NOT corrected
        assert torch.allclose(new_k_after, orig_k_after, atol=1e-6)

    def test_no_prune_just_removes_last(self):
        """When no batch element is pruned, only the last entry is removed."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        prune_map: dict[int, tuple[int, int]] = {}

        orig_k = _cache_keys(cache, 0)[:, :, :seq_len - 1, :].clone()

        new_seq = _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len, HEAD_DIM, ROPE_THETA, prune_aware=True,
        )

        assert new_seq == seq_len - 1
        assert torch.allclose(_cache_keys(cache, 0), orig_k)

    def test_prune_at_beginning(self):
        """Pruning a block starting at position 0."""
        seq_len = 15
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        # [THOUGHT] at 0, [SOLUTION] at 3 → remove positions 1,2,3
        prune_map = {0: (0, 3)}

        new_seq = _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len, HEAD_DIM, ROPE_THETA, prune_aware=True,
        )

        # Keep [0] ∪ [4..13] = 1 + 10 = 11
        assert new_seq == 11

    def test_empty_cache(self):
        """Empty cache should not crash."""
        cache = DynamicCache()
        prune_map = {0: (5, 10)}

        new_seq = _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, 0, HEAD_DIM, ROPE_THETA, prune_aware=True,
        )
        assert new_seq == 0


# ---------------------------------------------------------------------------
# _prune_model_inputs with retain_kv_cache
# ---------------------------------------------------------------------------

class TestPruneModelInputsWithCacheRetention:
    """Test that _prune_model_inputs correctly retains KV cache."""

    def test_cache_retained_prune_aware(self):
        """In prune-aware mode with retain_kv_cache=True, cache should be modified not discarded."""
        model = _make_mock_model()
        seq_len = 20

        input_ids = torch.arange(seq_len).unsqueeze(0).long()
        thought_pos, solution_pos, return_pos = 5, 10, 19

        cache = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        model_kwargs = {
            'past_key_values': cache,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }

        new_input_ids, new_kwargs = _prune_model_inputs(
            model,
            prune_input_candidates=[0],
            prune_input_locations=[[(thought_pos, solution_pos, return_pos)]],
            input_ids=input_ids,
            prune_aware=True,
            model_kwargs=model_kwargs,
            retain_kv_cache=True,
        )

        # Cache should still be present (not discarded)
        assert 'past_key_values' in new_kwargs
        retained_cache = new_kwargs['past_key_values']
        assert isinstance(retained_cache, DynamicCache)
        assert _cache_num_layers(retained_cache) == NUM_LAYERS

        # input_ids should be pruned
        pruned_len = seq_len - (solution_pos - thought_pos)  # 20 - 5 = 15
        assert new_input_ids.shape[1] == pruned_len

        # cache_position should point to last token (single-token decode)
        assert new_kwargs['cache_position'].tolist() == [pruned_len - 1]

        # Cache seq len should be pruned_len - 1 (last entry removed for re-processing)
        assert _cache_keys(retained_cache, 0).shape[2] == pruned_len - 1

    def test_cache_discarded_when_disabled(self):
        """With retain_kv_cache=False, cache should be discarded (old behavior)."""
        model = _make_mock_model()
        seq_len = 20

        input_ids = torch.arange(seq_len).unsqueeze(0).long()
        thought_pos, solution_pos, return_pos = 5, 10, 19

        cache = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        model_kwargs = {
            'past_key_values': cache,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }

        new_input_ids, new_kwargs = _prune_model_inputs(
            model,
            prune_input_candidates=[0],
            prune_input_locations=[[(thought_pos, solution_pos, return_pos)]],
            input_ids=input_ids,
            prune_aware=True,
            model_kwargs=model_kwargs,
            retain_kv_cache=False,
        )

        # Cache should be discarded
        assert 'past_key_values' not in new_kwargs

        # cache_position should be full range (for prefill)
        pruned_len = seq_len - (solution_pos - thought_pos)
        assert new_kwargs['cache_position'].tolist() == list(range(pruned_len))

    def test_input_ids_unchanged(self):
        """The pruned input_ids should be the same regardless of cache retention mode."""
        model = _make_mock_model()
        seq_len = 20

        input_ids = torch.arange(seq_len).unsqueeze(0).long()
        thought_pos, solution_pos, return_pos = 5, 10, 19

        # Run with cache retention
        cache1 = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        kwargs1 = {
            'past_key_values': cache1,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }
        ids_retained, _ = _prune_model_inputs(
            model, [0], [[(thought_pos, solution_pos, return_pos)]],
            input_ids.clone(), True, kwargs1, retain_kv_cache=True,
        )

        # Run without cache retention
        cache2 = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        kwargs2 = {
            'past_key_values': cache2,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }
        ids_discarded, _ = _prune_model_inputs(
            model, [0], [[(thought_pos, solution_pos, return_pos)]],
            input_ids.clone(), True, kwargs2, retain_kv_cache=False,
        )

        assert torch.equal(ids_retained, ids_discarded)


# ---------------------------------------------------------------------------
# End-to-end RoPE correctness: simulate full prune cycle
# ---------------------------------------------------------------------------

class TestEndToEndRopeCorrectness:
    """Simulate what happens during generation: build a cache with RoPE-encoded
    keys, prune it, and verify the corrected keys match a from-scratch encoding."""

    def test_full_prune_cycle(self):
        """Build cache → prune → verify RoPE positions are correct."""
        seq_len = 30
        thought_pos = 10
        solution_pos = 18
        shift = solution_pos - thought_pos

        # Simulate original keys (before RoPE)
        k_orig = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)

        # Apply RoPE at each position (simulating what the model does)
        k_cached = torch.cat([
            _apply_rope_at_position(k_orig[:, :, i:i+1, :], i, HEAD_DIM, ROPE_THETA)
            for i in range(seq_len)
        ], dim=2)

        # Build cache with these RoPE-encoded keys
        cache = DynamicCache()
        for layer_idx in range(NUM_LAYERS):
            cache.update(k_cached.clone(), torch.randn_like(k_cached), layer_idx=layer_idx)

        # Prune
        prune_map = {0: (thought_pos, solution_pos)}
        _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len, HEAD_DIM, ROPE_THETA, prune_aware=True,
        )

        # Build expected: keys at new contiguous positions after pruning
        # Keep [0..10] ∪ [19..28] (excluding 29 = last)
        # New positions: 0..10 stay, 19..28 become 11..20
        kept_before = list(range(thought_pos + 1))       # 0..10
        kept_after = list(range(solution_pos + 1, seq_len - 1))  # 19..28

        expected_keys = []
        # Before block: positions unchanged
        for orig_pos in kept_before:
            expected_keys.append(
                _apply_rope_at_position(k_orig[:, :, orig_pos:orig_pos+1, :], orig_pos, HEAD_DIM, ROPE_THETA)
            )
        # After block: positions shifted by -shift
        for orig_pos in kept_after:
            new_pos = orig_pos - shift
            expected_keys.append(
                _apply_rope_at_position(k_orig[:, :, orig_pos:orig_pos+1, :], new_pos, HEAD_DIM, ROPE_THETA)
            )
        expected = torch.cat(expected_keys, dim=2)

        # Compare
        actual = _cache_keys(cache, 0)[:, :, :expected.shape[2], :]
        assert torch.allclose(actual, expected, atol=1e-4), (
            f"End-to-end RoPE correction failed: max diff = {(actual - expected).abs().max().item()}"
        )
