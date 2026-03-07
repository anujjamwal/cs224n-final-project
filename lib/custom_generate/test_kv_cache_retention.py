"""Tests for KV cache retention after pruning.

These tests verify that the KV cache is correctly pruned to retain only the
prefix (tokens before the thought block) and that the caller receives the
right cache_position to re-process solution tokens.
"""

import torch
import pytest
from transformers import DynamicCache

from .generate import (
    _retain_and_prune_kv_cache,
    _prune_model_inputs,
)


# ---------------------------------------------------------------------------
# Test parameters matching Qwen2.5-Math-1.5B
# ---------------------------------------------------------------------------
HEAD_DIM = 128
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


def _make_mock_model(head_dim=HEAD_DIM, num_heads=12, pad_token_id=0):
    """Create a minimal mock model with the config attributes needed by _prune_model_inputs."""
    from unittest.mock import MagicMock
    model = MagicMock()
    model.config.hidden_size = head_dim * num_heads
    model.config.num_attention_heads = num_heads
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = 2
    return model


# ---------------------------------------------------------------------------
# _retain_and_prune_kv_cache
# ---------------------------------------------------------------------------

class TestRetainAndPruneKvCache:
    def test_keeps_only_prefix(self):
        """Should keep only [0..thought_pos] in the cache."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        new_seq = _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len,
        )

        # Only prefix [0..5] = 6 entries
        assert new_seq == thought_pos + 1
        assert _cache_keys(cache, 0).shape == (BATCH_SIZE, NUM_HEADS, 6, HEAD_DIM)
        assert _cache_values(cache, 0).shape == (BATCH_SIZE, NUM_HEADS, 6, HEAD_DIM)

    def test_prefix_keys_unchanged(self):
        """Prefix keys should be preserved exactly (no rotation applied)."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        orig_k = _cache_keys(cache, 0)[:, :, :thought_pos + 1, :].clone()

        _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        assert torch.allclose(_cache_keys(cache, 0)[:, :, :thought_pos + 1, :], orig_k)

    def test_prefix_values_unchanged(self):
        """Prefix values should be preserved exactly."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        orig_v = _cache_values(cache, 0)[:, :, :thought_pos + 1, :].clone()

        _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        assert torch.allclose(_cache_values(cache, 0)[:, :, :thought_pos + 1, :], orig_v)

    def test_non_pruned_keeps_all(self):
        """Non-pruned batch elements should keep all entries."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        prune_map: dict[int, tuple[int, int]] = {}

        orig_k = _cache_keys(cache, 0).clone()

        new_seq = _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        assert new_seq == seq_len
        assert torch.allclose(_cache_keys(cache, 0), orig_k)

    def test_prune_at_beginning(self):
        """Pruning a block starting at position 0."""
        seq_len = 15
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        # [THOUGHT] at 0, [SOLUTION] at 3 → keep only [0]
        prune_map = {0: (0, 3)}

        new_seq = _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        assert new_seq == 1

    def test_empty_cache(self):
        """Empty cache should not crash."""
        cache = DynamicCache()
        prune_map = {0: (5, 10)}

        new_seq = _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, 0)
        assert new_seq == 0

    def test_all_layers_pruned(self):
        """All layers should be pruned consistently."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        for layer_idx in range(NUM_LAYERS):
            assert _cache_keys(cache, layer_idx).shape[2] == thought_pos + 1
            assert _cache_values(cache, layer_idx).shape[2] == thought_pos + 1


# ---------------------------------------------------------------------------
# _prune_model_inputs with retain_kv_cache
# ---------------------------------------------------------------------------

class TestPruneModelInputsWithCacheRetention:
    """Test that _prune_model_inputs correctly retains KV cache."""

    def test_cache_retained(self):
        """With retain_kv_cache=True, cache should keep only the prefix."""
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

        # Cache should have only the prefix (thought_pos + 1 entries)
        assert _cache_keys(retained_cache, 0).shape[2] == thought_pos + 1

        # cache_position should cover solution tokens for re-processing
        expected_cache_pos = list(range(thought_pos + 1, pruned_len))
        assert new_kwargs['cache_position'].tolist() == expected_cache_pos

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

    def test_input_ids_same_regardless_of_cache_mode(self):
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
