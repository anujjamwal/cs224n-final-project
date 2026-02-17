"""Tests for FlexAttentionMaskMixin and MaterialisedMaskMixin.

We test the materialised path directly (always available) and verify that the
FlexAttention mixin produces equivalent boolean masks when FlexAttention is
importable.
"""

import pytest
import torch

from .masks import (
    FlexAttentionMaskMixin,
    MaterialisedMaskMixin,
    extract_cot_blocks,
    build_min_blocked_q,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THOUGHT_ID = 100
SOLUTION_ID = 101
RETURN_ID = 102


class _FakeMixin:
    """Lightweight stand-in that carries the three special-token IDs."""

    thought_token_id = THOUGHT_ID
    solution_token_id = SOLUTION_ID
    return_token_id = RETURN_ID


class MaterialisedTester(_FakeMixin, MaterialisedMaskMixin):
    pass


class FlexTester(_FakeMixin, FlexAttentionMaskMixin):
    pass


def _allowed(mask_4d, b, q, k):
    """Return True if query q is allowed to attend to key k in batch element b.

    Works with the materialised (batch, 1, seq, seq) bfloat16 mask: 0.0 means
    allowed, large-negative means blocked.
    """
    return mask_4d[b, 0, q, k].item() == 0.0


# ---------------------------------------------------------------------------
# extract_cot_blocks
# ---------------------------------------------------------------------------


class TestExtractCotBlocks:
    def test_single_block(self):
        ids = torch.tensor([[1, THOUGHT_ID, 2, 3, SOLUTION_ID, 4, RETURN_ID, 5]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        assert blocks == [[(1, 4, 6)]]

    def test_nested_blocks(self):
        # outer [T] ... inner [T] ... [S] ... [R] ... [S] ... [R]
        ids = torch.tensor([[
            THOUGHT_ID,      # 0 outer-thought
            10,              # 1
            THOUGHT_ID,      # 2 inner-thought
            20,              # 3
            SOLUTION_ID,     # 4 inner-solution
            21,              # 5
            RETURN_ID,       # 6 inner-return
            SOLUTION_ID,     # 7 outer-solution
            30,              # 8
            RETURN_ID,       # 9 outer-return
        ]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        # inner closes first (stack), then outer
        assert (2, 4, 6) in blocks[0]
        assert (0, 7, 9) in blocks[0]

    def test_no_blocks(self):
        ids = torch.tensor([[1, 2, 3]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        assert blocks == [[]]

    def test_incomplete_block_no_solution(self):
        ids = torch.tensor([[THOUGHT_ID, 1, 2, RETURN_ID]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        assert blocks == [[]]


# ---------------------------------------------------------------------------
# MaterialisedMaskMixin
# ---------------------------------------------------------------------------


class TestMaterialisedMask:
    def _build(self, input_ids, padding_mask=None):
        tester = MaterialisedTester()
        return tester._build_hierarchical_mask(input_ids, padding_mask=padding_mask)

    def test_pure_causal_no_special_tokens(self):
        """Without special tokens the mask should be standard causal."""
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = self._build(ids)
        seq = ids.shape[1]
        for q in range(seq):
            for k in range(seq):
                if k <= q:
                    assert _allowed(mask, 0, q, k), f"q={q}, k={k} should be allowed"
                else:
                    assert not _allowed(mask, 0, q, k), f"q={q}, k={k} should be blocked"

    def test_single_block_pruning(self):
        """Tokens after [RETURN] cannot attend to [THOUGHT]..[SOLUTION) span."""
        # positions:  0   1          2  3         4          5       6          7
        ids = torch.tensor([[1, THOUGHT_ID, 2, 3, SOLUTION_ID, 4, RETURN_ID, 5]])
        mask = self._build(ids)

        # Token at position 7 (after RETURN at 6) must NOT attend to positions 1,2,3
        # (the [THOUGHT]..[SOLUTION) span = positions 1..4 exclusive = 1,2,3)
        for k in [1, 2, 3]:
            assert not _allowed(mask, 0, 7, k), f"pos 7 should NOT attend to {k}"

        # But position 7 CAN attend to positions before the thought (0)
        # and to the solution summary (4, 5) and return (6)
        for k in [0, 4, 5, 6]:
            assert _allowed(mask, 0, 7, k), f"pos 7 SHOULD attend to {k}"

        # Tokens inside the reasoning span can still attend to each other (causal)
        assert _allowed(mask, 0, 3, 2)
        assert _allowed(mask, 0, 3, 1)

    def test_tokens_before_return_unaffected(self):
        """Tokens at or before [RETURN] still attend normally (causal)."""
        ids = torch.tensor([[1, THOUGHT_ID, 2, SOLUTION_ID, 3, RETURN_ID, 4]])
        mask = self._build(ids)

        # Position 5 = RETURN itself: can attend to everything before it
        for k in range(6):
            assert _allowed(mask, 0, 5, k), f"RETURN (5) should attend to {k}"

        # Position 4 (between SOLUTION and RETURN): causal access
        for k in range(5):
            assert _allowed(mask, 0, 4, k)

    def test_nested_blocks(self):
        """After outer [RETURN], both inner and outer reasoning spans are pruned."""
        ids = torch.tensor([[
            THOUGHT_ID,      # 0
            10,              # 1
            THOUGHT_ID,      # 2
            20,              # 3
            SOLUTION_ID,     # 4
            21,              # 5
            RETURN_ID,       # 6
            SOLUTION_ID,     # 7
            30,              # 8
            RETURN_ID,       # 9
            99,              # 10
        ]])
        mask = self._build(ids)

        # After outer return (pos 10):
        # - inner reasoning span [2,4) = positions 2,3 should be blocked
        # - outer reasoning span [0,7) = positions 0,1,2,3,4,5,6 should be blocked
        for k in [0, 1, 2, 3, 4, 5, 6]:
            assert not _allowed(mask, 0, 10, k), f"pos 10 should NOT attend to {k}"

        # But can attend to outer solution summary and beyond
        for k in [7, 8, 9]:
            assert _allowed(mask, 0, 10, k), f"pos 10 SHOULD attend to {k}"

        # After inner return (pos 7 onward but before outer return):
        # position 8 should NOT attend to inner reasoning [2,4) = 2,3
        for k in [2, 3]:
            assert not _allowed(mask, 0, 8, k), f"pos 8 should NOT attend to {k}"

        # But position 8 CAN attend to the outer thought region (0, 1) since
        # the outer block isn't closed yet at position 8
        for k in [0, 1]:
            assert _allowed(mask, 0, 8, k), f"pos 8 SHOULD attend to {k}"

    def test_padding_mask(self):
        """Padding positions should be masked out everywhere."""
        ids = torch.tensor([[1, 2, 3, 0]])  # last token is padding
        padding = torch.tensor([[1, 1, 1, 0]])
        mask = self._build(ids, padding_mask=padding)

        # No query should attend to position 3 (padded)
        for q in range(4):
            assert not _allowed(mask, 0, q, 3), f"q={q} should NOT attend to pad"

        # Non-pad positions still follow causal rule
        assert _allowed(mask, 0, 2, 0)
        assert _allowed(mask, 0, 2, 1)
        assert _allowed(mask, 0, 2, 2)

    def test_batch_independence(self):
        """Each batch element should have its own mask based on its tokens."""
        ids = torch.tensor([
            [1, THOUGHT_ID, 2, SOLUTION_ID, 3, RETURN_ID, 4],
            [1, 2, 3, 4, 5, 6, 7],  # no special tokens
        ])
        mask = self._build(ids)

        # Batch 0: post-return (pos 6) blocked from reasoning [1,3)
        assert not _allowed(mask, 0, 6, 1)
        assert not _allowed(mask, 0, 6, 2)

        # Batch 1: pure causal — pos 6 can attend to everything before it
        for k in range(7):
            assert _allowed(mask, 1, 6, k)

    def test_shape(self):
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = self._build(ids)
        assert mask.shape == (2, 1, 3, 3)
        assert mask.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# FlexAttentionMaskMixin
# ---------------------------------------------------------------------------


def _can_import_flex():
    try:
        from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


@pytest.mark.skipif(not _can_import_flex(), reason="FlexAttention not available")
class TestFlexAttentionMask:
    """Verify FlexAttention mask_mod produces the same dense mask as MaterialisedMaskMixin.

    We use ``create_mask`` (element-level) instead of ``create_block_mask``
    (block-level) so we can compare directly against the materialised output.
    """

    @staticmethod
    def _flex_dense(input_ids, padding_mask=None):
        """Build a dense boolean mask using the same logic as FlexAttentionMaskMixin."""
        from torch.nn.attention.flex_attention import create_mask

        batch_size, seq_len = input_ids.shape
        batch_blocks = extract_cot_blocks(input_ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        _mbq = build_min_blocked_q(input_ids, batch_blocks)
        _pm = padding_mask

        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            not_pruned = q_idx < _mbq[b, kv_idx]
            if _pm is not None:
                return causal & not_pruned & _pm[b, kv_idx].bool()
            return causal & not_pruned

        # create_mask returns (B, H, Q_LEN, KV_LEN) bool tensor
        return create_mask(mask_mod, B=batch_size, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=input_ids.device)

    def test_matches_materialised_single_block(self):
        ids = torch.tensor([[1, THOUGHT_ID, 2, 3, SOLUTION_ID, 4, RETURN_ID, 5]])
        mat = MaterialisedTester()._build_hierarchical_mask(ids)
        mat_bool = mat[:, 0, :, :] == 0.0
        flex_bool = self._flex_dense(ids)[:, 0, :, :]
        assert torch.equal(mat_bool, flex_bool)

    def test_matches_materialised_no_blocks(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        mat = MaterialisedTester()._build_hierarchical_mask(ids)
        mat_bool = mat[:, 0, :, :] == 0.0
        flex_bool = self._flex_dense(ids)[:, 0, :, :]
        assert torch.equal(mat_bool, flex_bool)

    def test_matches_materialised_nested(self):
        ids = torch.tensor([[
            THOUGHT_ID, 10, THOUGHT_ID, 20, SOLUTION_ID, 21,
            RETURN_ID, SOLUTION_ID, 30, RETURN_ID, 99,
        ]])
        mat = MaterialisedTester()._build_hierarchical_mask(ids)
        mat_bool = mat[:, 0, :, :] == 0.0
        flex_bool = self._flex_dense(ids)[:, 0, :, :]
        assert torch.equal(mat_bool, flex_bool)

    def test_block_mask_builds_without_error(self):
        """Smoke test: FlexAttentionMaskMixin._build_hierarchical_mask runs."""
        ids = torch.tensor([[1, THOUGHT_ID, 2, SOLUTION_ID, 3, RETURN_ID, 4]])
        mask = FlexTester()._build_hierarchical_mask(ids)
        assert mask is not None


# ---------------------------------------------------------------------------
# build_min_blocked_q
# ---------------------------------------------------------------------------


class TestBuildMinBlockedQ:
    def test_no_blocks(self):
        ids = torch.tensor([[1, 2, 3]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        mbq = build_min_blocked_q(ids, blocks)
        # All entries should be seq_len (never blocked)
        assert (mbq == 3).all()

    def test_single_block(self):
        ids = torch.tensor([[1, THOUGHT_ID, 2, SOLUTION_ID, 4, RETURN_ID, 5]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        mbq = build_min_blocked_q(ids, blocks)
        seq_len = ids.shape[1]
        # Positions 1,2 (thought..solution exclusive) should be blocked starting at return+1=6
        assert mbq[0, 1].item() == 6
        assert mbq[0, 2].item() == 6
        # Other positions remain at seq_len
        assert mbq[0, 0].item() == seq_len
        assert mbq[0, 3].item() == seq_len
