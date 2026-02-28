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
    build_position_ids,
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
        """Tokens after [RETURN] cannot attend to thought content and [SOLUTION], but keep [THOUGHT]."""
        # positions:  0   1          2  3         4          5       6          7
        ids = torch.tensor([[1, THOUGHT_ID, 2, 3, SOLUTION_ID, 4, RETURN_ID, 5]])
        mask = self._build(ids)

        # Token at position 7 (after RETURN at 6) must NOT attend to positions 2,3,4
        # (the thought content + [SOLUTION] = positions thought+1..solution+1 = 2,3,4)
        for k in [2, 3, 4]:
            assert not _allowed(mask, 0, 7, k), f"pos 7 should NOT attend to {k}"

        # But position 7 CAN attend to positions before the thought (0),
        # the [THOUGHT] token itself (1), the solution summary (5) and return (6)
        for k in [0, 1, 5, 6]:
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
        # - inner block masks thought+1..solution+1 = positions 3,4
        # - outer block masks thought+1..solution+1 = positions 1,2,3,4,5,6,7
        # Combined: positions 1..7 should be blocked (but [THOUGHT] tokens 0,2 kept)
        for k in [1, 2, 3, 4, 5, 6, 7]:
            assert not _allowed(mask, 0, 10, k), f"pos 10 should NOT attend to {k}"

        # pos 10 CAN attend to outer [THOUGHT] (0) and outer solution summary (8), return (9)
        for k in [0, 8, 9]:
            assert _allowed(mask, 0, 10, k), f"pos 10 SHOULD attend to {k}"

        # After inner return (pos 7 onward but before outer return):
        # position 8 should NOT attend to inner thought content + [SOLUTION] = 3,4
        for k in [3, 4]:
            assert not _allowed(mask, 0, 8, k), f"pos 8 should NOT attend to {k}"

        # But position 8 CAN attend to the outer thought region (0, 1) and
        # inner [THOUGHT] (2) since the outer block isn't closed yet at position 8
        for k in [0, 1, 2]:
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

        # Batch 0: post-return (pos 6) blocked from thought content + [SOLUTION] (pos 2, 3)
        # but [THOUGHT] (pos 1) is kept
        assert _allowed(mask, 0, 6, 1)
        assert not _allowed(mask, 0, 6, 2)
        assert not _allowed(mask, 0, 6, 3)

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
        # Positions 2,3 (thought+1..solution+1) should be blocked starting at return+1=6
        # [THOUGHT] at pos 1 is kept (not blocked)
        assert mbq[0, 2].item() == 6
        assert mbq[0, 3].item() == 6
        # Other positions remain at seq_len (including [THOUGHT] at pos 1)
        assert mbq[0, 0].item() == seq_len
        assert mbq[0, 1].item() == seq_len


# ---------------------------------------------------------------------------
# build_position_ids
# ---------------------------------------------------------------------------


class TestBuildPositionIds:
    def test_no_blocks(self):
        """Without blocks, position_ids should be identity [0, 1, 2, ...]."""
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        pos = build_position_ids(ids, blocks)
        expected = torch.arange(5).unsqueeze(0)
        assert torch.equal(pos, expected)

    def test_single_block(self):
        """After a single block, tokens beyond solution_pos are shifted back."""
        # positions:  0   1          2  3         4          5       6
        ids = torch.tensor([[1, THOUGHT_ID, 2, 3, SOLUTION_ID, 4, RETURN_ID]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        pos = build_position_ids(ids, blocks)
        # Block: thought=1, solution=4, return=6
        # span_len = solution - thought = 4 - 1 = 3
        # Positions 0..4 unchanged; positions 5,6 shifted by -3
        assert pos[0, :5].tolist() == [0, 1, 2, 3, 4]
        assert pos[0, 5].item() == 5 - 3  # = 2
        assert pos[0, 6].item() == 6 - 3  # = 3

    def test_single_block_with_trailing(self):
        """Tokens after [RETURN] also get shifted."""
        # positions:  0   1          2  3         4          5       6          7
        ids = torch.tensor([[1, THOUGHT_ID, 2, 3, SOLUTION_ID, 4, RETURN_ID, 5]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        pos = build_position_ids(ids, blocks)
        # Block: thought=1, solution=4, return=6; span_len=3
        # Positions 5,6,7 shifted by -3
        assert pos[0, 7].item() == 7 - 3  # = 4

    def test_multiple_blocks_cumulative_shift(self):
        """Multiple blocks accumulate position shifts."""
        # Block 1: positions 1(T) 2 3(S) 4 5(R)
        # Block 2: positions 7(T) 8 9(S) 10 11(R)
        # trailing: position 12
        ids = torch.tensor([[
            1,              # 0
            THOUGHT_ID,     # 1
            2,              # 2
            SOLUTION_ID,    # 3
            4,              # 4
            RETURN_ID,      # 5
            6,              # 6
            THOUGHT_ID,     # 7
            8,              # 8
            SOLUTION_ID,    # 9
            10,             # 10
            RETURN_ID,      # 11
            12,             # 12
        ]])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        pos = build_position_ids(ids, blocks)

        # Block 1: thought=1, solution=3, span_len=2
        # Positions 4+ shifted by -2
        # Block 2: thought=7, solution=9, span_len=2
        # Positions 10+ shifted by an additional -2 (total -4)

        # Positions inside block 1 span (2,3): unchanged
        assert pos[0, 2].item() == 2
        assert pos[0, 3].item() == 3
        # Position 4 (solution content, after block 1's solution_pos+1): shifted by -2
        assert pos[0, 4].item() == 4 - 2  # = 2
        assert pos[0, 5].item() == 5 - 2  # = 3
        assert pos[0, 6].item() == 6 - 2  # = 4
        # Positions inside block 2 span (8,9): shifted by -2 (from block 1)
        assert pos[0, 8].item() == 8 - 2  # = 6
        assert pos[0, 9].item() == 9 - 2  # = 7
        # Positions 10+ shifted by -2 (block 1) -2 (block 2) = -4
        assert pos[0, 10].item() == 10 - 4  # = 6
        assert pos[0, 11].item() == 11 - 4  # = 7
        assert pos[0, 12].item() == 12 - 4  # = 8

    def test_batch_independence(self):
        """Each batch element has independent position_ids."""
        ids = torch.tensor([
            [1, THOUGHT_ID, 2, SOLUTION_ID, 3, RETURN_ID, 4],
            [1, 2, 3, 4, 5, 6, 7],  # no blocks
        ])
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        pos = build_position_ids(ids, blocks)

        # Batch 0: thought=1, solution=3, span_len=2; positions 4+ shifted by -2
        assert pos[0, 4].item() == 2
        assert pos[0, 5].item() == 3
        assert pos[0, 6].item() == 4

        # Batch 1: identity (no blocks)
        assert pos[1].tolist() == list(range(7))


# ---------------------------------------------------------------------------
# MaterialisedMaskMixin._build_hierarchical_mask_and_position_ids
# ---------------------------------------------------------------------------


class TestMaterialisedMaskAndPositionIds:
    def _build(self, input_ids, padding_mask=None):
        tester = MaterialisedTester()
        return tester._build_hierarchical_mask_and_position_ids(
            input_ids, padding_mask=padding_mask
        )

    def test_returns_tuple(self):
        ids = torch.tensor([[1, THOUGHT_ID, 2, SOLUTION_ID, 3, RETURN_ID, 4]])
        result = self._build(ids)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mask_matches_plain_method(self):
        """The mask from the tuple method should match the standalone method."""
        ids = torch.tensor([[1, THOUGHT_ID, 2, SOLUTION_ID, 3, RETURN_ID, 4]])
        mask_plain = MaterialisedTester()._build_hierarchical_mask(ids)
        mask_tuple, _ = self._build(ids)
        assert torch.equal(mask_plain, mask_tuple)

    def test_position_ids_match_build_fn(self):
        """position_ids from the tuple method should match build_position_ids."""
        ids = torch.tensor([[1, THOUGHT_ID, 2, SOLUTION_ID, 3, RETURN_ID, 4]])
        _, pos = self._build(ids)
        blocks = extract_cot_blocks(ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID)
        expected = build_position_ids(ids, blocks)
        assert torch.equal(pos, expected)

    def test_no_blocks_identity_positions(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        _, pos = self._build(ids)
        expected = torch.arange(4).unsqueeze(0)
        assert torch.equal(pos, expected)
