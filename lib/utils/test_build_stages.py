"""Tests for build_stages — staged pruning sequence construction."""

import torch
import pytest

from .  import build_stages, find_cot_blocks

# Token IDs
THOUGHT_ID = 100
SOLUTION_ID = 101
RETURN_ID = 102
PAD_ID = 0


def _ids(*tokens):
    """Helper to create a 1-D tensor of token IDs."""
    return torch.tensor(tokens, dtype=torch.long)


def _mask(*vals):
    """Helper to create a 1-D attention mask tensor."""
    return torch.tensor(vals, dtype=torch.long)


def _find_blocks_1d(ids):
    """Run find_cot_blocks on a single (unbatched) sequence."""
    return find_cot_blocks(ids.unsqueeze(0), THOUGHT_ID, SOLUTION_ID, RETURN_ID)[0]


# ---------------------------------------------------------------------------
# Single block
# ---------------------------------------------------------------------------

class TestSingleBlock:
    """A B [TH] c d [SOL] e [RET] f g"""

    def setup_method(self):
        #               0  1   2     3  4   5     6   7     8  9
        self.ids = _ids(10, 11, THOUGHT_ID, 12, 13, SOLUTION_ID, 14, RETURN_ID, 15, 16)
        self.labels = self.ids.clone()
        self.mask = _mask(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        self.blocks = _find_blocks_1d(self.ids)

    def test_two_stages_produced(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        assert len(stages) == 2

    def test_stage0_is_full_sequence(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        ids, labels, mask = stages[0]
        # Stage 0 uses the full sequence (no pruning yet)
        assert ids.tolist() == self.ids.tolist()
        # Loss only for positions 0..7 (up to [RETURN] inclusive)
        for j in range(8):
            assert labels[j].item() == self.ids[j].item()
        for j in range(8, 10):
            assert labels[j].item() == -100

    def test_stage1_pruned_sequence(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        ids, labels, mask = stages[1]
        # Thought content (positions 3,4,5 = c, d, [SOL]) removed
        # Remaining: A(0) B(1) [TH](2) e(6) [RET](7) f(8) g(9)
        expected_ids = [10, 11, THOUGHT_ID, 14, RETURN_ID, 15, 16]
        assert ids.tolist() == expected_ids

    def test_stage1_labels(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        _, labels, _ = stages[1]
        # Loss only for original positions 8..9 (after [RETURN])
        # In pruned sequence: positions 0..4 are from original 0,1,2,6,7 → all < 8 → -100
        # Position 5 from original 8 → label = 15
        # Position 6 from original 9 → label = 16
        expected = [-100, -100, -100, -100, -100, 15, 16]
        assert labels.tolist() == expected

    def test_stage1_contiguous_positions(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        ids, _, mask = stages[1]
        # Positions are just 0..len-1 (contiguous) by default when model processes
        assert mask.sum().item() == len(ids)


# ---------------------------------------------------------------------------
# Nested blocks
# ---------------------------------------------------------------------------

class TestNestedBlocks:
    """A [TH1] c [TH2] d [SOL2] e [RET2] f [SOL1] g [RET1] h"""

    def setup_method(self):
        #               0   1          2   3          4   5           6   7          8   9           10  11          12
        self.ids = _ids(10, THOUGHT_ID, 12, THOUGHT_ID, 13, SOLUTION_ID, 14, RETURN_ID, 15, SOLUTION_ID, 16, RETURN_ID, 17)
        self.labels = self.ids.clone()
        self.mask = torch.ones(13, dtype=torch.long)
        self.blocks = _find_blocks_1d(self.ids)

    def test_three_stages_produced(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        assert len(stages) == 3

    def test_stage0_full_sequence_loss_up_to_inner_return(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        ids, labels, _ = stages[0]
        # Full sequence, loss for positions 0..7 (inner [RET2])
        assert ids.tolist() == self.ids.tolist()
        for j in range(8):
            assert labels[j].item() == self.ids[j].item()
        for j in range(8, 13):
            assert labels[j].item() == -100

    def test_stage1_inner_pruned(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        ids, labels, _ = stages[1]
        # Inner block (thought=3, solution=5) pruned: remove positions 4,5 (d, [SOL2])
        # Kept: 0,1,2,3,6,7,8,9,10,11,12
        expected_ids = [10, THOUGHT_ID, 12, THOUGHT_ID, 14, RETURN_ID, 15, SOLUTION_ID, 16, RETURN_ID, 17]
        assert ids.tolist() == expected_ids
        # Loss for original positions 8..11 (between [RET2]+1 and [RET1])
        # Mapped to pruned: orig 8→pruned 6, orig 9→7, orig 10→8, orig 11→9
        for j in range(6):
            assert labels[j].item() == -100
        assert labels[6].item() == 15   # orig 8
        assert labels[7].item() == SOLUTION_ID  # orig 9 = [SOL1]
        assert labels[8].item() == 16   # orig 10
        assert labels[9].item() == RETURN_ID  # orig 11 = [RET1]
        assert labels[10].item() == -100  # orig 12 → after loss range

    def test_stage2_fully_pruned(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        ids, labels, _ = stages[2]
        # Both blocks pruned:
        #   Inner removes positions 4,5
        #   Outer removes positions 2,3,4,5,6,7,8,9 (thought_pos1+1=2 to solution_pos1=9)
        #   Union: {2,3,4,5,6,7,8,9}
        # Kept: 0,1,10,11,12
        expected_ids = [10, THOUGHT_ID, 16, RETURN_ID, 17]
        assert ids.tolist() == expected_ids
        # Loss for original positions 12+ (after [RET1])
        # Only orig 12 maps to pruned position 4
        assert labels.tolist() == [-100, -100, -100, -100, 17]


# ---------------------------------------------------------------------------
# No blocks
# ---------------------------------------------------------------------------

class TestNoBlocks:
    def test_single_stage_standard_sft(self):
        ids = _ids(10, 11, 12, 13)
        labels = ids.clone()
        mask = _mask(1, 1, 1, 1)
        stages = build_stages(ids, labels, mask, [])
        assert len(stages) == 1
        s_ids, s_labels, s_mask = stages[0]
        assert s_ids.tolist() == ids.tolist()
        assert s_labels.tolist() == labels.tolist()


# ---------------------------------------------------------------------------
# Multiple non-overlapping blocks
# ---------------------------------------------------------------------------

class TestMultipleNonOverlapping:
    """A [TH1] c [SOL1] d [RET1] B [TH2] e [SOL2] f [RET2] g"""

    def setup_method(self):
        #               0   1          2   3           4   5          6   7          8   9           10  11          12
        self.ids = _ids(10, THOUGHT_ID, 12, SOLUTION_ID, 13, RETURN_ID, 11, THOUGHT_ID, 14, SOLUTION_ID, 15, RETURN_ID, 16)
        self.labels = self.ids.clone()
        self.mask = torch.ones(13, dtype=torch.long)
        self.blocks = _find_blocks_1d(self.ids)

    def test_three_stages(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        assert len(stages) == 3

    def test_stage0_loss_up_to_first_return(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        _, labels, _ = stages[0]
        # Loss for positions 0..5 ([RET1] at pos 5)
        for j in range(6):
            assert labels[j].item() == self.ids[j].item()
        for j in range(6, 13):
            assert labels[j].item() == -100

    def test_stage1_first_block_pruned(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        ids, labels, _ = stages[1]
        # Block 1 (thought=1, solution=3) pruned: remove positions 2,3 (c, [SOL1])
        # Kept: 0,1,4,5,6,7,8,9,10,11,12
        expected_ids = [10, THOUGHT_ID, 13, RETURN_ID, 11, THOUGHT_ID, 14, SOLUTION_ID, 15, RETURN_ID, 16]
        assert ids.tolist() == expected_ids
        # Loss for original positions 6..11 (between [RET1]+1 and [RET2])
        # orig 6→pruned 4, ..., orig 11→pruned 9
        for j in range(4):
            assert labels[j].item() == -100
        assert labels[4].item() == 11   # orig 6 = B
        assert labels[9].item() == RETURN_ID  # orig 11 = [RET2]
        assert labels[10].item() == -100  # orig 12 → after loss range

    def test_stage2_both_pruned(self):
        stages = build_stages(self.ids, self.labels, self.mask, self.blocks)
        ids, labels, _ = stages[2]
        # Both blocks pruned: remove {2,3} ∪ {8,9}
        # Kept: 0,1,4,5,6,7,10,11,12
        expected_ids = [10, THOUGHT_ID, 13, RETURN_ID, 11, THOUGHT_ID, 15, RETURN_ID, 16]
        assert ids.tolist() == expected_ids
        # Loss for positions 12+ → only orig 12 = pruned position 8
        assert labels.tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, 16]


# ---------------------------------------------------------------------------
# No tokens after last [RETURN]
# ---------------------------------------------------------------------------

class TestNoTokensAfterReturn:
    def test_last_stage_omitted(self):
        #              0   1          2   3           4   5
        ids = _ids(10, THOUGHT_ID, 12, SOLUTION_ID, 13, RETURN_ID)
        labels = ids.clone()
        mask = _mask(1, 1, 1, 1, 1, 1)
        blocks = _find_blocks_1d(ids)
        stages = build_stages(ids, labels, mask, blocks)
        # Stage 0: loss for positions 0..5 (up to [RETURN])
        # Stage 1 would cover positions 6+ → nothing → omitted
        assert len(stages) == 1


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

class TestPadding:
    def test_padding_excluded(self):
        #              0   1          2   3           4   5          6  7       8
        ids = _ids(10, THOUGHT_ID, 12, SOLUTION_ID, 13, RETURN_ID, 14, PAD_ID, PAD_ID)
        labels = _ids(10, THOUGHT_ID, 12, SOLUTION_ID, 13, RETURN_ID, 14, -100, -100)
        mask = _mask(1, 1, 1, 1, 1, 1, 1, 0, 0)
        blocks = _find_blocks_1d(ids)
        stages = build_stages(ids, labels, mask, blocks)
        assert len(stages) == 2

        # Stage 1: pruned, padding excluded
        s1_ids, s1_labels, s1_mask = stages[1]
        # Kept non-pad non-removed: 0,1,4,5,6 (removed 2,3; excluded 7,8 padding)
        assert s1_ids.tolist() == [10, THOUGHT_ID, 13, RETURN_ID, 14]
        # Loss for position 6 (after [RETURN])
        assert s1_labels.tolist() == [-100, -100, -100, -100, 14]
        # All real tokens
        assert s1_mask.sum().item() == 5


# ---------------------------------------------------------------------------
# Labels already partially masked (prompt masking)
# ---------------------------------------------------------------------------

class TestPromptMasking:
    def test_respects_existing_label_mask(self):
        """If original labels already have -100 (e.g. prompt), those stay masked."""
        #              0   1          2   3           4   5          6  7
        ids = _ids(10, THOUGHT_ID, 12, SOLUTION_ID, 13, RETURN_ID, 14, 15)
        # Positions 0,1 are prompt → labels = -100
        labels = _ids(-100, -100, 12, SOLUTION_ID, 13, RETURN_ID, 14, 15)
        mask = _mask(1, 1, 1, 1, 1, 1, 1, 1)
        blocks = _find_blocks_1d(ids)
        stages = build_stages(ids, labels, mask, blocks)

        # Stage 0: labels[0] and labels[1] stay -100 from original
        s0_labels = stages[0][1]
        assert s0_labels[0].item() == -100
        assert s0_labels[1].item() == -100
        # labels[2] = 12 (within loss range and not prompt-masked)
        assert s0_labels[2].item() == 12
