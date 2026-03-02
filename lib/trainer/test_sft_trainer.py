"""Tests for HCotSFTTrainer._build_pruned_inputs."""

import torch
import utils

# ---------------------------------------------------------------------------
# Token IDs used throughout tests
# ---------------------------------------------------------------------------

THOUGHT_ID = 100
SOLUTION_ID = 101
RETURN_ID = 102
PAD_ID = 0


def _build_pruned(input_ids, labels=None, pad_id=PAD_ID):
    """Standalone helper that mimics _build_pruned_inputs logic.

    We test the pruning logic directly rather than instantiating the full
    trainer (which requires a model, tokenizer, datasets, etc.).
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if labels is None:
        labels = input_ids.clone()

    batch_blocks = utils.find_cot_blocks(
        input_ids, THOUGHT_ID, SOLUTION_ID, RETURN_ID
    )

    pruned_rows: list[torch.Tensor] = []
    pruned_labels: list[torch.Tensor] = []
    for b in range(batch_size):
        keep = torch.ones(seq_len, dtype=torch.bool, device=device)
        for thought_pos, solution_pos, _return_pos in batch_blocks[b]:
            keep[thought_pos + 1 : solution_pos + 1] = False
        pruned_rows.append(input_ids[b][keep])
        pruned_labels.append(labels[b][keep])

    max_len = max(r.shape[0] for r in pruned_rows)
    new_input_ids = torch.full(
        (batch_size, max_len), pad_id, dtype=input_ids.dtype, device=device
    )
    new_labels = torch.full(
        (batch_size, max_len), -100, dtype=labels.dtype, device=device
    )
    new_attn = torch.zeros(
        (batch_size, max_len), dtype=torch.long, device=device
    )
    for b, (r, l) in enumerate(zip(pruned_rows, pruned_labels)):
        new_input_ids[b, : r.shape[0]] = r
        new_labels[b, : l.shape[0]] = l
        new_attn[b, : r.shape[0]] = 1

    return new_input_ids, new_labels, new_attn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildPrunedInputs:
    def test_no_blocks(self):
        """Without special tokens the output should equal the input."""
        ids = torch.tensor([[1, 2, 3, 4]])
        out_ids, out_labels, out_attn = _build_pruned(ids)
        assert out_ids.tolist() == [[1, 2, 3, 4]]
        assert out_attn.tolist() == [[1, 1, 1, 1]]

    def test_single_block(self):
        """Thinking tokens (thought+1 .. solution) should be removed."""
        #  index:  0   1          2   3         4          5       6          7
        ids = torch.tensor([[1, THOUGHT_ID, 2, 3, SOLUTION_ID, 4, RETURN_ID, 5]])
        out_ids, _, out_attn = _build_pruned(ids)
        # Removed: indices 2, 3, 4 (thought+1 .. solution inclusive)
        # Kept: 1, [THOUGHT], 4, [RETURN], 5
        assert out_ids[0].tolist() == [1, THOUGHT_ID, 4, RETURN_ID, 5]
        assert out_attn[0].tolist() == [1, 1, 1, 1, 1]

    def test_nested_blocks(self):
        """Both inner and outer thinking spans should be removed."""
        ids = torch.tensor([[
            THOUGHT_ID,      # 0 outer-thought
            10,              # 1 (outer thinking)
            THOUGHT_ID,      # 2 inner-thought
            20,              # 3 (inner thinking)
            SOLUTION_ID,     # 4 inner-solution
            21,              # 5 (inner solution content)
            RETURN_ID,       # 6 inner-return
            SOLUTION_ID,     # 7 outer-solution
            30,              # 8 (outer solution content)
            RETURN_ID,       # 9 outer-return
            99,              # 10 (visible after)
        ]])
        out_ids, _, out_attn = _build_pruned(ids)
        # Inner block (2,4,6) prunes indices 3,4 → tokens 20, SOLUTION_ID
        # Outer block (0,7,9) prunes indices 1,2,3,4,5,6,7 → 10, [T], 20, [S], 21, [R], [S]
        # Combined prune: indices 1,2,3,4,5,6,7
        # Kept: [THOUGHT](0), 30(8), [RETURN](9), 99(10)
        expected = [THOUGHT_ID, 30, RETURN_ID, 99]
        assert out_ids[0].tolist() == expected

    def test_two_sequential_blocks(self):
        """Two non-nested blocks in sequence."""
        ids = torch.tensor([[
            1,
            THOUGHT_ID, 10, 11, SOLUTION_ID, 20, RETURN_ID,  # block 1
            2,
            THOUGHT_ID, 30, SOLUTION_ID, 40, RETURN_ID,      # block 2
            3,
        ]])
        out_ids, _, _ = _build_pruned(ids)
        # Block 1 (1,4,6): prune indices 2,3,4 → 10, 11, [S]
        # Block 2 (8,10,12): prune indices 9,10 → 30, [S]
        # Kept: 1, [T], 20, [R], 2, [T], 40, [R], 3
        expected = [1, THOUGHT_ID, 20, RETURN_ID, 2, THOUGHT_ID, 40, RETURN_ID, 3]
        assert out_ids[0].tolist() == expected

    def test_empty_thinking(self):
        """[THOUGHT][SOLUTION] adjacent → nothing between to prune (just [S])."""
        ids = torch.tensor([[1, THOUGHT_ID, SOLUTION_ID, 2, RETURN_ID, 3]])
        out_ids, _, _ = _build_pruned(ids)
        # Block (1,2,4): prune index 2 → [SOLUTION]
        # Kept: 1, [T], 2, [R], 3
        expected = [1, THOUGHT_ID, 2, RETURN_ID, 3]
        assert out_ids[0].tolist() == expected

    def test_labels_pruned_consistently(self):
        """Labels should have the same tokens removed as input_ids."""
        ids = torch.tensor([[1, THOUGHT_ID, 10, SOLUTION_ID, 20, RETURN_ID, 5]])
        # Labels: -100 for non-generation (first token), real ids otherwise
        labels = torch.tensor([[-100, THOUGHT_ID, 10, SOLUTION_ID, 20, RETURN_ID, 5]])
        out_ids, out_labels, _ = _build_pruned(ids, labels=labels)
        # Pruned: indices 2,3 (10, [S])
        assert out_ids[0].tolist() == [1, THOUGHT_ID, 20, RETURN_ID, 5]
        assert out_labels[0].tolist() == [-100, THOUGHT_ID, 20, RETURN_ID, 5]

    def test_batch_mixed(self):
        """Batch with one element having a block, another without."""
        ids = torch.tensor([
            [1, THOUGHT_ID, 10, SOLUTION_ID, 20, RETURN_ID, 5],
            [1, 2, 3, 4, 5, 6, 7],  # no special tokens
        ])
        out_ids, _, out_attn = _build_pruned(ids)
        # Batch 0: prune indices 2,3 → [1, T, 20, R, 5] (len 5)
        # Batch 1: no prune → [1, 2, 3, 4, 5, 6, 7] (len 7)
        # Padded to max_len=7
        assert out_ids[0].tolist() == [1, THOUGHT_ID, 20, RETURN_ID, 5, PAD_ID, PAD_ID]
        assert out_ids[1].tolist() == [1, 2, 3, 4, 5, 6, 7]
        # Attention mask: 1s for real, 0s for padding
        assert out_attn[0].tolist() == [1, 1, 1, 1, 1, 0, 0]
        assert out_attn[1].tolist() == [1, 1, 1, 1, 1, 1, 1]

    def test_labels_padding_is_minus_100(self):
        """Padded label positions should be -100 (ignored by loss)."""
        ids = torch.tensor([
            [1, THOUGHT_ID, 10, SOLUTION_ID, 20, RETURN_ID, 5],
            [1, 2, 3, 4, 5, 6, 7],
        ])
        _, out_labels, _ = _build_pruned(ids)
        # Batch 0 is shorter (5 tokens) → padded positions should be -100
        assert out_labels[0, 5].item() == -100
        assert out_labels[0, 6].item() == -100
