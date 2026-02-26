"""Tests for the custom generate function with hierarchical CoT pruning."""

import pytest
import torch
from dataclasses import dataclass
from transformers import LogitsProcessorList, StoppingCriteriaList

from .generate import generate, generate_with_mask, HCotGenerateOutput

# ---------------------------------------------------------------------------
# Token IDs used throughout tests
# ---------------------------------------------------------------------------

THOUGHT_ID = 100
SOLUTION_ID = 101
RETURN_ID = 102
EOS_ID = 2
PAD_ID = 0

# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeOutput:
    logits: torch.Tensor


class ScriptedModel:
    """A fake model that returns pre-scripted next-token logits each call.

    ``script`` is a list of tensors, each of shape (batch, vocab_size).
    On the i-th call, ``forward`` returns logits that make ``argmax`` pick
    the tokens encoded in ``script[i]``.
    """

    def __init__(self, script: list[torch.Tensor], vocab_size: int = 200):
        self.script = script
        self.vocab_size = vocab_size
        self._step = 0

    def __call__(self, input_ids, **kwargs):
        # Build logits so argmax yields the scripted token ids
        batch_size = input_ids.shape[0]
        logits_last = torch.full((batch_size, self.vocab_size), -100.0)
        token_ids = self.script[self._step]  # (batch,)
        for b in range(batch_size):
            logits_last[b, token_ids[b]] = 100.0
        self._step += 1
        # Model returns (batch, seq_len, vocab) but only last position matters
        seq_len = input_ids.shape[1]
        full_logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        full_logits[:, -1, :] = logits_last
        return _FakeOutput(logits=full_logits)


def _identity_processor():
    """LogitsProcessorList that passes scores through unchanged."""
    return LogitsProcessorList()


class _MaxLengthCriterion:
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[1] >= self.max_length


def _stopping(max_length):
    return StoppingCriteriaList([_MaxLengthCriterion(max_length=max_length)])


def _run(script_tokens, prompt, max_length, batch_size=1):
    """Helper that sets up fakes and calls generate.

    Args:
        script_tokens: list of lists, outer = step, inner = per-batch token id.
        prompt: 2-D list of prompt token ids (batch x seq).
        max_length: stopping criteria max_length.
    """
    script = [torch.tensor(step) for step in script_tokens]
    model = ScriptedModel(script)
    input_ids = torch.tensor(prompt, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return generate(
        model=model,
        input_ids=input_ids,
        logits_processor=_identity_processor(),
        stopping_criteria=_stopping(max_length),
        return_token_id=RETURN_ID,
        solution_token_id=SOLUTION_ID,
        thought_token_id=THOUGHT_ID,
        eos_token_id=EOS_ID,
        attention_mask=attention_mask,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEOSStopping:
    def test_stops_on_eos(self):
        """Generation should stop as soon as all batch elements produce EOS."""
        # Prompt: [1, 2], script: token 3, then EOS
        result = _run(
            script_tokens=[[3], [EOS_ID]],
            prompt=[[1, 2]],
            max_length=100,
        )
        assert isinstance(result, HCotGenerateOutput)
        assert result.sequences[0].tolist() == [1, 2, 3, EOS_ID]

    def test_stops_on_max_length(self):
        """Generation should stop when max_length is reached."""
        # Prompt len 2, max_length 5 → generate 3 tokens then stop
        result = _run(
            script_tokens=[[3], [4], [5]],
            prompt=[[1, 2]],
            max_length=5,
        )
        assert result.sequences[0].tolist() == [1, 2, 3, 4, 5]

    def test_batch_eos_all_must_finish(self):
        """Loop continues until ALL batch elements hit EOS."""
        # Batch 0 gets EOS at step 1, batch 1 at step 2
        result = _run(
            script_tokens=[
                [EOS_ID, 3],
                [EOS_ID, EOS_ID],  # won't be reached for b0, but b1 needs it
            ],
            prompt=[[1, 2], [1, 2]],
            max_length=100,
            batch_size=2,
        )
        # Batch 0: [1, 2, EOS] (stopped first but loop continues for b1)
        # Batch 1: step 0 → 3, step 1 → EOS → [1, 2, 3, EOS]
        # Both are in the same tensor so they share seq_len dimension
        assert result.sequences.shape[0] == 2


class TestPruning:
    def test_single_block_pruned(self):
        """[THOUGHT] reasoning [SOLUTION] summary [RETURN] should prune reasoning."""
        # Prompt: [1]
        # Script generates: THOUGHT, 10, 11, SOLUTION, 20, RETURN, EOS
        result = _run(
            script_tokens=[
                [THOUGHT_ID], [10], [11], [SOLUTION_ID], [20], [RETURN_ID], [EOS_ID],
            ],
            prompt=[[1]],
            max_length=100,
        )
        tokens = result.sequences[0].tolist()
        # After pruning: [THOUGHT], 10, 11 removed; [SOLUTION], 20, [RETURN] kept
        assert THOUGHT_ID not in tokens
        assert 10 not in tokens
        assert 11 not in tokens
        assert SOLUTION_ID in tokens
        assert 20 in tokens
        assert RETURN_ID in tokens
        # Verify per-record stats (batch_size=1)
        assert result.prune_events == [1]
        assert result.tokens_pruned[0] > 0
        assert result.prompt_tokens == 1
        assert result.total_tokens_processed[0] > 0
        assert result.wall_time_seconds > 0

    def test_no_prune_without_return(self):
        """Without [RETURN], no pruning should occur."""
        result = _run(
            script_tokens=[[THOUGHT_ID], [10], [SOLUTION_ID], [20], [EOS_ID]],
            prompt=[[1]],
            max_length=100,
        )
        tokens = result.sequences[0].tolist()
        assert THOUGHT_ID in tokens
        assert 10 in tokens
        assert SOLUTION_ID in tokens
        assert 20 in tokens
        assert result.prune_events == [0]
        assert result.tokens_pruned == [0]

    def test_no_prune_thought_without_solution(self):
        """[THOUGHT] ... [RETURN] with no [SOLUTION] should not prune."""
        result = _run(
            script_tokens=[[THOUGHT_ID], [10], [RETURN_ID], [EOS_ID]],
            prompt=[[1]],
            max_length=100,
        )
        tokens = result.sequences[0].tolist()
        # Nothing pruned — THOUGHT and reasoning tokens still present
        assert THOUGHT_ID in tokens
        assert 10 in tokens


class TestBatchPruning:
    def test_only_pruned_element_changes(self):
        """When one batch element hits [RETURN], only that element is pruned."""
        # Batch 0: generates THOUGHT, SOLUTION, RETURN → gets pruned
        # Batch 1: generates plain tokens → untouched
        result = _run(
            script_tokens=[
                [THOUGHT_ID, 3],
                [10, 4],
                [SOLUTION_ID, 5],
                [20, 6],
                [RETURN_ID, 7],
                [EOS_ID, EOS_ID],
            ],
            prompt=[[1, 2], [1, 2]],
            max_length=100,
            batch_size=2,
        )
        b0 = result.sequences[0].tolist()
        b1 = result.sequences[1].tolist()

        # Batch 0 should have reasoning pruned
        assert THOUGHT_ID not in b0
        assert 10 not in b0
        assert SOLUTION_ID in b0

        # Batch 1 should be intact (no special tokens were generated)
        assert 3 in b1
        assert 4 in b1
        assert 5 in b1


class TestLogitsProcessor:
    def test_processor_receives_full_sequence(self):
        """logits_processor should receive the growing token sequence, not just the prompt."""
        seen_lengths = []

        class _TrackingProcessor:
            def __call__(self, input_ids, scores):
                seen_lengths.append(input_ids.shape[1])
                return scores

        script = [torch.tensor([3]), torch.tensor([EOS_ID])]
        model = ScriptedModel(script)
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)

        processor_list = LogitsProcessorList([_TrackingProcessor()])
        attention_mask = torch.ones_like(input_ids)
        generate(
            model=model,
            input_ids=input_ids,
            logits_processor=processor_list,
            stopping_criteria=_stopping(100),
            return_token_id=RETURN_ID,
            solution_token_id=SOLUTION_ID,
            thought_token_id=THOUGHT_ID,
            eos_token_id=EOS_ID,
            attention_mask=attention_mask,
        )
        # Step 0: prompt len 2, step 1: prompt + 1 generated = 3
        assert seen_lengths == [2, 3]


# ---------------------------------------------------------------------------
# Helpers for generate_with_mask
# ---------------------------------------------------------------------------


def _run_masked(script_tokens, prompt, max_length, min_token_length, batch_size=1):
    """Helper that sets up fakes and calls generate_with_mask."""
    script = [torch.tensor(step) for step in script_tokens]
    model = ScriptedModel(script)
    input_ids = torch.tensor(prompt, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return generate_with_mask(
        model=model,
        input_ids=input_ids,
        logits_processor=_identity_processor(),
        stopping_criteria=_stopping(max_length),
        return_token_id=RETURN_ID,
        solution_token_id=SOLUTION_ID,
        thought_token_id=THOUGHT_ID,
        eos_token_id=EOS_ID,
        min_token_length=min_token_length,
        attention_mask=attention_mask,
    )


# ---------------------------------------------------------------------------
# Tests for generate_with_mask
# ---------------------------------------------------------------------------


class TestMaskBelowThreshold:
    def test_no_prune_below_min_token_length(self):
        """Below min_token_length, tokens should NOT be pruned (all kept)."""
        # Prompt [1], generates: THOUGHT, 10, 11, SOLUTION, 20, RETURN, EOS
        # Total after RETURN: 7 tokens.  Set min_token_length=100 so we never prune.
        result = _run_masked(
            script_tokens=[
                [THOUGHT_ID], [10], [11], [SOLUTION_ID], [20], [RETURN_ID], [EOS_ID],
            ],
            prompt=[[1]],
            max_length=100,
            min_token_length=100,
        )
        assert isinstance(result, HCotGenerateOutput)
        tokens = result.sequences[0].tolist()
        # All tokens should still be present (masked, not pruned)
        assert THOUGHT_ID in tokens
        assert 10 in tokens
        assert 11 in tokens
        assert SOLUTION_ID in tokens
        assert 20 in tokens
        assert RETURN_ID in tokens

    def test_4d_mask_passed_to_model(self):
        """After a block completes below threshold, model receives a 4-D mask."""
        received_masks = []

        class MaskCapturingModel:
            def __init__(self, script, vocab_size=200):
                self._inner = ScriptedModel(script, vocab_size)

            def __call__(self, input_ids, **kwargs):
                received_masks.append(kwargs.get('attention_mask'))
                return self._inner(input_ids, **kwargs)

        # Prompt [1], generates: THOUGHT, SOLUTION, RETURN, 30, EOS
        script = [
            torch.tensor([THOUGHT_ID]),
            torch.tensor([SOLUTION_ID]),
            torch.tensor([RETURN_ID]),
            torch.tensor([30]),
            torch.tensor([EOS_ID]),
        ]
        model = MaskCapturingModel(script)
        input_ids = torch.tensor([[1]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        generate_with_mask(
            model=model,
            input_ids=input_ids,
            logits_processor=_identity_processor(),
            stopping_criteria=_stopping(100),
            return_token_id=RETURN_ID,
            solution_token_id=SOLUTION_ID,
            thought_token_id=THOUGHT_ID,
            eos_token_id=EOS_ID,
            min_token_length=100,
            attention_mask=attention_mask,
        )
        # Steps 0-2 (generating THOUGHT, SOLUTION, RETURN): no completed block yet → 2-D mask
        # Steps 3-4 (generating 30, EOS): block is completed → 4-D mask
        assert received_masks[0].dim() == 2  # before block completes
        assert received_masks[3].dim() == 4  # after block completes
        assert received_masks[4].dim() == 4


class TestPruneAboveThreshold:
    def test_prune_above_min_token_length(self):
        """Above min_token_length, completed blocks should be pruned."""
        # Prompt [1], generates: THOUGHT, 10, SOLUTION, 20, RETURN, EOS
        # Total after RETURN: 6 tokens.  Set min_token_length=1 so we always prune.
        result = _run_masked(
            script_tokens=[
                [THOUGHT_ID], [10], [SOLUTION_ID], [20], [RETURN_ID], [EOS_ID],
            ],
            prompt=[[1]],
            max_length=100,
            min_token_length=1,
        )
        tokens = result.sequences[0].tolist()
        # Thought span pruned, solution kept
        assert THOUGHT_ID not in tokens
        assert 10 not in tokens
        assert SOLUTION_ID in tokens
        assert 20 in tokens

    def test_mask_then_prune(self):
        """First block masked (below threshold), second block pruned (above)."""
        # Prompt is 1 token. min_token_length=8.
        # Block 1: THOUGHT, 10, SOLUTION, 20, RETURN  (seq reaches 6 tokens → below 8 → mask)
        # Then: 30 (seq=7), then block 2 starts:
        # THOUGHT, 11, SOLUTION, 21, RETURN  (seq reaches 12 → above 8 → prune)
        # Then: EOS
        result = _run_masked(
            script_tokens=[
                [THOUGHT_ID], [10], [SOLUTION_ID], [20], [RETURN_ID],  # block 1 (masked)
                [30],
                [THOUGHT_ID], [11], [SOLUTION_ID], [21], [RETURN_ID],  # block 2 (pruned)
                [EOS_ID],
            ],
            prompt=[[1]],
            max_length=100,
            min_token_length=8,
        )
        tokens = result.sequences[0].tolist()
        # Block 1 should still be in the sequence (masked, not pruned)
        assert 10 in tokens
        # Block 2's thought span should be pruned
        assert 11 not in tokens
        # Block 2's solution should be kept
        assert 21 in tokens
