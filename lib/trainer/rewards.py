"""Reward functions for GRPO training of Hierarchical Chain-of-Thought models.

All functions follow the TRL GRPOTrainer reward signature:
    reward_fn(completions, **kwargs) -> list[float]

Completions are in conversational format (list of list of message dicts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from trainer import THOUGHT_TOKEN, SOLUTION_TOKEN, RETURN_TOKEN
from eval.benchmarks import extract_boxed_last, check_answer_math_verify


# ---------------------------------------------------------------------------
# String-based block parser
# ---------------------------------------------------------------------------

@dataclass
class ThoughtBlock:
    """A parsed [THOUGHT]...[SOLUTION]...[RETURN] block."""
    thought_start: int          # char position of [THOUGHT]
    solution_start: int         # char position of [SOLUTION]
    return_end: int             # char position of [RETURN]
    depth: int                  # nesting depth (0 = root-level)
    thought_content: str        # text between [THOUGHT] and [SOLUTION]
    solution_content: str       # text between [SOLUTION] and [RETURN]
    is_leaf: bool = True        # True if no nested [THOUGHT] in thought_content
    children: list["ThoughtBlock"] = field(default_factory=list)


def parse_thought_blocks(text: str) -> tuple[list[ThoughtBlock], bool, int]:
    """Parse [THOUGHT]/[SOLUTION]/[RETURN] blocks from raw text.

    Uses a scanner + stack approach (mirrors the token-based
    ``find_cot_blocks`` in ``utils`` but operates on strings).

    Returns:
        blocks: list of top-level ``ThoughtBlock`` objects (children nested).
        is_valid: ``True`` when every opener has a matching [SOLUTION] and
                  [RETURN] and there are no orphaned markers.
        max_depth: maximum nesting depth observed (0 if no blocks).
    """
    tokens = (THOUGHT_TOKEN, SOLUTION_TOKEN, RETURN_TOKEN)
    tok_len = {t: len(t) for t in tokens}

    # Scan for all marker positions
    markers: list[tuple[int, str]] = []
    for tok in tokens:
        start = 0
        while True:
            idx = text.find(tok, start)
            if idx == -1:
                break
            markers.append((idx, tok))
            start = idx + tok_len[tok]
    markers.sort(key=lambda m: m[0])

    # Stack-based parse
    stack: list[dict] = []            # working entries
    finished: list[dict] = []         # completed blocks (flat)
    is_valid = True
    max_depth = 0

    for pos, tok in markers:
        if tok == THOUGHT_TOKEN:
            depth = len(stack)
            max_depth = max(max_depth, depth + 1)
            stack.append({
                "thought_start": pos,
                "solution_start": None,
                "depth": depth,
            })
        elif tok == SOLUTION_TOKEN:
            if not stack or stack[-1]["solution_start"] is not None:
                is_valid = False
                continue
            stack[-1]["solution_start"] = pos
        elif tok == RETURN_TOKEN:
            if not stack or stack[-1]["solution_start"] is None:
                is_valid = False
                continue
            entry = stack.pop()
            entry["return_end"] = pos
            finished.append(entry)

    # Unclosed blocks → invalid
    if stack:
        is_valid = False

    # Build ThoughtBlock objects with content extraction
    blocks_by_depth: dict[int, list[ThoughtBlock]] = {}
    for e in finished:
        t_start = e["thought_start"]
        s_start = e["solution_start"]
        r_end = e["return_end"]
        thought_content = text[t_start + tok_len[THOUGHT_TOKEN]: s_start]
        solution_content = text[s_start + tok_len[SOLUTION_TOKEN]: r_end]
        has_nested = THOUGHT_TOKEN in thought_content
        blk = ThoughtBlock(
            thought_start=t_start,
            solution_start=s_start,
            return_end=r_end,
            depth=e["depth"],
            thought_content=thought_content,
            solution_content=solution_content,
            is_leaf=not has_nested,
        )
        blocks_by_depth.setdefault(e["depth"], []).append(blk)

    # Wire parent-child relationships
    all_blocks = [b for bs in blocks_by_depth.values() for b in bs]
    all_blocks.sort(key=lambda b: b.thought_start)

    # Assign children: a block B is child of A if B.depth == A.depth+1 and
    # B is contained within A's thought span.
    for blk in all_blocks:
        if blk.depth == 0:
            continue
        for candidate in all_blocks:
            if candidate.depth != blk.depth - 1:
                continue
            if candidate.thought_start < blk.thought_start < candidate.solution_start:
                candidate.children.append(blk)
                break

    top_level = [b for b in all_blocks if b.depth == 0]
    return top_level, is_valid, max_depth


def _get_completion_text(completion) -> str:
    """Extract plain text from a TRL completion (conversational format)."""
    if isinstance(completion, list):
        # conversational: [{role: ..., content: ...}]
        return completion[0]["content"] if completion else ""
    return str(completion)


# ---------------------------------------------------------------------------
# Reward 1: Correctness
# ---------------------------------------------------------------------------

def correctness_reward(completions, expected_answer, **kwargs) -> list[float]:
    """1.0 if the \\boxed{} answer matches the expected answer, else 0.0.

    Uses ``math-verify`` for robust mathematical comparison with a
    normalised-string fallback.
    """
    rewards = []
    for completion, answer in zip(completions, expected_answer):
        text = _get_completion_text(completion)
        predicted = extract_boxed_last(text)
        correct = check_answer_math_verify(predicted, str(answer))
        rewards.append(1.0 if correct else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward 2: Syntax validity
# ---------------------------------------------------------------------------

def syntax_reward(completions, **kwargs) -> list[float]:
    """1.0 if [THOUGHT]/[SOLUTION]/[RETURN] nesting is valid, 0.0 otherwise."""
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        _blocks, is_valid, _depth = parse_thought_blocks(text)
        rewards.append(1.0 if is_valid else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward 3: Leaf thought length
# ---------------------------------------------------------------------------

def leaf_length_reward(completions, max_leaf_chars: int = 2000, **kwargs) -> list[float]:
    """Penalise if any leaf thought exceeds *max_leaf_chars* characters.

    Returns 1.0 when all leaves are within the limit.  Beyond the limit
    the reward decays linearly to 0.0 at 2× the threshold.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        blocks, is_valid, _ = parse_thought_blocks(text)
        if not is_valid or not blocks:
            rewards.append(0.0)
            continue

        # Collect all blocks (including nested children)
        all_blocks = _flatten_blocks(blocks)
        leaves = [b for b in all_blocks if b.is_leaf]
        if not leaves:
            rewards.append(1.0)
            continue

        worst = max(len(b.thought_content) for b in leaves)
        if worst <= max_leaf_chars:
            rewards.append(1.0)
        else:
            # Linear decay: 1.0 at threshold → 0.0 at 2× threshold
            ratio = (worst - max_leaf_chars) / max_leaf_chars
            rewards.append(max(0.0, 1.0 - ratio))
    return rewards


# ---------------------------------------------------------------------------
# Reward 4: Tree depth
# ---------------------------------------------------------------------------

def depth_reward(completions, max_depth: int = 4, **kwargs) -> list[float]:
    """Penalise excessive nesting depth.

    Returns 1.0 when depth ≤ *max_depth*.  Linearly decays to 0.0 at
    *max_depth + 2*.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        _blocks, is_valid, depth = parse_thought_blocks(text)
        if not is_valid:
            rewards.append(0.0)
            continue
        if depth <= max_depth:
            rewards.append(1.0)
        else:
            excess = depth - max_depth
            rewards.append(max(0.0, 1.0 - excess / 2.0))
    return rewards


# ---------------------------------------------------------------------------
# Reward 5: Compression ratio
# ---------------------------------------------------------------------------

def compression_reward(completions, **kwargs) -> list[float]:
    """Reward good solution-to-thought compression ratio.

    Sweet spot is 0.1–0.5.  Below 0.05 (too terse) or above 0.5 (not
    compressing) the reward decays toward 0.0.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        blocks, is_valid, _ = parse_thought_blocks(text)
        if not is_valid or not blocks:
            rewards.append(0.0)
            continue

        all_blocks = _flatten_blocks(blocks)
        total_thought = sum(len(b.thought_content) for b in all_blocks)
        total_solution = sum(len(b.solution_content) for b in all_blocks)

        if total_thought == 0:
            rewards.append(0.0)
            continue

        ratio = total_solution / total_thought
        if 0.1 <= ratio <= 0.5:
            rewards.append(1.0)
        elif ratio < 0.1:
            # Linear decay from 1.0 at 0.1 to 0.0 at 0.0
            rewards.append(max(0.0, ratio / 0.1))
        else:
            # Linear decay from 1.0 at 0.5 to 0.0 at 1.0
            rewards.append(max(0.0, 1.0 - (ratio - 0.5) / 0.5))
    return rewards


# ---------------------------------------------------------------------------
# Reward 6: Format compliance
# ---------------------------------------------------------------------------

def format_reward(completions, **kwargs) -> list[float]:
    """Check structural format: <think>/</think>, \\boxed{}, and ≥1 block.

    Awards ~0.33 for each element present, summing to 1.0 when all three
    are found.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        score = 0.0

        # Check <think>...</think>
        if "<think>" in text and "</think>" in text:
            score += 1 / 3

        # Check \boxed{}
        if r"\boxed{" in text:
            score += 1 / 3

        # Check at least one complete thought block
        blocks, is_valid, _ = parse_thought_blocks(text)
        if blocks and is_valid:
            score += 1 / 3

        rewards.append(round(score, 4))
    return rewards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_blocks(blocks: list[ThoughtBlock]) -> list[ThoughtBlock]:
    """Recursively flatten a tree of ThoughtBlocks into a flat list."""
    result = []
    for b in blocks:
        result.append(b)
        if b.children:
            result.extend(_flatten_blocks(b.children))
    return result
