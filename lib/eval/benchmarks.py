"""Benchmark dataset loaders.

Each loader returns a list of EvalProblem — a uniform representation that
the runner consumes.  Adding a new benchmark (e.g. GSM8K) means adding a
new ``load_*`` function here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class EvalProblem:
    """One evaluation problem, benchmark-agnostic."""
    id: str
    problem: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_boxed(text: str) -> str | None:
    """Extract content from the last ``\\boxed{...}``, handling nested braces."""
    pattern = r"boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def load_math(
    split: str = "test",
    *,
    dataset_id: str = "lighteval/MATH",
    subjects: list[str] | None = None,
    levels: list[int] | None = None,
    level_key="level",
    problem_key="problem",
    subject_key="subject",
    solution_key="solution"
) -> list[EvalProblem]:
    """Load the MATH benchmark and return a list of :class:`EvalProblem`.

    Parameters
    ----------
    split:
        HuggingFace split name (``"test"`` or ``"train"``).
    dataset_id:
        HuggingFace dataset identifier.  Override if using a mirror.
    subjects:
        Optional filter — keep only these subject types
        (e.g. ``["Algebra", "Number Theory"]``).
    levels:
        Optional filter — keep only these difficulty levels (1–5).
        The raw dataset stores them as ``"Level 1"`` etc.; pass plain ints.
    """
    ds = load_dataset(dataset_id, split=split)

    level_strings = {f"Level {l}" for l in levels} if levels else None
    subject_set = set(subjects) if subjects else None

    problems: list[EvalProblem] = []
    for idx, row in enumerate(ds):
        if level_strings and row[level_key] not in level_strings:
            continue
        if subject_set and row[subject_key] not in subject_set:
            continue

        answer = _extract_boxed(row[solution_key])
        if answer is None:
            answer = ""

        problems.append(EvalProblem(
            id=f"math_{split}_{idx:05d}",
            problem=row[problem_key],
            expected_answer=answer,
            metadata={
                "level": row[level_key],
                "subject": row[subject_key],
                "solution": row[solution_key],
            },
        ))

    return problems


# Polymath difficulty level ordering (top = easiest, low = hardest)
POLYMATH_LEVELS = ("top", "high", "medium", "low")


def load_polymath(
    language: str = "en",
    *,
    dataset_id: str = "Qwen/PolyMath",
    levels: list[str] | None = None,
) -> list[EvalProblem]:
    """Load the Qwen/PolyMath benchmark and return a list of :class:`EvalProblem`.

    PolyMath is a multilingual mathematical reasoning benchmark with 18
    languages, 4 difficulty levels, and 500 problems per language.

    Parameters
    ----------
    language:
        Language config to load (e.g. ``"en"``, ``"zh"``, ``"fr"``).
        Defaults to ``"en"`` (English).
    dataset_id:
        HuggingFace dataset identifier.  Override if using a mirror.
    levels:
        Optional filter — keep only these difficulty levels.
        Valid values: ``"top"`` (easiest), ``"high"``, ``"medium"``,
        ``"low"`` (hardest).
    """
    ds = load_dataset(dataset_id, language)

    level_set = set(levels) if levels else None

    problems: list[EvalProblem] = []
    for split_name in POLYMATH_LEVELS:
        if level_set and split_name not in level_set:
            continue
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            problems.append(EvalProblem(
                id=row["id"],
                problem=row["question"],
                expected_answer=row["answer"].strip(),
                metadata={
                    "level": split_name,
                    "subject": "math",
                    "language": language,
                },
            ))

    return problems
