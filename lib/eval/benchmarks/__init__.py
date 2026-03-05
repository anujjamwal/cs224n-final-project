"""Benchmark abstraction layer.

Each benchmark provides its own dataset loader, answer extractor,
answer checker, and system prompt.  The runner works with any object
that satisfies the :class:`Benchmark` protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class EvalProblem:
    """One evaluation problem, benchmark-agnostic."""
    id: str
    problem: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Benchmark(Protocol):
    """Interface that every benchmark must implement."""

    name: str

    @property
    def system_prompt(self) -> str:
        """System prompt prepended to every problem."""
        ...

    def load(self, **kwargs: Any) -> list[EvalProblem]:
        """Load problems from the dataset."""
        ...

    def extract_answer(self, model_output: str) -> str | None:
        """Extract the predicted answer from raw model output."""
        ...

    def check_answer(self, predicted: str | None, expected: str) -> bool:
        """Return ``True`` if *predicted* matches *expected*."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_boxed_last(text: str) -> str | None:
    """Extract content from the **last** ``\\boxed{…}``, handling nested braces."""
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


def check_answer_math_verify(predicted: str | None, expected: str) -> bool:
    """Compare answers using HuggingFace ``math-verify`` library."""
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        gold = parse(expected, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred = parse(predicted, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        return verify(gold, pred)
    except Exception:
        pass
    return _normalize(predicted) == _normalize(expected)


def _normalize(s: str) -> str:
    """Basic normalization for fallback string comparison."""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    for cmd in (r"\text", r"\mathrm", r"\displaystyle"):
        s = s.replace(cmd, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


from .math import MathBenchmark
from .polymath import PolyMathBenchmark, POLYMATH_LEVELS

# ---------------------------------------------------------------------------
# Legacy function-based loaders for backward compatibility with notebooks.
# ---------------------------------------------------------------------------

_math = MathBenchmark()
_polymath = PolyMathBenchmark()


def load_math(
    split="test",
    *,
    dataset_id="lighteval/MATH",
    subjects=None,
    levels=None,
    **kwargs,
):
    return _math.load(
        split=split,
        dataset_id=dataset_id,
        subjects=subjects,
        levels=levels,
        **kwargs,
    )


def load_polymath(
    language="en",
    *,
    dataset_id="Qwen/PolyMath",
    levels=None,
):
    return _polymath.load(
        language=language,
        dataset_id=dataset_id,
        levels=levels,
    )


__all__ = [
    "Benchmark",
    "EvalProblem",
    "MathBenchmark",
    "PolyMathBenchmark",
    "POLYMATH_LEVELS",
    "extract_boxed_last",
    "check_answer_math_verify",
    "load_math",
    "load_polymath",
]
