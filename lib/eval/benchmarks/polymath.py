"""PolyMath benchmark (Qwen/PolyMath).

Uses the official QwenLM/PolyMath evaluation logic (added as a git submodule
under ``third_party/PolyMath``) for answer extraction and comparison.
Falls back to ``math-verify`` when the official code is unavailable.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

from . import EvalProblem, check_answer_math_verify, extract_boxed_last

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the official PolyMath evaluation helpers.
# ---------------------------------------------------------------------------
_POLYMATH_EVAL_DIR = str(
    Path(__file__).resolve().parents[3] / "third_party" / "PolyMath" / "eval"
)

_has_official = False
try:
    if _POLYMATH_EVAL_DIR not in sys.path:
        sys.path.insert(0, _POLYMATH_EVAL_DIR)
    from scripts import math_equal as _official_math_equal  # type: ignore[import-untyped]

    _has_official = True
    logger.info("Using official PolyMath evaluation from %s", _POLYMATH_EVAL_DIR)
except ImportError as exc:
    logger.warning(
        "Could not import official PolyMath evaluation (%s). "
        "Falling back to math-verify.",
        exc,
    )

# Difficulty level ordering (top = easiest, low = hardest).
POLYMATH_LEVELS = ("top", "high", "middle", "low")


class PolyMathBenchmark:
    """The PolyMath multilingual math benchmark (Qwen/PolyMath)."""

    name = "PolyMath"

    @property
    def system_prompt(self) -> str:
        return (
            "Solve the following math problem. "
            "Make sure to put the answer (and only answer) inside \\boxed{}."
        )

    def load(
        self,
        *,
        language: str = "en",
        dataset_id: str = "Qwen/PolyMath",
        levels: list[str] | None = None,
        **kwargs: Any,
    ) -> list[EvalProblem]:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, language)
        level_set = set(levels) if levels else None

        problems: list[EvalProblem] = []
        for split_name in POLYMATH_LEVELS:
            if level_set and split_name not in level_set:
                continue
            if split_name not in ds:
                continue
            for row in ds[split_name]:
                problems.append(
                    EvalProblem(
                        id=row["id"],
                        problem=row["question"],
                        expected_answer=row["answer"].strip(),
                        metadata={
                            "level": split_name,
                            "subject": "math",
                            "language": language,
                        },
                    )
                )

        return problems

    def extract_answer(self, model_output: str) -> str | None:
        """Extract ``\\boxed{…}`` using the official logic.

        The official implementation strips all whitespace before searching
        and returns the **first** match.
        """
        if _has_official:
            text = model_output.replace(" ", "")
            pattern = re.compile(r"boxed{")
            results: list[str] = []
            for match in pattern.finditer(text):
                start_pos = match.end()
                brace_count = 1
                i = start_pos
                while i < len(text) and brace_count > 0:
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    results.append(text[start_pos : i - 1])
            return results[0] if results else None

        return extract_boxed_last(model_output)

    def check_answer(self, predicted: str | None, expected: str) -> bool:
        """Compare answers using the official ``math_equal``.

        The official implementation performs extensive normalisation,
        numeric comparison with tolerance, percentage equivalence,
        unit stripping, and symbolic equality via sympy.
        """
        if predicted is None:
            return False

        if _has_official:
            try:
                return _official_math_equal(predicted, expected)
            except Exception:
                logger.debug(
                    "official math_equal raised; falling back", exc_info=True
                )

        return check_answer_math_verify(predicted, expected)
