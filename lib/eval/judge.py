"""Answer extraction and verification for math benchmarks.

Uses HuggingFace's ``math-verify`` library for robust LaTeX comparison.
"""

from __future__ import annotations

import logging
import re

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

logger = logging.getLogger(__name__)


def extract_answer(model_output: str) -> str | None:
    """Extract the answer from model output.

    Looks for the last ``\\boxed{...}`` in the output, handling nested braces.
    """
    pattern = r"boxed\{"
    matches = list(re.finditer(pattern, model_output))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(model_output) and depth > 0:
        if model_output[i] == "{":
            depth += 1
        elif model_output[i] == "}":
            depth -= 1
        i += 1
    return model_output[start : i - 1].strip() if depth == 0 else None


def check_answer(predicted: str | None, expected: str) -> bool:
    """Compare predicted answer against expected using math-verify.

    Falls back to normalized string comparison if parsing fails on both sides.
    """
    if predicted is None:
        return False

    try:
        gold = parse(expected, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred = parse(predicted, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        return verify(gold, pred)
    except Exception:
        pass

    # Fallback: strip whitespace and compare
    return _normalize(predicted) == _normalize(expected)


def _normalize(s: str) -> str:
    """Basic normalization for fallback string comparison."""
    s = s.strip()
    # Remove surrounding $...$
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Remove common LaTeX wrappers
    for cmd in (r"\text", r"\mathrm", r"\displaystyle"):
        s = s.replace(cmd, "")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
