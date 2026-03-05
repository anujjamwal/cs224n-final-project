"""MATH benchmark (lighteval/MATH)."""

from __future__ import annotations

from typing import Any

from . import EvalProblem, check_answer_math_verify, extract_boxed_last


class MathBenchmark:
    """The MATH benchmark (https://github.com/hendrycks/math)."""

    name = "MATH"

    @property
    def system_prompt(self) -> str:
        return (
            "Solve the following math problem. "
            "Make sure to put the answer (and only answer) inside \\boxed{}."
        )

    def load(
        self,
        *,
        split: str = "test",
        dataset_id: str = "lighteval/MATH",
        subjects: list[str] | None = None,
        levels: list[int] | None = None,
        level_key: str = "level",
        problem_key: str = "problem",
        subject_key: str = "subject",
        solution_key: str = "solution",
        **kwargs: Any,
    ) -> list[EvalProblem]:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split=split)

        level_strings = {f"Level {l}" for l in levels} if levels else None
        subject_set = set(subjects) if subjects else None

        problems: list[EvalProblem] = []
        for idx, row in enumerate(ds):
            if level_strings and row[level_key] not in level_strings:
                continue
            if subject_set and row[subject_key] not in subject_set:
                continue

            answer = extract_boxed_last(row[solution_key])
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

    def extract_answer(self, model_output: str) -> str | None:
        return extract_boxed_last(model_output)

    def check_answer(self, predicted: str | None, expected: str) -> bool:
        return check_answer_math_verify(predicted, expected)
