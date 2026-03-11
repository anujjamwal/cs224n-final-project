"""Evaluation runner with JSONL checkpointing.

Results are written incrementally to ``{output_dir}/results.jsonl`` so that
long runs can be interrupted and resumed.  A summary JSON is also updated
after each batch.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm.auto import tqdm

from .benchmarks import Benchmark, EvalProblem
from trainer.dataset import convert_to_trl_prompt

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Solve the following math problem. "
    "Make sure to put the answer (and only answer) inside \\boxed{}."
)


@dataclass
class EvalResult:
    problem_id: str
    predicted: str | None
    expected: str
    correct: bool
    generated_tokens: int = 0
    wall_time: float = 0.0
    raw_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _results_path(output_dir: str) -> Path:
    return Path(output_dir) / "results.jsonl"


def _summary_path(output_dir: str) -> Path:
    return Path(output_dir) / "summary.json"


def _config_path(output_dir: str) -> Path:
    return Path(output_dir) / "config.json"


def load_results(output_dir: str) -> list[EvalResult]:
    """Load previously saved results from the JSONL checkpoint."""
    path = _results_path(output_dir)
    if not path.exists():
        return []
    results: list[EvalResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.pop("mode", None) # For backward compat
            results.append(EvalResult(**d))
    return results


def _append_results(output_dir: str, results: list[EvalResult]) -> None:
    path = _results_path(output_dir)
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
        f.flush()


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    """Compute aggregate accuracy statistics from a list of results."""
    if not results:
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    total = len(results)
    correct = sum(1 for r in results if r.correct)

    by_level: dict[str, dict] = {}
    by_subject: dict[str, dict] = {}

    for r in results:
        # By level (from metadata)
        level = r.metadata.get("level", "unknown")
        lv = by_level.setdefault(level, {"total": 0, "correct": 0})
        lv["total"] += 1
        lv["correct"] += int(r.correct)

        # By subject
        subject = r.metadata.get("subject", "unknown")
        sv = by_subject.setdefault(subject, {"total": 0, "correct": 0})
        sv["total"] += 1
        sv["correct"] += int(r.correct)

    def _add_accuracy(d: dict) -> dict:
        for v in d.values():
            v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] else 0.0
        return d

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "by_level": _add_accuracy(by_level),
        "by_subject": _add_accuracy(by_subject),
    }


def _save_summary(output_dir: str, results: list[EvalResult]) -> None:
    summary = summarize_results(results)
    with open(_summary_path(output_dir), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eval(
    model,
    tokenizer,
    problems: list[EvalProblem],
    output_dir: str,
    *,
    benchmark: Benchmark | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 4096,
    generate_kwargs: dict[Any, Any],
) -> list[EvalResult]:
    """Run evaluation with JSONL checkpointing.

    Parameters
    ----------
    model:
        HuggingFace causal LM (already on device, in eval mode).
    tokenizer:
        Matching tokenizer with special tokens already added.
    problems:
        List of :class:`EvalProblem` to evaluate.
    output_dir:
        Directory for checkpoint files.  Created if it doesn't exist.
    benchmark:
        A :class:`Benchmark` instance that provides ``extract_answer``,
        ``check_answer``, and ``system_prompt``.  When ``None``, falls
        back to a default boxed-extraction with ``math-verify``.
    batch_size:
        Number of problems per forward batch.
    max_new_tokens:
        Maximum tokens to generate per problem.

    Returns
    -------
    All results (including previously checkpointed ones).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve benchmark-specific functions, falling back to defaults.
    if benchmark is not None:
        _extract = benchmark.extract_answer
        _check = benchmark.check_answer
        system_prompt = benchmark.system_prompt
    else:
        from .benchmarks import check_answer_math_verify, extract_boxed_last
        _extract = extract_boxed_last
        _check = check_answer_math_verify
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Save config
    config = {
        "benchmark": benchmark.name if benchmark else "unknown",
        "num_problems": len(problems),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    }
    with open(_config_path(output_dir), "w") as f:
        json.dump(config, f, indent=2)

    # Load existing checkpoint
    all_results = load_results(output_dir)
    done_keys = set([r.problem_id for r in all_results])
    if done_keys:
        logger.info("Resuming: %d results already checkpointed", len(done_keys))

    device = next(model.parameters()).device
    pending = [p for p in problems if p.id not in done_keys]

    if not pending:
        logger.info("All %d problems already done, skipping", len(problems))
        return all_results

    logger.info("%d/%d problems remaining", len(pending), len(problems))
    
    if generate_kwargs is None:
        generate_kwargs = {}
    generate_kwargs['max_new_tokens'] = max_new_tokens

    for batch_start in tqdm(
        range(0, len(pending), batch_size),
        desc="Eval",
        total=(len(pending) + batch_size - 1) // batch_size,
    ):
        batch = pending[batch_start : batch_start + batch_size]
        prompts = [convert_to_trl_prompt(p, system_prompt)["prompt"] for p in batch]
        inp = tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inp, **generate_kwargs)
        wall_time = time.time() - t0

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        batch_results: list[EvalResult] = []
        for j, prob in enumerate(batch):
            decoded = tokenizer.decode(sequences[j], skip_special_tokens=False)
            predicted = _extract(decoded)
            correct = _check(predicted, prob.expected_answer)

            batch_results.append(EvalResult(
                problem_id=prob.id,
                predicted=predicted,
                expected=prob.expected_answer,
                correct=correct,
                generated_tokens=sequences[j].shape[0] - inp["input_ids"].shape[1],
                wall_time=wall_time / len(batch),
                raw_output=decoded,
                metadata=prob.metadata,
            ))

        _append_results(output_dir, batch_results)
        all_results.extend(batch_results)
        done_keys.update(r.problem_id for r in batch_results)

        _save_summary(output_dir, all_results)

    logger.info("Evaluation complete: %d total results", len(all_results))
    return all_results
