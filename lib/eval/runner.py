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

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Solve the following math problem. "
    "Make sure to put the answer (and only answer) inside \\boxed{}."
)


@dataclass
class EvalResult:
    problem_id: str
    mode: str
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

    by_mode: dict[str, dict] = {}
    by_level: dict[str, dict] = {}
    by_subject: dict[str, dict] = {}

    for r in results:
        # By mode
        m = by_mode.setdefault(r.mode, {"total": 0, "correct": 0})
        m["total"] += 1
        m["correct"] += int(r.correct)

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
        "by_mode": _add_accuracy(by_mode),
        "by_level": _add_accuracy(by_level),
        "by_subject": _add_accuracy(by_subject),
    }


def _save_summary(output_dir: str, results: list[EvalResult]) -> None:
    summary = summarize_results(results)
    with open(_summary_path(output_dir), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _build_prompt_messages(problem: EvalProblem, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem.problem},
    ]


def _batch_tokenize(tokenizer, prompts: list[list[dict]], device) -> dict:
    """Tokenize multiple chat prompts with left-padding for batched generation."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    encoded = [
        tokenizer.apply_chat_template(
            p, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        for p in prompts
    ]

    max_len = max(e["input_ids"].shape[1] for e in encoded)
    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)

    for i, e in enumerate(encoded):
        seq_len = e["input_ids"].shape[1]
        input_ids[i, max_len - seq_len :] = e["input_ids"][0]
        attention_mask[i, max_len - seq_len :] = 1

    tokenizer.padding_side = prev_side
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class GenerationMode:
    """Describes one generation mode to evaluate."""
    name: str
    generate_fn: Callable | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


def run_eval(
    model,
    tokenizer,
    problems: list[EvalProblem],
    output_dir: str,
    modes: list[GenerationMode],
    *,
    benchmark: Benchmark | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 4096,
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
    modes:
        List of :class:`GenerationMode`.  Each problem is evaluated once
        per mode.  Set ``generate_fn=None`` for standard HF generation;
        otherwise pass e.g. ``custom_generate._sample``.
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
        "modes": [m.name for m in modes],
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    }
    with open(_config_path(output_dir), "w") as f:
        json.dump(config, f, indent=2)

    # Load existing checkpoint
    all_results = load_results(output_dir)
    done_keys = {(r.problem_id, r.mode) for r in all_results}
    if done_keys:
        logger.info("Resuming: %d results already checkpointed", len(done_keys))

    device = next(model.parameters()).device

    for mode in modes:
        # Filter to problems not yet done for this mode
        pending = [p for p in problems if (p.id, mode.name) not in done_keys]
        if not pending:
            logger.info("Mode '%s': all %d problems already done, skipping", mode.name, len(problems))
            continue

        logger.info("Mode '%s': %d/%d problems remaining", mode.name, len(pending), len(problems))

        for batch_start in tqdm(
            range(0, len(pending), batch_size),
            desc=f"Eval [{mode.name}]",
            total=(len(pending) + batch_size - 1) // batch_size,
        ):
            batch = pending[batch_start : batch_start + batch_size]
            prompts = [_build_prompt_messages(p, system_prompt) for p in batch]
            inp = _batch_tokenize(tokenizer, prompts, device)

            gen_kwargs: dict[str, Any] = {
                **inp,
                "max_new_tokens": max_new_tokens,
                **mode.kwargs,
            }
            if mode.generate_fn is not None:
                gen_kwargs["custom_generate"] = mode.generate_fn
                gen_kwargs["processing_class"] = tokenizer

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**gen_kwargs)
            wall_time = time.time() - t0

            sequences = out if isinstance(out, torch.Tensor) else out.sequences
            batch_results: list[EvalResult] = []
            for j, prob in enumerate(batch):
                decoded = tokenizer.decode(sequences[j], skip_special_tokens=False)
                predicted = _extract(decoded)
                correct = _check(predicted, prob.expected_answer)

                batch_results.append(EvalResult(
                    problem_id=prob.id,
                    mode=mode.name,
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
            done_keys.update((r.problem_id, r.mode) for r in batch_results)

            _save_summary(output_dir, all_results)

    logger.info("Evaluation complete: %d total results", len(all_results))
    return all_results
