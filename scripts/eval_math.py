#!/usr/bin/env python3
"""Evaluate a model on the MATH benchmark with checkpointing.

Usage examples:

    # Baseline (standard HF generation) on Level 1 problems:
    python scripts/eval_math.py \\
        --model-path nvidia/OpenMath-Nemotron-1.5B \\
        --output-dir results/math-baseline \\
        --modes baseline \\
        --levels 1 \\
        --batch-size 4

    # HCoT model with pruning on all levels:
    python scripts/eval_math.py \\
        --model-path anujjamwal/OpenMath-Nemotron-1.5B-hcot \\
        --output-dir results/math-hcot \\
        --modes hcot \\
        --batch-size 4

    # Resume an interrupted run (same command — skips already-done problems):
    python scripts/eval_math.py \\
        --model-path anujjamwal/OpenMath-Nemotron-1.5B-hcot \\
        --output-dir results/math-hcot \\
        --modes hcot \\
        --batch-size 4
"""

import argparse
import logging
import sys
import os

import torch

# Ensure lib/ is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import MathBenchmark, run_eval, summarize_results
from eval.runner import GenerationMode
from custom_generate.generate import _sample
from trainer import prepare_base_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


AVAILABLE_MODES = {
    "baseline": GenerationMode(name="Baseline", generate_fn=None, kwargs={"use_cache": True}),
    "hcot": GenerationMode(name="HCoT Prune", generate_fn=_sample, kwargs={"use_cache": False}),
    "hcot-cached": GenerationMode(name="HCoT Prune (Cached)", generate_fn=_sample, kwargs={"use_cache": True}),
}


def parse_args():
    p = argparse.ArgumentParser(description="MATH benchmark evaluation")
    p.add_argument("--model-path", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--output-dir", required=True, help="Directory for checkpoint/result files")
    p.add_argument("--split", default="test", help="Dataset split (default: test)")
    p.add_argument("--dataset-id", default="lighteval/MATH", help="HuggingFace dataset ID")
    p.add_argument("--modes", nargs="+", default=["baseline"], choices=list(AVAILABLE_MODES.keys()),
                    help="Generation modes to evaluate")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--levels", nargs="*", type=int, default=None,
                    help="Filter by difficulty levels (1-5)")
    p.add_argument("--subjects", nargs="*", default=None,
                    help="Filter by subject (e.g. 'Algebra' 'Number Theory')")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
                    help="Model dtype (default: bfloat16)")
    p.add_argument("--prepare-base-model", action="store_true",
                    help="Run prepare_base_model to add special tokens (needed for HCoT models)")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    logger.info("Loading model: %s (dtype=%s)", args.model_path, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.prepare_base_model:
        logger.info("Preparing base model (adding special tokens)")
        model, tokenizer = prepare_base_model(model, tokenizer)

    model.eval()

    benchmark = MathBenchmark()

    logger.info("Loading MATH dataset (split=%s, dataset=%s)", args.split, args.dataset_id)
    problems = benchmark.load(
        split=args.split,
        dataset_id=args.dataset_id,
        subjects=args.subjects,
        levels=args.levels,
    )
    logger.info("Loaded %d problems", len(problems))

    modes = [AVAILABLE_MODES[m] for m in args.modes]

    results = run_eval(
        model, tokenizer, problems, args.output_dir, modes,
        benchmark=benchmark,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    summary = summarize_results(results)
    print("\n=== MATH Evaluation Summary ===")
    print(f"Total: {summary['correct']}/{summary['total']} ({summary['accuracy']:.1%})")
    for mode, stats in summary.get("by_mode", {}).items():
        print(f"  {mode}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
