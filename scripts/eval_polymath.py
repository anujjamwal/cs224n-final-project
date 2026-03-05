#!/usr/bin/env python3
"""Evaluate a model on the Qwen/PolyMath benchmark with checkpointing.

PolyMath is a multilingual math reasoning benchmark with 18 languages,
4 difficulty levels, and 500 problems per language (125 per level).

Usage examples:

    # Baseline on English, all difficulty levels:
    python scripts/eval_polymath.py \\
        --model-path nvidia/OpenMath-Nemotron-1.5B \\
        --output-dir results/polymath-baseline \\
        --modes baseline \\
        --batch-size 4

    # HCoT model on easy problems only:
    python scripts/eval_polymath.py \\
        --model-path anujjamwal/OpenMath-Nemotron-1.5B-hcot \\
        --output-dir results/polymath-hcot-top \\
        --modes hcot \\
        --levels top \\
        --batch-size 4

    # Resume an interrupted run (same command — skips already-done problems):
    python scripts/eval_polymath.py \\
        --model-path nvidia/OpenMath-Nemotron-1.5B \\
        --output-dir results/polymath-baseline \\
        --modes baseline \\
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

from eval import load_polymath, run_eval, summarize_results
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

VALID_LEVELS = ["top", "high", "medium", "low"]


def parse_args():
    p = argparse.ArgumentParser(description="PolyMath benchmark evaluation")
    p.add_argument("--model-path", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--output-dir", required=True, help="Directory for checkpoint/result files")
    p.add_argument("--language", default="en", help="Language config (default: en)")
    p.add_argument("--dataset-id", default="Qwen/PolyMath", help="HuggingFace dataset ID")
    p.add_argument("--modes", nargs="+", default=["baseline"], choices=list(AVAILABLE_MODES.keys()),
                    help="Generation modes to evaluate")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--levels", nargs="*", default=None, choices=VALID_LEVELS,
                    help="Filter by difficulty level (top=easiest, low=hardest)")
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

    logger.info("Loading PolyMath dataset (language=%s, dataset=%s)", args.language, args.dataset_id)
    problems = load_polymath(
        args.language,
        dataset_id=args.dataset_id,
        levels=args.levels,
    )
    logger.info("Loaded %d problems", len(problems))

    modes = [AVAILABLE_MODES[m] for m in args.modes]

    results = run_eval(
        model, tokenizer, problems, args.output_dir, modes,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    summary = summarize_results(results)
    print("\n=== PolyMath Evaluation Summary ===")
    print(f"Language: {args.language}")
    print(f"Total: {summary['correct']}/{summary['total']} ({summary['accuracy']:.1%})")
    for mode, stats in summary.get("by_mode", {}).items():
        print(f"  {mode}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    for level, stats in summary.get("by_level", {}).items():
        print(f"  Level [{level}]: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
