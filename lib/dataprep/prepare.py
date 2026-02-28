#!/usr/bin/env python3
"""Prepare the OpenMathReasoning dataset with hierarchical chain-of-thought segmentation.

Usage examples:
    python prepare.py --model claude-opus-4-6 --method cli --offset 0 --limit 100 --mode append
    python prepare.py --model claude-opus-4-6 --method api --offset 0 --limit 100 --mode overwrite
    python prepare.py --model claude-sonnet-4-6 --method cli --offset 0 --limit 50 --mode append --parallelism 8
    python prepare.py --model gemini-3.1-pro --method api --offset 0 --limit 100 --mode append

Modes:
    append    Load existing HF dataset, skip already-processed records, process the
              remainder, then merge and push.
    overwrite Process all records in [offset, offset+limit) and push, replacing any
              prior version of the dataset.
"""

import argparse
import logging
import os
import sys

from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv

import segment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOURCE_REPO = "davidanugraha/OpenMathReasoning-Sampled"
HF_REPO = "anujjamwal/OpenMathReasoning-Sampled-Hierarchical-Cot"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare hierarchical CoT dataset from OpenMathReasoning"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier passed to the API or CLI",
    )
    parser.add_argument(
        "--method",
        choices=["api", "cli"],
        default="cli",
        help="Use Python API or CLI subprocess (default: cli)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index into the source dataset (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Number of source records to consider",
    )
    parser.add_argument(
        "--mode",
        choices=["append", "overwrite"],
        required=True,
        help=(
            "append: skip records already on HF and merge results; "
            "overwrite: process all records and push a fresh dataset"
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=segment.DEFAULT_PARALLELISM,
        help=f"Parallel workers for CLI method (default: {segment.DEFAULT_PARALLELISM})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory to cache LLM outputs (default: {segment.DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_source_slice(offset, limit):
    logger.info(
        "Loading source records [%d, %d) from %s", offset, offset + limit, SOURCE_REPO
    )
    ds = load_dataset(SOURCE_REPO, split="train", streaming=True)
    records = list(ds.skip(offset).take(limit))
    logger.info("Loaded %d source records", len(records))
    return Dataset.from_list(records)


def load_existing():
    try:
        ds = load_dataset(HF_REPO, split="train")
        logger.info("Loaded %d existing records from %s", len(ds), HF_REPO)
        return ds
    except Exception:
        logger.info("No existing dataset found at %s", HF_REPO)
        return None


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_with_api(examples, model, parallelism, output_dir):
    if output_dir is None:
        output_dir = segment.DEFAULT_OUTPUT_DIR

    def _segment(x, idx):
        output_file = os.path.join(output_dir, f"example_{x["id"]}.txt")
        x["hcot_model"] = model
        try:
            x["hierarchical_cot"], x["hierarchical_cot_raw"] = segment.segment_chain_of_thought(
                x["question"], x["generated_solution"], x["expected_answer"], model=model, output_file=output_file
            )
        except Exception as e:
            logger.error("Failed to segment problem %r (index %s): %s", x["question"], x["id"], e)
            x["hierarchical_cot"], x["hierarchical_cot_raw"] = "", ""
        
        return x

    return examples.map(_segment, with_indices=True, num_proc=parallelism)


def process_with_cli(examples, model, parallelism, output_dir):
    raw = [
        {
            "problem_statement": ex["question"],
            "chain_of_thought": ex["generated_solution"],
            "final_solution": ex["expected_answer"],
        }
        for ex in examples
    ]
    results = segment.process_examples_parallel(raw, parallelism=parallelism, model=model, output_dir=output_dir)

    def _attach(x, idx):
        x["hierarchical_cot"], x["hierarchical_cot_raw"] = results[idx]
        x["hcot_model"] = model
        return x

    return examples.map(_attach, with_indices=True, num_proc=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    args = parse_args()

    source = load_source_slice(args.offset, args.limit)
    source_problems = set(source["question"])

    existing = None
    if args.mode == "append":
        existing = load_existing()

    # Determine which records still need processing.
    if existing is not None:
        done_problems = set(
            existing.filter(lambda x: len(x["hierarchical_cot"]) > 50)["question"]
        )
        to_process = source.filter(lambda x: x["question"] not in done_problems)
        logger.info(
            "%d already processed, %d remaining to process",
            len(source) - len(to_process),
            len(to_process),
        )
    else:
        to_process = source

    # Process.
    if len(to_process) == 0:
        logger.info("Nothing new to process.")
        newly_processed = None
    else:
        logger.info("Processing %d records via %s", len(to_process), args.method)
        if args.method == "api":
            newly_processed = process_with_api(to_process, args.model, args.parallelism, args.output_dir)
        else:
            newly_processed = process_with_cli(to_process, args.model, args.parallelism, args.output_dir)

    if newly_processed is not None:
        newly_processed = newly_processed.filter(lambda x: len(x["hierarchical_cot"]) > 50)

    # Build the final dataset to push.
    if args.mode == "append" and existing is not None:
        parts = []

        # Existing records outside the current source window → keep untouched.
        kept_outside = existing.filter(lambda x: x["question"] not in source_problems)
        if len(kept_outside) > 0:
            parts.append(kept_outside)

        # Existing records inside the source window that were already done → keep.
        kept_inside = existing.filter(
            lambda x: x["question"] in source_problems and bool(x["hierarchical_cot"])
        )

        if newly_processed is not None:
            newly_processed_questions = set(newly_processed["question"])
            kept_inside = kept_inside.filter(lambda x: x["question"] not in newly_processed_questions)
            parts.append(newly_processed)

        if len(kept_inside) > 0:
            parts.append(kept_inside)

        if not parts:
            logger.info("Nothing to push.")
            return

        final = concatenate_datasets(parts)
    else:
        if newly_processed is None:
            logger.info("Nothing to push.")
            return
        final = newly_processed

    logger.info("Pushing %d records to %s", len(final), HF_REPO)
    final.push_to_hub(HF_REPO)
    logger.info("Done.")


if __name__ == "__main__":
    main()
