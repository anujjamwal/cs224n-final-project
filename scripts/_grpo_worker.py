"""Standalone GRPO training worker.

Launched directly for single-GPU or via ``accelerate launch`` for multi-GPU.
All configuration is passed through CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/root/lib")


def main():
    parser = argparse.ArgumentParser(description="GRPO training worker")
    parser.add_argument("--model-name", default="anujjamwal/OpenMath-Nemotron-1.5B-PruneAware")
    parser.add_argument("--grpo-repo-id", default="anujjamwal/OpenMath-Nemotron-1.5B-PruneAware-grpo")
    parser.add_argument("--dataset", default="davidanugraha/OpenMathReasoning-Sampled")
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument("--dataset-limit", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--sample-every-n-steps", type=int, default=50)
    parser.add_argument("--output-dir", default="/checkpoints/PruneAware-nemotron-1.5b-grpo")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA (PEFT) instead of full fine-tuning")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    args = parser.parse_args()

    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from datasets import load_dataset
    from trl import GRPOTrainer, GRPOConfig

    from trainer import prepare_base_model
    from trainer.rewards import (
        correctness_reward,
        syntax_reward,
        leaf_length_reward,
        depth_reward,
        compression_reward,
        format_reward,
    )

    # ---- PEFT / LoRA ----
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    # ---- W&B init (main process only) ----
    reward_weights = [2.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    if is_main:
        wandb.init(
            project="hcot-grpo",
            config={
                "model": args.model_name,
                "method": "GRPO",
                "platform": "modal",
                "distributed": is_distributed,
                "num_generations": args.num_generations,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "learning_rate": args.learning_rate,
                "max_completion_length": args.max_completion_length,
                "reward_weights": reward_weights,
                "use_lora": args.use_lora,
                "lora_r": args.lora_r if args.use_lora else None,
                "lora_alpha": args.lora_alpha if args.use_lora else None,
            },
        )

    # ---- Model ----
    print(f"[rank {local_rank}] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        # No device_map="auto" — let Accelerate/DeepSpeed handle placement
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- Dataset ----
    SYSTEM_PROMPT = (
        "Solve the following math problem. "
        "Make sure to put the answer (and only answer) inside \\boxed{}."
    )

    raw_dataset = load_dataset(
        args.dataset,
        split="train",
    ).skip(args.dataset_offset).take(args.dataset_limit)

    if is_main:
        print(f"Dataset size: {len(raw_dataset)}")

    def to_grpo_format(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "expected_answer": example["expected_answer"],
        }

    dataset = raw_dataset.map(
        to_grpo_format,
        remove_columns=[c for c in raw_dataset.column_names if c != "expected_answer"],
    )

    # ---- Reward functions ----
    reward_funcs = [
        correctness_reward,
        syntax_reward,
        leaf_length_reward,
        depth_reward,
        compression_reward,
        format_reward,
    ]
    reward_names = [
        "correctness", "syntax", "leaf_length",
        "depth", "compression", "format",
    ]

    # ---- Sample logging callback (main process only) ----
    class WandbSampleCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if not is_main:
                return
            if state.global_step % self.every_n != 0 or model is None:
                return

            model.eval()
            num_samples = min(3, len(dataset))
            samples = dataset.select(range(num_samples))
            table = wandb.Table(columns=[
                "step", "prompt", "completion", "expected_answer",
                *reward_names, "total_reward",
            ])

            for sample in samples:
                inputs = tokenizer.apply_chat_template(
                    sample["prompt"],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(model.device)

                with torch.no_grad():
                    gen = model.generate(
                        **inputs, max_new_tokens=2048,
                        do_sample=True, temperature=0.7,
                    )

                completion_text = tokenizer.decode(
                    gen[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=False,
                )
                completion_msg = [{"role": "assistant", "content": completion_text}]

                scores = {}
                for name, fn in zip(reward_names, reward_funcs):
                    if name == "correctness":
                        r = fn([completion_msg], expected_answer=[sample["expected_answer"]])
                    else:
                        r = fn([completion_msg])
                    scores[name] = r[0]

                total = sum(w * scores[n] for w, n in zip(reward_weights, reward_names))

                table.add_data(
                    state.global_step,
                    sample["prompt"][-1]["content"][:200],
                    completion_text[:1000],
                    sample["expected_answer"],
                    *[scores[n] for n in reward_names],
                    total,
                )

            wandb.log({"sample_completions": table}, step=state.global_step)
            model.train()

    sample_cb = WandbSampleCallback()
    sample_cb.every_n = args.sample_every_n_steps

    # ---- Training config ----
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        hub_model_id=args.grpo_repo_id,

        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        generation_kwargs={
            "processing_class": tokenizer,
            "return_unpruned_output": True,
        },

        # Optimisation
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        beta=0.1,
        bf16=True,
        gradient_checkpointing=True,
        mask_truncated_completions=True,
        torch_compile=True,

        # Reward weighting
        reward_weights=reward_weights,

        # Logging & checkpointing
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="wandb" if is_main else "none",
        push_to_hub=is_main,
    )

    # ---- Train ----
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        callbacks=[sample_cb],
        peft_config=peft_config,
    )

    torch.cuda.empty_cache()
    if is_main:
        print("Starting GRPO training...")
    trainer.train()

    # ---- Push & cleanup (main process only) ----
    if is_main:
        if args.use_lora:
            # Merge LoRA weights into the base model and push the full model
            print("Merging LoRA adapter into base model...")
            merged_model = trainer.model.merge_and_unload()
            merged_model.push_to_hub(args.grpo_repo_id)
        else:
            trainer.push_to_hub()
        tokenizer.push_to_hub(args.grpo_repo_id)
        wandb.finish()
        print("Training complete! Model pushed to HuggingFace Hub.")


if __name__ == "__main__":
    main()
