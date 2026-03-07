"""GRPO training on Modal with W&B logging.

Usage:
    # Run as a detached job (keeps running after you close terminal):
    modal run --detach scripts/modal_grpo_train.py

    # Run interactively:
    modal run scripts/modal_grpo_train.py

    # Override defaults:
    modal run scripts/modal_grpo_train.py --num-generations 8 --gpu a100-80gb

Requires Modal and W&B secrets configured:
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal secret create wandb WANDB_API_KEY=xxx
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "trl",
        "accelerate",
        "sentencepiece",
        "math-verify",
        "wandb",
        "huggingface_hub",
    )
    .add_local_dir("lib", remote_path="/root/lib")
)

app = modal.App("hcot-grpo-training", image=image)

# Persistent volume for checkpoints (survives across runs)
vol = modal.Volume.from_name("hcot-grpo-checkpoints", create_if_missing=True)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="a100",  # A100 40GB by default; use "a100-80gb" or "h100" for larger runs
    timeout=6 * 3600,  # 6 hours max
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    volumes={"/checkpoints": vol},
)
def train(
    model_name: str = "anujjamwal/OpenMath-Nemotron-1.5B-hcot",
    grpo_repo_id: str = "anujjamwal/OpenMath-Nemotron-1.5B-hcot-grpo",
    num_generations: int = 4,
    num_epochs: int = 1,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-7,
    max_completion_length: int = 4096,
    sample_every_n_steps: int = 50,
):
    import sys
    import os

    sys.path.insert(0, "/root/lib")
    os.environ.setdefault("WANDB_PROJECT", "hcot-grpo")

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

    # ---- W&B init ----
    reward_weights = [2.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    wandb.init(
        project="hcot-grpo",
        config={
            "model": model_name,
            "method": "GRPO",
            "platform": "modal",
            "num_generations": num_generations,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "learning_rate": learning_rate,
            "max_completion_length": max_completion_length,
            "reward_weights": reward_weights,
        },
    )

    # ---- Model ----
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model, tokenizer = prepare_base_model(model, tokenizer)

    # ---- Dataset ----
    SYSTEM_PROMPT = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}."

    raw_dataset = load_dataset(
        "anujjamwal/OpenMathReasoning-Sampled-Hierarchical-Cot",
        split="train",
    ).filter(lambda ex: len(ex["hierarchical_cot"]) > 50)
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

    # ---- Sample logging callback ----
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

    class WandbSampleCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % sample_every_n_steps != 0 or model is None:
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

    # ---- Training config ----
    output_dir = "/checkpoints/hcot-nemotron-1.5b-grpo"

    training_args = GRPOConfig(
        output_dir=output_dir,
        hub_model_id=grpo_repo_id,

        # Generation
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=512,

        # Optimisation
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        beta=0.1,
        bf16=True,
        gradient_checkpointing=True,

        # Reward weighting
        reward_weights=reward_weights,

        # Logging & checkpointing
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="wandb",
        push_to_hub=True,
    )

    # ---- Train ----
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        callbacks=[WandbSampleCallback()],
    )

    torch.cuda.empty_cache()
    print("Starting GRPO training...")
    trainer.train()

    # ---- Push & cleanup ----
    trainer.push_to_hub()
    tokenizer.push_to_hub(grpo_repo_id)
    vol.commit()
    wandb.finish()
    print("Training complete! Model pushed to HuggingFace Hub.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    num_generations: int = 4,
    gpu: str = "a100",
):
    train.remote(num_generations=num_generations)
