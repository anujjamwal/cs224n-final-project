"""GRPO training on Modal with W&B logging and multi-GPU support.

Usage:
    # Single GPU (default):
    modal run scripts/modal_grpo_train.py

    # Multi-GPU:
    modal run scripts/modal_grpo_train.py --num-gpus 4

    # Detached (keeps running after you close terminal):
    modal run --detach scripts/modal_grpo_train.py --num-gpus 4

    # With LoRA:
    modal run --detach scripts/modal_grpo_train.py --num-gpus 4 --use-lora

Requires Modal and W&B secrets configured:
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal secret create wandb WANDB_API_KEY=xxx
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------
flash_attn_release = (
    "https://github.com/lesj0610/flash-attention/releases/download/"
    "v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "sentencepiece",
        "math-verify",
        "wandb",
        "huggingface_hub",)
    .add_local_dir("lib", remote_path="/root/lib")
    .add_local_file(
        "scripts/_grpo_worker.py", remote_path="/root/scripts/_grpo_worker.py"
    )
)

app = modal.App("hcot-grpo-training", image=image)

# Persistent volume for checkpoints (survives across runs)
vol = modal.Volume.from_name("hcot-grpo-checkpoints", create_if_missing=True)

SECRETS = [
    modal.Secret.from_name("huggingface"),
    modal.Secret.from_name("wandb"),
]


# ---------------------------------------------------------------------------
# Training function — single entry point for all GPU counts
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100:8",
    timeout=6 * 3600,
    secrets=SECRETS,
    volumes={"/checkpoints": vol},
)
def train(
    num_gpus: int = 4,
    model_name: str = "anujjamwal/OpenMath-Nemotron-1.5B-PruneAware",
    grpo_repo_id: str = "anujjamwal/OpenMath-Nemotron-1.5B-PruneAware-grpo",
    dataset: str = "davidanugraha/OpenMathReasoning-Sampled",
    dataset_offset: int = 0,
    dataset_limit: int = 1000,
    num_generations: int = 4,
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum_steps: int = 2,
    learning_rate: float = 1e-5,
    max_completion_length: int = 4096,
    sample_every_n_steps: int = 50,
    use_lora: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
):
    import subprocess

    output_dir = "/checkpoints/hcot-nemotron-1.5b-grpo"

    # Build worker CLI args
    worker_args = [
        "--model-name", model_name,
        "--grpo-repo-id", grpo_repo_id,
        "--dataset", dataset,
        "--dataset-offset", str(dataset_offset),
        "--dataset-limit", str(dataset_limit),
        "--num-generations", str(num_generations),
        "--num-epochs", str(num_epochs),
        "--batch-size", str(batch_size),
        "--grad-accum-steps", str(grad_accum_steps),
        "--learning-rate", str(learning_rate),
        "--max-completion-length", str(max_completion_length),
        "--sample-every-n-steps", str(sample_every_n_steps),
        "--output-dir", output_dir,
    ]

    if use_lora:
        worker_args += [
            "--use-lora",
            "--lora-r", str(lora_r),
            "--lora-alpha", str(lora_alpha),
            "--lora-dropout", str(lora_dropout),
        ]

    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            "--master_port", "29500",
            "/root/scripts/_grpo_worker.py",
        ] + worker_args
    else:
        cmd = ["python", "/root/scripts/_grpo_worker.py"] + worker_args

    print(f"Launching: {' '.join(cmd)}")
    # Stream output directly to Modal logs (no capture)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Training command failed (exit {result.returncode})")

    # Commit volume after training
    from modal import Volume
    vol = Volume.from_name("hcot-grpo-checkpoints")
    vol.commit()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    num_gpus: int = 8,
    # Dataset
    dataset: str = "davidanugraha/OpenMathReasoning-Sampled",
    dataset_offset: int = 0,
    dataset_limit: int = 1000,
    # Training
    num_generations: int = 4,
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum_steps: int = 2,
    learning_rate: float = 1e-5,
    max_completion_length: int = 4096,
    sample_every_n_steps: int = 50,
    # LoRA
    use_lora: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
):
    train.remote(
        num_gpus=num_gpus,
        dataset=dataset,
        dataset_offset=dataset_offset,
        dataset_limit=dataset_limit,
        num_generations=num_generations,
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        max_completion_length=max_completion_length,
        sample_every_n_steps=sample_every_n_steps,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
