"""GRPO training on Modal with W&B logging and multi-GPU support.

Usage:
    # Single GPU (default):
    modal run scripts/modal_grpo_train.py

    # Multi-GPU with DeepSpeed ZeRO-2:
    modal run scripts/modal_grpo_train.py --num-gpus 4

    # Detached (keeps running after you close terminal):
    modal run --detach scripts/modal_grpo_train.py --num-gpus 4

    # Custom GPU type:
    modal run scripts/modal_grpo_train.py --num-gpus 2 --gpu a100-80gb

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
        "peft",
        "accelerate",
        "deepspeed",
        "sentencepiece",
        "math-verify",
        "wandb",
        "huggingface_hub",
    )
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
# GPU-parameterised training functions
#
# Modal's @app.function decorator requires static gpu= values, so we define
# entry points for common GPU counts. The local entrypoint dispatches to the
# right one based on --num-gpus.
# ---------------------------------------------------------------------------

_COMMON_KWARGS = dict(
    timeout=6 * 3600,
    secrets=SECRETS,
    volumes={"/checkpoints": vol},
)


_GPU_FUNC_DEFAULTS = dict(
    model_name="anujjamwal/OpenMath-Nemotron-1.5B-hcot",
    grpo_repo_id="anujjamwal/OpenMath-Nemotron-1.5B-hcot-grpo",
    dataset="davidanugraha/OpenMathReasoning-Sampled",
    dataset_offset=0,
    dataset_limit=1000,
    num_generations=4,
    num_epochs=1,
    batch_size=2,
    grad_accum_steps=4,
    learning_rate=5e-7,
    max_completion_length=4096,
    sample_every_n_steps=50,
    use_lora=False,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
)


def _forward_to_run_training(num_gpus: int, **kwargs):
    merged = {**_GPU_FUNC_DEFAULTS, **kwargs}
    _run_training(num_gpus=num_gpus, **merged)


@app.function(gpu="a100", **_COMMON_KWARGS)
def train_1gpu(**kwargs):
    _forward_to_run_training(num_gpus=1, **kwargs)


@app.function(gpu="a100:2", **_COMMON_KWARGS)
def train_2gpu(**kwargs):
    _forward_to_run_training(num_gpus=2, **kwargs)


@app.function(gpu="a100:4", **_COMMON_KWARGS)
def train_4gpu(**kwargs):
    _forward_to_run_training(num_gpus=4, **kwargs)


@app.function(gpu="a100:8", **_COMMON_KWARGS)
def train_8gpu(**kwargs):
    _forward_to_run_training(num_gpus=8, **kwargs)


# ---------------------------------------------------------------------------
# Core launcher (shared by all GPU variants)
# ---------------------------------------------------------------------------

def _run_training(
    *,
    num_gpus: int,
    model_name: str,
    grpo_repo_id: str,
    dataset: str,
    dataset_offset: int,
    dataset_limit: int,
    num_generations: int,
    num_epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    max_completion_length: int,
    sample_every_n_steps: int,
    use_lora: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
):
    import json
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
        # Write DeepSpeed ZeRO-2 config
        ds_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": int(2e8),
                "reduce_scatter": True,
                "reduce_bucket_size": int(2e8),
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
        }
        ds_config_path = "/tmp/ds_config.json"
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f)

        worker_args += ["--deepspeed", ds_config_path]

        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_gpus),
            "--use_deepspeed",
            "--deepspeed_config_file", ds_config_path,
            "/root/scripts/_grpo_worker.py",
        ] + worker_args
    else:
        cmd = ["python", "/root/scripts/_grpo_worker.py"] + worker_args

    print(f"Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    # Commit volume after training
    from modal import Volume
    vol = Volume.from_name("hcot-grpo-checkpoints")
    vol.commit()

    return result.returncode


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

_TRAIN_FNS = {1: train_1gpu, 2: train_2gpu, 4: train_4gpu, 8: train_8gpu}


@app.local_entrypoint()
def main(
    num_gpus: int = 1,
    # Dataset
    dataset: str = "davidanugraha/OpenMathReasoning-Sampled",
    dataset_offset: int = 0,
    dataset_limit: int = 1000,
    # Training
    num_generations: int = 4,
    num_epochs: int = 1,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-7,
    max_completion_length: int = 4096,
    sample_every_n_steps: int = 50,
    # LoRA
    use_lora: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
):
    if num_gpus not in _TRAIN_FNS:
        raise ValueError(f"num_gpus must be one of {list(_TRAIN_FNS.keys())}, got {num_gpus}")

    fn = _TRAIN_FNS[num_gpus]
    fn.remote(
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
