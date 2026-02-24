#!/usr/bin/env python3
"""Launch random-search EAI jobs for pi_distill tau-bench training.

Each iteration samples hyperparameters, generates a bash script with hydra
overrides, and submits via `eai job new`.
"""

import subprocess
import os
import time
import random

# ============================================================================
# SWEEP CONFIG
# ============================================================================
TOTAL_RUNS = 30

TRAIN_SCRIPT = "/home/toolkit/emi_home/Privileged-Information-Distillation/gem/examples/train_verl_tau_bench/train_tau_bench.py"
VENV_ACTIVATE = "/home/toolkit/emi_home/Privileged-Information-Distillation/tau-bench/opsd/bin/activate"
PROJECT_ROOT = "/home/toolkit/emi_home/Privileged-Information-Distillation"

# ============================================================================
# EAI JOB SETTINGS (shared across all runs)
# ============================================================================
EAI_SETTINGS = {
    "account": "snow.research.adea",
    "image": "registry.toolkit-sp.yul201.service-now.com/snow.research.adea/ui_copilot_playwright",
    "cpu": 60,
    "mem": 512,
    "gpu": 3,
}


# ============================================================================
# Helpers
# ============================================================================

def build_overrides(params: dict) -> list[str]:
    overrides = []
    for key, val in params.items():
        if val is None:
            continue
        # Convert Python bools to hydra-friendly lowercase
        if isinstance(val, bool):
            val = str(val).lower()
        # Quote strings so Hydra doesn't misparse (e.g. "20260217_215816" as int)
        # But don't quote hydra list literals like [retail,airline]
        if isinstance(val, str) and not val.startswith("["):
            overrides.append(f'{key}="{val}"')
        elif isinstance(val, str):
            overrides.append(f"{key}={val}")
        else:
            overrides.append(f"{key}={val}")
    return overrides


def extract_job_id(eai_output: str) -> str | None:
    """Parse the UUID job ID from eai's table output (second line, first column)."""
    import re
    for line in eai_output.strip().splitlines():
        match = re.match(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", line)
        if match:
            return match.group(1)
    return None


def build_hints_tag(use_human_privileged, use_only_hints):
    """Build a short tag describing the privileged info configuration."""
    parts = []
    if use_human_privileged:
        parts.append("human")
    if use_only_hints:
        parts.append("hints_only")
    return "_".join(parts) if parts else "no_pi"


def submit_job(params: dict, run_id: str, tag: str):
    """Generate bash script and submit EAI job."""
    overrides = build_overrides(params)

    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"source {VENV_ACTIVATE}",
        f"export TAU_BENCH_PATH={PROJECT_ROOT}/tau-bench",
        "export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1",
        f"python {TRAIN_SCRIPT} \\",
    ]
    for i, ov in enumerate(overrides):
        suffix = " \\" if i < len(overrides) - 1 else ""
        script_lines.append(f"    {ov}{suffix}")

    script_content = "\n".join(script_lines) + "\n"

    tmp_path = f"/home/toolkit/emi_home/tmp/tau_bench_{run_id}_{tag}.sh"
    with open(tmp_path, "w") as f:
        f.write(script_content)
    os.chmod(tmp_path, 0o755)

    job_name = f"tau_bench_{run_id}_{tag}".replace(".", "_").replace("-", "_").lower()
    cmd = [
        "eai", "job", "new",
        "--name", job_name,
        "--account", EAI_SETTINGS["account"],
        "--restartable",
        "--image", EAI_SETTINGS["image"],
        "--data", "snow.research.adea.data:/mnt/adea/data_rw:rw",
        "--data", "snow.research.adea.data:/mnt/adea/data:ro",
        "--data", "snow.research.adea.dheeraj_home:/home/toolkit:rw",
        "--env", "HOME=/home/toolkit",
        "--cpu", str(EAI_SETTINGS["cpu"]),
        "--mem", str(EAI_SETTINGS["mem"]),
        "--gpu", str(EAI_SETTINGS["gpu"]),
        "--", "bash", tmp_path,
    ]

    print(f"Run ID:  {run_id}")
    print(f"Job:     {job_name}")
    print(f"Script:  {tmp_path}")
    print(f"eai command:\n  {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    job_id = extract_job_id(result.stdout)
    if job_id and result.returncode == 0:
        print(f"eai job logs -f {job_id}")
    else:
        print(f"Job submission failed (exit code {result.returncode})")
    print("=" * 60)
    return job_id


# ============================================================================
# Main: random search loop
# ============================================================================

def main():
    for i in range(TOTAL_RUNS):
        run_id = time.strftime("%Y%m%d_%H%M%S")

        # --- Sample hyperparameters ---
        lr = random.choice([1e-6, 5e-6, 1e-5])
        gamma = random.choice([0.9, 0.95, 1.0])
        temperature = random.choice([0.5, 0.7, 0.9])
        ppo_epochs = random.choice([1, 2, 3])
        n_repeat = random.choice([3, 4, 5])
        distill_coef = random.choice([0.1, 0.25, 0.5])
        distill_alpha = random.choice([0.5, 1.0, 2.0])
        unique_tasks = random.choice([40, 50, 64])
        use_only_hints = random.choice([True, False])
        use_human_privileged = random.choice([True, False])
        num_env = unique_tasks * n_repeat
        hints_tag = build_hints_tag(use_human_privileged, use_only_hints)
        tag = (
            f"pi_distill__beta{distill_coef}__alpha{distill_alpha}"
            f"__lr{lr}__gamma{gamma}__temp{temperature}"
            f"__epochs{ppo_epochs}__nr{n_repeat}__ut{unique_tasks}__{hints_tag}"
        )

        params = {
            # --- Model (path in ppo_trainer base, gradient_checkpointing is NOT) ---
            "actor_rollout_ref.model.path": "/mnt/adea/data_rw/finetuning/base/models--Qwen--Qwen3-4B",
            "actor_rollout_ref.model.enable_gradient_checkpointing": True,

            # --- Trainer (all in ppo_trainer base) ---
            "trainer.n_gpus_per_node": 2,
            "trainer.total_training_steps": 600,
            "trainer.eval_freq": 1,
            "trainer.save_freq": 999999,
            "trainer.project_name": "verl-tau-bench",
            "trainer.experiment_name": f"{run_id}_{tag}",

            # --- Environment (keys in config.yaml don't need +) ---
            "actor_rollout_ref.env.num_env": num_env,
            "actor_rollout_ref.env.rollout_batch_size": num_env,
            "actor_rollout_ref.env.n_repeat": n_repeat,
            "actor_rollout_ref.env.env_name": "retail",
            "actor_rollout_ref.env.user_model": "/mnt/adea/data_rw/finetuning/base/models--Qwen--Qwen3-14B",
            "actor_rollout_ref.env.user_strategy": "llm",
            "actor_rollout_ref.env.user_temperature": 0.7,
            "actor_rollout_ref.env.user_vllm_gpu": 2,
            "actor_rollout_ref.env.user_vllm_max_model_len": 25000,
            "actor_rollout_ref.env.use_reasoning": True,
            "actor_rollout_ref.env.debug": False,
            # Keys NOT in config.yaml — need + prefix
            "++actor_rollout_ref.env.max_turns": 15,
            "++actor_rollout_ref.env.filter_zero_adv_groups": True,
            "++actor_rollout_ref.env.eval_env_names": "[retail,airline]",
            "++actor_rollout_ref.env.save_training_data": True,

            # --- Privileged info (in config.yaml) ---
            "actor_rollout_ref.env.use_only_hints": use_only_hints,
            "actor_rollout_ref.env.use_human_privileged": use_human_privileged,
            "actor_rollout_ref.env.use_self_generated_hints": False,

            # --- Rollout (temperature/max_model_len/response_length/enable_chunked_prefill/gpu_memory_utilization in config.yaml) ---
            "actor_rollout_ref.gamma": gamma,
            "actor_rollout_ref.apply_chat_template": False,
            "actor_rollout_ref.prompt_template": "na",
            "actor_rollout_ref.rollout.temperature": temperature,
            "actor_rollout_ref.rollout.max_model_len": 25000,
            "actor_rollout_ref.rollout.response_length": 2058,
            "actor_rollout_ref.rollout.enable_chunked_prefill": True,
            "actor_rollout_ref.rollout.gpu_memory_utilization": 0.25,
            # In rollout.yaml base config — no prefix needed
            "actor_rollout_ref.rollout.max_num_batched_tokens": 16384,
            "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 4096,

            # --- Actor (optim.lr/ppo_epochs/ppo_micro_batch_size in config.yaml) ---
            "actor_rollout_ref.actor.optim.lr": lr,
            "actor_rollout_ref.actor.ppo_epochs": ppo_epochs,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
            # Keys NOT in config.yaml — need + prefix
            "++actor_rollout_ref.actor.student_clip_ratio": 0.2,
            "++actor_rollout_ref.alpha_anneal_epochs": 15,
            "++actor_rollout_ref.actor.distill_mode": "pi_distill",
            "++actor_rollout_ref.actor.distill_coef": distill_coef,
            "++actor_rollout_ref.actor.distill_alpha": distill_alpha,
        }

        print(f"\n{'=' * 60}")
        print(f"Run {i + 1}/{TOTAL_RUNS}")
        print(f"{'=' * 60}")
        submit_job(params, run_id, tag)

        if i < TOTAL_RUNS - 1:
            time.sleep(2)


if __name__ == "__main__":
    main()
