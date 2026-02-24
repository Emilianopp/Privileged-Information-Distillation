#!/usr/bin/env bash
# Launch script for tau-bench RL training with verl.
#
# The user-simulator vLLM server is now managed dynamically by the training
# script (started before rollout, stopped before FSDP training) so that all
# GPUs can be used for training.
#
# Usage:
#   bash launch.sh [OVERRIDES...]
#
# Example:
#   bash launch.sh \
#     actor_rollout_ref.env.privileged_trajectories_path=/path/to/privileged.json \
#     actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
#     trainer.total_training_steps=100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Activate opsd venv
source "$PROJECT_ROOT/tau-bench/opsd/bin/activate"

# ---------------------------------------------------------------------------
# Configuration — override these via environment variables as needed
# ---------------------------------------------------------------------------
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

export TAU_BENCH_PATH="${TAU_BENCH_PATH:-$PROJECT_ROOT/tau-bench}"

# Prevent Ray from overriding CUDA_VISIBLE_DEVICES for workers
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ---------------------------------------------------------------------------
# Run training (user-sim vLLM server is managed inside train_tau_bench.py)
# ---------------------------------------------------------------------------
echo "==> Starting tau-bench RL training on GPUs: $TRAIN_GPUS..."

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python "$SCRIPT_DIR/train_tau_bench.py" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    actor_rollout_ref.model.path=/mnt/adea/data_rw/finetuning/base/models--Qwen--Qwen3-4B \
    "$@"

echo "==> Done."
