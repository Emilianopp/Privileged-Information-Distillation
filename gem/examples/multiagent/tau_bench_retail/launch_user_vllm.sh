#!/bin/bash
# Launch a vLLM server for the user simulator (Qwen3-14B) on a single GPU.
#
# Usage:
#   bash launch_user_vllm.sh              # defaults: GPU 0, port 8001
#   CUDA_VISIBLE_DEVICES=1 bash launch_user_vllm.sh   # use GPU 1
#   USER_VLLM_PORT=9001 bash launch_user_vllm.sh      # custom port
#
# The run_eval.py script expects the user model at http://127.0.0.1:8001/v1
# (configurable via USER_VLLM_API_BASE env var).

set -euo pipefail

MODEL="${USER_VLLM_MODEL:-Qwen/Qwen3-14B}"
PORT="${USER_VLLM_PORT:-8001}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_MODEL_LEN="${USER_VLLM_MAX_MODEL_LEN:-8192}"

echo "=========================================="
echo " Launching vLLM user-simulator server"
echo "  Model:   ${MODEL}"
echo "  Port:    ${PORT}"
echo "  GPU:     ${GPU}"
echo "  Max len: ${MAX_MODEL_LEN}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU}" python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.90
