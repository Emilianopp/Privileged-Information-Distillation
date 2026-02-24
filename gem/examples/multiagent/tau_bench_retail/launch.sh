#!/usr/bin/env bash
# =============================================================================
# TAU-bench Privileged Information Distillation – Launch Script
#
# Launches two vLLM servers and runs evaluation:
#   GPU 0 → Qwen3-8B  (agent model)       on port 8000
#   GPU 1 → Qwen3-14B (user simulator)    on port 8001
#
# Usage:
#   bash launch.sh                          # defaults
#   bash launch.sh --task-ids 0 1 2         # run specific tasks
#   bash launch.sh --end-index 10           # first 10 tasks
#   bash launch.sh --agent-strategy react   # non-privileged react
#   OUTPUT_DIR=/my/path bash launch.sh      # custom output directory
# =============================================================================
set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TAU_BENCH_PATH="${TAU_BENCH_PATH:-$REPO_ROOT/tau-bench}"
VENV_PATH="${VENV_PATH:-$TAU_BENCH_PATH/opsd}"

# ── Models ──────────────────────────────────────────────────────────────────
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-8B}"
USER_MODEL="${USER_MODEL:-Qwen/Qwen3-14B}"

# ── GPU assignment ──────────────────────────────────────────────────────────
AGENT_GPU="${AGENT_GPU:-0}"
USER_GPU="${USER_GPU:-1}"

# ── vLLM ports ──────────────────────────────────────────────────────────────
AGENT_PORT="${AGENT_PORT:-8000}"
USER_PORT="${USER_PORT:-8001}"

# ── Privileged info ─────────────────────────────────────────────────────────
PRIVILEGED_PATH="${PRIVILEGED_PATH:-$REPO_ROOT/files/privileged_actions.json}"

# ── Eval defaults ───────────────────────────────────────────────────────────
AGENT_STRATEGY="${AGENT_STRATEGY:-chat-react-priv}"
TASK_SPLIT="${TASK_SPLIT:-train}"
ENV_NAME="${ENV_NAME:-retail}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
BASE_RESULTS_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${BASE_RESULTS_DIR}/${ENV_NAME}_${RUN_TIMESTAMP}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
log() { echo -e "${BLUE}[launch]${NC} $*"; }
err() { echo -e "${RED}[launch] ERROR:${NC} $*" >&2; }

cleanup() {
    log "Shutting down vLLM servers..."
    [[ -n "${AGENT_PID:-}" ]] && kill "$AGENT_PID" 2>/dev/null && log "Killed agent vLLM (PID $AGENT_PID)"
    [[ -n "${USER_PID:-}" ]]  && kill "$USER_PID"  2>/dev/null && log "Killed user vLLM (PID $USER_PID)"
    wait 2>/dev/null
    log "Done."
}
trap cleanup EXIT INT TERM

# ── Activate venv ───────────────────────────────────────────────────────────
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    log "Activating venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    err "Virtual env not found at $VENV_PATH"
    err "Set VENV_PATH or create it: uv venv $VENV_PATH --python 3.12.3"
    exit 1
fi

# ── Validate paths ──────────────────────────────────────────────────────────
[[ -d "$TAU_BENCH_PATH" ]] || { err "tau-bench not found at $TAU_BENCH_PATH"; exit 1; }
[[ -f "$PRIVILEGED_PATH" ]] || { err "Privileged actions not found at $PRIVILEGED_PATH"; exit 1; }

# ── Create run output directory ────────────────────────────────────────────
mkdir -p "$LOG_DIR"
log "Created run directory: $LOG_DIR"

log "──────────────────────────────────────────────────────────"
log "Agent model:    ${GREEN}$AGENT_MODEL${NC}  (GPU $AGENT_GPU, port $AGENT_PORT)"
log "User model:     ${GREEN}$USER_MODEL${NC}   (GPU $USER_GPU, port $USER_PORT)"
log "Strategy:       ${YELLOW}$AGENT_STRATEGY${NC}"
log "Privileged:     $PRIVILEGED_PATH"
log "Task split:     $TASK_SPLIT"
log "Log dir:        $LOG_DIR"
log "──────────────────────────────────────────────────────────"

# ── Launch agent vLLM server (GPU 0) ────────────────────────────────────────
log "Starting agent vLLM server ($AGENT_MODEL on GPU $AGENT_GPU, port $AGENT_PORT)..."
CUDA_VISIBLE_DEVICES="$AGENT_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$AGENT_MODEL" \
    --port "$AGENT_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOG_DIR/vllm_agent.log" 2>&1 &
AGENT_PID=$!
log "Agent vLLM PID: $AGENT_PID (log: $LOG_DIR/vllm_agent.log)"

# ── Launch user vLLM server (GPU 1) ────────────────────────────────────────
log "Starting user vLLM server ($USER_MODEL on GPU $USER_GPU, port $USER_PORT)..."
CUDA_VISIBLE_DEVICES="$USER_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$USER_MODEL" \
    --port "$USER_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOG_DIR/vllm_user.log" 2>&1 &
USER_PID=$!
log "User vLLM PID: $USER_PID (log: $LOG_DIR/vllm_user.log)"

# ── Wait for servers to be ready ────────────────────────────────────────────
wait_for_server() {
    local name="$1" port="$2" pid="$3" max_wait="${4:-600}"
    local elapsed=0
    log "Waiting for $name server (port $port)..."
    while ! curl -s "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            err "$name server (PID $pid) died. Check logs."
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $max_wait ]]; then
            err "$name server didn't start within ${max_wait}s"
            exit 1
        fi
        printf "  ${YELLOW}%ds...${NC}\n" "$elapsed"
    done
    log "${GREEN}$name server ready!${NC} (took ${elapsed}s)"
}

wait_for_server "Agent" "$AGENT_PORT" "$AGENT_PID"
wait_for_server "User"  "$USER_PORT"  "$USER_PID"

# ── Set env vars for the fork ──────────────────────────────────────────────
export TAU_BENCH_PATH
export USER_VLLM_API_BASE="http://127.0.0.1:${USER_PORT}/v1"

# ── Run evaluation ──────────────────────────────────────────────────────────
log "Starting evaluation..."
python "$SCRIPT_DIR/run_eval.py" \
    --model "$AGENT_MODEL" \
    --model-provider hosted_vllm \
    --user-model "$USER_MODEL" \
    --user-model-provider hosted_vllm \
    --user-vllm-port "$USER_PORT" \
    --agent-strategy "$AGENT_STRATEGY" \
    --privileged-trajectories-path "$PRIVILEGED_PATH" \
    --env "$ENV_NAME" \
    --task-split "$TASK_SPLIT" \
    --temperature "$TEMPERATURE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --log-dir "$LOG_DIR" \
    "$@"

log "${GREEN}Evaluation complete!${NC}"
