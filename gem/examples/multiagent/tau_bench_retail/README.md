# TAU-bench – GEM MultiAgentEnv Integration

Thin GEM wrapper around the [Vattikondadheeraj/tau-bench](https://github.com/Vattikondadheeraj/tau-bench) fork (`dheeraj/local-sync` branch).

All environment logic, agents, user simulation, privileged-info handling, and reward calculation are delegated to the fork. The files here are adapters that expose those capabilities via GEM's `MultiAgentEnv` API.

## Setup

### 1. Install the tau-bench fork

The fork is expected at `<repo-root>/tau-bench` (auto-detected) or at the path set via `TAU_BENCH_PATH`:

```bash
cd /path/to/Privileged-Information-Distillation/tau-bench
git checkout dheeraj/local-sync
pip install -e .

# Or, if the fork lives elsewhere:
export TAU_BENCH_PATH=/path/to/tau-bench
```

### 2. Install GEM

```bash
cd /path/to/Privileged-Information-Distillation/gem
pip install -e .
```

### 3. Set API Keys

```bash
export OPENAI_API_KEY="your-key"

# Optional – for OpenRouter / hosted vLLM
export OPENROUTER_API_KEY="your-key"
```

### 4. Run Evaluation

```bash
# Basic (OpenAI, tool-calling)
python run_eval.py --model gpt-4o --model-provider openai

# React with privileged info from human labels
python run_eval.py --model gpt-4o --model-provider openai \
    --agent-strategy chat-react-priv --use-human-privileged

# React with privileged info from JSON
python run_eval.py --model gpt-4o --model-provider openai \
    --agent-strategy chat-react-priv \
    --privileged-trajectories-path /path/to/privileged_actions.json

# Hosted vLLM (local model)
python run_eval.py --model meta-llama/Llama-3.1-8B-Instruct \
    --model-provider hosted_vllm --agent-strategy react

# Subset of tasks
python run_eval.py --model gpt-4o --model-provider openai \
    --start-index 0 --end-index 10
```

## Files

| File | Role |
|---|---|
| `tau_bench_env.py` | GEM `MultiAgentEnv` adapter – delegates to the fork's native `Env` via `get_env()` |
| `tau_bench_agent.py` | Agent wrapper – delegates to the fork's `agent_factory` (supports all strategies) |
| `run_eval.py` | CLI runner – builds a `RunConfig` and calls the fork's `run()` pipeline |

## Agent Strategies

| Strategy | Description |
|---|---|
| `tool-calling` | Native function-calling (default) |
| `react` | ReAct with `<think>` reasoning |
| `act` | ReAct without reasoning |
| `chat-react-priv` | ReAct + privileged information |
| `few-shot` | Few-shot in-context examples |

## Model / Provider Support

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, …
- **OpenRouter**: `google/gemini-2.0-flash-001`, `deepseek/deepseek-chat`, `anthropic/claude-3.5-sonnet`, …
- **hosted_vllm**: Any model served locally via vLLM
