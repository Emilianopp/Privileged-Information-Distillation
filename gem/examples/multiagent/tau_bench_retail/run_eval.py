#!/usr/bin/env python3
"""Evaluation runner for TAU-bench via the fork's native ``run()`` pipeline.

This script is a thin CLI that builds a ``RunConfig`` and delegates to the
tau-bench fork's ``tau_bench.run.run()``.  All fork features are exposed:
  • agent strategies: tool-calling, react, act, chat-react-priv, few-shot
  • privileged information (human or from JSON)
  • hosted-vllm / openrouter / openai providers
  • self-generated hints

Usage examples
--------------
# Basic (OpenAI, tool-calling):
    python run_eval.py --model gpt-4o --model-provider openai

# React with privileged info from human labels:
    python run_eval.py --model gpt-4o --model-provider openai \\
        --agent-strategy chat-react-priv --use-human-privileged

# Hosted vLLM:
    python run_eval.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --model-provider hosted_vllm --agent-strategy react

# Only a subset of tasks:
    python run_eval.py --model gpt-4o --model-provider openai \\
        --start-index 0 --end-index 10
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the fork is importable
# ---------------------------------------------------------------------------
TAU_BENCH_PATH = os.environ.get(
    "TAU_BENCH_PATH",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tau-bench")
    ),
)
if TAU_BENCH_PATH not in sys.path:
    sys.path.insert(0, TAU_BENCH_PATH)

from tau_bench.types import RunConfig  # noqa: E402
from tau_bench.run import run  # noqa: E402
from tau_bench.envs.user import UserStrategy  # noqa: E402
from litellm import provider_list  # noqa: E402


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run TAU-bench evaluation (delegates to the fork's run pipeline)."
    )

    # ── Model / provider ────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name for the agent")
    parser.add_argument("--model-provider", type=str, default="openai",
                        choices=provider_list,
                        help="LiteLLM provider for the agent")
    parser.add_argument("--user-model", type=str, default="Qwen/Qwen3-14B",
                        help="Model for the user simulator")
    parser.add_argument("--user-model-provider", type=str, default="hosted_vllm",
                        choices=provider_list,
                        help="LiteLLM provider for the user simulator")
    parser.add_argument("--user-vllm-port", type=int, default=8001,
                        help="Port for the user vLLM server (sets USER_VLLM_API_BASE)")

    # ── Agent strategy ──────────────────────────────────────────────────
    parser.add_argument("--agent-strategy", type=str, default="tool-calling",
                        choices=["tool-calling", "act", "react",
                                 "chat-react-priv", "few-shot"],
                        help="Agent strategy")

    # ── Environment ─────────────────────────────────────────────────────
    parser.add_argument("--env", type=str, default="retail",
                        choices=["retail", "airline"])
    parser.add_argument("--task-split", type=str, default="test",
                        choices=["train", "test", "dev"])
    parser.add_argument("--user-strategy", type=str, default="llm",
                        choices=[item.value for item in UserStrategy])

    # ── Task range ──────────────────────────────────────────────────────
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1,
                        help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+",
                        help="(Optional) run only these task IDs")
    parser.add_argument("--num-trials", type=int, default=1)

    # ── Sampling ────────────────────────────────────────────────────────
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)

    # ── Privileged information ──────────────────────────────────────────
    parser.add_argument("--privileged-trajectories-path", type=str, default=None,
                        help="Path to privileged trajectories JSON file")
    parser.add_argument("--use-human-privileged", action="store_true",
                        help="Use human-authored privileged info from tasks module")
    parser.add_argument("--use-only-hints", action="store_true",
                        help="Only use tool-name hints (no full arguments)")
    parser.add_argument("--use-self-generated-hints", action="store_true",
                        help="Use self-generated hints")
    parser.add_argument("--self-generated-hints-path", type=str, default=None,
                        help="Path to self-generated hints JSON")

    # ── Few-shot ────────────────────────────────────────────────────────
    parser.add_argument("--few-shot-displays-path", type=str, default=None,
                        help="Path to JSONL with few-shot displays")

    # ── Output / concurrency ────────────────────────────────────────────
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--max-concurrency", type=int, default=1,
                        help="Number of tasks to run in parallel")

    args = parser.parse_args()
    print(args)

    # Set the user vLLM API base so the fork's LLMUserSimulationEnv picks it up
    if args.user_model_provider == "hosted_vllm":
        os.environ.setdefault(
            "USER_VLLM_API_BASE", f"http://127.0.0.1:{args.user_vllm_port}/v1"
        )

    return RunConfig(
        model=args.model,
        model_provider=args.model_provider,
        user_model=args.user_model,
        user_model_provider=args.user_model_provider,
        agent_strategy=args.agent_strategy,
        env=args.env,
        task_split=args.task_split,
        user_strategy=args.user_strategy,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        num_trials=args.num_trials,
        temperature=args.temperature,
        seed=args.seed,
        shuffle=args.shuffle,
        privileged_trajectories_path=args.privileged_trajectories_path,
        use_human_privileged=args.use_human_privileged,
        use_only_hints=args.use_only_hints,
        use_self_generated_hints=args.use_self_generated_hints,
        self_generated_hints_path=args.self_generated_hints_path,
        few_shot_displays_path=args.few_shot_displays_path,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
    )


def main():
    config = parse_args()
    results = run(config)
    print(f"\nFinished – {len(results)} results collected.")


if __name__ == "__main__":
    main()
