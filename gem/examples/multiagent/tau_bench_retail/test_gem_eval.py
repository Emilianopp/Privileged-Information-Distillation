#!/usr/bin/env python3
"""GEM-style evaluation of TAU-bench retail tasks.

Runs TAU-bench evaluation *through* gem's register/make abstractions and the
TauBenchAgent wrapper, validating that the full gem integration works.

Prerequisites
-------------
- vLLM servers must already be running (agent on port 8000, user on port 8001).
  Use ``launch.sh`` to start them, or start them manually.
- The tau-bench fork must be available (set ``TAU_BENCH_PATH`` or place it at
  ``<repo>/tau-bench``).

Usage
-----
::

    source tau-bench/opsd/bin/activate
    python gem/examples/multiagent/tau_bench_retail/test_gem_eval.py --task-ids 1 2 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Resolve paths before any gem/tau-bench imports ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR / ".." / ".." / ".." / ".."
REPO_ROOT = REPO_ROOT.resolve()

# Ensure TAU_BENCH_PATH is set so the adapters can find the fork
if "TAU_BENCH_PATH" not in os.environ:
    os.environ["TAU_BENCH_PATH"] = str(REPO_ROOT / "tau-bench")

# Ensure USER_VLLM_API_BASE is set for the user simulator
if "USER_VLLM_API_BASE" not in os.environ:
    os.environ["USER_VLLM_API_BASE"] = "http://127.0.0.1:8001/v1"

# Add gem to the Python path so ``import gem`` works when running standalone
GEM_ROOT = str(REPO_ROOT / "gem")
if GEM_ROOT not in sys.path:
    sys.path.insert(0, GEM_ROOT)

# ── Imports ─────────────────────────────────────────────────────────────────
import importlib.util  # noqa: E402

import gem  # noqa: E402


def _import_from_file(module_name: str, file_path: Path):
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_env_mod = _import_from_file("tau_bench_env", SCRIPT_DIR / "tau_bench_env.py")
_agent_mod = _import_from_file("tau_bench_agent", SCRIPT_DIR / "tau_bench_agent.py")
TauBenchEnv = _env_mod.TauBenchEnv
TauBenchAgent = _agent_mod.TauBenchAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="GEM-style TAU-bench retail evaluation"
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Task indices to evaluate (default: 0 1 2)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Agent model (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--provider",
        default="hosted_vllm",
        help="Agent model provider (default: hosted_vllm)",
    )
    parser.add_argument(
        "--user-model",
        default="Qwen/Qwen3-14B",
        help="User simulator model (default: Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--user-provider",
        default="hosted_vllm",
        help="User simulator provider (default: hosted_vllm)",
    )
    parser.add_argument(
        "--agent-strategy",
        default="chat-react-priv",
        help="Agent strategy (default: chat-react-priv)",
    )
    parser.add_argument(
        "--privileged-trajectories-path",
        default=str(REPO_ROOT / "files" / "privileged_actions.json"),
        help="Path to privileged_actions.json",
    )
    parser.add_argument(
        "--task-split",
        default="train",
        help="Task split (default: train)",
    )
    parser.add_argument(
        "--env-name",
        default="retail",
        help="Environment name (default: retail)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Max steps per task (default: 30)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Output directory for traces. Defaults to "
             "results/<env>_<YYYYMMDD_HHMMSS>/",
    )
    return parser.parse_args()


def _build_log_dir(args) -> Path:
    """Create ``results/<env>_<YYYYMMDD_HHMMSS>/`` matching launch.sh convention."""
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = SCRIPT_DIR / "results" / f"{args.env_name}_{run_ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _ckpt_filename(args) -> str:
    """Build checkpoint filename matching tau-bench's convention:

    ``<strategy>-<agent_model_short>-<temp>_user-<user_model_safe>-llm.json``
    """
    agent_short = args.model.split("/")[-1]
    user_safe = args.user_model.replace("/", "_")
    return (
        f"{args.agent_strategy}-{agent_short}-{args.temperature}"
        f"_user-{user_safe}-llm.json"
    )


def main():
    args = parse_args()

    # ── 0. Set up output directory ──────────────────────────────────────────
    log_dir = _build_log_dir(args)
    ckpt_path = log_dir / _ckpt_filename(args)
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint:    {ckpt_path}")

    env_id = f"tau-bench:{args.env_name}-{args.task_split}-v0"

    # ── 1. Register the environment with gem ────────────────────────────────
    print(f"\nRegistering environment: {env_id}")
    gem.register(
        env_id,
        entry_point=TauBenchEnv,
        env_name=args.env_name,
        task_split=args.task_split,
        user_model=args.user_model,
        user_provider=args.user_provider,
        user_strategy="llm",
    )

    # ── 2. Create the environment via gem.make() ────────────────────────────
    print("Creating environment via gem.make()...")
    env = gem.make(env_id)
    print(f"  Environment type: {type(env).__name__}")
    print(f"  Number of tasks:  {len(env.tasks)}")
    print(f"  Possible agents:  {env.possible_agents}")

    # ── 3. Check privileged data availability ───────────────────────────────
    priv_path = args.privileged_trajectories_path
    priv_available = os.path.isfile(priv_path)
    print(f"  Privileged data:  {priv_path}")
    print(f"  Available:        {priv_available}")

    if priv_available:
        with open(priv_path) as f:
            priv_data = json.load(f)
        print(f"  Privileged tasks: {len(priv_data)} entries")

    # ── 4. Build the agent ──────────────────────────────────────────────────
    print(f"\nCreating TauBenchAgent (strategy={args.agent_strategy})...")
    agent = TauBenchAgent(
        model=args.model,
        provider=args.provider,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        env_name=args.env_name,
        privileged_trajectories_path=priv_path if priv_available else None,
    )

    # ── 5. Run tasks ────────────────────────────────────────────────────────
    results = []
    print(f"\nRunning {len(args.task_ids)} task(s): {args.task_ids}")
    print("=" * 60)

    for task_idx in args.task_ids:
        if task_idx >= len(env.tasks):
            print(f"\n[Task {task_idx}] SKIPPED — index out of range "
                  f"(max {len(env.tasks) - 1})")
            continue

        print(f"\n[Task {task_idx}] Starting...")
        t0 = time.time()

        result = agent.solve(env, task_index=task_idx, max_num_steps=args.max_steps)

        elapsed = time.time() - t0
        reward = result["reward"]
        n_messages = len(result.get("messages", []))

        results.append({
            "task_index": task_idx,
            "reward": reward,
            "n_messages": n_messages,
            "elapsed": elapsed,
        })

        # Incremental save — append this result to the checkpoint file
        # (same pattern as tau_bench.run.run)
        existing = []
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                existing = json.load(f)
        existing.append({
            "task_id": task_idx,
            "reward": reward,
            "info": result.get("info", {}),
            "traj": result.get("messages", []),
            "trial": 0,
        })
        with open(ckpt_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"  Reward:           {reward}")
        print(f"  Messages:         {n_messages}")
        print(f"  Time:             {elapsed:.1f}s")
        print(f"  (saved to {ckpt_path})")

    # ── 6. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not results:
        print("No tasks were executed.")
        return

    rewards = [r["reward"] for r in results]
    avg_reward = sum(rewards) / len(rewards)
    successes = sum(1 for r in rewards if r > 0)

    print(f"  Tasks run:        {len(results)}")
    print(f"  Avg reward:       {avg_reward:.3f}")
    print(f"  Success rate:     {successes}/{len(results)} "
          f"({100 * successes / len(results):.0f}%)")
    print(f"  Privileged info:  {'enabled' if priv_available else 'disabled'}")
    print(f"  Agent strategy:   {args.agent_strategy}")
    print(f"  Model:            {args.model}")
    print(f"  Results saved to: {ckpt_path}")

    for r in results:
        status = "PASS" if r["reward"] > 0 else "FAIL"
        print(f"  Task {r['task_index']:3d}: reward={r['reward']:.2f}  "
              f"msgs={r['n_messages']:3d}  time={r['elapsed']:.1f}s  [{status}]")


if __name__ == "__main__":
    main()
