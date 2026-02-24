#!/usr/bin/env python3
"""Print agent prompts from a TAU-bench results JSON file.

Usage:
    python print_prompts.py results/retail_20260213_165834/chat-react-priv-Qwen3-8B-0.0_user-Qwen_Qwen3-14B-llm.json
    python print_prompts.py results/retail_*/chat-react-priv*.json --task-id 0
    python print_prompts.py results.json --task-id 0 --message-index 0  # system prompt only
"""

import argparse
import glob
import json
import sys


def print_prompt(result, msg_idx=None):
    task_id = result.get("task_id", "?")
    reward = result.get("reward", "?")
    traj = result.get("traj", [])
    priv = result.get("info", {}).get("privileged", {})
    priv_available = priv.get("available", "?")

    print(f"{'='*80}")
    print(f"TASK {task_id} | reward={reward} | privileged available={priv_available} | traj len={len(traj)}")
    print(f"{'='*80}")

    if msg_idx is not None:
        indices = [msg_idx] if msg_idx < len(traj) else []
    else:
        indices = range(len(traj))

    for i in indices:
        msg = traj[i]
        role = msg.get("role", "?")
        content = msg.get("content", "")
        print(f"\n--- [{i}] role: {role} ---")
        print(content)

    print()


def main():
    parser = argparse.ArgumentParser(description="Print agent prompts from TAU-bench results.")
    parser.add_argument("path", help="Path to results JSON (supports glob patterns)")
    parser.add_argument("--task-id", type=int, default=None, help="Only print this task ID")
    parser.add_argument("--message-index", "-m", type=int, default=None,
                        help="Only print this message index (0=system prompt)")
    args = parser.parse_args()

    files = sorted(glob.glob(args.path))
    if not files:
        print(f"No files matched: {args.path}", file=sys.stderr)
        sys.exit(1)

    for fpath in files:
        if len(files) > 1:
            print(f"\n{'#'*80}")
            print(f"# FILE: {fpath}")
            print(f"{'#'*80}")

        with open(fpath) as f:
            data = json.load(f)

        for result in data:
            if args.task_id is not None and result.get("task_id") != args.task_id:
                continue
            print_prompt(result, args.message_index)


if __name__ == "__main__":
    main()
