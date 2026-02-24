# Copyright Sierra
from tqdm import tqdm
import os
import json
import random
import re
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import importlib.util

from finetuning.configs.llm_configs import CHAT_MODEL_ARGS_DICT
from finetuning.core.process_checkpoints import process_checkpoints, has_valid_checkpoints
from finetuning.utils.model_mapping import alias_for
from tau_bench.envs.retail.tasks_test import TASKS_TEST as retail_test_tasks
from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as retail_train_tasks
from tau_bench.envs.airline.tasks_test import TASKS as airline_test_tasks

# TauBench imports
from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from tau_bench.envs.user import UserStrategy
from litellm import provider_list

# GEM imports
import gem
from gem.envs.registration import ENV_REGISTRY

# ---------------------------------------------------------------------------
# Import GEM tau-bench wrappers
# ---------------------------------------------------------------------------
_GEM_TAUBENCH_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "gem" / "examples" / "multiagent" / "tau_bench_retail"


def _import_from_file(module_name: str, file_path: Path):
    """Import a module directly from a file path."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_env_mod = _import_from_file("tau_bench_env", _GEM_TAUBENCH_DIR / "tau_bench_env.py")
_agent_mod = _import_from_file("tau_bench_agent", _GEM_TAUBENCH_DIR / "tau_bench_agent.py")
TauBenchEnv = _env_mod.TauBenchEnv
TauBenchAgent = _agent_mod.TauBenchAgent


# --------------------------- Metrics ---------------------------


def compute_metrics(results: List[EnvRunResult]) -> Dict[str, Any]:
    """Compute metrics from results and return as a dictionary."""

    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    if not results:
        return {}

    num_trials = len(set(r.trial for r in results))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)

    c_per_task_id: Dict[int, int] = {}
    for r in results:
        c_per_task_id.setdefault(r.task_id, 0)
        c_per_task_id[r.task_id] += 1 if is_successful(r.reward) else 0

    pass_hat_ks: Dict[int, float] = {}
    for k in range(1, num_trials + 1):
        total = 0.0
        for c in c_per_task_id.values():
            total += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = total / max(1, len(c_per_task_id))

    # Build metrics dictionary
    metrics = {"avg_reward": float(avg_reward)}
    for k, v in pass_hat_ks.items():
        metrics[f"pass^k_{k}"] = float(v)

    return metrics


def display_metrics(results: List[EnvRunResult]) -> None:
    """Display metrics to console."""
    metrics = compute_metrics(results)

    if not metrics:
        print("No results to summarize.")
        return

    print(f"🏆 Average reward: {metrics['avg_reward']:.4f}")
    print("📈 Pass^k")
    for key, value in metrics.items():
        if key.startswith("pass^k_"):
            k = key.split("_")[-1]
            print(f"  k={k}: {value:.4f}")


def prepare_tasks(split: str, evaluation_args=None, env: str = "retail"):
    """
    retail:
      - train: retail_train_tasks minus a small held-out head
      - test : retail_test_tasks + held-out head from retail_train_tasks
      - dev  : small head slice of retail_train_tasks

    airline (no train tasks):
      - train: []  (skip airline for training; "train only do retail")
      - test : all airline_test_tasks
      - dev  : first dev_k of airline_test_tasks
    """
    heldout_k = getattr(evaluation_args, "heldout_k", 5)
    dev_k = getattr(evaluation_args, "dev_k", 20)

    if env == "retail":
        all_seeds = list(range(1000))
        heldout_seeds = all_seeds[:heldout_k]
        train_seeds = [s for s in all_seeds if s not in heldout_seeds]

        if split == "test":
            tasks = list(retail_test_tasks)
            return tasks
        elif split == "train":
            idxs = [i for i in train_seeds if i < len(retail_train_tasks)]
            return [retail_train_tasks[i] for i in idxs]
        elif split == "dev":
            k = min(dev_k, len(retail_train_tasks))
            return list(retail_train_tasks[:k])
        else:
            raise ValueError(f"Unknown split: {split!r}")

    if env == "airline":
        if split == "train":
            # Train only on retail; airline has no train set ⇒ skip.
            return []
        elif split == "test":
            return list(airline_test_tasks)
        elif split == "dev":
            k = min(dev_k, len(airline_test_tasks))
            return list(airline_test_tasks[:k])
        else:
            raise ValueError(f"Unknown split: {split!r}")

    raise ValueError(f"Unknown env: {env!r}")


# --------------------------- GEM-based single-env runner ---------------------------


def _gem_register_safe(env_id: str, **kwargs):
    """Register a GEM environment, replacing any existing registration with the same ID."""
    if env_id in ENV_REGISTRY:
        del ENV_REGISTRY[env_id]
    gem.register(env_id, **kwargs)


def run_env(config: RunConfig, evaluation_args) -> List[EnvRunResult]:
    """Run tasks for a single environment using GEM wrappers."""
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert (
        config.model_provider in provider_list
    ), f"Invalid model provider: {config.model_provider}"
    assert (
        config.user_model_provider in provider_list
    ), f"Invalid user model provider: {config.user_model_provider}"
    assert config.agent_strategy in [
        "tool-calling",
        "act",
        "react",
        "few-shot",
        "chat-react-priv",
    ], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    print(f"[{config.env}/{config.task_split}] user strategy: {config.user_strategy}")

    # Prepare tasks
    tasks = prepare_tasks(config.task_split, evaluation_args, env=config.env)

    # Create a reference GEM environment (for metadata: tasks, tools_info, wiki)
    env_id = f"tau-bench:{config.env}-{config.task_split}-{id(config)}"
    _gem_register_safe(
        env_id,
        entry_point=TauBenchEnv,
        env_name=config.env,
        task_split=config.task_split,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        user_strategy=config.user_strategy,
        tasks=tasks,
    )
    ref_env = gem.make(env_id)

    # Create GEM agent (lazily builds inner agent from ref_env metadata)
    agent = TauBenchAgent(
        model=config.model,
        provider=config.model_provider,
        agent_strategy=config.agent_strategy,
        temperature=config.temperature,
        env_name=config.env,
        privileged_trajectories_path=(
            str(config.privileged_trajectories_path) if config.privileged_trajectories_path else None
        ),
        use_human_privileged=config.use_human_privileged,
        use_only_hints=config.use_only_hints,
        use_self_generated_hints=config.use_self_generated_hints,
        self_generated_hints_path=(
            str(config.self_generated_hints_path) if config.self_generated_hints_path else None
        ),
        few_shot_displays_path=(
            str(config.few_shot_displays_path) if config.few_shot_displays_path else None
        ),
    )

    # Determine task indices (use ref_env for metadata)
    end_index = len(ref_env.tasks) if config.end_index == -1 else min(config.end_index, len(ref_env.tasks))
    results: List[EnvRunResult] = []
    debug = [1, 2, 3, 4, 5] == config.task_ids
    if config.task_ids is not None and len(config.task_ids) == 1 and config.task_split == "train":
        # random tasks
        num_random_tasks = config.task_ids[0]
        idxs = random.sample(
            range(config.start_index, end_index),
            min(num_random_tasks, end_index - config.start_index),
        )
        random.shuffle(idxs)
    elif config.task_ids and (config.task_split == "train" or debug):
        idxs = list(config.task_ids)  # may contain repeats if caller provided them
        print(f"Running explicit task_ids (n={len(idxs)})")
    else:
        idxs = list(range(config.start_index, end_index))
        print(f"Running tasks {config.start_index}..{end_index-1}")

    for trial in range(config.num_trials):
        trial_indices = list(idxs)
        if config.shuffle:
            random.shuffle(trial_indices)

        def _run(idx: int) -> EnvRunResult:
            # Create an isolated env per task for thread safety
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
                tasks=tasks,
            )
            print(f"→ task {idx} (trial {trial})")
            try:
                result = agent.solve(isolated_env, task_index=idx)
                out = EnvRunResult(
                    task_id=idx,
                    reward=result["reward"],
                    info=result.get("info", {}),
                    traj=result.get("messages", []),
                    trial=trial,
                )
            except Exception as e:
                out = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=trial,
                )

            # tag which env this result came from
            try:
                out.info["env"] = config.env
            except Exception:
                out.info = {"env": config.env}

            print("✅" if abs(out.reward - 1.0) <= 1e-6 else "❌", f"task_id={idx}", out.info)
            print("-----")
            return out

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as ex:
            for res in tqdm(
                ex.map(_run, trial_indices), total=len(trial_indices), desc=f"trial {trial}"
            ):
                results.append(res)

    results = _fix_rewared_hack(results)
    display_metrics(results)
    return results


# --------------------------- Public entry ---------------------------


def _fix_rewared_hack(all_results) -> List[EnvRunResult]:
    filter_output = {"role": "user", "content": "API output: Transfer successful"}
    for result in all_results:
        # If the last trajectory message exactly matches the filter, zero the reward.
        if result.traj and result.traj[-1] == filter_output:
            result.reward = 0.0
    return all_results


def taubench_run(evaluation_args,process_checkpoint = True) -> List[EnvRunResult]:
    """
    Boots vLLM servers for both agent and user models,
    sets OPENAI_* envs to your local servers, and runs TauBench across envs.
    Writes a single combined JSON with all envs' results and prints combined accuracy.
    """

    # 1) Boot/reuse local vLLM for agent model using your config dict
    if process_checkpoint:
        process_checkpoints(evaluation_args.base_model_dir, evaluation_args.ckpt_dirs)
    if re.search(
        "openrouter",
        evaluation_args.model_name,
    ):
        agent_vllm_args = CHAT_MODEL_ARGS_DICT["openrouter"]
        agent_vllm_args.model_name = evaluation_args.model_name
        agent_vllm_args.vllm_cpus = evaluation_args.vllm_cpus
        agent_vllm_args.n_gpus = evaluation_args.n_gpus
    else:
        agent_vllm_args = CHAT_MODEL_ARGS_DICT[evaluation_args.model_name]
        agent_vllm_args.model_name = alias_for(agent_vllm_args.model_name)
        agent_vllm_args.model_path = evaluation_args.model_path
        agent_vllm_args.vllm_cpus = evaluation_args.vllm_cpus
        # Force agent to use only GPU 0 (single GPU setup for dual-server config)
        agent_vllm_args.gpu_ids = [0]
        agent_vllm_args.n_gpus = 1
        agent_vllm_args.port = 8000  # Explicit agent port
        agent_vllm_args.prepare_server()

    # 2) Boot user model vLLM server (if enabled)
    user_vllm_args = None
    if getattr(evaluation_args, "use_local_user_model", False):
        from finetuning.toolkit_utils.chat_api import VLLMModelArgs

        # Determine GPU allocation for user model
        user_gpu_ids = None
        if hasattr(evaluation_args, "user_gpu_ids") and evaluation_args.user_gpu_ids:
            user_gpu_ids = evaluation_args.user_gpu_ids
        elif hasattr(evaluation_args, "user_n_gpus") and evaluation_args.user_n_gpus:
            # Use next available GPUs after agent GPUs
            agent_gpu_count = evaluation_args.n_gpus
            user_gpu_ids = list(
                range(agent_gpu_count, agent_gpu_count + evaluation_args.user_n_gpus)
            )

        user_vllm_args = VLLMModelArgs(
            model_name=getattr(evaluation_args, "user_model_name", "Qwen/Qwen3-14B"),
            model_path=getattr(evaluation_args, "user_model_path", None),
            port=getattr(evaluation_args, "user_vllm_port", 8001),
            gpu_ids=user_gpu_ids,
            n_gpus=len(user_gpu_ids) if user_gpu_ids else 1,
            vllm_cpus=getattr(evaluation_args, "user_vllm_cpus", None),
            max_total_tokens=getattr(evaluation_args, "user_max_total_tokens", 4096),
            max_batch_total_tokens=getattr(evaluation_args, "user_max_batch_total_tokens", 4096),
            temperature=getattr(evaluation_args, "user_temperature", 0.7),
            tensor_parallel_size=len(user_gpu_ids) if user_gpu_ids else 1,
            enable_thinking=False,
        )
        user_vllm_args.prepare_server(verbose=True)

    try:

        # 3) Set environment variables
        OR_key = getattr(evaluation_args, "OPENROUTER_API_KEY", None)
        openai_key = getattr(evaluation_args, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["OPENROUTER_API_KEY"] = OR_key
        os.environ["HOSTED_VLLM_API_BASE"] = "http://127.0.0.1:8000/v1"

        # User model vLLM base URL
        if user_vllm_args:
            os.environ["USER_VLLM_API_BASE"] = f"http://127.0.0.1:{user_vllm_args.port}/v1"

        # 4) Defaults
        cfg = getattr(evaluation_args, "task_configs", {}) or {}

        model_provider = cfg["model_provider"]
        model_name = alias_for(agent_vllm_args.model_name)

        # Set user model provider to hosted_vllm if using local user model
        if user_vllm_args:
            user_model_provider = "hosted_vllm"
            user_model_name = user_vllm_args.model_name
        else:
            user_model_provider = cfg.get("user_model_provider", model_provider)
            user_model_name = cfg.get("user_model_name", model_name)

        start_index = int(cfg.get("start_index", 0))
        end_index = int(cfg.get("end_index", -1))
        explicit_task_ids = (
            cfg.get("task_ids", []) if not evaluation_args.debug else [1, 2, 3, 4, 5]
        )
        num_trials = int(cfg.get("num_trials", 1))
        max_concurrency = int(cfg.get("max_concurrency", 4))
        shuffle = bool(cfg.get("shuffle", False))
        seed = int(cfg.get("seed", 0))
        temperature = float(cfg.get("temperature", 0.0))
        log_dir = str(getattr(evaluation_args, "exp_root", "./runs")) + "/runs"

        agent_strategy = cfg.get("agent_strategy", "react")
        user_strategy = cfg.get("user_strategy", "llm")
        task_split = getattr(evaluation_args, "benchmark_split", "dev")
        few_shot_displays_path = cfg.get("few_shot_displays_path", None)
        privileged_actions_path = evaluation_args.extra_agent_args.get(
            "privaleged_actions_path", None
        )
        use_human_privileged = evaluation_args.extra_agent_args.get("use_human_privileged", False)
        use_only_hints = evaluation_args.extra_agent_args.get("use_only_hints", False)
        use_self_generated_hints = evaluation_args.extra_agent_args.get(
            "use_self_generated_hints", False
        )
        self_generated_hints_path = evaluation_args.extra_agent_args.get(
            "self_generated_hints_path"
        )

        if task_split == "train":
            num_trials = evaluation_args.n_repeats_train
        elif task_split == "test":
            num_trials = evaluation_args.n_repeats_eval

        all_results: List[EnvRunResult] = []

        # Accept either .env (your change) or .envs (older style). Support str or list.
        envs_attr = getattr(evaluation_args, "env", None)
        if isinstance(envs_attr, str):
            envs = [envs_attr]
        elif envs_attr is None:
            envs = []
        else:
            envs = list(envs_attr)

        # run each requested env and gather results in memory
        for env_name in envs:
            if model_provider not in provider_list:
                raise ValueError(f"Invalid model provider: {model_provider}")

            cfg_run = RunConfig(
                env=env_name,
                task_split=task_split,
                agent_strategy=agent_strategy,
                temperature=temperature,
                model=model_name,
                model_provider=model_provider,
                user_model=user_model_name,
                user_model_provider=user_model_provider,
                user_strategy=user_strategy,
                start_index=start_index,
                end_index=end_index,
                task_ids=explicit_task_ids,
                num_trials=num_trials,
                max_concurrency=max_concurrency,
                shuffle=shuffle,
                seed=seed,
                log_dir=log_dir,
                few_shot_displays_path=few_shot_displays_path,
                privileged_trajectories_path=(
                    str(privileged_actions_path) if privileged_actions_path else None
                ),
                use_human_privileged=use_human_privileged,
                use_only_hints=use_only_hints,
                use_self_generated_hints=use_self_generated_hints,
                self_generated_hints_path=(
                    str(self_generated_hints_path) if self_generated_hints_path else None
                ),
            )

            print(
                f"\n=== TauBench: env='{env_name}' split='{task_split}' "
                f"model='{model_name}' provider='{model_provider}' base='http://127.0.0.1:8000/v1' ==="
            )
            # Skip airline/train as per your rule
            if cfg_run.env == "airline" and task_split == "train":
                continue

            all_results.extend(run_env(cfg_run, evaluation_args))

        # ---------- ONE combined JSON + combined metrics ----------
        if not all_results:
            print("No results to save.")
            if user_vllm_args:
                user_vllm_args.close_server()
            agent_vllm_args.close_server()

            return all_results

        # show combined metrics across both envs (Pass^1 ≙ accuracy)
        print("\n=== Combined metrics across all requested envs ===")
        display_metrics(all_results)

        # Compute metrics dictionary for logging - split by environment
        metrics = {}

        # Split results by environment
        results_by_env = {}
        for r in all_results:
            env_name = (r.info or {}).get("env", "unknown")
            if env_name not in results_by_env:
                results_by_env[env_name] = []
            results_by_env[env_name].append(r)

        # Compute metrics for each environment separately
        for env_name, env_results in results_by_env.items():
            env_metrics = compute_metrics(env_results)
            metrics[env_name] = env_metrics
            print(f"\n=== Metrics for {env_name} ===")
            print(f"Average reward: {env_metrics.get('avg_reward', 0.0):.4f}")
            for k, v in env_metrics.items():
                if k.startswith("pass^k_"):
                    print(f"  {k}: {v:.4f}")

        # Save one combined JSON (both envs). Each record includes its env tag via result.info["env"].
        time_str = datetime.now().strftime("%m%d%H%M%S")
        env_tag = "-".join(sorted(envs)) if envs else "none"
        model_stub = model_name.split("/")[-1]
        os.makedirs(log_dir, exist_ok=True)
        combined_path = (
            f"{log_dir}/combined_{env_tag}_{task_split}_"
            f"{agent_strategy}-{model_stub}-{temperature}_"
            f"range_{start_index}-{end_index}_"
            f"user-{user_strategy}_{time_str}.json"
        )

        combined_records = []
        for r in all_results:
            rec = r.model_dump()
            rec["env"] = (r.info or {}).get("env", "unknown")  # keep env tag in the JSON
            combined_records.append(rec)

        with open(combined_path, "w") as f:
            json.dump(combined_records, f, indent=2)

        print("\n📄 Combined results saved to:", combined_path)

        # Save metrics to a summary.json file that can be picked up by the processing pipeline
        # Now includes separate metrics for each environment
        metrics_path = combined_path.replace(".json", "_summary.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"📊 Metrics saved to: {metrics_path}")
    finally:
        # 6) Cleanup both servers
        if user_vllm_args:
            user_vllm_args.close_server()
        agent_vllm_args.close_server()
    return all_results
