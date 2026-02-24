"""
Entry script for using VeRL to train LLM agents on tau-bench via GEM.

Adapted from gem/examples/train_verl/train_verl.py with these key changes:
  1. Manual env construction: TauBenchSingleAgentEnv + TauBenchObservationWrapper
     instead of gem.make_vec().
  2. Action extraction: <action>...</action> regex instead of \\boxed{}.
  3. Prompt handling: prompt_template=na, apply_chat_template=False — the
     observation wrapper already produces a fully formatted prompt.
  4. SyncVectorEnv (tau-bench user sim is blocking).

Modifications are labeled with "[TAU-BENCH]" (Ctrl+F to navigate).
"""

import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pprint import pprint
from typing import List, Optional, Sequence, Tuple

import hydra
import numpy as np
import ray
import torch
import torch.utils
import torch.utils.data
import tree
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from tqdm import tqdm
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    Dataset,
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.device import is_cuda_available
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.fsdp_workers import ActorRolloutRefWorker

# ---------------------------------------------------------------------------
# [TAU-BENCH] Ensure tau-bench is importable
# ---------------------------------------------------------------------------
TAU_BENCH_PATH = os.environ.get(
    "TAU_BENCH_PATH",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "tau-bench")
    ),
)
if TAU_BENCH_PATH not in sys.path:
    sys.path.insert(0, TAU_BENCH_PATH)

# [TAU-BENCH] Import tau-bench specific env and wrapper
from tau_bench_single_agent_env import TauBenchSingleAgentEnv
from tau_bench_observation_wrapper import TauBenchObservationWrapper

from gem.vector.sync_vector_env import SyncVectorEnv

WorkerType = type[Worker]

logger = logging.getLogger(__file__)


@hydra.main(config_path="./", config_name="config", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env=PPO_RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    if (
        is_cuda_available
        and OmegaConf.select(config.trainer, "profile_steps") is not None
        and len(OmegaConf.select(config.trainer, "profile_steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.trainer.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


# Invalid action to be sent to the env to trigger format error penalty.
INVALID_ACTION = "<｜INVALID_ACTION｜>"


# [TAU-BENCH] prompt_template=na means pass-through (observation wrapper already formatted)
def apply_no_template(observation: str) -> str:
    return observation


TEMPLATE_FACTORY = {
    "na": apply_no_template,
}


# ---------------------------------------------------------------------------
# [TAU-BENCH] Dynamic vLLM user-model server lifecycle
# ---------------------------------------------------------------------------
class VLLMServerManager:
    """Manages the lifecycle of a vLLM OpenAI API server as a subprocess."""

    def __init__(self, model, port, gpu, gpu_memory_utilization, max_model_len, dtype="auto"):
        self.model = model
        self.port = port
        self.gpu = gpu
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._process = None

    def start(self, timeout=300):
        """Start the vLLM server and wait until it's healthy."""
        if self._process is not None and self._process.poll() is None:
            print(f"[VLLMServerManager] Server already running (PID {self._process.pid})")
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
        ]

        self._log_file = open("/tmp/user_vllm_server.log", "w")
        print(f"[VLLMServerManager] Starting vLLM server: GPU={self.gpu}, port={self.port}, model={self.model}")
        print(f"[VLLMServerManager] Server logs: /tmp/user_vllm_server.log")
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=self._log_file,
            preexec_fn=os.setsid,
        )

        # Poll health endpoint until ready
        import requests
        health_url = f"http://localhost:{self.port}/health"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    print(f"[VLLMServerManager] Server ready (PID {self._process.pid})")
                    return
            except requests.ConnectionError:
                pass
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"[VLLMServerManager] Server process exited with code {self._process.returncode}"
                )
            time.sleep(5)

        self.stop()
        raise TimeoutError(
            f"[VLLMServerManager] Server did not become healthy within {timeout}s"
        )

    def stop(self):
        """Stop the vLLM server gracefully, falling back to SIGKILL."""
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._process = None
            return

        pid = self._process.pid
        print(f"[VLLMServerManager] Stopping server (PID {pid})...")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            self._process.wait(timeout=30)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                self._process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        self._process = None
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()
            self._log_file = None
        print(f"[VLLMServerManager] Server stopped.")

    def is_running(self):
        return self._process is not None and self._process.poll() is None


@dataclass
class Transition:
    obs: str
    action: str
    reward: float
    done: bool

    prompt: str
    prompt_ids: list
    response: str
    response_ids: list

    attention_mask: list
    position_ids: list

    response_is_truncated: bool
    action_is_formatted: bool

    prompt_ids_with_priv: list = None     # with-PI prompt IDs (set in OPSD mode)
    prompt_ids_without_priv: list = None  # without-PI prompt IDs (set in pi_distill mode)

    def format(self):
        return {
            "obs": self.obs,
            "action": self.action,
            "reward": self.reward,
            "done": int(self.done),
            "prompt": self.prompt,
            "response": self.response,
        }


class GEMActorRolloutRefWorker(ActorRolloutRefWorker):
    pass


class DummyPromptDataset(Dataset):
    """Empty dataset to satisfy VeRL's requirements without actually loading data."""

    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        del idx
        return ""


def extract_action_tags(text: str) -> Optional[str]:
    """[TAU-BENCH] Extract content from <action>...</action> tags."""
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class ReinforceGEMTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name="cuda",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                self.config.algorithm.kl_ctrl
            )

        # [TAU-BENCH] We only support multi-turn REINFORCE, not Actor-Critic.
        self.use_critic = False

        self.train_dataloader = torch.utils.data.DataLoader(
            DummyPromptDataset(int(1e9)),
        )
        self.total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps

        # [TAU-BENCH] Build vectorized tau-bench environment manually
        env_cfg = self.config.actor_rollout_ref.env
        seed = int(time.time_ns())

        # [TAU-BENCH] Set up dynamic user model server
        os.environ["USER_VLLM_API_BASE"] = f"http://localhost:{env_cfg.get('user_vllm_port', 8100)}/v1"
        if env_cfg.get("user_temperature", None) is not None:
            os.environ["USER_TEMPERATURE"] = str(env_cfg.user_temperature)
        self.user_server = VLLMServerManager(
            model=env_cfg.user_model,
            port=env_cfg.get("user_vllm_port", 8100),
            gpu=env_cfg.get("user_vllm_gpu", 0),
            gpu_memory_utilization=env_cfg.get("user_vllm_gpu_mem_util", 0.45),
            max_model_len=env_cfg.get("user_vllm_max_model_len", 8192),
            dtype="auto",
        )
        # User sim vLLM runs on a dedicated GPU — start once, keep running.
        # LLMUserSimulationEnv calls litellm completion() during __init__.
        self.user_server.start()

        # Debug mode: 2 train tasks (× n_repeat), 2 eval tasks
        self.debug = env_cfg.get("debug", False)
        _tasks = None
        if self.debug:
            _tmp = TauBenchSingleAgentEnv(
                env_name=env_cfg.env_name,
                task_split=env_cfg.task_split,
                user_model=env_cfg.user_model,
                user_provider=env_cfg.user_provider,
                user_strategy=env_cfg.get("user_strategy", "llm"),
            )
            _tasks = _tmp.tasks[:2]
            del _tmp
            print(f"[DEBUG] Train: {len(_tasks)} tasks × n_repeat={env_cfg.get('n_repeat', 1)}")

        envs = []
        for j in range(env_cfg.num_env):
            base_env = TauBenchSingleAgentEnv(
                env_name=env_cfg.env_name,
                task_split=env_cfg.task_split,
                user_model=env_cfg.user_model,
                user_provider=env_cfg.user_provider,
                user_strategy=env_cfg.get("user_strategy", "llm"),
                tasks=_tasks,
            )
            wrapped_env = TauBenchObservationWrapper(
                env=base_env,
                tokenizer=self.tokenizer,
                model_name=env_cfg.get("model_name", "qwen"),
                privileged_path=env_cfg.get("privileged_trajectories_path", None),
                use_human_privileged=env_cfg.get("use_human_privileged", False),
                env_name=env_cfg.env_name,
                use_only_hints=env_cfg.get("use_only_hints", False),
                use_reasoning=env_cfg.get("use_reasoning", True),
                max_history_length=env_cfg.get("max_history_length", None),
                use_self_generated_hints=env_cfg.get("use_self_generated_hints", False),
                self_generated_hints_path=env_cfg.get("self_generated_hints_path", None),
            )
            envs.append(wrapped_env)

        self.env = SyncVectorEnv(
            env_ids=[env_cfg.env_name] * env_cfg.num_env,
            env_fns=[lambda e=e: e for e in envs],
        )

        # [TAU-BENCH] Build eval environments (test split, no privileged info)
        # Evaluate on each env in eval_env_names (default: just the training env)
        eval_env_names = env_cfg.get("eval_env_names", [env_cfg.env_name])
        if isinstance(eval_env_names, str):
            eval_env_names = [eval_env_names]
        self.eval_envs = {}
        for eval_env_name in eval_env_names:
            # Get the full task list for this eval env
            _tmp_env = TauBenchSingleAgentEnv(
                env_name=eval_env_name,
                task_split="test",
                user_model=env_cfg.user_model,
                user_provider=env_cfg.user_provider,
                user_strategy=env_cfg.get("user_strategy", "llm"),
            )
            all_eval_tasks = list(_tmp_env.tasks)
            del _tmp_env
            if self.debug:
                all_eval_tasks = all_eval_tasks[:2]
            n_eval_tasks = len(all_eval_tasks)
            print(f"[Eval] {eval_env_name}: {n_eval_tasks} test tasks")

            # One env per task — each env gets exactly one task for deterministic coverage
            eval_envs = []
            for j in range(n_eval_tasks):
                base_env = TauBenchSingleAgentEnv(
                    env_name=eval_env_name,
                    task_split="test",
                    user_model=env_cfg.user_model,
                    user_provider=env_cfg.user_provider,
                    user_strategy=env_cfg.get("user_strategy", "llm"),
                    tasks=[all_eval_tasks[j]],
                )
                wrapped_env = TauBenchObservationWrapper(
                    env=base_env,
                    tokenizer=self.tokenizer,
                    model_name=env_cfg.get("model_name", "qwen"),
                    privileged_path=None,
                    use_human_privileged=False,
                    env_name=eval_env_name,
                    use_only_hints=False,
                    use_reasoning=env_cfg.get("use_reasoning", True),
                    max_history_length=env_cfg.get("max_history_length", None),
                )
                eval_envs.append(wrapped_env)

            self.eval_envs[eval_env_name] = SyncVectorEnv(
                env_ids=[eval_env_name] * n_eval_tasks,
                env_fns=[lambda e=e: e for e in eval_envs],
            )

    def _init_trace_dir(self):
        """Set up a structured trace directory for the run.

        Layout:
            {trace_dir}/{YYYYMMDD_HHMMSS}/
                config.yaml
                training_summary.jsonl   (appended each step)
                step_0001.json
                step_0002.json
                ...
        """
        from omegaconf import OmegaConf

        base_trace_dir = self.config.actor_rollout_ref.env.get("trace_dir", None)
        if not base_trace_dir:
            self._trace_run_dir = None
            return

        # Use experiment_name as run dir name if set, otherwise generate timestamp
        run_name = self.config.trainer.get("experiment_name", None)
        if not run_name or run_name == "tau-bench":
            run_name = time.strftime("%Y%m%d_%H%M%S")
        run_name = str(run_name)
        self._trace_run_dir = os.path.join(base_trace_dir, run_name)
        os.makedirs(self._trace_run_dir, exist_ok=True)

        # Point checkpoint saving/loading at this run directory
        with open_dict(self.config):
            self.config.trainer.default_local_dir = os.path.join(
                self._trace_run_dir, "checkpoints"
            )

        # Save config snapshot (after updating default_local_dir)
        config_path = os.path.join(self._trace_run_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.config, resolve=True))

        print(f"[Traces] Run directory: {self._trace_run_dir}")
        print(f"[Traces] Checkpoints: {self.config.trainer.default_local_dir}")

    def _save_step_traces(self, all_finished_episodes):
        """Save per-step traces with structured sub-directories."""
        if self._trace_run_dir is None:
            return

        step_dir = os.path.join(self._trace_run_dir, f"step_{self.global_steps:04d}")
        os.makedirs(step_dir, exist_ok=True)

        # Build episode records
        episodes_data = []
        for ep_idx, ep in enumerate(all_finished_episodes):
            ep_reward = sum(t.reward for t in ep)
            episodes_data.append({
                "episode_idx": ep_idx,
                "num_turns": len(ep),
                "reward": ep_reward,
                "success": bool(ep[-1].reward == 1) if ep else False,
                "turns": [t.format() for t in ep],
            })

        # Save episodes
        with open(os.path.join(step_dir, "episodes.json"), "w") as f:
            json.dump(episodes_data, f, indent=2)

        # Compute and save per-step summary
        rewards = [e["reward"] for e in episodes_data]
        successes = [e["success"] for e in episodes_data]
        turn_counts = [e["num_turns"] for e in episodes_data]
        step_summary = {
            "step": self.global_steps,
            "num_episodes": len(episodes_data),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "mean_turns": float(np.mean(turn_counts)) if turn_counts else 0.0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(step_dir, "summary.json"), "w") as f:
            json.dump(step_summary, f, indent=2)

        # Append to running training summary (one JSON object per line)
        summary_path = os.path.join(self._trace_run_dir, "training_summary.jsonl")
        with open(summary_path, "a") as f:
            f.write(json.dumps(step_summary) + "\n")

        print(
            f"[Traces] Step {self.global_steps}: "
            f"{len(episodes_data)} episodes, "
            f"success={step_summary['success_rate']:.2%}, "
            f"mean_reward={step_summary['mean_reward']:.3f} "
            f"-> {step_dir}"
        )

    def _evaluate(self):
        """Run eval episodes on each eval env's test split (no privileged info) and return metrics."""
        env_cfg = self.config.actor_rollout_ref.env
        eval_batch_size = env_cfg.get("eval_batch_size", 8)
        if self.debug:
            eval_batch_size = 2

        all_metrics = {}
        total_episodes = 0
        total_successes = 0

        for env_name, eval_env in self.eval_envs.items():
            # Run until every env (one per task) completes at least one episode
            n_eval_tasks = eval_env.num_envs
            finished_episodes, collection_info = self.collect_experience(
                eval_env, min_steps=0, min_episodes=n_eval_tasks
            )

            mean_episode_len = np.mean([len(ep) for ep in finished_episodes])
            mean_episode_return = np.mean(
                [sum(t.reward for t in ep) for ep in finished_episodes]
            )
            mean_episode_success = np.mean(
                [ep[-1].reward == 1 for ep in finished_episodes]
            )
            n_episodes = len(finished_episodes)
            n_successes = sum(ep[-1].reward == 1 for ep in finished_episodes)

            all_metrics[f"eval/{env_name}/mean_episode_len"] = float(mean_episode_len)
            all_metrics[f"eval/{env_name}/mean_episode_return"] = float(mean_episode_return)
            all_metrics[f"eval/{env_name}/success"] = float(mean_episode_success)
            all_metrics[f"eval/{env_name}/num_episodes"] = n_episodes
            for k, v in collection_info.items():
                all_metrics[f"eval/{env_name}/{k}"] = v

            print(
                f"[Eval] Step {self.global_steps} | {env_name}: "
                f"{n_episodes} episodes, "
                f"success={mean_episode_success:.2%} ({n_successes}/{n_episodes}), "
                f"mean_reward={mean_episode_return:.3f}"
            )

            total_episodes += n_episodes
            total_successes += n_successes

        # Overall aggregate
        if total_episodes > 0:
            all_metrics["eval/overall_success"] = total_successes / total_episodes
            all_metrics["eval/total_episodes"] = total_episodes

        return all_metrics

    def fit(self):
        """The training loop of REINFORCE."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        self._init_trace_dir()

        # Use the trace run folder name (timestamp) as wandb experiment name
        experiment_name = self.config.trainer.experiment_name
        if self._trace_run_dir is not None:
            experiment_name = os.path.basename(self._trace_run_dir)

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Run tau-bench eval before training starts
        eval_freq = self.config.trainer.get("eval_freq", -1)
        if eval_freq > 0:
            eval_metrics = self._evaluate()
            pprint(f"Initial eval metrics: {eval_metrics}")
            logger.log(data=eval_metrics, step=self.global_steps)

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for _ in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(
                            role="e2e", profile_step=self.global_steps
                        )
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.run_agent_env_loop()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    gen_batch_output.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(gen_batch_output.batch))],
                        dtype=object,
                    )
                    batch = gen_batch_output

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    print("Trainer batch size:", len(gen_batch_output.batch))

                    # --- KL-for-reward-shaping ---
                    # Compute KL between with-PI and without-PI prompts per turn,
                    # aggregate to per-episode KL, then: new_reward = G - distill_coef * KL_episode
                    # pi_distill: forward KL = log pi(a|with_PI) - log pi(a|without_PI)
                    # OPSD:       reverse KL = log pi(a|without_PI) - log pi(a|with_PI)
                    distill_mode = self.config.actor_rollout_ref.actor.get("distill_mode", "none")
                    distill_coef = self.config.actor_rollout_ref.actor.get("distill_coef", 0.0)
                    response_mask = batch.batch["response_mask"]

                    do_kl_pi_distill = (
                        distill_coef > 0
                        and distill_mode == "pi_distill"
                        and "input_ids_without_priv" in batch.batch.keys()
                    )
                    do_kl_opsd = (
                        distill_coef > 0
                        and distill_mode == "opsd"
                        and "input_ids_with_priv" in batch.batch.keys()
                    )

                    if do_kl_pi_distill or do_kl_opsd:
                        with marked_timer("kl_reward", timing_raw, color="cyan"):
                            if do_kl_pi_distill:
                                # With-PI log probs (main input_ids already have PI)
                                wp_batch = DataProto(batch=TensorDict({
                                    "input_ids": batch.batch["input_ids"],
                                    "attention_mask": batch.batch["attention_mask"],
                                    "position_ids": batch.batch["position_ids"],
                                    "responses": batch.batch["responses"],
                                }, batch_size=len(batch.batch)))
                                wp_output = self.actor_rollout_wg.compute_log_prob(wp_batch)
                                with_priv_lp = wp_output.batch["old_log_probs"]

                                # Without-PI log probs
                                np_batch = DataProto(batch=TensorDict({
                                    "input_ids": batch.batch["input_ids_without_priv"],
                                    "attention_mask": batch.batch["attention_mask_without_priv"],
                                    "position_ids": batch.batch["position_ids_without_priv"],
                                    "responses": batch.batch["responses"],
                                }, batch_size=len(batch.batch)))
                                np_output = self.actor_rollout_wg.compute_log_prob(np_batch)
                                without_priv_lp = np_output.batch["old_log_probs"]

                                # Forward KL per token
                                kl_per_token = with_priv_lp - without_priv_lp

                            else:  # do_kl_opsd
                                # In OPSD, main input_ids are WITHOUT-PI
                                np_batch = DataProto(batch=TensorDict({
                                    "input_ids": batch.batch["input_ids"],
                                    "attention_mask": batch.batch["attention_mask"],
                                    "position_ids": batch.batch["position_ids"],
                                    "responses": batch.batch["responses"],
                                }, batch_size=len(batch.batch)))
                                np_output = self.actor_rollout_wg.compute_log_prob(np_batch)
                                without_priv_lp = np_output.batch["old_log_probs"]

                                wp_batch = DataProto(batch=TensorDict({
                                    "input_ids": batch.batch["input_ids_with_priv"],
                                    "attention_mask": batch.batch["attention_mask_with_priv"],
                                    "position_ids": batch.batch["position_ids_with_priv"],
                                    "responses": batch.batch["responses"],
                                }, batch_size=len(batch.batch)))
                                wp_output = self.actor_rollout_wg.compute_log_prob(wp_batch)
                                with_priv_lp = wp_output.batch["old_log_probs"]

                                # Reverse KL per token
                                kl_per_token = without_priv_lp - with_priv_lp

                            # Sum KL over response tokens per turn
                            kl_per_turn = (kl_per_token * response_mask).sum(dim=-1)  # (N,)

                            # Aggregate per-turn KL to per-episode KL
                            ep_ids = batch.batch["episode_ids"]  # (N,)
                            unique_eps = ep_ids.unique()
                            kl_per_episode = torch.zeros_like(ep_ids, dtype=torch.float32)
                            for ep_id in unique_eps:
                                mask = ep_ids == ep_id
                                kl_ep = kl_per_turn[mask].sum()  # total KL for this episode
                                kl_per_episode[mask] = kl_ep

                            # new_reward = G_t - distill_coef * KL_episode
                            batch.batch["advantages"] = batch.batch["advantages"] - distill_coef * kl_per_episode.unsqueeze(-1)

                            mean_kl_ep = kl_per_episode[~kl_per_episode.isnan()].mean().item() if len(unique_eps) > 0 else 0.0
                            metrics["train/kl_per_episode"] = mean_kl_ep
                            metrics["train/kl_per_turn"] = kl_per_turn.mean().item()

                            # Log both forward and reverse KL for monitoring
                            # forward KL = log pi(with_PI) - log pi(without_PI)
                            # reverse KL = log pi(without_PI) - log pi(with_PI)
                            fwd_kl_per_token = with_priv_lp - without_priv_lp
                            rev_kl_per_token = without_priv_lp - with_priv_lp
                            resp_counts = response_mask.sum(dim=-1).clamp(min=1).float()
                            metrics["train/forward_kl"] = ((fwd_kl_per_token * response_mask).sum(dim=-1) / resp_counts).mean().item()
                            metrics["train/reverse_kl"] = ((rev_kl_per_token * response_mask).sum(dim=-1) / resp_counts).mean().item()

                            print(f"[KL-reward] {distill_mode}: mean episode KL={mean_kl_ep:.4f}, forward_kl={metrics['train/forward_kl']:.4f}, reverse_kl={metrics['train/reverse_kl']:.4f}")

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(
                                rollout_probs_diff, response_mask.bool()
                            )
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.config.actor_rollout_ref.norm_return:
                        batch.batch["advantages"] = (
                            batch.batch["advantages"] - batch.batch["advantages"].mean()
                        ) / (batch.batch["advantages"].std() + 1e-9)

                    # Dummy metric logging to satisfy verl's compute_data_metrics
                    batch.batch["token_level_scores"] = batch.batch["advantages"]
                    batch.batch["token_level_rewards"] = batch.batch["advantages"]
                    batch.batch["returns"] = batch.batch["advantages"]

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer(
                            "dump_rollout_generations", timing_raw, color="green"
                        ):
                            inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores = batch.batch["advantages"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict={},
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # [TAU-BENCH] Evaluate on test split
                    eval_freq = self.config.trainer.get("eval_freq", self.config.trainer.test_freq)
                    if eval_freq > 0 and (
                        is_last_step
                        or self.global_steps % eval_freq == 0
                    ):
                        with marked_timer("eval", timing_raw, color="green"):
                            eval_metrics = self._evaluate()
                        metrics.update(eval_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print(
                                "Force saving checkpoint: ESI instance expiration approaching."
                            )
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(batch.meta_info["metrics"])
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    # Flush and finish wandb before Ray tears down the actor
                    if "wandb" in logger.logger:
                        logger.logger["wandb"].finish()
                    return

    def run_agent_env_loop(self):
        """Generate experiences by making the agent interact with tau-bench environments."""
        generate_st = time.time()

        env_cfg = self.config.actor_rollout_ref.env
        n_repeat = env_cfg.get("n_repeat", 1)
        batch_size = env_cfg.rollout_batch_size
        if self.debug:
            # Debug: 2 tasks × n_repeat episodes
            batch_size = 2 * n_repeat
        # Collect batch_size episodes total across n_repeat rounds
        per_repeat_size = max(1, batch_size // n_repeat)

        all_finished_episodes = []
        all_collection_infos = []
        for _ in range(n_repeat):
            finished_episodes, collection_info = self.collect_experience(
                self.env, per_repeat_size
            )
            all_finished_episodes.extend(finished_episodes)
            all_collection_infos.append(collection_info)

        # Save traces to disk (structured under timestamped run directory)
        self._save_step_traces(all_finished_episodes)

        all_trajectories = []
        for ep_idx, ep in enumerate(all_finished_episodes):
            for turn in self.prepare_trajectories(ep):
                turn["episode_id"] = ep_idx
                all_trajectories.append(turn)

        mean_episode_len = np.mean([len(ep) for ep in all_finished_episodes])
        mean_episode_return = np.mean(
            [
                sum(transition.reward for transition in episode)
                for episode in all_finished_episodes
            ]
        )
        mean_episode_success = np.mean(
            [episode[-1].reward == 1 for episode in all_finished_episodes]
        )

        # Subsample trajectories if they exceed the batch size
        if len(all_trajectories) > batch_size:
            subsample_indices = np.random.choice(
                len(all_trajectories),
                batch_size,
                replace=False,
            )
            all_trajectories = [all_trajectories[si] for si in subsample_indices]

        pad_token_id = self.tokenizer.pad_token_id
        ids = []
        attention_mask = []
        position_ids = []
        prompts = []
        responses = []
        adv = []
        max_prompt_len = max([len(x["prompt_ids"]) for x in all_trajectories])
        max_resp_len = max([len(x["response_ids"]) for x in all_trajectories])
        for transition in all_trajectories:
            num_to_pad = max_prompt_len - len(transition["prompt_ids"])
            transition["prompt_ids"] = [
                pad_token_id
            ] * num_to_pad + transition["prompt_ids"]
            transition["attention_mask"] = [0] * num_to_pad + transition[
                "attention_mask"
            ]
            transition["position_ids"] = [0] * num_to_pad + transition["position_ids"]
            ids.append(transition["prompt_ids"] + transition["response_ids"])
            attention_mask.append(transition["attention_mask"])
            position_ids.append(transition["position_ids"])
            responses.append(transition["response_ids"])
            prompts.append(transition["prompt_ids"])
            adv.append(transition["adv"])

        episode_ids = [t["episode_id"] for t in all_trajectories]

        batch_dict = {
            "input_ids": torch.tensor(ids),
            "responses": torch.tensor(responses),
            "attention_mask": torch.tensor(attention_mask),
            "position_ids": torch.tensor(position_ids),
            "advantages": torch.tensor(adv)[:, None],
            "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
        }

        # Build alt-prompt tensors for KL / distillation loss:
        #   OPSD:       prompt_ids_with_priv + response_ids
        #   pi_distill: prompt_ids_without_priv + response_ids
        has_with_priv = all_trajectories and all_trajectories[0].get("prompt_ids_with_priv") is not None
        has_without_priv = all_trajectories and all_trajectories[0].get("prompt_ids_without_priv") is not None

        def _build_alt_turn_tensors(alt_prompt_key):
            """Build alt-prompt input tensors with proper padding."""
            max_alt_plen = max(len(t[alt_prompt_key]) for t in all_trajectories)
            alt_ids, alt_am = [], []
            for t in all_trajectories:
                alt_p = t[alt_prompt_key]
                r = t["response_ids"]
                ap_pad = max_alt_plen - len(alt_p)
                r_pad = max_resp_len - len(r)
                alt_ids.append([pad_token_id] * ap_pad + alt_p + r + [pad_token_id] * r_pad)
                alt_am.append([0] * ap_pad + [1] * len(alt_p) + [1] * len(r) + [0] * r_pad)
            alt_am_t = torch.tensor(alt_am)
            return torch.tensor(alt_ids), alt_am_t, compute_position_id_with_mask(alt_am_t)

        if has_with_priv:
            wp_ids, wp_am, wp_pos = _build_alt_turn_tensors("prompt_ids_with_priv")
            batch_dict["input_ids_with_priv"] = wp_ids
            batch_dict["attention_mask_with_priv"] = wp_am
            batch_dict["position_ids_with_priv"] = wp_pos

        if has_without_priv:
            np_ids, np_am, np_pos = _build_alt_turn_tensors("prompt_ids_without_priv")
            batch_dict["input_ids_without_priv"] = np_ids
            batch_dict["attention_mask_without_priv"] = np_am
            batch_dict["position_ids_without_priv"] = np_pos

        batch = TensorDict(
            batch_dict,
            batch_size=len(ids),
        )
        out = DataProto(batch=batch)
        out.meta_info["timing"] = {"actor_time": time.time() - generate_st}
        out.meta_info["metrics"] = {
            "train/mean_episode_len": mean_episode_len,
            "train/mean_episode_return": mean_episode_return,
            "train/mean_episode_success": mean_episode_success,
        }
        return out

    def collect_experience(self, env, min_steps: int, min_episodes: int = None):
        max_turns = self.config.actor_rollout_ref.env.get("max_turns", None)
        obs, _ = env.reset()
        done = False
        episodes = [[] for _ in range(env.num_envs)]
        turn_counts = [0] * env.num_envs
        finished_episodes = []
        finished_episodes_tool_uses = []
        finished_episodes_tool_success = []
        num_generation_failed = 0
        while True:
            action, extra = self.agent_act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(env.num_envs):
                turn_counts[i] += 1

                # Force-truncate if max_turns reached
                if max_turns is not None and turn_counts[i] >= max_turns and not done[i]:
                    done[i] = True
                    truncated[i] = True
                    reward[i] = 0.0

                if extra[i]["generation_failed"]:
                    num_generation_failed += 1
                    if self.config.actor_rollout_ref.keep_generation_failed:
                        episodes[i][-1].reward += reward[i]
                        episodes[i][-1].done = True
                        finished_episodes.append(deepcopy(episodes[i]))
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                            if done[i]
                            else info[i].get("tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                            if done[i]
                            else info[i].get("tool_success_counter", 0)
                        )
                    episodes[i].clear()
                    turn_counts[i] = 0
                    if not done[i]:
                        next_obs[i] = env.envs[i].reset()[0]
                else:
                    transition = Transition(
                        obs=obs[i],
                        action=action[i],
                        reward=reward[i],
                        done=done[i],
                        prompt=extra[i]["formatted_observation"],
                        prompt_ids=extra[i]["prompt_ids"],
                        response=extra[i]["response"],
                        response_ids=extra[i]["response_ids"],
                        attention_mask=extra[i]["attention_mask"],
                        position_ids=extra[i]["position_ids"],
                        response_is_truncated=extra[i]["response_is_truncated"],
                        action_is_formatted=extra[i]["action_is_formatted"],
                        prompt_ids_with_priv=extra[i].get("prompt_ids_with_priv"),
                        prompt_ids_without_priv=extra[i].get("prompt_ids_without_priv"),
                    )
                    episodes[i].append(transition)
                    if done[i]:
                        finished_episodes.append(deepcopy(episodes[i]))
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                        )
                        episodes[i].clear()
                        turn_counts[i] = 0

            obs = next_obs
            if min_episodes is not None:
                if len(finished_episodes) >= min_episodes:
                    break
            elif len(tree.flatten(finished_episodes)) >= min_steps:
                break

        collection_info = {
            "actor/num_generation_failed": num_generation_failed,
            "actor/prop_generation_failed": (
                num_generation_failed / len(finished_episodes)
                if self.config.actor_rollout_ref.keep_generation_failed
                else num_generation_failed
                / (len(finished_episodes) + num_generation_failed)
            ),
            "actor/num_tool_uses": np.mean(finished_episodes_tool_uses),
            "actor/num_tool_success": np.mean(finished_episodes_tool_success),
        }
        return finished_episodes, collection_info

    @staticmethod
    def _strip_privileged_info(text: str) -> str:
        """Strip <Secret information>...</Secret information> blocks from text."""
        return re.sub(
            r"<Secret information>.*?</Secret information>",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

    def agent_act(self, vec_observation: List[str]) -> Tuple[str, dict]:
        """Use the current LLM as a policy to act.

        [TAU-BENCH] The observation is already a fully formatted prompt from
        TauBenchObservationWrapper. We use prompt_template=na (pass-through)
        and apply_chat_template=False.
        """
        formatted_observations = []
        for observation in vec_observation:
            # [TAU-BENCH] prompt_template=na → pass-through
            observation = TEMPLATE_FACTORY[
                self.config.actor_rollout_ref.prompt_template
            ](observation)
            # [TAU-BENCH] apply_chat_template=False — already formatted
            if self.config.actor_rollout_ref.apply_chat_template:
                observation = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": observation}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_observations.append(observation)

        # Build with-PI and without-PI versions of observations
        obs_with_priv = formatted_observations
        obs_without_priv = [
            self._strip_privileged_info(obs) for obs in formatted_observations
        ]

        # Decide which version to use for generation:
        #   OPSD:       student generates → use WITHOUT-PI
        #   pi_distill: teacher generates → use WITH-PI
        #   none:       standard RL → use WITH-PI (or whatever env provides)
        distill_mode = self.config.actor_rollout_ref.actor.get("distill_mode", "none")
        if distill_mode == "opsd":
            gen_observations = obs_without_priv
        else:
            gen_observations = obs_with_priv

        # Subsample to remove observations that exceed max model length
        idss = self.tokenizer(gen_observations).input_ids
        exceeds_lengths = [
            len(ids) >= self.config.actor_rollout_ref.rollout.max_model_len
            for ids in idss
        ]
        sub_gen_observations = [
            o for o, e in zip(gen_observations, exceeds_lengths) if not e
        ]

        outs = self.tokenizer(
            sub_gen_observations,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            padding_side="left",
        )
        outs["position_ids"] = compute_position_id_with_mask(outs.attention_mask)
        batch: DataProto = DataProto.from_single_dict(outs)

        prompts = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[],
        )

        output = self.actor_rollout_wg.generate_sequences(prompts)

        # Tokenize alt-prompt versions for KL computation:
        #   OPSD:       need WITH-PI tokens
        #   pi_distill: need WITHOUT-PI tokens
        with_priv_idss = None
        without_priv_idss = None
        if distill_mode == "opsd":
            sub_with_priv = [o for o, e in zip(obs_with_priv, exceeds_lengths) if not e]
            with_priv_idss = self.tokenizer(sub_with_priv).input_ids
        elif distill_mode == "pi_distill":
            sub_without_priv = [o for o, e in zip(obs_without_priv, exceeds_lengths) if not e]
            without_priv_idss = self.tokenizer(sub_without_priv).input_ids

        executable_actions = []
        extras = []
        sub_i = 0

        for i, exceeds_length in enumerate(exceeds_lengths):
            if exceeds_length:
                executable_actions.append(INVALID_ACTION)
                extras.append({"generation_failed": True})
            else:
                token_ids = output.batch["responses"][sub_i].tolist()
                prompt_token_ids = output.batch["prompts"][sub_i].tolist()
                attention_mask = output.batch["attention_mask"][sub_i].tolist()
                position_ids = output.batch["position_ids"][sub_i].tolist()
                raw_action = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                response_is_truncated = self.tokenizer.eos_token_id not in token_ids

                # [TAU-BENCH] Check for valid <action> tags
                extracted_action = (
                    INVALID_ACTION
                    if response_is_truncated
                    else self.extract_action(raw_action)
                )
                executable_actions.append(
                    INVALID_ACTION if response_is_truncated else raw_action
                )
                extras.append(
                    {
                        "formatted_observation": formatted_observations[i],
                        "prompt_ids": prompt_token_ids,
                        "prompt_ids_with_priv": with_priv_idss[sub_i] if with_priv_idss else None,
                        "prompt_ids_without_priv": without_priv_idss[sub_i] if without_priv_idss else None,
                        "response": raw_action,
                        "response_ids": token_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "response_is_truncated": response_is_truncated,
                        "action_is_formatted": extracted_action != INVALID_ACTION,
                        "generation_failed": False,
                        "generation_max_length_reached": (
                            len(prompt_token_ids) + len(token_ids)
                            >= self.config.actor_rollout_ref.rollout.max_model_len
                        ),
                    }
                )
                sub_i += 1
        return executable_actions, extras

    def extract_action(self, text: str) -> str:
        """[TAU-BENCH] Extract action from <action>...</action> tags."""
        if not text:
            return INVALID_ACTION
        result = extract_action_tags(text)
        if result is None:
            return INVALID_ACTION
        return result

    def prepare_trajectories(self, episode: Sequence[Transition]) -> List[dict]:
        """Prepare language trajectories (transitions of episode)."""
        trajectory_data = []
        rewards = [t.reward for t in episode]

        # Compute returns
        returns = np.zeros_like(rewards, dtype=np.float32)
        cur = 0.0
        for i in reversed(range(len(rewards))):
            cur = rewards[i] + self.config.actor_rollout_ref.gamma * cur
            returns[i] = cur

        for i, step_data in enumerate(episode):
            turn = dict(
                prompt=step_data.prompt,
                prompt_ids=step_data.prompt_ids,
                response=step_data.response,
                response_ids=step_data.response_ids,
                attention_mask=step_data.attention_mask,
                position_ids=step_data.position_ids,
                adv=returns[i],
                info={
                    "actor/action_is_formatted": step_data.action_is_formatted,
                    "actor/step_reward": rewards[i],
                    "actor/discount_factor": self.config.actor_rollout_ref.gamma,
                    "actor/discounted_step_return": returns[i],
                    "actor/response_is_truncated": step_data.response_is_truncated,
                },
            )
            if step_data.prompt_ids_with_priv is not None:
                turn["prompt_ids_with_priv"] = step_data.prompt_ids_with_priv
            if step_data.prompt_ids_without_priv is not None:
                turn["prompt_ids_without_priv"] = step_data.prompt_ids_without_priv
            trajectory_data.append(turn)

        return trajectory_data


@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote class for executing distributed training tasks."""

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                try:
                    from verl.utils.vllm_utils import is_version_ge
                    if not is_version_ge(pkg="vllm", minver="0.7.3"):
                        raise NotImplementedError(
                            "PPO LoRA is not supported before vllm 0.7.3"
                        )
                except ImportError:
                    pass  # Skip version check if module not available

        actor_rollout_cls = GEMActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        trainer = ReinforceGEMTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            device_name=config.trainer.device,
        )
        try:
            trainer.init_workers()
            trainer.fit()
        finally:
            trainer.user_server.stop()


if __name__ == "__main__":
    main()
