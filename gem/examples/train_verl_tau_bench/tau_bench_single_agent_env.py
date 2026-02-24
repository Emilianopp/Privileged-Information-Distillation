"""Thin gem.Env wrapper that converts TauBenchEnv's MultiAgentEnv dict interface
to a standard scalar (single-agent) interface suitable for verl training.
"""

import importlib.util
import os
import sys
import random
from typing import Any, Optional, Tuple

from gem.core import Env

# ---------------------------------------------------------------------------
# Make sure the tau-bench fork is importable
# ---------------------------------------------------------------------------
TAU_BENCH_PATH = os.environ.get(
    "TAU_BENCH_PATH",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "tau-bench")
    ),
)
if TAU_BENCH_PATH not in sys.path:
    sys.path.insert(0, TAU_BENCH_PATH)

# Import TauBenchEnv from the GEM examples directory (not a proper package)
_tau_bench_env_path = os.path.join(
    os.path.dirname(__file__), "..", "multiagent", "tau_bench_retail", "tau_bench_env.py"
)
_spec = importlib.util.spec_from_file_location("tau_bench_env", _tau_bench_env_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TauBenchEnv = _mod.TauBenchEnv


class TauBenchSingleAgentEnv(Env):
    """Wraps TauBenchEnv (MultiAgentEnv with one agent 'assistant') into scalar Env.

    On each reset a random task is chosen (for training diversity).
    """

    def __init__(
        self,
        env_name: str = "retail",
        task_split: str = "train",
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = "openai",
        user_strategy: str = "llm",
        tasks=None,
    ):
        super().__init__()
        self._inner = TauBenchEnv(
            env_name=env_name,
            task_split=task_split,
            user_model=user_model,
            user_provider=user_provider,
            user_strategy=user_strategy,
            tasks=tasks,
        )
        # Expose attributes needed by the observation wrapper
        self.wiki = self._inner.wiki
        self.tools_info = self._inner.tools_info
        self.tasks = self._inner.tasks

    @property
    def inner_env(self):
        return self._inner._inner_env

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> Tuple[str, dict[str, Any]]:
        super().reset(seed=seed)
        # Pick a random task for training diversity
        task_index = kwargs.get("task_index", None)
        if task_index is None:
            task_index = random.randint(0, len(self.tasks) - 1)

        obs_dict, info_dict = self._inner.reset(seed=seed, task_index=task_index)
        obs = obs_dict["assistant"]
        info = info_dict.get("assistant", {})
        # Store the task instruction for privileged info lookup
        info["task_instruction"] = self.tasks[task_index].instruction
        info["task_index"] = task_index
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        actions = {"assistant": action}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._inner.step(
            actions
        )
        obs = obs_dict.get("assistant", "")
        reward = rew_dict.get("assistant", 0.0)
        terminated = term_dict.get("assistant", False)
        truncated = trunc_dict.get("assistant", False)
        info = info_dict.get("assistant", {})
        return obs, reward, terminated, truncated, info
