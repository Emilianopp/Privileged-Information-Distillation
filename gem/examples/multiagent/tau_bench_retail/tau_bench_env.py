#!/usr/bin/env python3
"""GEM MultiAgentEnv wrapper that delegates to the tau-bench fork's native Env.

This is a thin adapter: the real environment logic lives in the tau-bench fork
at ``tau_bench.envs``.  We import the fork's ``get_env`` factory so that all
features (privileged info, hosted-vllm user simulator, custom task lists, …)
are available out of the box.

Environment variable
    TAU_BENCH_PATH  –  absolute path to the cloned tau-bench fork.
                       Falls back to ``<this-dir>/../../../../tau-bench``.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from gem.envs.multiagent import MultiAgentEnv
from gem.envs.multiagent.multi_agent_env import AgentSelector

# ---------------------------------------------------------------------------
# Make sure the tau-bench fork is importable
# ---------------------------------------------------------------------------
TAU_BENCH_PATH = os.environ.get(
    "TAU_BENCH_PATH",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tau-bench")
    ),
)

if not os.path.isdir(TAU_BENCH_PATH):
    raise FileNotFoundError(
        f"tau-bench fork not found at {TAU_BENCH_PATH}.\n"
        "Please set the TAU_BENCH_PATH environment variable to the cloned fork."
    )

if TAU_BENCH_PATH not in sys.path:
    sys.path.insert(0, TAU_BENCH_PATH)

from tau_bench.envs import get_env  # noqa: E402
from tau_bench.types import Action, Task, RESPOND_ACTION_NAME  # noqa: E402


class TauBenchEnv(MultiAgentEnv):
    """GEM MultiAgentEnv adapter around the tau-bench fork's native ``Env``.

    Parameters
    ----------
    env_name : str
        ``"retail"`` or ``"airline"``.
    task_split : str
        ``"train"``, ``"test"``, or ``"dev"``.
    user_model : str
        Model name for the user simulator (e.g. ``"gpt-4o"``).
    user_provider : str | None
        LiteLLM provider for the user simulator.
    user_strategy : str
        One of the ``UserStrategy`` values (default ``"llm"``).
    tasks : list[Task] | None
        Optional custom task list; overrides ``task_split``.
    """

    def __init__(
        self,
        env_name: str = "retail",
        task_split: str = "test",
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = "openai",
        user_strategy: str = "llm",
        tasks: Optional[List[Task]] = None,
    ):
        super().__init__()

        self.possible_agents = ["assistant"]
        self.agent_selector = AgentSelector(self.possible_agents, mode="sequential")

        # Build the native tau-bench env via its factory
        self._inner_env = get_env(
            env_name=env_name,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_split=task_split,
            tasks=tasks,
        )

        # Expose useful attributes from the inner env
        self.wiki = self._inner_env.wiki
        self.tools_info = self._inner_env.tools_info
        self.tool_definitions = self._inner_env.tools_info
        self.tasks = self._inner_env.tasks
        self.terminate_tools = self._inner_env.terminate_tools

    # -- properties that proxy the inner env ----------------------------------

    @property
    def task(self):
        return self._inner_env.task

    @property
    def task_index(self):
        return self._inner_env.task_index

    @property
    def data(self):
        return self._inner_env.data

    @property
    def inner_env(self):
        """Direct access to the underlying tau-bench ``Env`` instance."""
        return self._inner_env

    # -- reset ----------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task_index: Optional[int] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        observations, infos = super().reset(seed=seed)

        response = self._inner_env.reset(task_index=task_index)

        observations["assistant"] = response.observation
        infos["assistant"] = response.info.model_dump()
        return observations, infos

    # -- observe --------------------------------------------------------------

    def observe(self, agent: str) -> str:  # noqa: D401
        """Last observation for *agent* (only ``"assistant"`` is supported)."""
        return ""

    # -- step -----------------------------------------------------------------

    def _process_actions(
        self, actions: Dict[str, str]
    ) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        observations: Dict[str, str] = {}
        rewards: Dict[str, float] = {"assistant": 0.0}
        terminations: Dict[str, bool] = {"assistant": False}
        truncations: Dict[str, bool] = {"assistant": False}
        infos: Dict[str, dict] = {"assistant": {}}

        if "assistant" not in actions:
            return observations, rewards, terminations, truncations, infos

        action_str = actions["assistant"]

        # Parse the action – accept JSON (with "kwargs" or "arguments") or plain text
        try:
            action_dict = json.loads(action_str)
            action = Action(
                name=action_dict["name"],
                kwargs=action_dict.get("kwargs", action_dict.get("arguments", {})),
            )
        except Exception:
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": action_str})

        response = self._inner_env.step(action)

        observations["assistant"] = response.observation
        rewards["assistant"] = response.reward
        terminations["assistant"] = response.done
        infos["assistant"] = response.info.model_dump()

        return observations, rewards, terminations, truncations, infos
