#!/usr/bin/env python3
"""Agent wrapper that delegates to the tau-bench fork's ``agent_factory``.

Supports all agent strategies from the fork:
  - tool-calling  (native function-calling)
  - react         (ReAct with reasoning)
  - act           (ReAct without reasoning)
  - chat-react-priv  (ReAct + privileged info)
  - few-shot

Two usage modes
---------------
1. **Via GEM MultiAgentEnv** – call :pymeth:`solve_gem` with a ``TauBenchEnv``.
   The agent resets the *inner* env, runs the fork's ``agent.solve()``, and
   returns the ``SolveResult`` directly.

2. **Via native tau-bench Env** – call :pymeth:`solve_native` (or the fork's
   agent directly).
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from tau_bench.run import agent_factory  # noqa: E402
from tau_bench.types import RunConfig, SolveResult  # noqa: E402


def build_agent(
    tools_info: List[Dict[str, Any]],
    wiki: str,
    model: str = "gpt-4o",
    model_provider: str = "openai",
    agent_strategy: str = "tool-calling",
    temperature: float = 0.0,
    env: str = "retail",
    privileged_trajectories_path: Optional[str] = None,
    use_human_privileged: bool = False,
    use_only_hints: bool = False,
    use_self_generated_hints: bool = False,
    self_generated_hints_path: Optional[str] = None,
    few_shot_displays_path: Optional[str] = None,
):
    """Create a tau-bench agent via the fork's ``agent_factory``.

    Returns an ``Agent`` subclass instance whose ``.solve(env, task_index)``
    method works directly with the fork's native ``Env``.
    """
    config = RunConfig(
        model=model,
        model_provider=model_provider,
        user_model_provider=model_provider,  # not used by agent, just required
        agent_strategy=agent_strategy,
        temperature=temperature,
        env=env,
        privileged_trajectories_path=privileged_trajectories_path,
        use_human_privileged=use_human_privileged,
        use_only_hints=use_only_hints,
        use_self_generated_hints=use_self_generated_hints,
        self_generated_hints_path=self_generated_hints_path,
        few_shot_displays_path=few_shot_displays_path,
    )
    return agent_factory(tools_info=tools_info, wiki=wiki, config=config)


class TauBenchAgent:
    """Convenience wrapper that builds a fork agent and can run against either
    the GEM ``TauBenchEnv`` or the native ``Env``."""

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: str = "openai",
        agent_strategy: str = "tool-calling",
        temperature: float = 0.0,
        env_name: str = "retail",
        privileged_trajectories_path: Optional[str] = None,
        use_human_privileged: bool = False,
        use_only_hints: bool = False,
        use_self_generated_hints: bool = False,
        self_generated_hints_path: Optional[str] = None,
        few_shot_displays_path: Optional[str] = None,
    ):
        self.model = model
        self.provider = provider
        self.agent_strategy = agent_strategy
        self.temperature = temperature
        self.env_name = env_name

        # Privileged-info related
        self.privileged_trajectories_path = privileged_trajectories_path
        self.use_human_privileged = use_human_privileged
        self.use_only_hints = use_only_hints
        self.use_self_generated_hints = use_self_generated_hints
        self.self_generated_hints_path = self_generated_hints_path
        self.few_shot_displays_path = few_shot_displays_path

        self._agent = None  # lazily built on first solve()

    def _ensure_agent(self, tools_info, wiki):
        """Build the underlying fork agent (once)."""
        if self._agent is None:
            self._agent = build_agent(
                tools_info=tools_info,
                wiki=wiki,
                model=self.model,
                model_provider=self.provider,
                agent_strategy=self.agent_strategy,
                temperature=self.temperature,
                env=self.env_name,
                privileged_trajectories_path=self.privileged_trajectories_path,
                use_human_privileged=self.use_human_privileged,
                use_only_hints=self.use_only_hints,
                use_self_generated_hints=self.use_self_generated_hints,
                self_generated_hints_path=self.self_generated_hints_path,
                few_shot_displays_path=self.few_shot_displays_path,
            )

    def solve(self, env, task_index: int = 0, max_num_steps: int = 30) -> Dict[str, Any]:
        """Run the agent on a ``TauBenchEnv`` (GEM wrapper).

        Internally uses the *inner* native env so that all fork features
        (privileged info, reward calculation, etc.) work correctly.
        """
        inner = getattr(env, "_inner_env", None) or getattr(env, "inner_env", env)
        self._ensure_agent(inner.tools_info, inner.wiki)

        result: SolveResult = self._agent.solve(
            env=inner,
            task_index=task_index,
            max_num_steps=max_num_steps,
        )

        return {
            "reward": result.reward,
            "messages": result.messages,
            "info": result.info,
            "total_cost": result.total_cost,
            "task_index": task_index,
        }

    def solve_native(self, env, task_index: int = 0, max_num_steps: int = 30) -> SolveResult:
        """Run the agent directly on a native tau-bench ``Env``."""
        self._ensure_agent(env.tools_info, env.wiki)
        return self._agent.solve(env=env, task_index=task_index, max_num_steps=max_num_steps)
