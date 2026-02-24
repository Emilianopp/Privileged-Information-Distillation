"""Custom EnvWrapper that builds the full chat-format prompt with system message
(wiki + tools + privileged info) and multi-turn conversation history for tau-bench.

The observation returned is a fully formatted prompt string produced by
``tokenizer.apply_chat_template``, ready to be tokenized directly by the
training loop (with ``apply_chat_template=False`` and ``prompt_template=na``).
"""

import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env, EnvWrapper

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

from tau_bench.agents.privileged_mixin import PrivilegedTrajectoryMixin
from tau_bench.agents.prompts import get_model_prompt


class TauBenchObservationWrapper(EnvWrapper, PrivilegedTrajectoryMixin):
    """Builds fully-formatted chat prompts for tau-bench RL training.

    On ``reset()``: constructs a system prompt from wiki + tools + ReAct
    instruction + optional privileged info, and an initial user message.
    Returns ``tokenizer.apply_chat_template(messages)``.

    On ``step(raw_action)``: parses ``<action>...</action>`` from the raw
    model output to get the JSON action string for the underlying env.
    Appends assistant/user messages to history and returns the updated
    chat-template prompt.

    Parameters
    ----------
    env : Env
        A ``TauBenchSingleAgentEnv`` instance.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer with ``apply_chat_template`` support.
    model_name : str
        Model name used to select the appropriate ReAct prompt variant.
    privileged_path : str or Path or None
        Path to privileged trajectories JSON file.  ``None`` disables privileged info.
    use_human_privileged : bool
        If True, use the human-authored privileged trajectories from task definitions.
    env_name : str
        ``"retail"`` or ``"airline"`` — needed by the privileged mixin.
    use_only_hints : bool
        If True, only expose tool names (not full calls) as hints.
    use_reasoning : bool
        Whether to use the ReAct (think+action) prompt variant.
    max_history_length : int or None
        Cap on conversation turns kept in the sliding window.
    use_self_generated_hints : bool
        Whether to use self-generated hints.
    self_generated_hints_path : str or Path or None
        Path to self-generated hints JSON file.
    """

    def __init__(
        self,
        env: Env,
        tokenizer,
        model_name: str = "qwen",
        privileged_path: Optional[str] = None,
        use_human_privileged: bool = False,
        env_name: str = "retail",
        use_only_hints: bool = False,
        use_reasoning: bool = True,
        max_history_length: Optional[int] = None,
        use_self_generated_hints: bool = False,
        self_generated_hints_path: Optional[str] = None,
    ):
        EnvWrapper.__init__(self, env)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.use_reasoning = use_reasoning
        self.env_name = env_name

        # Conversation history
        self._messages: list = []
        self._max_history_length = max_history_length

        # Initialize privileged data (may be a no-op if paths are None)
        self._has_privileged = (
            privileged_path is not None or use_human_privileged
        )
        if self._has_privileged:
            self._init_privileged_data(
                privileged_path=Path(privileged_path) if privileged_path else None,
                use_human_privileged=use_human_privileged,
                env=env_name,
                use_only_hints=use_only_hints,
                use_self_generated_hints=use_self_generated_hints,
                self_generated_hints_path=(
                    Path(self_generated_hints_path)
                    if self_generated_hints_path
                    else None
                ),
            )

    # ------------------------------------------------------------------
    # Build system prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, task_instruction: Optional[str] = None) -> str:
        parts = []

        # 1. Wiki / policy
        wiki = getattr(self.env, "wiki", None)
        if wiki:
            parts.append(wiki)

        # 2. Available tools
        tools_info = getattr(self.env, "tools_info", None)
        if tools_info:
            parts.append("# Available tools\n" + json.dumps(tools_info, indent=2))

        # 3. ReAct instruction prompt
        instruction = get_model_prompt(self.model_name, use_reasoning=self.use_reasoning)
        parts.append(instruction)

        # 4. Privileged info (only during training)
        if self._has_privileged and task_instruction:
            priv_prompt, _ = self.get_privileged_prompt_for_goal(task_instruction)
            if priv_prompt:
                parts.append(priv_prompt)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(raw_action: str) -> str:
        """Extract JSON from ``<action>...</action>`` tags.

        Falls back to wrapping the raw text as a ``respond`` action.
        """
        match = re.search(r"<action>\s*(.*?)\s*</action>", raw_action, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: treat entire output as a respond action
        return json.dumps({"name": "respond", "arguments": {"content": raw_action}})

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> Tuple[str, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, **kwargs)

        task_instruction = info.get("task_instruction", None)
        system_prompt = self._build_system_prompt(task_instruction)

        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs},
        ]

        formatted = self.tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted, info

    def step(
        self, raw_action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        # Parse the action from model output
        action_json = self._parse_action(raw_action)

        # Step the underlying env with the parsed JSON string
        next_obs, reward, terminated, truncated, info = self.env.step(action_json)

        # Append assistant turn (full raw output) and env response to history
        self._messages.append({"role": "assistant", "content": raw_action})
        self._messages.append({"role": "user", "content": next_obs})

        # Trim history if needed (keep system prompt + last N turn pairs)
        if self._max_history_length is not None:
            # Each turn pair = 2 messages (assistant + user), plus system + initial user = 2
            max_messages = 2 + self._max_history_length * 2
            if len(self._messages) > max_messages:
                # Keep system message + trim the rest
                system = self._messages[0]
                self._messages = [system] + self._messages[-(max_messages - 1):]

        formatted = self.tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted, reward, terminated, truncated, info
