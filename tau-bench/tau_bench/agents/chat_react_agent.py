# Copyright Sierra

import json
from pathlib import Path
from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple
import re

from tau_bench.agents.prompts import get_model_prompt
from tau_bench.agents.privileged_mixin import PrivilegedTrajectoryMixin


START_TAG = "[BEGIN FINAL RESPONSE]"
END_TAG = "[END FINAL RESPONSE]"
END_RESPONSE_TAG = "<|end|>"

LLAMA_REASONING_PREFIX = "Lets think step by step"


def ensure_reasoning_prefix_block(text: str, prefix: str) -> str:
    """Ensure the first <think> block begins with the desired prefix."""

    if not text:
        return text

    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text

    inner = match.group(1).strip()
    if inner.startswith(prefix):
        return text

    new_inner_lines = [prefix]
    if inner:
        new_inner_lines.append(inner)
    replacement = "\n" + "\n".join(new_inner_lines) + "\n"

    start, end = match.span(1)
    return text[:start] + replacement + text[end:]


def _content_from_reasoning_only(reasoning_content: str) -> Optional[str]:
    """Build a think/action block when only reasoning_content is populated."""

    if not reasoning_content:
        return None

    action_match = re.search(
        r"<action>(.*?)</action>", reasoning_content, flags=re.DOTALL | re.IGNORECASE
    )
    if not action_match:
        return None

    think_text = reasoning_content[: action_match.start()].strip()
    action_body = action_match.group(1).strip()

    action_block = f"<action>\n{action_body}\n</action>"
    if think_text:
        return f"<think>\n{think_text}\n</think>\n{action_block}"
    else:
        print('no think')
    return action_block

def split_reasoning_and_action(s: str) -> Tuple[str, str]:
    """Return (reasoning_wrapped, action_wrapped) from a single string.
    reasoning_wrapped -> '<think>\\n{reasoning}\\n</think>' or '' if none
    action_wrapped    -> '\\n\\n<action>\\n{content}\\n</action>'
    """
    txt = s.strip()

    # Locate tags
    i = txt.find(START_TAG)
    j = txt.find(END_TAG, i + len(START_TAG)) if i != -1 else -1

    if i != -1 and j != -1:
        reasoning = txt[:i].strip()
        content = txt[i + len(START_TAG):j].strip()
    else:
        reasoning = ""
        content = txt

    # Clean accidental echoes
    if reasoning.endswith(START_TAG):
        reasoning = reasoning[:-len(START_TAG)].rstrip()
    if content.startswith(START_TAG):
        content = content[len(START_TAG):].lstrip()
    if content.endswith(END_TAG):
        content = content[:-len(END_TAG)].rstrip()
    if content.endswith(END_RESPONSE_TAG):
        content = content[:-len(END_RESPONSE_TAG)].rstrip()

    # Normalize existing <think> wrappers
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>"):].lstrip()
    if reasoning.endswith("</think>"):
        reasoning = reasoning[:-len("</think>")].rstrip()

    # Strip any action wrappers inside content before re-wrapping
    content = content.replace("<action>", "").replace("<end_action>", "").strip()

    reasoning_wrapped = f"<think>\n{reasoning}\n</think>" if reasoning else ""
    action_wrapped = f"\n\n<action>\n{content}\n</action>"

    return reasoning_wrapped, action_wrapped


class ChatReActAgent(PrivilegedTrajectoryMixin, Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        privieleged_path: Optional[Path] = None,
        use_human_privileged: bool = False,
        env: str = "retail",
        use_only_hints: bool = False,
        use_self_generated_hints: bool = False,
        self_generated_hints_path: Optional[Path] = None,
    ) -> None:
        instruction = get_model_prompt(model, use_reasoning)
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        self.use_only_hints = use_only_hints
        self.use_self_generated_hints = use_self_generated_hints

        if use_human_privileged or privieleged_path is not None or use_self_generated_hints:
            self._init_privileged_data(
                privieleged_path,
                use_human_privileged=use_human_privileged,
                env=env,
                use_only_hints=use_only_hints,
                use_self_generated_hints=use_self_generated_hints,
                self_generated_hints_path=self_generated_hints_path,
            )

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        if self.provider == "hosted_vllm":
            kwargs = {"api_base": "http://localhost:8000/v1", "api_key": "EMPTY"}
        elif self.provider == "openrouter":
            kwargs = {
                "reasoning": {"enabled": True, "effort": "medium"},
                "include_reasoning": True,
            }
        else:
            kwargs = {}

        # Check if this is a Llama model
        is_llama = 'llama' in self.model.lower()

        # If Llama, add prefix message and adjust parameters
        if is_llama:
            prefix_message = {
                'role': 'assistant',
                'content': '<think>\n Lets think step by step',
                'prefix': True
            }
            messages_with_prefix = messages + [prefix_message]

            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=messages_with_prefix,
                temperature=0.6,
                top_p=0.95,
                continue_final_message=True,
                add_generation_prompt=False,
                **kwargs,
            )
        else:
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=messages,
                temperature=self.temperature,
                **kwargs,
            )

        if 'apriel' in self.model.lower():
            res.choices[0].reasoning_content, res.choices[0].message.content = split_reasoning_and_action(res.choices[0].message.content)

        message = res.choices[0].message
        message.role = "assistant"

        content = message.content or ""

        # Find all action blocks
        action_matches = list(
            re.finditer(
                r"<action>(.*?)</action>", content, flags=re.DOTALL | re.IGNORECASE
            )
        )

        # Check for reasoning content in different locations
        reasoning_content = ""
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning_content = message.reasoning_content
            del message.reasoning_content

        if is_llama and not content.strip() and reasoning_content:
            rebuilt = _content_from_reasoning_only(reasoning_content)
            if rebuilt:
                content = rebuilt
                message.content = content
                reasoning_content = ""
                action_matches = list(
                re.finditer(
                    r"<action>(.*?)</action>", content, flags=re.DOTALL | re.IGNORECASE
                )
              )

        # Combine both content sources for think block detection
        if reasoning_content:
            content = f"<think>\n{reasoning_content}\n</think>\n{content}"
            message.content = content
            content = message.content or ""

        if is_llama:
            updated_content = ensure_reasoning_prefix_block(message.content or "", LLAMA_REASONING_PREFIX)
            if updated_content != (message.content or ""):
                message.content = updated_content
            content = message.content or ""

        think_blocks = re.findall(
            r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE
        )

        if not action_matches:
            return (
                message.model_dump(),
                False,
                res._hidden_params.get("response_cost", 0.0),
            )

        if len(think_blocks) > 1:
            return (
                message.model_dump(),
                False,
                res._hidden_params.get("response_cost", 0.0),
            )

        # If multiple actions, take only the first one and modify content
        if len(action_matches) > 1:
            first_action = action_matches[0]
            # Keep content up to and including the first action
            content_up_to_first_action = content[: first_action.end()]
            message.content = content_up_to_first_action
            content = content_up_to_first_action

        # Use the first action
        m = action_matches[0]
        action_str = m.group(1).strip()

        # Remove surrounding code fences if present
        action_str = re.sub(r"^```(?:json)?\s*", "", action_str, flags=re.IGNORECASE)
        action_str = re.sub(r"\s*```$", "", action_str)
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])

        return message.model_dump(), action, res._hidden_params["response_cost"]


    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 20
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        goal = getattr(env.task, "instruction", None)
        privileged_prompt = ""
        privileged_raw = None

        privileged_prompt, privileged_raw = self.get_privileged_prompt_for_goal(
            goal
        )
        initial_role = "system"
        initial_prompt = self.prompt
        if "llama" in self.model.lower():
            initial_role = "user"
            initial_prompt = "This is system not the user.\n" + self.prompt
        messages: List[Dict[str, Any]] = [
            {"role": initial_role, "content": initial_prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info = {}
        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step(messages)
            if not action:
                # there was an action parse error
                messages.extend([message])
                reward = 0.0
                # Build a richer info payload with safe defaults
                module = getattr(env.__class__, "__module__", "")
                env_name = None
                try:
                    env_name = module.split(".")[-2]
                except Exception:
                    env_name = None

                info["error"] = "Action parse error"
                info["task"] = {
                    "user_id": getattr(env.task, "user_id", None),
                    "actions": [],
                    "instruction": getattr(env.task, "instruction", None),
                    "index": env.task_index,
                }
                break
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            if cost == None:
                cost = 0.0
            total_cost += cost
            if response.done:
                break
        # Ensure env name is included in info for success paths as well
        merged_info = info
        merged_info = self._merge_privileged_info(
            info,
            goal,
            privileged_prompt,
            privileged_raw,
        )

        return SolveResult(
            messages=messages,
            reward=reward,
            info=merged_info,
            total_cost=total_cost,
        )
