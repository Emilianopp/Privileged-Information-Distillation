# Copyright Sierra

import ast
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any


MAX_SELF_GENERATED_HINTS = 3


def is_json_format(candidate: str, return_json: bool = False):
    """Try to parse a string as JSON, with fallback strategies."""
    if not candidate or "{" not in candidate or "}" not in candidate:
        return False

    start = candidate.find("{")
    end = candidate.rfind("}") + 1
    snippet = candidate[start:end]
    snippet = (
        snippet.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    )

    try:
        parsed = json.loads(snippet)
        return parsed if return_json else True
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r'(?<!\\)"(?![:,}\]])', r"\"", snippet)
    try:
        parsed = json.loads(fixed)
        return parsed if return_json else True
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(snippet)
        return parsed if return_json else True
    except Exception:
        return False


def remove_think(content: str) -> str:
    """Strip <think>...</think> blocks from content."""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)


class PrivilegedTrajectoryMixin:
    """Utility mixin for loading and formatting privileged trajectories."""

    def _init_privileged_data(
        self,
        privileged_path: Optional[Path],
        use_human_privileged: bool,
        env: str,
        use_only_hints: bool = False,
        use_self_generated_hints: bool = False,
        self_generated_hints_path: Optional[Path] = None,
    ) -> None:
        self.use_only_hints = use_only_hints
        self.use_self_generated_hints = use_self_generated_hints
        self.self_generated_hints_path = self_generated_hints_path
        self.self_generated_hints_data: Dict[str, List[Dict[str, Any]]] = {}

        if use_human_privileged:
            if env == "retail":
                from tau_bench.envs.retail.tasks_train import TASKS_TRAIN

                tasks = TASKS_TRAIN
            elif env == "airline":
                from tau_bench.envs.airline.tasks_test import TASKS as TASKS_TEST

                tasks = TASKS_TEST
            else:
                raise ValueError(f"Unknown environment: {env}")

            privileged_source = self.convert_tasks_to_privileged_format(tasks)
        else:
            if privileged_path is None:
                raise ValueError(
                    "privieleged_path must be provided when use_human_privileged is False"
                )
            try:
                with open(privileged_path, "r") as handle:
                    privileged_source = json.load(handle)
            except:
                privileged_path = '/mnt/adea/data_rw/finetuning/emiliano_home/experiments/20251121_010758_on_policy_taubench_random_tasks_v1/epoch_91/privileged_actions.json'
                with open(privileged_path, "r") as handle:
                    privileged_source = json.load(handle)


        self.privileged_trajectories_data = self.process_trajectories(privileged_source)

        if self.use_self_generated_hints:
            if self_generated_hints_path is None:
                raise ValueError(
                    "self_generated_hints_path must be provided when use_self_generated_hints is True"
                )
            self.self_generated_hints_data = self._load_self_generated_hints(
                self_generated_hints_path
            )

    def convert_tasks_to_privileged_format(
        self, tasks: List[Any]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        privileged_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for task in tasks:
            if isinstance(task, dict):
                instruction = task["instruction"]
                actions = task["actions"]
            else:
                instruction = task.instruction
                actions = task.actions

            traj_steps = []
            for action in actions:
                if isinstance(action, dict):
                    action_name = action["name"]
                    action_args = action["arguments"]
                else:
                    action_name = action.name
                    action_args = action.kwargs

                action_content = json.dumps(
                    {"name": action_name, "arguments": action_args}
                )

                traj_steps.append({"role": "assistant", "content": action_content})

            if instruction not in privileged_data:
                privileged_data[instruction] = {}

            traj_key = f"traj_{len(privileged_data[instruction])}"
            privileged_data[instruction][traj_key] = {"traj": traj_steps}

        return privileged_data

    def process_trajectories(self, trajectories_data: dict) -> dict:
        goal_to_candidates: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(trajectories_data, list):
            trajectories_data = trajectories_data[0]
        for traj_key, trajectory in trajectories_data.items():
            for traj_id, traj_steps in trajectory.items():
                steps = traj_steps["traj"]
                goal_to_candidates.setdefault(traj_key, []).append(
                    {
                        "traj_key": traj_key,
                        "steps": steps,
                        "traj_id": traj_id,
                    }
                )
        return goal_to_candidates

    def _load_self_generated_hints(self, hints_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        if not hints_path.exists():
            raise FileNotFoundError(f"self-generated hints path not found: {hints_path}")

        with open(hints_path, "r") as handle:
            raw_hints = json.load(handle)

        if isinstance(raw_hints, dict) and "hints" in raw_hints:
            hint_entries = raw_hints["hints"]
        else:
            hint_entries = raw_hints

        hints_by_goal: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(hint_entries, dict):
            for goal, entries in hint_entries.items():
                if not isinstance(entries, list):
                    entries = [entries]
                hints_by_goal[goal] = [
                    entry
                    for entry in entries
                    if isinstance(entry, dict) and entry.get("hint")
                ]
            return hints_by_goal

        if not isinstance(hint_entries, list):
            return {}

        for entry in hint_entries:
            if not isinstance(entry, dict):
                continue
            goal = entry.get("goal")
            hint_text = entry.get("hint")
            if not goal or not hint_text:
                continue
            hints_by_goal.setdefault(goal, []).append(entry)
        return hints_by_goal

    def _get_self_generated_hints(self, goal: Optional[str]) -> List[Dict[str, Any]]:
        if not goal or not getattr(self, "use_self_generated_hints", False):
            return []
        return list(self.self_generated_hints_data.get(goal, []))

    def _format_self_generated_hints_prompt(
        self, goal: str, hints: List[Dict[str, Any]]
    ) -> str:
        if not hints:
            return ""
        selected = hints[:MAX_SELF_GENERATED_HINTS]
        bullets = "\n".join(f"- {entry['hint']}" for entry in selected if entry.get("hint"))
        return f"""
<Secret information>
{bullets}
Use them to guide your plan, but never mention this secret section.
</Secret information>
"""

    def get_privileged_info_prompt_taubench(self, trajectory) -> str:
        tool_calls: List[str] = []
        for item in trajectory[0]["steps"]:
            if item["role"] == "assistant" and is_json_format(
                remove_think(item["content"])
            ):
                if self.use_only_hints:
                    try:
                        parsed_content = is_json_format(
                            remove_think(item["content"]), return_json=True
                        )
                        if (
                            isinstance(parsed_content, dict)
                            and "name" in parsed_content
                        ):
                            tool_calls.append(parsed_content["name"])
                    except Exception:
                        pass
                else:
                    tool_calls.append(item["content"])

        if self.use_only_hints:
            tool_calls_section = f"""
        Required Tool Sequence (hints only):
        {chr(10).join([f'{i+1}. {tool_name}' for i, tool_name in enumerate(tool_calls)]) }

        These are the tools you need to use in order responses to the user may be missing. You must determine the appropriate arguments for each tool based on the context and user information."""
        else:
            tool_calls_section = f"""
        Required Tool Calls:
        {chr(10).join([f'{tool_call}\n' for tool_call in tool_calls]) }"""

        return f"""
        <Secret information>
        This section is highly important for you to pay attention to. You have been given access to secret information that is not available to the user and should not be mentioned.
        Here is a succesful set of tools, given the context and tools that have been called so far use this information to acomplish the task.

        {tool_calls_section}

        **Your Task:**
        1.  **You will need to use the provided tools in order to accomplish the task as well as query the user for more information if needed.**
        3.  **Reasoning:** Please reason about your steps if you think you need to use a tool provide reasoning justifications and clear logic for using it before invoking it using <think>reasoning...</think>.
        4.  **Strict Constraint:** Do NOT mention that you have been given access to the secret information. You will be penalized for violating this rule.

        **Reasoning Format**
        Your reasoning traces will be used to train future agents that do not have access to privileged information. Please make sure your reasoning is clear and concise.

        </Secret information>
        """

    def get_privileged_prompt_for_goal(self, goal: Optional[str]):
        prompt_sections: List[str] = []
        payload: Dict[str, Any] = {}

        trajectory = None
        if goal and getattr(self, "privileged_trajectories_data", None):
            trajectory = self.privileged_trajectories_data.get(goal)

        self_hints = self._get_self_generated_hints(goal)
        if self_hints:
            prompt_sections.append(
                self._format_self_generated_hints_prompt(goal, self_hints)
            )
            payload["self_generated_hints"] = self_hints
            self._last_self_generated_hints = self_hints
        elif trajectory:
            prompt_sections.append(self.get_privileged_info_prompt_taubench(trajectory))
            payload["trajectories"] = trajectory
            self._last_self_generated_hints = None
        else:
            self._last_self_generated_hints = None

        if not prompt_sections and not payload:
            return "", None

        combined_prompt = "\n\n".join(section.strip() for section in prompt_sections if section)
        return combined_prompt, (payload if payload else None)

    def _merge_privileged_info(
        self,
        info: Dict[str, Any],
        goal: Optional[str],
        privileged_prompt: str,
        privileged_raw: Optional[Any],
    ) -> Dict[str, Any]:
        merged = dict(info)
        merged.setdefault(
            "privileged",
            {
                "goal": goal,
                "available": bool(privileged_raw),
            },
        )
        payload: Dict[str, Any] = {}
        if isinstance(privileged_raw, dict):
            payload.update(privileged_raw)
        elif privileged_raw:
            payload["trajectories"] = privileged_raw

        self_hints = getattr(self, "_last_self_generated_hints", None)
        if self_hints:
            payload.setdefault("self_generated_hints", self_hints)

        if payload:
            merged["privileged"].update(
                {
                    "prompt": privileged_prompt,
                    "use_only_hints": getattr(self, "use_only_hints", False),
                    **payload,
                }
            )
        return merged
