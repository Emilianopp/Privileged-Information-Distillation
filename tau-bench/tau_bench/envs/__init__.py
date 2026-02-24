# Copyright Sierra

from typing import Optional, Union
from tau_bench.envs.base import Env
from tau_bench.envs.user import UserStrategy
from typing import Any, Callable, Dict, List
from tau_bench.types import (
    Task,
)


def filter_tasks(tasks: List[Any]) -> List[Any]:
    """Exclude tasks whose actions list is empty or contains transfer_to_human_agents."""
    filtered = []
    for task in tasks:
        actions = task.get("actions", []) if isinstance(task, dict) else getattr(task, "actions", [])
        if not actions:
            continue
        if any(
            (a.get("name") if isinstance(a, dict) else getattr(a, "name", None))
            == "transfer_to_human_agents"
            for a in actions
        ):
            continue
        filtered.append(task)
    return filtered


def get_env(
    env_name: str,
    user_strategy: Union[str, UserStrategy],
    user_model: str,
    task_split: str,
    user_provider: Optional[str] = None,
    task_index: Optional[int] = None,
    tasks: List[Task] = None
) -> Env:
    if env_name == "retail":
        from tau_bench.envs.retail import MockRetailDomainEnv

        return MockRetailDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
            tasks=tasks
        )
    elif env_name == "airline":
        from tau_bench.envs.airline import MockAirlineDomainEnv

        return MockAirlineDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
