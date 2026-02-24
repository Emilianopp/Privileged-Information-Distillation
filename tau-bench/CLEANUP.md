# tau-bench Cleanup Log

This document records the structural cleanup applied to the tau-bench fork.

## 1. Agent file split

`tau_bench/agents/chat_react_agent.py` was ~1000 lines mixing four concerns. It has been split into three files:

| File | Contents |
|------|----------|
| `tau_bench/agents/prompts.py` | `MODEL_PROMPTS` dict and `get_model_prompt()` function. Pure data/config. |
| `tau_bench/agents/privileged_mixin.py` | `PrivilegedTrajectoryMixin` class (11 methods) plus shared utilities `is_json_format()` and `remove_think()`. |
| `tau_bench/agents/chat_react_agent.py` | `ChatReActAgent` class and module-level parsing helpers (`ensure_reasoning_prefix_block`, `_content_from_reasoning_only`, `split_reasoning_and_action`). |

`chat_react_agent_priv.py` was updated to import from the new locations. Its duplicate `_content_from_reasoning_only()` was removed in favor of importing from `chat_react_agent.py`.

## 2. Removed `ChatReActPrivCollectorAgent`

This class (previously in `chat_react_agent.py`, lines 682-713) inherited from `ChatReActAgent` and overrode nothing -- its `__init__` simply called `super().__init__()` with identical arguments. It was not referenced anywhere in the codebase. Deleted.

## 3. `TransferToHumanAgents` tool removed from tool lists

The `TransferToHumanAgents` import and list entry were removed from both:
- `tau_bench/envs/retail/tools/__init__.py`
- `tau_bench/envs/airline/tools/__init__.py`

Previously the import was present but the tool was commented out of `ALL_TOOLS`. Now the import is removed entirely since the tool is not exposed to agents.

## 4. Task files restored; runtime filtering added

The task definition files (`retail/tasks.py`, `airline/tasks.py`) previously had tasks commented out when they contained `transfer_to_human_agents` actions or had empty action lists. This made the files drift from upstream.

**Fix:** Task files were restored to their original (upstream-compatible) state with all tasks uncommented. A `filter_tasks()` function was added to `tau_bench/envs/__init__.py` and applied in both `MockRetailDomainEnv` and `MockAirlineDomainEnv` constructors. It excludes tasks where:
- The `actions` list is empty, or
- Any action has `name == "transfer_to_human_agents"`

This keeps the task files upstream-compatible while making the filtering logic explicit and centralized.

## 5. Hardcoded URLs replaced with env vars

| File | Before | After |
|------|--------|-------|
| `model_utils/model/vllm_chat.py` | `BASE_URL = "http://127.0.0.1:8000/v1"` | `os.environ.get("AGENT_VLLM_API_BASE", "http://127.0.0.1:8000/v1")` |
| `model_utils/model/vllm_completion.py` | `BASE_URL = "http://127.0.0.1:8000/v1"` | Same pattern |
| `envs/user.py` | Hardcoded `"http://localhost:8001/v1"` in `generate_next_message` | Uses `self.api_base` (already set from `USER_VLLM_API_BASE` env var) |

## 6. Deleted `OPENROUTER.py` (uppercase)

`tau_bench/model_utils/model/OPENROUTER.py` was an empty (0-byte) file. The actual implementation lives in the lowercase `openrouter.py`. The empty uppercase file was deleted.

## 7. `/nothink` in user.py

The `/nothink` suffix in the user simulation system prompt is intentional -- it suppresses Qwen3's internal reasoning during user simulation. An inline comment was added to document this.
