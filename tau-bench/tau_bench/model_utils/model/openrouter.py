import os
from collections import defaultdict

from tau_bench.model_utils.api.datapoint import Datapoint
from tau_bench.model_utils.model.completion import (
    approx_cost_for_datapoint,
    approx_prompt_str,
)
from tau_bench.model_utils.model.utils import approx_num_tokens
from .openai import OpenAIModel as _BaseOpenAIModel

DEFAULT_OPENAI_MODEL = "gpt-4o-2024-08-06"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"


PRICE_PER_INPUT_TOKEN_MAP = defaultdict(
    lambda: 5 / 1000000,
    {
        "gpt-4o-2024-08-06": 2.5 / 1000000,
        "gpt-4o": 5 / 1000000,
        "gpt-4o-2024-08-06": 2.5 / 1000000,
        "gpt-4o-2024-05-13": 5 / 1000000,
        "gpt-4-turbo": 10 / 1000000,
        "gpt-4-turbo-2024-04-09": 10 / 1000000,
        "gpt-4": 30 / 1000000,
        "gpt-4o-mini": 0.15 / 1000000,
        "gpt-4o-mini-2024-07-18": 0.15 / 1000000,
        "gpt-3.5-turbo": 0.5 / 1000000,
        "gpt-3.5-turbo-0125": 0.5 / 1000000,
        "gpt-3.5-turbo-instruct": 1.5 / 1000000,
    },
)
INPUT_PRICE_PER_TOKEN_FALLBACK = 5 / 1000000


CAPABILITY_SCORE_MAP = defaultdict(
    lambda: 0.5,
    {
        "gpt-4o-2024-08-06": 0.8,
        "gpt-4o": 0.8,
        "gpt-4o-2024-08-06": 0.8,
        "gpt-4o-2024-05-13": 0.8,
        "gpt-4-turbo": 0.9,
        "gpt-4-turbo-2024-04-09": 0.9,
        "gpt-4": 0.8,
        "gpt-4o-mini": 0.5,
        "gpt-4o-mini-2024-07-18": 0.5,
        "gpt-3.5-turbo": 0.3,
        "gpt-3.5-turbo-0125": 0.3,
    },
)
CAPABILITY_SCORE_FALLBACK = 0.5

# TODO: implement
LATENCY_MS_PER_OUTPUT_TOKEN_MAP = {}
# TODO: implement
LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK = 0.0


MAX_CONTEXT_LENGTH_MAP = defaultdict(
    lambda: 128000,
    {
        "gpt-4o-2024-08-06": 128000,
        "gpt-4o": 128000,
        "gpt-4o-2024-08-06": 128000,
        "gpt-4o-2024-05-13": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-2024-04-09": 128000,
        "gpt-4": 8192,
        "gpt-4o-mini": 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-0125": 16385,
    },
)
MAX_CONTEXT_LENGTH_FALLBACK = 128000


DEFAULT_OPENROUTER_MODEL = DEFAULT_OPENAI_MODEL


class OpenRouterModel(_BaseOpenAIModel):
    """OpenRouter variant of OpenAIModel using a different API key env var and
    custom pricing/capability/context fallback defaults defined in this module.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        if model is None:
            model = DEFAULT_OPENROUTER_MODEL
        if api_key is None:
            api_key = os.getenv(API_KEY_ENV_VAR)
            if api_key is None:
                raise ValueError(f"{API_KEY_ENV_VAR} environment variable is not set")
        super().__init__(model=model, api_key=api_key, temperature=temperature)

    # Override cost/capability/latency/context methods to use OpenRouter maps
    def get_approx_cost(self, dp: Datapoint) -> float:  # type: ignore[override]
        cost_per_token = PRICE_PER_INPUT_TOKEN_MAP.get(
            self.model, INPUT_PRICE_PER_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=cost_per_token)

    def get_latency(self, dp: Datapoint) -> float:  # type: ignore[override]
        latency_per_output_token = LATENCY_MS_PER_OUTPUT_TOKEN_MAP.get(
            self.model, LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(
            dp=dp, price_per_input_token=latency_per_output_token
        )

    def get_capability(self) -> float:  # type: ignore[override]
        return CAPABILITY_SCORE_MAP.get(self.model, CAPABILITY_SCORE_FALLBACK)

    def supports_dp(self, dp: Datapoint) -> bool:  # type: ignore[override]
        prompt = approx_prompt_str(dp)
        return approx_num_tokens(prompt) <= MAX_CONTEXT_LENGTH_MAP.get(
            self.model, MAX_CONTEXT_LENGTH_FALLBACK
        )
