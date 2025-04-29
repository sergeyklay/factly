from __future__ import annotations

from typing import TYPE_CHECKING, Union

from deepeval.models.llms import GPTModel
from deepeval.models.llms.openai_model import (
    json_mode_models,
    log_retry_error,
    model_pricing,
    retryable_exceptions,
    structured_outputs_models,
    valid_gpt_models,
)
from deepeval.models.llms.utils import trim_and_load_json
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from typing import Iterable

    from openai.types.chat import ChatCompletionMessageParam


FACTLY_MODELS = {
    "openai/gpt-4o": {
        "base_model": "gpt-4o",
        "pricing": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    },
    "openai/gpt-4.1": {
        "base_model": "gpt-4.1",
        "pricing": {"input": 2.00 / 1e6, "output": 8.00 / 1e6},
    },
    "openai/gpt-4.1-mini": {
        "base_model": "gpt-4.1-mini",
        "pricing": {"input": 0.4 / 1e6, "output": 1.60 / 1e6},
    },
    "openai/gpt-4.1-nano": {
        "base_model": "gpt-4.1-nano",
        "pricing": {"input": 0.1 / 1e6, "output": 0.4 / 1e6},
    },
}
"""LiteLLM's OpenAI models with their pricing details."""


def register_factly_models() -> None:
    """Register Factly's custom models with DeepEval."""
    for model_name, config in FACTLY_MODELS.items():
        # Add pricing information
        model_pricing[model_name] = model_pricing[config["base_model"]]

        # Add to valid models
        if model_name not in valid_gpt_models:
            valid_gpt_models.append(model_name)

        # Add to structured outputs models
        if model_name not in structured_outputs_models:
            structured_outputs_models.append(model_name)

        # Add to JSON mode models
        if model_name not in json_mode_models:
            json_mode_models.append(model_name)


# Register the models
register_factly_models()


class FactlyGptModel(GPTModel):
    """Factly GPT model."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        prompt_name: str,
        *args,
        **kwargs,
    ):
        """Initialize the Factly GPT model."""
        super().__init__(model, *args, **kwargs)

        self.model_name = model
        self.system_prompt = system_prompt
        self.prompt_name = prompt_name

    def _create_messages(self, prompt: str) -> Iterable[ChatCompletionMessageParam]:
        """Create messages for the chat completion."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate(
        self, prompt: str, schema: BaseModel | None = None
    ) -> tuple[Union[str, dict, BaseModel], float]:
        """Generate a response from the model."""
        messages = self._create_messages(prompt)
        client = self.load_model(async_mode=False)

        if schema:
            if self.model_name in structured_outputs_models:
                completion = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=schema,
                )
                structured_output: BaseModel = completion.choices[0].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return structured_output, cost

            if self.model_name in json_mode_models:
                completion = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                json_output = trim_and_load_json(completion.choices[0].message.content)
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return schema.model_validate(json_output), cost

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )

        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return str(output), cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: BaseModel | None = None
    ) -> tuple[Union[str, dict, BaseModel], float]:
        """Generate a response from the model asynchronously."""
        messages = self._create_messages(prompt)
        client = self.load_model(async_mode=True)

        if schema:
            if self.model_name in structured_outputs_models:
                completion = await client.beta.chat.completions.parse(  # type: ignore
                    model=self.model_name,
                    messages=messages,
                    response_format=schema,
                )
                structured_output: BaseModel = completion.choices[0].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return structured_output, cost
            if self.model_name in json_mode_models:
                completion = await client.beta.chat.completions.parse(  # type: ignore
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                json_output = trim_and_load_json(completion.choices[0].message.content)
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return schema.model_validate(json_output), cost

        completion = await client.chat.completions.create(  # type: ignore
            model=self.model_name,
            messages=messages,
        )

        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )

        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return str(output), cost

    def load_model(self, async_mode: bool = False) -> Union[OpenAI, AsyncOpenAI]:
        if not async_mode:
            return OpenAI(api_key=self._openai_api_key, base_url=self.base_url)
        return AsyncOpenAI(api_key=self._openai_api_key, base_url=self.base_url)
