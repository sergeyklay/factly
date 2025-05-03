from __future__ import annotations

from typing import TYPE_CHECKING, Union

from deepeval.models.llms import GPTModel
from deepeval.models.llms.openai_model import (
    json_mode_models,
    log_retry_error,
    retryable_exceptions,
    structured_outputs_models,
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


def get_actual_model_name(model_name: str) -> str:
    """Get the actual model name.

    Returns:
        The actual model name
    """
    if "/" in model_name:
        # For LiteLLM format "provider/model", extract just the model part
        _, model_name = model_name.split("/", 1)
        return model_name
    # For direct provider models, return as is
    return model_name


class FactlyGptModel(GPTModel):
    """Factly GPT model."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        prompt_name: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1,
        *args,
        **kwargs,
    ):
        """Initialize the Factly GPT model.

        Args:
            model: Model identifier, can be "<provider>/<model>" for LiteLLM or
                "<model>" for direct provider models
            system_prompt: System prompt to use for generating responses
            prompt_name: Display name for this model configuration in reports
            temperature: Sampling temperature between 0.0 and 2.0
            top_p: Nucleus sampling parameter between 0.0 and 1.0
            max_tokens: Maximum number of tokens to generate
        """
        actual_model_name = get_actual_model_name(model)
        super().__init__(actual_model_name, *args, **kwargs)

        self.model_name = model  # Redefine the model name
        self.actual_model_name = actual_model_name
        self.system_prompt = system_prompt
        self.prompt_name = prompt_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def create_messages(self, prompt: str) -> Iterable[ChatCompletionMessageParam]:
        """Create messages for the chat completion."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def get_display_model_name(self) -> str:
        """Get the display model name.

        Returns:
            The display model name
        """
        return self.actual_model_name

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def ainvoke(
        self, prompt: str, schema: BaseModel | None = None
    ) -> Union[str, dict, BaseModel]:
        """Generate a response from the model asynchronously."""
        messages = self.create_messages(prompt)
        client = self.load_model(async_mode=True)

        if schema:
            if self.actual_model_name in structured_outputs_models:
                completion = await client.beta.chat.completions.parse(  # type: ignore
                    model=self.model_name,
                    messages=messages,
                    response_format=schema,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                return completion.choices[0].message.parsed

            if self.actual_model_name in json_mode_models:
                completion = await client.beta.chat.completions.parse(  # type: ignore
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                json_output = trim_and_load_json(completion.choices[0].message.content)
                return schema.model_validate(json_output)

        completion = await client.chat.completions.create(  # type: ignore
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        output = completion.choices[0].message.content or ""

        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output)

        return str(completion.choices[0].message.content)

    def load_model(self, async_mode: bool = False) -> Union[OpenAI, AsyncOpenAI]:
        """Load the OpenAI client in sync or async mode.

        Args:
            async_mode: Whether to load the async client

        Returns:
            OpenAI client instance
        """
        if not async_mode:
            return OpenAI(api_key=self._openai_api_key, base_url=self.base_url)
        return AsyncOpenAI(api_key=self._openai_api_key, base_url=self.base_url)
