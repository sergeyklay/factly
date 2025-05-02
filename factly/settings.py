"""
Settings module for Factly CLI.

Defines configuration models for API, inference, and overall application settings.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """
    Configuration for the LLM API connection and model selection.

    Attributes:
        api_base (str): Base URL for the model API endpoint.
        model (str): Model name or identifier (e.g., 'gpt-4o').
        api_key (Optional[str]): API key for authenticating with the model provider.
            Set to None for local models that don't require authentication.
    """

    api_base: str = Field(default="https://api.openai.com/v1")
    model: str = Field(default="gpt-4o")
    api_key: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPENAI_",
        extra="ignore",
    )

    @classmethod
    def create(cls, **kwargs) -> "ModelSettings":
        """
        Create a ModelSettings instance with optional overrides.

        Args:
            **kwargs: Override default settings values.

        Returns:
            ModelSettings: A settings instance.
        """
        return cls(**kwargs)


class InferenceSettings(BaseSettings):
    """
    Inference-time parameters for LLM decoding, following MMLU best practices.

    Attributes:
        temperature (float): Sampling temperature. Default set to 0.0 to ensure
            deterministic, reproducible outputs by disabling sampling randomness.
        top_p (float): Nucleus sampling parameter. Controls how much of the probability
            mass the model is allowed to sample from. Default set to 1.0 to disable
            nucleus sampling, guaranteeing the model always selects the most probable
            token.
        max_tokens (int): Maximum tokens to generate. Default set to 256 to allow
            sufficient space for model reasoning. For standard MMLU, you typically want
            just 1 token (A/B/C/D answers), but setting max_tokens: 1 will break
            benchmarks if your prompts expect structured outputs (e.g., JSON) or
            encourage reasoning before answering. With higher max_tokens, you may need
            to post-process results to extract final answers.
        n_shots (int): Number of examples for few-shot learning. Default set to 0 for
            zero-shot evaluation. Increasing this value provides more demonstration
            examples in prompts to help the model understand the task format.
    """

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=256, gt=0)
    n_shots: int = Field(default=0, ge=0)

    @classmethod
    def create(cls, **kwargs) -> "InferenceSettings":
        """
        Create an InferenceSettings instance with optional overrides.

        Args:
            **kwargs: Override default settings values.

        Returns:
            InferenceSettings: A settings instance.
        """
        return cls(**kwargs)

    @classmethod
    def for_mmlu(cls) -> "InferenceSettings":
        """
        Create inference settings configured for traditional MMLU benchmarking.

        Uses max_tokens=1 for single-letter answers, which is the canonical setup for
        standard MMLU evaluation where only a single token (A/B/C/D) is expected.

        Returns:
            InferenceSettings: MMLU-optimized settings
            (temperature=0, top_p=1, max_tokens=1).
        """
        return cls(temperature=0.0, top_p=1.0, max_tokens=1)


class FactlySettings(BaseSettings):
    """
    Aggregated settings for the Factly CLI, including model and inference configuration.

    Attributes:
        model (ModelSettings): Model API and authentication settings.
        inference (InferenceSettings): Inference-time decoding parameters.
    """

    model: ModelSettings = Field(default_factory=lambda: ModelSettings())
    inference: InferenceSettings = Field(
        default_factory=lambda: InferenceSettings.for_mmlu()
    )

    @classmethod
    def create(cls, **kwargs) -> "FactlySettings":
        """
        Create FactlySettings with optional overrides.

        This factory method handles nested configuration with dictionaries.

        Args:
            **kwargs: Configuration overrides including nested dictionaries
                     for model and inference settings.

        Returns:
            FactlySettings: A settings instance.

        Example:
            >>> # Create with API key and custom temperature
            >>> settings = FactlySettings.create(
            ...     model={"api_key": "sk-abc123", "model": "gpt-4o"},
            ...     inference={"temperature": 0.1}
            ... )
            >>>
            >>> # Alternatively, update settings after creation:
            >>> settings = FactlySettings()
            >>> settings.model.api_key = "sk-abc123"
            >>> settings.inference.temperature = 0.1
        """
        return cls(**kwargs)

    @classmethod
    def from_cli(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        n_shots: int | None = None,
    ) -> "FactlySettings":
        """
        Create settings by combining CLI arguments with environment variables.

        CLI arguments take precedence over environment variables and defaults.
        Only non-None CLI values will override settings from the environment.

        Args:
            model: Model name (e.g., "gpt-4o")
            api_key: API key for the model provider
            api_base: Base URL for the API
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            n_shots: Number of examples for few-shot learning

        Returns:
            FactlySettings: Combined settings with proper priority
        """
        inference_kwargs = {}
        if temperature is not None:
            inference_kwargs["temperature"] = temperature
        if top_p is not None:
            inference_kwargs["top_p"] = top_p
        if max_tokens is not None:
            inference_kwargs["max_tokens"] = max_tokens
        if n_shots is not None:
            inference_kwargs["n_shots"] = n_shots

        _inference = InferenceSettings.create(**inference_kwargs)

        model_kwargs = {}
        if api_key is not None:
            model_kwargs["api_key"] = api_key
        if api_base is not None:
            model_kwargs["api_base"] = api_base
        if model is not None:
            model_kwargs["model"] = model

        _model = ModelSettings.create(**model_kwargs)

        # First load settings from environment in the following order:
        settings = cls(
            model=_model,
            inference=_inference,
        )

        return settings
