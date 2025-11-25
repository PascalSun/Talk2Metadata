"""OpenAI LLM provider."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from talk2metadata.agent.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        """Initialize OpenAI provider.

        Args:
            model: Model name (defaults to gpt-4)
            **kwargs: Configuration options including:
                - api_key: OpenAI API key (or OPENAI_API_KEY env var)
                - base_url: Optional custom API base URL
                - organization: Optional organization ID
                - temperature: Default temperature
                - max_tokens: Default max tokens
        """
        if OpenAI is None:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        # Default model
        model = model or "gpt-4"

        # Extract client-specific kwargs
        allowed_client_keys = {
            "api_key",
            "base_url",
            "organization",
            "project",
            "timeout",
            "http_client",
        }
        client_kwargs = {k: v for k, v in kwargs.items() if k in allowed_client_keys}

        # Use env var as fallback for API key
        if "api_key" not in client_kwargs:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                client_kwargs["api_key"] = api_key

        # Initialize client
        self.client = OpenAI(**client_kwargs)

        # Store config (non-client kwargs become defaults)
        config_kwargs = {
            k: v for k, v in kwargs.items() if k not in allowed_client_keys
        }

        super().__init__(model=model, **config_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" or None)
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Multi-turn chat completion.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" or None)
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        # Check if model requires max_completion_tokens instead of max_tokens
        # Models that require max_completion_tokens:
        # - o1 series: o1, o1-preview, o1-mini
        # - gpt-5 series: gpt-5.1, etc. (newer models)
        requires_max_completion_tokens = False
        if self.model:
            requires_max_completion_tokens = (
                self.model.startswith("o1")
                or self.model.startswith("gpt-5")
                or self.model.startswith("gpt-4.5")  # Future-proofing
            )

        # Merge config defaults with call-time parameters
        # Note: We exclude max_tokens and max_completion_tokens from initial merge
        # to handle them separately based on model requirements
        default_keys = {
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "logit_bias",
            "seed",
        }
        merged_kwargs = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }

        # Call-time parameters override defaults
        if temperature is not None:
            merged_kwargs["temperature"] = temperature

        # Handle max_tokens vs max_completion_tokens based on model
        # Priority: call-time max_tokens > config max_tokens > config max_completion_tokens
        effective_max_tokens = max_tokens
        if effective_max_tokens is None:
            # Check config for max_tokens or max_completion_tokens
            config_max_tokens = self.config.get("max_tokens")
            config_max_completion_tokens = self.config.get("max_completion_tokens")
            if config_max_tokens is not None:
                effective_max_tokens = config_max_tokens
            elif config_max_completion_tokens is not None:
                effective_max_tokens = config_max_completion_tokens

        # Convert to the correct parameter based on model
        if effective_max_tokens is not None:
            if requires_max_completion_tokens:
                # Use max_completion_tokens for models that require it
                merged_kwargs["max_completion_tokens"] = effective_max_tokens
                # Ensure max_tokens is not present
                merged_kwargs.pop("max_tokens", None)
            else:
                # Use max_tokens for other models
                merged_kwargs["max_tokens"] = effective_max_tokens
                # Ensure max_completion_tokens is not present
                merged_kwargs.pop("max_completion_tokens", None)

        # Handle JSON mode
        if response_format == "json":
            merged_kwargs["response_format"] = {"type": "json_object"}

        # Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **merged_kwargs
            )

            # Extract content
            content = response.choices[0].message.content or ""

            # Build usage dict
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            error_str = str(e)
            # Fallback: Check if error is about max_tokens not being supported
            # This handles cases where we didn't detect the model correctly
            if (
                "max_tokens" in error_str
                and "not supported" in error_str.lower()
                and "max_completion_tokens" in error_str.lower()
                and "max_tokens" in merged_kwargs
                and not requires_max_completion_tokens  # Only retry if we didn't already switch
            ):
                # Retry with max_completion_tokens instead
                self.logger.warning(
                    f"Model {self.model} requires max_completion_tokens instead of max_tokens. Retrying..."
                )
                retry_kwargs = merged_kwargs.copy()
                if "max_tokens" in retry_kwargs:
                    retry_kwargs["max_completion_tokens"] = retry_kwargs.pop(
                        "max_tokens"
                    )

                try:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, **retry_kwargs
                    )

                    content = response.choices[0].message.content or ""
                    usage = {}
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }

                    return LLMResponse(
                        content=content,
                        model=response.model,
                        usage=usage,
                        metadata={"finish_reason": response.choices[0].finish_reason},
                    )
                except Exception as retry_error:
                    self.logger.error(f"OpenAI API error (retry failed): {retry_error}")
                    raise

            self.logger.error(f"OpenAI API error: {e}")
            raise
