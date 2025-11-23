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
        # Merge config defaults with call-time parameters
        default_keys = {
            "temperature",
            "max_tokens",
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
        if max_tokens is not None:
            merged_kwargs["max_tokens"] = max_tokens

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
            self.logger.error(f"OpenAI API error: {e}")
            raise
