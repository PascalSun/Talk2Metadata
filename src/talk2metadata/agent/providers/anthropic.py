"""Anthropic/Claude LLM provider."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore

from talk2metadata.agent.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic/Claude provider implementation."""

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        """Initialize Anthropic provider.

        Args:
            model: Model name (defaults to claude-sonnet-4-5-20250929)
            **kwargs: Configuration options including:
                - api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
                - base_url: Optional custom API base URL
                - temperature: Default temperature
                - max_tokens: Default max tokens
        """
        if Anthropic is None:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        # Default model
        model = model or "claude-sonnet-4-5-20250929"

        # Extract client-specific kwargs
        allowed_client_keys = {"api_key", "base_url", "timeout", "max_retries"}
        client_kwargs = {k: v for k, v in kwargs.items() if k in allowed_client_keys}

        # Use env var as fallback for API key
        if "api_key" not in client_kwargs:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                client_kwargs["api_key"] = api_key

        # Initialize client
        self.client = Anthropic(**client_kwargs)

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
        messages = [{"role": "user", "content": prompt}]

        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            system_prompt=system_prompt,
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
            **kwargs: Additional parameters including system_prompt

        Returns:
            LLMResponse object
        """
        # Extract system_prompt from kwargs
        system_prompt = kwargs.pop("system_prompt", None)

        # Merge config defaults with call-time parameters
        default_keys = {
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "stop_sequences",
        }
        merged_kwargs = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }

        # Call-time parameters override defaults
        if temperature is not None:
            merged_kwargs["temperature"] = temperature
        if max_tokens is not None:
            merged_kwargs["max_tokens"] = max_tokens

        # Set default max_tokens if not specified
        if "max_tokens" not in merged_kwargs:
            merged_kwargs["max_tokens"] = 4096

        # Handle JSON mode via prompt engineering
        if response_format == "json":
            if system_prompt:
                system_prompt = f"{system_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text."
            else:
                system_prompt = (
                    "You must respond with valid JSON only, no additional text."
                )

        # Build system blocks
        system_blocks = None
        if system_prompt:
            system_blocks = [{"type": "text", "text": system_prompt}]

        # Prepare create kwargs
        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **merged_kwargs,
        }
        if system_blocks:
            create_kwargs["system"] = system_blocks

        # Call API
        try:
            response = self.client.messages.create(**create_kwargs)

            # Extract content
            content = response.content[0].text if response.content else ""

            # Build usage dict
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            }

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                metadata={"stop_reason": response.stop_reason},
            )

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
