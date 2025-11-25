"""Google Gemini provider implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

from talk2metadata.agent.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Gemini provider.

        Args:
            model: Model name (e.g., 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro')
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            **kwargs: Additional Gemini configuration
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package is required. Install with: pip install google-generativeai"
            )

        # Persist generation defaults in provider config
        super().__init__(model, **kwargs)
        # Try to get API key from parameter (from config) first, then fallback to env var
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set it in config.yml (agent.gemini.api_key) or set GOOGLE_API_KEY env var."
            )

        genai.configure(api_key=api_key)
        # Do not pass generation kwargs into model constructor; store them in self.config
        self.client = genai.GenerativeModel(model_name=self.model)

    def _build_generation_config(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs: Any,
    ) -> Optional[genai.types.GenerationConfig]:
        """Build generation config for Gemini API."""
        default_keys = {"temperature", "max_output_tokens", "top_p", "top_k"}
        generation_config = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }
        if temperature is not None:
            generation_config["temperature"] = temperature
        generation_config["max_output_tokens"] = max_tokens or generation_config.get(
            "max_output_tokens"
        )
        # Filter out None values from kwargs
        generation_config.update({k: v for k, v in kwargs.items() if v is not None})

        return (
            genai.types.GenerationConfig(**generation_config)
            if generation_config
            else None
        )

    def _extract_usage_metadata(self, response: Any) -> Dict[str, int]:
        """Extract usage metadata from Gemini response."""
        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
                "total_tokens": getattr(
                    response.usage_metadata, "total_token_count", 0
                ),
            }
        return usage

    def _extract_response_content(self, response: Any) -> tuple[str, Optional[int]]:
        """Extract content and finish reason from Gemini response."""
        finish_reason = None
        content = ""
        if not response.candidates:
            return content, finish_reason

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)

        # finish_reason 2 = SAFETY (content filtered)
        if finish_reason in (2, 3, 4):
            try:
                content = response.text
            except ValueError:
                content = "[Content filtered by safety filters]"
                self.logger.warning(f"Gemini response was filtered: {finish_reason}")
        else:
            try:
                content = response.text
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Failed to extract text: {e}")
                content = "[Failed to extract response content]"

        return content, finish_reason

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using Gemini API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Gemini JSON mode via prompt engineering
        if response_format == "json":
            full_prompt = f"{full_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

        generation_config = self._build_generation_config(
            temperature, max_tokens, **kwargs
        )

        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config,
        )

        usage = self._extract_usage_metadata(response)
        content, finish_reason = self._extract_response_content(response)

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            metadata={"finish_reason": finish_reason},
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with message history."""
        # Gemini uses a chat history format
        chat = self.client.start_chat(history=[])

        # Build conversation history
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat.send_message(msg["content"])

        # Send the last message
        last_message = messages[-1]["content"]
        # Gemini JSON mode via prompt engineering
        if response_format == "json":
            last_message = f"{last_message}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

        generation_config = self._build_generation_config(
            temperature, max_tokens, **kwargs
        )

        response = chat.send_message(
            last_message,
            generation_config=generation_config,
        )

        usage = self._extract_usage_metadata(response)
        content, finish_reason = self._extract_response_content(response)

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            metadata={"finish_reason": finish_reason},
        )
