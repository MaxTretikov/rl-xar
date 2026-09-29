"""Chat model client abstractions."""

from __future__ import annotations

import os
from typing import Protocol


class ChatClient(Protocol):
    """Small interface for synchronous system/user chat requests."""

    def complete(self, system: str, user: str) -> str:
        """Return the assistant response for a system and user message."""
        ...


class LiteLLMClient:
    """Chat client backed by LiteLLM's provider-agnostic completion API.

    The optional API key is looked up on every call so callers can configure or
    rotate credentials after constructing the client. If ``api_key_env`` is
    omitted, LiteLLM uses its normal provider-specific environment variables.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLM is required for LiteLLMClient; install the project's llm dependency."
            ) from exc

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if self.base_url is not None:
            kwargs["api_base"] = self.base_url
        if self.api_key_env is not None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key: set environment variable {self.api_key_env}."
                )
            kwargs["api_key"] = api_key

        try:
            response = completion(**kwargs)
        except Exception as exc:
            # SDK exceptions may include request headers or credential values.
            # Report only the exception type and setup hints, never its message.
            detail = type(exc).__name__
            raise RuntimeError(
                f"LiteLLM request failed ({detail}) for model {self.model!r}; "
                "check the model name, provider credentials, and base_url."
            ) from None

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "LiteLLM returned an unexpected response; expected a first choice with message content."
            ) from exc
        if not isinstance(content, str):
            raise RuntimeError(
                "LiteLLM returned no text content; expected a string response."
            )
        return content
