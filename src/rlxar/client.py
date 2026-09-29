"""Chat model client abstractions."""

from __future__ import annotations

from typing import Protocol


class ChatClient(Protocol):
    """Small interface for synchronous system/user chat requests."""

    def complete(self, system: str, user: str) -> str:
        """Return the assistant response for a system and user message."""
        ...
