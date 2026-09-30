"""Pydantic AI judge clients backed by locally discovered Trance candidates.

Discovery is deliberately separate from use. :func:`discover_judges` delegates
to ``trance.scan`` and does not send a model inference request. A
:class:`TranceChatClient` creates an agent without running it, and sends a
request only when its ``complete`` method is called by an explicitly selected
judge.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from .client import ChatClient

if TYPE_CHECKING:
    from trance import Candidate


def discover_judges() -> list[Candidate]:
    """Return candidates found in local chat-interface configuration.

    Trance's scan does not perform model inference. It may inspect configured
    chat interfaces and their local authentication state. Importing Trance
    here keeps it optional until discovery is requested.
    """
    try:
        from trance import scan
    except ImportError as exc:
        raise RuntimeError(
            "Trance discovery is unavailable; install the trance dependency."
        ) from exc
    try:
        return list(scan())
    except Exception as exc:  # noqa: BLE001
        # Scanner details can contain paths, provider metadata, or other local
        # configuration. Expose only a stable type-level diagnostic.
        raise RuntimeError(f"Trance discovery failed ({type(exc).__name__}).") from None


class TranceChatClient(ChatClient):
    """Use a user-selected Trance candidate through Pydantic AI.

    Constructing this client performs no inference. The Pydantic AI agent is
    created lazily as well, so a caller can display discovered models and ask
    for consent before any provider adapter is initialized.
    """

    def __init__(self, candidate: Candidate) -> None:
        self._candidate = candidate
        self._model: object = _UNSET
        self._selection = MappingProxyType(
            {
                "provider": candidate.provider,
                "model_name": candidate.model_name,
                "auth_kind": candidate.auth_kind,
                "source": candidate.source,
            }
        )

    @property
    def selection(self) -> Mapping[str, str]:
        """Safe metadata for displaying the approved model selection.

        This mapping contains identifiers only; the discovered model object
        and any credential material are intentionally excluded.
        """
        return self._selection

    def complete(self, system: str, user: str) -> str:
        """Send one system/user request and return its text output."""
        if self._model is _UNSET:
            try:
                self._model = self._candidate.to_pydantic_ai()
            except Exception as exc:  # noqa: BLE001
                # Adapter errors can include URLs, headers, or credentials.
                raise RuntimeError(f"Judge adapter failed ({type(exc).__name__}).") from None

        try:
            from pydantic_ai import Agent
        except ImportError:
            raise RuntimeError(
                "Pydantic AI is required for TranceChatClient; install it in the "
                "application environment."
            ) from None
        try:
            agent = Agent(self._model, system_prompt=system)
            result = agent.run_sync(user)
        except Exception as exc:  # noqa: BLE001
            # Provider and adapter errors can include URLs, headers, prompts,
            # or secrets.
            raise RuntimeError(f"Judge request failed ({type(exc).__name__}).") from None

        try:
            output = result.output
        except AttributeError:
            raise RuntimeError("Judge returned an unexpected response.") from None
        if not isinstance(output, str):
            raise RuntimeError(  # noqa: TRY004
                "Judge returned non-text output; expected a string response."
            )
        return output


_UNSET = object()
