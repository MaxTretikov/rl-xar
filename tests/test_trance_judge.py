from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from rlxar.trance_judge import TranceChatClient, discover_judges


@dataclass
class FakeCandidate:
    model: object = object()
    provider: str = ""
    model_name: str = ""
    auth_kind: str = ""
    source: str = ""

    def to_pydantic_ai(self) -> object:
        return self.model


def test_discover_judges_delegates_to_local_trance_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    found = [FakeCandidate()]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def scan(*args: object, **kwargs: object) -> list[FakeCandidate]:
        calls.append((args, kwargs))
        return found

    monkeypatch.setitem(sys.modules, "trance", types.SimpleNamespace(scan=scan))

    assert discover_judges() == found
    assert calls == [((), {})]


def test_client_does_not_request_until_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []

    class FakeAgent:
        def __init__(self, model: object, *, system_prompt: str) -> None:
            events.append(("construct", model, system_prompt))

        def run_sync(self, user: str) -> object:
            events.append(("request", user))
            return types.SimpleNamespace(output="judge response")

    monkeypatch.setitem(sys.modules, "pydantic_ai", types.SimpleNamespace(Agent=FakeAgent))
    model = object()
    candidate = FakeCandidate(model)
    client = TranceChatClient(candidate)
    assert events == []
    assert client.selection == {
        "provider": "",
        "model_name": "",
        "auth_kind": "",
        "source": "",
    }

    assert client.complete("system", "user") == "judge response"
    assert events == [("construct", model, "system"), ("request", "user")]


def test_client_caches_adapter_model_but_recreates_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []

    class FakeCandidateWithAdapter(FakeCandidate):
        def to_pydantic_ai(self) -> object:
            events.append("adapt")
            return self.model

    class FakeAgent:
        def __init__(self, model: object, *, system_prompt: str) -> None:
            events.append(("construct", model, system_prompt))

        def run_sync(self, user: str) -> object:
            events.append(("request", user))
            return types.SimpleNamespace(output=user)

    monkeypatch.setitem(sys.modules, "pydantic_ai", types.SimpleNamespace(Agent=FakeAgent))
    model = object()
    client = TranceChatClient(FakeCandidateWithAdapter(model))

    assert client.complete("system one", "user one") == "user one"
    assert client.complete("system two", "user two") == "user two"
    assert events == [
        "adapt",
        ("construct", model, "system one"),
        ("request", "user one"),
        ("construct", model, "system two"),
        ("request", "user two"),
    ]


def test_adapter_errors_are_sanitized() -> None:
    secret = "api-key-secret"

    @dataclass
    class FailingCandidate(FakeCandidate):
        def to_pydantic_ai(self) -> object:
            raise ValueError(f"adapter leaked {secret}")

    with pytest.raises(RuntimeError, match=r"Judge adapter failed \(ValueError\)") as error:
        TranceChatClient(FailingCandidate()).complete("system", "user")
    assert secret not in str(error.value)


def test_request_errors_are_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "api-key-secret"

    class FakeAgent:
        def __init__(self, model: object, *, system_prompt: str) -> None:
            pass

        def run_sync(self, user: str) -> object:
            raise ValueError(f"request leaked {secret}")

    monkeypatch.setitem(sys.modules, "pydantic_ai", types.SimpleNamespace(Agent=FakeAgent))
    with pytest.raises(RuntimeError, match=r"Judge request failed \(ValueError\)") as error:
        TranceChatClient(FakeCandidate(object())).complete("system", "user")
    assert secret not in str(error.value)


def test_client_rejects_non_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        def __init__(self, model: object, *, system_prompt: str) -> None:
            pass

        def run_sync(self, user: str) -> object:
            return types.SimpleNamespace(output={"score": 1})

    monkeypatch.setitem(sys.modules, "pydantic_ai", types.SimpleNamespace(Agent=FakeAgent))
    with pytest.raises(RuntimeError, match="Judge returned non-text output"):
        TranceChatClient(FakeCandidate(object())).complete("system", "user")
