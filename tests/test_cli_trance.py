"""Offline tests for Trance discovery and the CLI's explicit consent gate."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rlxar import cli


class _FakeClient:
    def __init__(self, candidate: object) -> None:
        self.candidate = candidate


def _install_trance(monkeypatch, candidates: list[object]) -> None:
    module = types.ModuleType("rlxar.trance_judge")
    module.discover_judges = lambda: candidates
    module.TranceChatClient = _FakeClient
    monkeypatch.setitem(sys.modules, "rlxar.trance_judge", module)


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(
        'dataset_path = "examples.jsonl"\nwriter_model_id = "writer"\n',
        encoding="utf-8",
    )
    (tmp_path / "examples.jsonl").write_text("", encoding="utf-8")
    return path


def test_noninteractive_run_requires_model_provider(
    monkeypatch, tmp_path: Path
) -> None:
    candidates = [{"provider": "openai", "model_name": "judge", "auth_kind": "env", "source": "trance"}]
    _install_trance(monkeypatch, candidates)
    calls: list[object] = []
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    pipeline = types.SimpleNamespace(run_pipeline=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "rlxar.pipeline", pipeline)

    result = CliRunner().invoke(
        cli.app,
        ["run", "--config", str(_config(tmp_path))],
    )

    assert result.exit_code == 2
    assert "provider=openai" in result.output
    assert "model=judge" in result.output
    assert "auth=env" in result.output
    assert "source=trance" in result.output
    assert "--model-provider" in result.output
    assert calls == []


def test_denied_interactive_judge_never_starts_pipeline(monkeypatch, tmp_path: Path) -> None:
    candidates = [{"provider": "openai", "model_name": "judge", "auth_kind": "env", "source": "trance"}]
    _install_trance(monkeypatch, candidates)
    calls: list[object] = []
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    pipeline = types.SimpleNamespace(run_pipeline=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "rlxar.pipeline", pipeline)
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)

    result = CliRunner().invoke(
        cli.app,
        ["run", "--config", str(_config(tmp_path))],
        input="1\nn\n",
    )

    assert result.exit_code == 0
    assert "not approved" in result.output
    assert calls == []


def test_model_provider_selects_exact_client_without_prompt(
    monkeypatch, tmp_path: Path
) -> None:
    first = {"provider": "openai", "model_name": "first", "auth_kind": "env", "source": "one"}
    second = {"provider": "anthropic", "model_name": "second", "auth_kind": "keychain", "source": "two"}
    _install_trance(monkeypatch, [first, second])
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(cli, "load_config", lambda path: "config")
    pipeline = types.SimpleNamespace(
        run_pipeline=lambda *args, **kwargs: calls.append((args, kwargs)) or Path("run-dir")
    )
    monkeypatch.setitem(sys.modules, "rlxar.pipeline", pipeline)

    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    monkeypatch.setattr(cli.typer, "prompt", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompted")))
    monkeypatch.setattr(cli.typer, "confirm", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompted")))
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--config",
            str(_config(tmp_path)),
            "--model-provider",
            "anthropic",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == ("config",)
    client = calls[0][1]["judge_client"]
    assert isinstance(client, _FakeClient)
    assert client.candidate is second


def test_model_provider_no_match_never_starts_pipeline(monkeypatch, tmp_path: Path) -> None:
    candidates = [{"provider": "openai", "model_name": "judge", "auth_kind": "env", "source": "trance"}]
    _install_trance(monkeypatch, candidates)
    calls: list[object] = []
    pipeline = types.SimpleNamespace(run_pipeline=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "rlxar.pipeline", pipeline)

    result = CliRunner().invoke(
        cli.app,
        ["run", "--config", str(_config(tmp_path)), "--model-provider", "anthropic"],
    )

    assert result.exit_code == 2
    assert "matches provider 'anthropic'" in result.output
    assert calls == []


@pytest.mark.parametrize(
    ("requested", "provider"),
    [("codex", "openai-codex"), ("grok", "grok-consumer")],
)
def test_model_provider_alias_selects_matching_provider(
    monkeypatch, tmp_path: Path, requested: str, provider: str
) -> None:
    candidate = {"provider": provider, "model_name": "judge", "auth_kind": "env", "source": "trance"}
    _install_trance(monkeypatch, [candidate])
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "load_config", lambda path: "config")
    monkeypatch.setitem(
        sys.modules,
        "rlxar.pipeline",
        types.SimpleNamespace(run_pipeline=lambda *args, **kwargs: calls.append(kwargs) or Path("run-dir")),
    )

    result = CliRunner().invoke(
        cli.app,
        ["run", "--config", str(_config(tmp_path)), "--model-provider", requested],
    )

    assert result.exit_code == 0
    assert "Choose one only" not in result.output
    assert calls[0]["judge_client"].candidate is candidate


def test_model_provider_uses_first_matching_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    first = {"provider": "OpenAI", "model_name": "first", "auth_kind": "env", "source": "one"}
    second = {"provider": "openai", "model_name": "second", "auth_kind": "env", "source": "two"}
    _install_trance(monkeypatch, [first, second])
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "load_config", lambda path: "config")
    monkeypatch.setitem(
        sys.modules,
        "rlxar.pipeline",
        types.SimpleNamespace(run_pipeline=lambda *args, **kwargs: calls.append(kwargs) or Path("run-dir")),
    )

    result = CliRunner().invoke(
        cli.app,
        ["run", "--config", str(_config(tmp_path)), "--model-provider", "OPENAI"],
    )

    assert result.exit_code == 0
    assert calls[0]["judge_client"].candidate is first


def test_no_discovered_judges_never_starts_pipeline(monkeypatch, tmp_path: Path) -> None:
    _install_trance(monkeypatch, [])
    calls: list[object] = []
    pipeline = types.SimpleNamespace(run_pipeline=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "rlxar.pipeline", pipeline)

    result = CliRunner().invoke(cli.app, ["run", "--config", str(_config(tmp_path))])

    assert result.exit_code == 1
    assert "found no configured chat interfaces" in result.output
    assert calls == []


def test_judges_is_read_only_and_lists_discovery(monkeypatch) -> None:
    _install_trance(
        monkeypatch,
        [{"provider": "anthropic", "model_name": "claude", "auth_kind": "keychain", "source": "trance"}],
    )

    result = CliRunner().invoke(cli.app, ["judges"])

    assert result.exit_code == 0
    assert "anthropic" in result.output
    assert "claude" in result.output
    assert "No model calls have been made" in result.output
