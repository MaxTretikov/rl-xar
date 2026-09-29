"""Offline tests for Trance discovery and the CLI's explicit consent gate."""

from __future__ import annotations

import sys
import types
from pathlib import Path

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


def test_run_displays_candidates_and_requires_both_noninteractive_flags(
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
        ["run", "--config", str(_config(tmp_path)), "--judge-index", "1"],
    )

    assert result.exit_code == 2
    assert "provider=openai" in result.output
    assert "model=judge" in result.output
    assert "auth=env" in result.output
    assert "source=trance" in result.output
    assert "--approve-judge" in result.output
    assert calls == []


def test_denied_judge_never_starts_pipeline(monkeypatch, tmp_path: Path) -> None:
    candidates = [{"provider": "openai", "model_name": "judge", "auth_kind": "env", "source": "trance"}]
    _install_trance(monkeypatch, candidates)
    calls: list[object] = []
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    pipeline = types.SimpleNamespace(run_pipeline=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "rlxar.pipeline", pipeline)
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)

    result = CliRunner().invoke(
        cli.app,
        ["run", "--config", str(_config(tmp_path)), "--judge-index", "1"],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "not approved" in result.output
    assert calls == []


def test_noninteractive_approval_passes_exact_selected_client_to_pipeline(
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

    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--config",
            str(_config(tmp_path)),
            "--judge-index",
            "2",
            "--approve-judge",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == ("config",)
    client = calls[0][1]["judge_client"]
    assert isinstance(client, _FakeClient)
    assert client.candidate is second


def test_invalid_judge_index_never_starts_pipeline(monkeypatch, tmp_path: Path) -> None:
    candidates = [{"provider": "openai", "model_name": "judge", "auth_kind": "env", "source": "trance"}]
    _install_trance(monkeypatch, candidates)
    calls: list[object] = []
    pipeline = types.SimpleNamespace(run_pipeline=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "rlxar.pipeline", pipeline)

    result = CliRunner().invoke(
        cli.app,
        ["run", "--config", str(_config(tmp_path)), "--judge-index", "2", "--approve-judge"],
    )

    assert result.exit_code == 2
    assert "between 1 and 1" in result.output
    assert calls == []


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
