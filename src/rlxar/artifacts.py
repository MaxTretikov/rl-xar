"""Safe, reproducible storage helpers for RL-XAR run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_RUNS_ROOT = Path("/mnt/archive/runs/rl-xar")


def _json_value(value: Any) -> Any:
    """Convert common structured values to JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _encoded_json(value: Any, *, pretty: bool = True) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "allow_nan": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(_json_value(value), **options) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def write_json(path: str | Path, value: Any) -> Path:
    """Atomically write one canonical, human-readable JSON document."""
    return _atomic_write(Path(path), _encoded_json(value))


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    """Atomically write JSON Lines, one JSON-compatible value per line."""
    content = b"".join(_encoded_json(row, pretty=False) for row in rows)
    return _atomic_write(Path(path), content)


def sha256_json(value: Any) -> str:
    """Return SHA-256 for canonical JSON, independent of mapping key order."""
    payload = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_checkout(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def create_run_dir(
    name: str,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    provenance: str,
    commands: str | Iterable[str] = (),
    checkpoint_paths: Iterable[str | Path] = (),
    rubric_hash: str | None = None,
    config_hash: str | None = None,
) -> Path:
    """Create an external run directory and its initial provenance manifest.

    ``runs_root`` defaults to ``/mnt/archive/runs/rl-xar``. The generated
    directory is named ``<name>-<UTC timestamp>``; names cannot contain paths.
    """
    if not name or name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("name must contain only letters, digits, dot, underscore, or hyphen")
    if not provenance.strip():
        raise ValueError("provenance must be nonempty")

    root = Path(runs_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkout = _git_checkout(root)
    if checkout is not None:
        raise ValueError(f"run storage must be outside a Git checkout ({checkout})")

    now = datetime.now(UTC)
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{name}-{now.strftime('%Y%m%dT%H%M%SZ')}-",
            dir=root,
        )
    )
    if isinstance(commands, str):
        command_list = [commands] if commands else []
    else:
        command_list = list(commands)
    manifest = {
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "name": name,
        "provenance": provenance,
        "commands": command_list,
        "checkpoint_paths": [str(Path(path)) for path in checkpoint_paths],
        "rubric_sha256": rubric_hash,
        "config_sha256": config_hash,
    }
    try:
        write_json(run_dir / "manifest.json", manifest)
    except BaseException:
        run_dir.rmdir()
        raise
    return run_dir
