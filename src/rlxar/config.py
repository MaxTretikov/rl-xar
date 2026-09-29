"""Run configuration loading and validation for RL-XAR."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ARCHIVE_OUTPUT_DIR = Path("/mnt/archive/runs/rl-xar")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Immutable settings for one reproducible training run.

    Judge configuration is deliberately absent. The runtime supplies an
    already-approved chat client after presenting any discovered interfaces to
    the user.
    """

    dataset_path: Path
    output_dir: Path
    writer_model_id: str
    rubric_iterations: int
    outer_rounds: int
    grpo_steps: int
    generations: int
    seed: int


_ALLOWED_KEYS = frozenset(RunConfig.__dataclass_fields__)


def load_config(path: Path) -> RunConfig:
    """Load a strict TOML run config and resolve relative paths beside it."""
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            values: dict[str, Any] = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML configuration {config_path}: {exc}") from exc

    unknown = set(values) - _ALLOWED_KEYS
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration key(s): {keys}")

    required = {
        "dataset_path",
        "writer_model_id",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Missing required configuration key(s): {', '.join(sorted(missing))}")

    base_dir = config_path.resolve().parent
    dataset_path = _path_value(values["dataset_path"], "dataset_path", base_dir)
    output_dir = _path_value(
        values.get("output_dir", str(_default_output_dir())), "output_dir", base_dir
    )

    writer_model_id = _nonempty_string(values["writer_model_id"], "writer_model_id")
    return RunConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        writer_model_id=writer_model_id,
        rubric_iterations=_positive_int(values.get("rubric_iterations", 3), "rubric_iterations"),
        outer_rounds=_positive_int(values.get("outer_rounds", 3), "outer_rounds"),
        grpo_steps=_positive_int(values.get("grpo_steps", 100), "grpo_steps"),
        generations=_positive_int(values.get("generations", 4), "generations"),
        seed=_nonnegative_int(values.get("seed", 42), "seed"),
    )


def _path_value(value: object, name: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _default_output_dir() -> Path:
    """Choose an output root suited to the current host."""
    override = os.environ.get("RL_XAR_OUTPUT_DIR")
    if override is not None:
        return Path(override).expanduser()
    archive_root = _ARCHIVE_OUTPUT_DIR.parents[1]
    if archive_root.is_dir() and os.access(archive_root, os.W_OK):
        return _ARCHIVE_OUTPUT_DIR
    data_home = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    return data_home / "rl-xar" / "runs"


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value
