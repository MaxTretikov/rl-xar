"""Loading and prompt construction for RL-XAR examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schema import Example

_REQUIRED_SPLITS = frozenset({"train", "validation", "test"})


def load_examples(path: Path) -> list[Example]:
    """Load and validate a JSONL dataset of examples.

    IDs must be globally unique, and all examples in a group must belong to
    the same split. A complete dataset must contain each required split.
    Errors identify the source line whenever they concern a particular row.
    """
    examples: list[Example] = []
    ids: set[str] = set()
    group_splits: dict[str, str] = {}

    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot open examples file {path}: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row: Any = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            try:
                example = Example.model_validate(row)
            except ValidationError as exc:
                details = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc']) or '<row>'}: {error['msg']}"
                    for error in exc.errors()
                )
                raise ValueError(f"{path}:{line_number}: invalid example: {details}") from exc

            if example.id in ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate example id {example.id!r}"
                )
            prior_split = group_splits.get(example.group_id)
            if prior_split is not None and prior_split != example.split:
                raise ValueError(
                    f"{path}:{line_number}: group_id {example.group_id!r} appears in "
                    f"both {prior_split!r} and {example.split!r} splits"
                )

            ids.add(example.id)
            group_splits[example.group_id] = example.split
            examples.append(example)

    present_splits = {example.split for example in examples}
    missing_splits = sorted(_REQUIRED_SPLITS - present_splits)
    if missing_splits:
        raise ValueError(
            f"{path}: dataset is missing required split(s): {', '.join(missing_splits)}"
        )
    return examples


def writer_prompt(example: Example) -> str:
    """Build a writer prompt from the example's context only."""
    return example.context
