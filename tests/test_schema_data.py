"""Validation and leakage guards for examples, rubrics, and judgments."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rlxar.data import load_examples, writer_prompt
from rlxar.rubrics import generate_rubric
from rlxar.schema import Example, Judgment, Rubric


def _example(example_id: str, group_id: str, split: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": example_id,
        "group_id": group_id,
        "context": f"Task context for {example_id}",
        "expert": f"Private expert answer for {example_id}",
        "split": split,
    }
    row.update(overrides)
    return row


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _complete_splits() -> list[dict[str, object]]:
    return [
        _example("train-1", "train-group", "train"),
        _example("validation-1", "validation-group", "validation"),
        _example("test-1", "test-group", "test"),
    ]


def test_example_rejects_missing_or_blank_task_fields() -> None:
    row = _example("e1", "g1", "train")
    row["expert"] = "   "

    with pytest.raises(ValidationError):
        Example.model_validate(row)


def test_rubric_rejects_duplicate_criteria_and_unbalanced_weights() -> None:
    duplicate = {
        "id": "r1",
        "criteria": [
            {"id": "accuracy", "instruction": "Check accuracy", "weight": 0.5},
            {"id": "accuracy", "instruction": "Check coverage", "weight": 0.5},
        ],
        "provenance": "task context",
    }
    unbalanced = {
        "id": "r2",
        "criteria": [
            {"id": "a", "instruction": "Check A", "weight": 0.4},
            {"id": "b", "instruction": "Check B", "weight": 0.4},
        ],
        "provenance": "task context",
    }

    with pytest.raises(ValidationError, match="unique"):
        Rubric.model_validate(duplicate)
    with pytest.raises(ValidationError, match="sum to 1"):
        Rubric.model_validate(unbalanced)


def test_judgment_rejects_out_of_range_scores_and_blank_rationale() -> None:
    with pytest.raises(ValidationError):
        Judgment(criterion_scores={"accuracy": 1.1}, total=0.8, rationale="supported")
    with pytest.raises(ValidationError):
        Judgment(criterion_scores={"accuracy": 0.8}, total=0.8, rationale="  ")


def test_load_examples_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = _complete_splits()
    rows.append(_example("train-1", "another-group", "test"))
    path = tmp_path / "examples.jsonl"
    _write_dataset(path, rows)

    with pytest.raises(ValueError, match=r"examples\.jsonl:4: duplicate example id 'train-1'"):
        load_examples(path)


def test_load_examples_rejects_group_split_leakage(tmp_path: Path) -> None:
    rows = _complete_splits()
    rows.append(_example("same-group-copy", "train-group", "test"))
    path = tmp_path / "examples.jsonl"
    _write_dataset(path, rows)

    with pytest.raises(ValueError, match="appears in both 'train' and 'test' splits"):
        load_examples(path)


def test_writer_prompt_omits_expert_answer() -> None:
    example = Example.model_validate(_example("e1", "g1", "train"))

    prompt = writer_prompt(example)

    assert prompt == example.context
    assert example.expert not in prompt


class _RecordingClient:
    def __init__(self) -> None:
        self.messages: tuple[str, str] | None = None

    def complete(self, system: str, user: str) -> str:
        self.messages = (system, user)
        return json.dumps(
            {
                "id": "r1",
                "criteria": [
                    {"id": "a", "instruction": "Check A", "weight": 0.34},
                    {"id": "b", "instruction": "Check B", "weight": 0.33},
                    {"id": "c", "instruction": "Check C", "weight": 0.33},
                ],
                "provenance": "task context",
            }
        )


def test_rubric_generation_prompt_never_receives_expert_answer() -> None:
    row = _example(
        "e1", "g1", "train", expert="CANARY expert answer that must not reach the rubric model"
    )
    example = Example.model_validate(row)
    client = _RecordingClient()

    generate_rubric(
        client, "Write a useful rubric", example.context, rubric_id="r1"
    )

    assert client.messages is not None
    system, user = client.messages
    assert example.expert not in system
    assert example.expert not in user
    assert example.context in user
