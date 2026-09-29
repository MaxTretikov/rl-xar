"""Offline integration coverage for the full outer-loop orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rlxar import artifacts
from rlxar.config import RunConfig
from rlxar.pipeline import run_pipeline
from rlxar.schema import Criterion, Example, Rubric


def _write_dataset(path: Path) -> dict[str, list[Example]]:
    rows = {
        "train": [
            Example(id="train-1", group_id="g-train-1", context="train context one", expert="train expert one", split="train"),
            Example(id="train-2", group_id="g-train-2", context="train context two", expert="train expert two", split="train"),
        ],
        "validation": [
            Example(id="validation-1", group_id="g-validation", context="validation context", expert="validation expert", split="validation"),
        ],
        "test": [
            Example(id="test-1", group_id="g-test", context="held out context", expert="held out expert", split="test"),
        ],
    }
    path.write_text(
        "".join(example.model_dump_json() + "\n" for split in rows.values() for example in split),
        encoding="utf-8",
    )
    return rows


def test_run_pipeline_orchestrates_rounds_without_split_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    dataset_path = repo / "examples.jsonl"
    splits = _write_dataset(dataset_path)
    runs_root = tmp_path / "archive" / "runs"
    # The sandbox mounts an empty /tmp/.git marker, which the production
    # directory walk mistakes for a repository. Keep this test scoped to the
    # actual fixture checkout while preserving the external-path assertion.
    monkeypatch.setattr(
        artifacts,
        "_git_checkout",
        lambda path: repo if Path(path).is_relative_to(repo) else None,
    )
    config = RunConfig(
        dataset_path=dataset_path,
        output_dir=runs_root,
        writer_model_id="fake/writer",
        judge_provider="litellm",
        judge_model="fake/judge",
        judge_base_url=None,
        judge_api_key_env="UNUSED_KEY",
        rubric_iterations=1,
        outer_rounds=2,
        grpo_steps=1,
        generations=2,
        seed=17,
    )

    calls: dict[str, list[Any]] = {
        "rollout": [], "optimizer": [], "rubric": [], "reward": [],
        "train": [], "evaluate": [], "clients": [],
    }

    def client_factory(model: str, **kwargs: Any) -> object:
        calls["clients"].append((model, kwargs))
        return object()

    def judge_factory(client: object) -> object:
        return {"client": client}

    def rollout(model: str, examples: list[Example], *, adapter_path: Path | None, seed: int) -> dict[str, str]:
        calls["rollout"].append((model, [row.id for row in examples], adapter_path, seed))
        # The injected writer receives only prompts/examples and never an expert answer.
        assert all("expert" not in row.context for row in examples)
        continuations = {row.id: f"response to: {row.context}" for row in examples}
        assert all(row.expert not in continuations[row.id] for row in examples)
        return continuations

    def optimizer(client: object, judge: object, train_pairs: list[tuple[Example, str]],
                  validation_pairs: list[tuple[Example, str]], initial_prompt: str,
                  *, iterations: int) -> dict[str, Any]:
        calls["optimizer"].append((train_pairs, validation_pairs, initial_prompt, iterations))
        return {
            "best_prompt": f"meta-{len(calls['optimizer'])}",
            "best_validation_gap": 0.25,
            "history": [{"iteration": len(calls["optimizer"]), "path": Path("trace.json")}],
        }

    def generate_rubric(client: object, meta_prompt: str, context: str, *, rubric_id: str) -> Rubric:
        calls["rubric"].append((meta_prompt, context, rubric_id))
        return Rubric(
            id=rubric_id,
            criteria=[Criterion(id="quality", instruction="Assess quality", weight=1.0)],
            provenance=f"for context: {context}",
        )

    def make_reward(judge: object, *, rubric_column: str) -> str:
        calls["reward"].append((judge, rubric_column))
        return f"reward-{len(calls['reward'])}"

    def train_grpo(model: str, records: list[dict[str, Any]], reward: str, output: Path,
                   *, steps: int, generations: int, seed: int,
                   adapter_path: Path | None) -> Path:
        calls["train"].append((model, records, reward, output, steps, generations, seed, adapter_path))
        output.mkdir(parents=True)
        result = output.parent / f"trained-adapter-{len(calls['train'])}"
        result.mkdir()
        return result

    def evaluate_policy(client: object, judge: object, meta_prompts: list[str],
                        examples: list[Example], outputs: dict[str, str]) -> dict[str, Any]:
        calls["evaluate"].append((meta_prompts, examples, outputs))
        return {"heldout_ids": [row.id for row in examples], "score": 0.75}

    run_dir = run_pipeline(config, dependencies={
        "rollout": rollout,
        "optimizer": optimizer,
        "generate_rubric": generate_rubric,
        "make_reward": make_reward,
        "train_grpo": train_grpo,
        "evaluate_policy": evaluate_policy,
        "client_factory": client_factory,
        "judge_factory": judge_factory,
    })

    # Train and validation data are the only inputs to meta-prompt optimization.
    assert len(calls["optimizer"]) == 2
    for train_pairs, validation_pairs, _, _ in calls["optimizer"]:
        assert [example.split for example, _ in train_pairs] == ["train", "train"]
        assert [example.split for example, _ in validation_pairs] == ["validation"]
        assert all("expert" not in continuation for _, continuation in train_pairs + validation_pairs)

    # Each round uses the prior round's adapter for both rollout and training.
    assert len(calls["train"]) == 2
    assert calls["train"][0][-1] is None
    expected_prior = run_dir / "round-001" / "trained-adapter-1"
    assert calls["train"][1][-1] == expected_prior
    train_rollouts = [call for call in calls["rollout"] if call[1] == ["train-1", "train-2"]]
    assert [call[2] for call in train_rollouts] == [None, expected_prior]
    assert [call[2] for call in calls["rollout"]] == [
        None, None, expected_prior, expected_prior,
        run_dir / "round-002" / "trained-adapter-2",
    ]
    assert calls["optimizer"][1][2] == "meta-1"

    # Rubrics are generated from the exact example context represented in each record.
    assert len(calls["rubric"]) == 4
    for round_index, train_call in enumerate(calls["train"], start=1):
        records = train_call[1]
        assert [record["context"] for record in records] == [row.context for row in splits["train"]]
        for record, source in zip(records, splits["train"], strict=True):
            assert record["prompt"] == [{"role": "user", "content": source.context}]
            rubric = json.loads(record["rubric_json"])
            assert rubric["provenance"] == f"for context: {source.context}"
            assert source.expert not in json.dumps(record)

    # The only evaluation rows and rollout outputs are held-out test examples.
    assert len(calls["evaluate"]) == 1
    meta_prompts, test_examples, test_outputs = calls["evaluate"][0]
    assert meta_prompts == ["meta-1", "meta-2"]
    assert [row.id for row in test_examples] == ["test-1"]
    assert set(test_outputs) == {"test-1"}
    assert [call[1] for call in calls["rollout"] if call[1] == ["test-1"]] == [["test-1"]]
    final_adapter = run_dir / "round-002" / "trained-adapter-2"
    assert next(call[2] for call in calls["rollout"] if call[1] == ["test-1"]) == final_adapter

    # All run outputs are outside the repository and metadata is JSON serializable.
    assert run_dir.is_relative_to(runs_root)
    assert not run_dir.is_relative_to(repo)
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["split_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert metadata["outer_rounds"][0]["adapter_path"] == str(expected_prior)
    assert json.loads((run_dir / "round-001" / "meta_prompt.json").read_text())["history"] == [
        {"iteration": 1, "path": "trace.json"}
    ]
    assert json.loads((run_dir / "test_evaluation.json").read_text()) == {
        "heldout_ids": ["test-1"], "score": 0.75,
    }
    assert (run_dir / "manifest.json").is_file()
