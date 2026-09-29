"""Offline regression tests for prompt selection and held-out evaluation."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from rlxar import evaluation, optimizer, rubrics
from rlxar.schema import Example, Judgment, Rubric


class FakeClient:
    """Return deterministic rubrics and revised prompts without network calls."""

    def __init__(self, revised_prompts: list[str] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.revised_prompts = iter(revised_prompts or [])

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if system.startswith("You improve rubric-generation instructions"):
            return next(self.revised_prompts)

        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            payload = {}
        rubric_id = payload.get("rubric_id")
        if rubric_id is None:
            # Both rubric generators include the requested ID in their user data.
            import re

            match = re.search(r"Requested rubric ID: ([^\n]+)", user)
            if match is None:
                raise AssertionError(f"No rubric ID in request: {user}")
            rubric_id = match.group(1)
        if "context_untrusted_data" in payload:
            return json.dumps(
                {
                    "criteria": [
                        {"id": f"c{i}", "instruction": f"assess {i}", "weight": 1 / 3}
                        for i in range(3)
                    ]
                }
            )
        return json.dumps(
            {
                "id": rubric_id,
                "criteria": [
                    {"id": f"c{i}", "instruction": f"assess {i}", "weight": 1 / 3}
                    for i in range(3)
                ],
                "provenance": "fake context-only rubric",
            }
        )


def example(example_id: str, split: str, *, context: str = "task context") -> Example:
    return Example(
        id=example_id,
        group_id=f"group-{example_id}",
        context=context,
        expert=f"expert answer {example_id}",
        split=split,
    )


def judgment(total: float) -> Judgment:
    return Judgment(criterion_scores={"c0": total}, total=total)


class OptimizationJudge:
    """Make validation gaps differ from both training gaps and the last gap."""

    validation_gaps: ClassVar[dict[int, float]] = {0: 0.2, 1: 0.7, 2: 0.4}

    def score(self, context: str, candidate: str, rubric: Rubric) -> Judgment:
        if rubric.id.startswith("validation-"):
            iteration = int(rubric.id.split("-")[1])
            gap = self.validation_gaps[iteration]
            return judgment(0.9 if candidate.startswith("expert") else 0.9 - gap)
        return judgment(0.1 if candidate.startswith("expert") else 0.9)


def test_optimizer_keeps_prompt_with_best_validation_gap() -> None:
    client = FakeClient(["revised-1", "revised-2"])
    train = [(example("train-1", "train"), "weak train answer")]
    validation = [(example("validation-1", "validation"), "weak validation answer")]

    result = optimizer.optimize_meta_prompt(
        client,
        OptimizationJudge(),
        train,
        validation,
        "initial",
        iterations=2,
    )

    assert [item.prompt for item in result.history] == ["initial", "revised-1", "revised-2"]
    assert [item.validation_gap for item in result.history] == pytest.approx([0.2, 0.7, 0.4])
    assert result.best_prompt == "revised-1"
    assert result.best_validation_gap == pytest.approx(0.7)


def test_rubric_generation_request_contains_context_but_no_answers() -> None:
    client = FakeClient()
    context = "Please explain how photosynthesis works."
    rubric = rubrics.generate_rubric(client, "Assess correctness and completeness.", context, rubric_id="r1")

    assert rubric.id == "r1"
    assert len(rubric.criteria) == 3
    request = "\n".join(system + "\n" + user for system, user in client.calls)
    assert context in request
    assert "expert answer x" not in request
    assert "candidate answer x" not in request
    assert "context-only" in rubric.provenance


class EvaluationJudge:
    scores: ClassVar[dict[tuple[str, str], float]] = {
        ("meta-1-example-a", "expert answer a"): 1.0,
        ("meta-1-example-a", "policy a"): 0.5,
        ("meta-1-example-b", "expert answer b"): 0.0,
        ("meta-1-example-b", "policy b"): 0.0,
        ("meta-2-example-a", "expert answer a"): 0.8,
        ("meta-2-example-a", "policy a"): 0.4,
        ("meta-2-example-b", "expert answer b"): 0.0,
        ("meta-2-example-b", "policy b"): 0.2,
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def score(self, context: str, candidate: str, rubric: Rubric) -> Judgment:
        self.calls.append((context, candidate, rubric.id))
        return judgment(self.scores[(rubric.id, candidate)])


def test_held_out_evaluation_scores_both_answers_on_same_rubric_and_reports_worst_set() -> None:
    client = FakeClient()
    judge = EvaluationJudge()
    examples = [example("a", "test", context="context a"), example("b", "test", context="context b")]

    result = evaluation.evaluate_policy(
        client,
        judge,
        ["meta one", "meta two"],
        examples,
        {"a": "policy a", "b": "policy b"},
    )

    assert len(judge.calls) == 8
    for rubric_id in {call[2] for call in judge.calls}:
        paired = [call for call in judge.calls if call[2] == rubric_id]
        assert len(paired) == 2
        assert paired[0][0] == paired[1][0]
        assert {paired[0][1], paired[1][1]} in (
            {"expert answer a", "policy a"},
            {"expert answer b", "policy b"},
        )
    assert result["rubric_count"] == 4
    assert result["mean_human_normalized_policy_score"] == pytest.approx(0.625)
    assert result["worst_human_normalized_policy_score"] == pytest.approx(0.5)
    first_set = result["rubric_sets"][0]
    assert first_set["examples"][1]["human_normalized_policy_score"] == 1.0
    second_set = result["rubric_sets"][1]
    assert second_set["examples"][1]["human_normalized_policy_score"] is None
    assert second_set["mean_human_normalized_policy_score"] == pytest.approx(0.5)
