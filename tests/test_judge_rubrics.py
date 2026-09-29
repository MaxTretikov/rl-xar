"""Offline tests for rubric generation and rubric-based judgment."""

from __future__ import annotations

import json

import pytest

from rlxar.rubrics import generate_rubric, revise_meta_prompt
from rlxar.schema import Rubric
from rlxar.scoring import RubricJudge


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


def make_rubric() -> Rubric:
    return Rubric.model_validate(
        {
            "id": "quality-v1",
            "criteria": [
                {"id": "accuracy", "instruction": "Is it accurate?", "weight": 0.7},
                {"id": "clarity", "instruction": "Is it clear?", "weight": 0.3},
            ],
            "provenance": "test fixture",
        }
    )


def test_rubric_judge_computes_weighted_total_locally() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "criterion_scores": {"accuracy": 0.8, "clarity": 0.5},
                "rationale": "The answer is accurate and mostly clear.",
            }
        )
    )

    judgment = RubricJudge(client).score("task context", "candidate answer", make_rubric())

    assert judgment.criterion_scores == {"accuracy": 0.8, "clarity": 0.5}
    assert judgment.total == pytest.approx(0.71)
    sent = json.loads(client.calls[0][1])
    assert "total" not in sent


@pytest.mark.parametrize(
    "scores",
    [
        {"accuracy": 0.8},  # missing a criterion
        {"accuracy": 0.8, "clarity": 0.5, "extra": 0.2},  # unknown criterion
        {"accuracy": "0.8", "clarity": 0.5},  # non-numeric
        {"accuracy": True, "clarity": 0.5},  # bool is not a score
        {"accuracy": 1.01, "clarity": 0.5},  # above the score range
        {"accuracy": -0.01, "clarity": 0.5},  # below the score range
        {"accuracy": float("nan"), "clarity": 0.5},  # non-finite
    ],
)
def test_rubric_judge_rejects_invalid_criterion_scores(scores: dict[str, object]) -> None:
    response = json.dumps({"criterion_scores": scores, "rationale": "reason"})

    with pytest.raises(ValueError):
        RubricJudge(FakeChatClient(response)).score("context", "candidate", make_rubric())


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        json.dumps({"criterion_scores": {"accuracy": 0.5, "clarity": 0.5}}),
        json.dumps(
            {
                "criterion_scores": {"accuracy": 0.5, "clarity": 0.5},
                "rationale": None,
                "total": 0.5,
            }
        ),
    ],
)
def test_rubric_judge_rejects_malformed_response(response: str) -> None:
    with pytest.raises(ValueError):
        RubricJudge(FakeChatClient(response)).score("context", "candidate", make_rubric())


def valid_generated_rubric(rubric_id: str = "generated-v1") -> str:
    return json.dumps(
        {
            "id": rubric_id,
            "criteria": [
                {"id": "coverage", "instruction": "Covers the requested points.", "weight": 0.5},
                {"id": "correctness", "instruction": "Uses correct facts.", "weight": 0.3},
                {"id": "reasoning", "instruction": "Explains the reasoning.", "weight": 0.2},
            ],
            "provenance": "Generated from task context.",
        }
    )


def test_generate_rubric_uses_context_and_never_sends_reference_answer() -> None:
    client = FakeChatClient(valid_generated_rubric())
    context = "Explain how photosynthesis converts light into stored energy."
    secret_reference = "REFERENCE ANSWER: chlorophyll captures photons in chloroplasts."

    rubric = generate_rubric(
        client,
        "Write a rubric for the task.",
        context,
        rubric_id="generated-v1",
    )

    assert rubric.id == "generated-v1"
    assert isinstance(rubric, Rubric)
    assert len(client.calls) == 1
    system, user = client.calls[0]
    assert context in user
    assert "context" in system.lower()
    assert secret_reference not in system + user
    assert "expert answer" in system.lower() or "examples" in system.lower()


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        json.dumps({"id": "generated-v1", "criteria": [], "provenance": "none"}),
        json.dumps(
            {
                "id": "generated-v1",
                "criteria": [
                    {"id": "only", "instruction": "One criterion.", "weight": 1.0}
                ],
                "provenance": "none",
            }
        ),
    ],
)
def test_generate_rubric_rejects_invalid_json_or_criterion_set(response: str) -> None:
    with pytest.raises(ValueError):
        generate_rubric(FakeChatClient(response), "Make rubric", "Task context", rubric_id="generated-v1")


def test_generate_rubric_rejects_wrong_rubric_id() -> None:
    with pytest.raises(ValueError, match="does not match"):
        generate_rubric(
            FakeChatClient(valid_generated_rubric("wrong-id")),
            "Make rubric",
            "Task context",
            rubric_id="generated-v1",
        )


def test_revise_meta_prompt_enforces_response_length_bound() -> None:
    client = FakeChatClient("x" * 21)

    with pytest.raises(ValueError, match="character limit"):
        revise_meta_prompt(client, "Current instructions", 0.2, ["Missed a required point."], max_chars=20)


def test_revise_meta_prompt_accepts_response_within_length_bound() -> None:
    client = FakeChatClient("Use clear and observable criteria.")

    revised = revise_meta_prompt(client, "Current instructions", 0.2, ["Missed a required point."], max_chars=40)

    assert revised == "Use clear and observable criteria."
    assert len(revised) <= 40
