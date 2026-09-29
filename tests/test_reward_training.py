"""Offline tests for reward batching and GRPO training wiring."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlxar.reward import DemoJudge, make_grpo_reward
from rlxar.schema import Judgment, Rubric


def rubric(rubric_id: str = "quality") -> Rubric:
    return Rubric.model_validate(
        {
            "id": rubric_id,
            "criteria": [
                {"id": "coverage", "instruction": "Cover the task.", "weight": 0.75},
                {"id": "clarity", "instruction": "Be clear.", "weight": 0.25},
            ],
            "provenance": "test fixture",
        }
    )


class RecordingJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Rubric]] = []

    def score(self, context: str, candidate: str, used_rubric: Rubric) -> Judgment:
        self.calls.append((context, candidate, used_rubric))
        value = 0.2 if context == "first context" else 0.8
        return Judgment(
            criterion_scores={"coverage": value, "clarity": value},
            total=value,
            rationale="fixture",
        )


def test_grpo_reward_maps_chat_completions_to_aligned_rubrics_and_contexts() -> None:
    judge = RecordingJudge()
    reward = make_grpo_reward(judge)
    rubrics = [rubric("first"), rubric("second")]
    completions = [
        [{"role": "assistant", "content": "first response"}],
        [{"role": "assistant", "content": "second response"}],
    ]

    scores = reward(
        prompts=[
            [{"role": "user", "content": "first prompt"}],
            [{"role": "user", "content": "second prompt"}],
        ],
        completions=completions,
        context=["first context", "second context"],
        rubric_json=[item.model_dump_json() for item in rubrics],
    )

    assert scores == [0.2, 0.8]
    assert judge.calls == [
        ("first context", "first response", rubrics[0]),
        ("second context", "second response", rubrics[1]),
    ]


def test_grpo_reward_accepts_rubric_objects_and_uses_prompt_as_context_fallback() -> None:
    judge = RecordingJudge()

    scores = make_grpo_reward(judge)(
        prompts=[[{"role": "user", "content": "prompt context"}]],
        completions=[[{"role": "assistant", "content": "candidate"}]],
        rubric_json=[rubric()],
    )

    assert scores == [0.8]
    assert judge.calls[0][0:2] == ("prompt context", "candidate")


@pytest.mark.parametrize(
    "bad_rubric",
    [
        "not json",
        json.dumps({"id": "x", "criteria": [], "provenance": "empty"}),
        json.dumps(
            {
                "id": "x",
                "criteria": [
                    {"id": "a", "instruction": "A", "weight": 0.4},
                    {"id": "b", "instruction": "B", "weight": 0.4},
                ],
                "provenance": "weights do not sum to one",
            }
        ),
    ],
)
def test_grpo_reward_rejects_malformed_rubrics_with_row_index(bad_rubric: str) -> None:
    judge = RecordingJudge()

    with pytest.raises(ValueError, match="batch row 0"):
        make_grpo_reward(judge)(
            prompts=["prompt"], completions=["candidate"], rubric_json=[bad_rubric]
        )
    assert judge.calls == []


@pytest.mark.parametrize("total", [float("nan"), float("inf"), -0.01, 1.01])
def test_grpo_reward_rejects_nonfinite_or_out_of_range_judge_total(total: float) -> None:
    class InvalidJudge:
        def score(self, context: str, candidate: str, used_rubric: Rubric) -> Judgment:
            # Bypass model validation to verify the reward boundary itself.
            return Judgment.model_construct(
                criterion_scores={"coverage": 0.5, "clarity": 0.5}, total=total
            )

    with pytest.raises(ValueError, match=r"score|total|finite|\[0, 1\]"):
        make_grpo_reward(InvalidJudge())(
            prompts=["prompt"], completions=["candidate"], rubric_json=[rubric()]
        )


def test_demo_judge_produces_a_nonconstant_signal() -> None:
    demo = DemoJudge()
    task = rubric()

    empty = demo.score("photosynthesis chlorophyll light", "", task)
    relevant = demo.score(
        "photosynthesis chlorophyll light", "Photosynthesis uses light and chlorophyll.", task
    )

    assert 0 <= empty.total < relevant.total <= 1
    assert empty.total != relevant.total


def test_train_grpo_wires_dataset_and_trl_without_loading_real_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rlxar import training

    events: list[tuple[str, object]] = []
    fake_tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=2,
        pad_token=None,
        chat_template="fixture template",
        save_pretrained=lambda path: events.append(("tokenizer_save", path)),
    )

    class FakeDataset:
        @classmethod
        def from_list(cls, records: list[dict[str, object]]) -> FakeDataset:
            instance = cls()
            instance.records = records
            return instance

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            events.append(("trainer_init", kwargs))
            self.kwargs = kwargs

        def train(self) -> None:
            events.append(("train", None))

        def save_model(self, path: str) -> None:
            events.append(("model_save", path))

    monkeypatch.setattr(training, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda model_id: fake_tokenizer))
    monkeypatch.setattr(training, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda model_id: "fake-model"))
    monkeypatch.setattr(training, "Dataset", FakeDataset)
    monkeypatch.setattr(training, "LoraConfig", lambda **kwargs: ("lora", kwargs))
    monkeypatch.setattr(training, "GRPOConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(training, "GRPOTrainer", FakeTrainer)
    def reward(**kwargs: object) -> list[float]:
        return [0.5]

    output = tmp_path / "adapter"

    result = training.train_grpo(
        "fixture-model",
        [
            {
                "prompt": "first prompt",
                "context": "first context",
                "rubric_json": rubric("r1").model_dump_json(),
            }
        ],
        reward,
        output,
        use_cpu=True,
    )

    trainer_kwargs = next(value for name, value in events if name == "trainer_init")
    assert result == output
    assert trainer_kwargs["train_dataset"].records[0]["prompt"] == [
        {"role": "user", "content": "first prompt"}
    ]
    assert trainer_kwargs["train_dataset"].records[0]["context"] == "first context"
    assert trainer_kwargs["train_dataset"].records[0]["rubric_json"] == rubric("r1").model_dump_json()
    assert trainer_kwargs["reward_funcs"] is reward
    assert trainer_kwargs["args"].use_cpu is True
    assert [name for name, _ in events][-3:] == ["train", "model_save", "tokenizer_save"]
