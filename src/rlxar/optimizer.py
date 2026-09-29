"""Train and select meta-prompts for context-grounded rubric evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .client import ChatClient
from .rubrics import DEFAULT_META_PROMPT, generate_rubric, revise_meta_prompt
from .schema import Example, Judgment, Rubric
from .scoring import Judge


@dataclass(frozen=True, slots=True)
class TrainingFailure:
    """A training case where the rubric failed to favor the expert."""

    example_id: str
    context: str
    expert: str
    candidate: str
    rubric: Rubric
    expert_judgment: Judgment
    candidate_judgment: Judgment


@dataclass(frozen=True, slots=True)
class PromptIteration:
    """Validation measurements for one evaluated meta-prompt."""

    iteration: int
    prompt: str
    validation_gap: float
    mean_expert_score: float
    mean_candidate_score: float
    train_gap: float
    training_failures: int


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """The best prompt found and the complete validation history."""

    best_prompt: str
    best_validation_gap: float
    history: list[PromptIteration]


def optimize_meta_prompt(
    client: ChatClient,
    judge: Judge,
    train_pairs: Sequence[tuple[Example, str]],
    validation_pairs: Sequence[tuple[Example, str]],
    initial_meta_prompt: str = DEFAULT_META_PROMPT,
    *,
    iterations: int = 3,
) -> OptimizationResult:
    """Iteratively revise a rubric meta-prompt and keep its best validation gap.

    Each pair contains an ``Example`` (whose ``expert`` response is the reference)
    and a model continuation. A rubric is generated once from each pair's context
    and shared by the expert and candidate score calls. The validation gap is the
    mean expert score minus mean candidate score. Positive gaps mean that the
    rubric favors the expert over the current model.
    """
    if not initial_meta_prompt.strip():
        raise ValueError("initial_meta_prompt must be nonempty")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not train_pairs or not validation_pairs:
        raise ValueError("train_pairs and validation_pairs must both be nonempty")
    _validate_pairs(train_pairs, expected_split="train", name="train_pairs")
    _validate_pairs(validation_pairs, expected_split="validation", name="validation_pairs")

    history: list[PromptIteration] = []
    best_prompt = initial_meta_prompt
    best_gap = float("-inf")
    prompt = initial_meta_prompt

    for iteration in range(iterations + 1):
        failures, train_gap = _training_evaluation(
            client, judge, prompt, train_pairs, iteration
        )
        validation_expert, validation_candidate = _score_pairs(
            client, judge, prompt, validation_pairs, "validation", iteration
        )
        expert_mean = sum(validation_expert) / len(validation_expert)
        candidate_mean = sum(validation_candidate) / len(validation_candidate)
        gap = expert_mean - candidate_mean
        history.append(
            PromptIteration(
                iteration=iteration,
                prompt=prompt,
                validation_gap=gap,
                mean_expert_score=expert_mean,
                mean_candidate_score=candidate_mean,
                train_gap=train_gap,
                training_failures=len(failures),
            )
        )
        if gap > best_gap:
            best_gap = gap
            best_prompt = prompt

        if iteration < iterations and failures:
            prompt = revise_meta_prompt(
                client,
                prompt,
                train_gap,
                _failure_descriptions(failures),
            )

    return OptimizationResult(
        best_prompt=best_prompt,
        best_validation_gap=best_gap,
        history=history,
    )


def _training_evaluation(
    client: ChatClient,
    judge: Judge,
    meta_prompt: str,
    pairs: Sequence[tuple[Example, str]],
    iteration: int,
) -> tuple[list[TrainingFailure], float]:
    failures: list[TrainingFailure] = []
    expert_scores: list[float] = []
    candidate_scores: list[float] = []
    for example, candidate in pairs:
        rubric = generate_rubric(
            client,
            meta_prompt,
            example.context,
            rubric_id=f"train-{iteration}-{example.id}",
        )
        expert_judgment = judge.score(example.context, example.expert, rubric)
        candidate_judgment = judge.score(example.context, candidate, rubric)
        expert_scores.append(expert_judgment.total)
        candidate_scores.append(candidate_judgment.total)
        if candidate_judgment.total >= expert_judgment.total - 1e-6:
            failures.append(
                TrainingFailure(
                    example_id=example.id,
                    context=example.context,
                    expert=example.expert,
                    candidate=candidate,
                    rubric=rubric,
                    expert_judgment=expert_judgment,
                    candidate_judgment=candidate_judgment,
                )
            )
    train_gap = sum(expert_scores) / len(expert_scores) - sum(candidate_scores) / len(
        candidate_scores
    )
    return failures, train_gap


def _failure_descriptions(failures: Sequence[TrainingFailure]) -> list[str]:
    """Describe rubric misses with answer excerpts and observed scores."""
    descriptions: list[str] = []
    for failure in failures:
        criterion_scores = [
            {
                "criterion": criterion.id,
                "instruction": criterion.instruction,
                "expert_score": failure.expert_judgment.criterion_scores.get(criterion.id),
                "candidate_score": failure.candidate_judgment.criterion_scores.get(
                    criterion.id
                ),
            }
            for criterion in failure.rubric.criteria
        ]
        descriptions.append(
            f"Training example {failure.example_id}: rubric did not favor the expert. "
            f"Expert total={failure.expert_judgment.total:.4f}; "
            f"candidate total={failure.candidate_judgment.total:.4f}. "
            f"Criterion scores: {criterion_scores!r}. "
            f"Context excerpt: {failure.context[:500]!r}. "
            f"Expert response excerpt: {failure.expert[:500]!r}. "
            f"Candidate response excerpt: {failure.candidate[:500]!r}. "
            f"Expert rationale: {failure.expert_judgment.rationale!r}. "
            f"Candidate rationale: {failure.candidate_judgment.rationale!r}."
        )
    return descriptions


def _score_pairs(
    client: ChatClient,
    judge: Judge,
    meta_prompt: str,
    pairs: Sequence[tuple[Example, str]],
    split: str,
    iteration: int,
) -> tuple[list[float], list[float]]:
    expert_scores: list[float] = []
    candidate_scores: list[float] = []
    for example, candidate in pairs:
        rubric = generate_rubric(
            client,
            meta_prompt,
            example.context,
            rubric_id=f"{split}-{iteration}-{example.id}",
        )
        expert_scores.append(judge.score(example.context, example.expert, rubric).total)
        candidate_scores.append(judge.score(example.context, candidate, rubric).total)
    return expert_scores, candidate_scores


def _validate_pairs(
    pairs: Sequence[tuple[Example, str]], *, expected_split: str, name: str
) -> None:
    seen: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{name} entries must be (Example, continuation) pairs")
        example, continuation = pair
        if example.split != expected_split:
            raise ValueError(
                f"{name} example {example.id!r} must have split {expected_split!r}"
            )
        if example.id in seen:
            raise ValueError(f"{name} contains duplicate example id {example.id!r}")
        if not isinstance(continuation, str) or not continuation.strip():
            raise ValueError(f"{name} continuation for {example.id!r} must be nonempty")
        seen.add(example.id)
