"""Held-out evaluation of policy continuations against expert responses."""

from __future__ import annotations

import math
from collections.abc import Sequence

from . import rubrics
from .client import ChatClient
from .schema import Example, Judgment
from .scoring import Judge


def evaluate_policy(
    client: ChatClient,
    judge: Judge,
    meta_prompts: list[str],
    examples: list[Example],
    continuations: dict[str, str],
) -> dict[str, object]:
    """Evaluate held-out continuations against expert answers.

    A fresh rubric is generated from each example's context for every
    meta-prompt. The caller chooses validation or test data; training examples
    are rejected to guard against accidental leakage.
    """
    if not examples:
        raise ValueError("held-out evaluation requires at least one example")
    if any(example.split == "train" for example in examples):
        raise ValueError("evaluation examples must be from validation or test")
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation example IDs must be unique")
    missing = sorted(set(ids) - continuations.keys())
    if missing:
        raise ValueError(f"missing policy continuation(s) for example IDs: {missing}")
    if not meta_prompts:
        raise ValueError("at least one meta-prompt is required")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in meta_prompts):
        raise ValueError("meta-prompts must be nonempty strings")

    rubric_sets: list[dict[str, object]] = []
    all_gaps: list[float] = []
    for prompt_index, meta_prompt in enumerate(meta_prompts):
        details: list[dict[str, object]] = []
        normalized_scores: list[float] = []
        for example in examples:
            rubric = rubrics.generate_rubric(
                client,
                meta_prompt,
                example.context,
                rubric_id=f"meta-{prompt_index + 1}-example-{example.id}",
            )
            expert_judgment = judge.score(example.context, example.expert, rubric)
            policy_judgment = judge.score(
                example.context, continuations[example.id], rubric
            )
            expert_score = _finite_score(expert_judgment, example.id, "expert")
            policy_score = _finite_score(policy_judgment, example.id, "policy")
            gap = expert_score - policy_score
            all_gaps.append(gap)
            normalized = _human_normalized(policy_score, expert_score)
            if normalized is not None:
                normalized_scores.append(normalized)
            details.append(
                {
                    "rubric_id": rubric.id,
                    "rubric": rubric.model_dump(mode="json"),
                    "example_id": example.id,
                    "expert_score": expert_score,
                    "policy_score": policy_score,
                    "expert_gap": gap,
                    "human_normalized_policy_score": normalized,
                    "expert_judgment": expert_judgment.model_dump(mode="json"),
                    "policy_judgment": policy_judgment.model_dump(mode="json"),
                }
            )
        rubric_sets.append(
            {
                "rubric_set_id": f"meta-{prompt_index + 1}",
                "meta_prompt": meta_prompt,
                "mean_expert_gap": _mean([float(row["expert_gap"]) for row in details]),
                "mean_human_normalized_policy_score": (
                    _mean(normalized_scores) if normalized_scores else None
                ),
                "worst_human_normalized_policy_score": (
                    min(normalized_scores) if normalized_scores else None
                ),
                "examples": details,
            }
        )

    set_scores = [
        float(score)
        for result in rubric_sets
        if (score := result["mean_human_normalized_policy_score"]) is not None
    ]
    return {
        "example_count": len(examples),
        "rubric_count": len(meta_prompts) * len(examples),
        "mean_expert_gap": _mean(all_gaps),
        "mean_human_normalized_policy_score": _mean(set_scores) if set_scores else None,
        "worst_human_normalized_policy_score": min(set_scores) if set_scores else None,
        "rubric_sets": rubric_sets,
    }


def _finite_score(judgment: Judgment, example_id: str, kind: str) -> float:
    score = float(judgment.total)
    if not math.isfinite(score):
        raise ValueError(f"{kind} judgment for {example_id!r} has a non-finite total")
    return score


def _mean(values: Sequence[float]) -> float:
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise ValueError("evaluation produced a non-finite mean")
    return result


def _human_normalized(policy_score: float, expert_score: float) -> float | None:
    """Return policy/expert; a positive score over a zero expert is undefined."""
    if expert_score == 0:
        return 1.0 if policy_score == 0 else None
    result = policy_score / expert_score
    if not math.isfinite(result):
        raise ValueError("evaluation produced a non-finite normalized score")
    return result
