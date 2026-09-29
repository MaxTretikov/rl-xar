"""TRL-compatible reward functions for rubric-guided optimization."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .schema import Judgment, Rubric


class Judge(Protocol):
    """Object that scores a candidate response against a rubric."""

    def score(self, context: str, candidate: str, rubric: Rubric) -> Judgment:
        ...


def make_grpo_reward(judge: Judge, rubric_column: str = "rubric_json"):
    """Create a reward callable compatible with TRL's GRPOTrainer.

    TRL supplies prompts and completions as batches and forwards dataset columns
    as keyword arguments. Each row must include a serialized rubric in
    ``rubric_column``; the prompt is used as context unless a ``context`` column
    is also supplied.
    """

    if not rubric_column.strip():
        raise ValueError("rubric_column must be a nonempty column name")

    def reward(prompts: Any, completions: Any, **kwargs: Any) -> list[float]:
        prompt_rows = _batch(prompts, "prompts")
        completion_rows = _batch(completions, "completions")
        if len(prompt_rows) != len(completion_rows):
            raise ValueError("prompts and completions must have the same batch length")

        rubric_values = _column(kwargs, rubric_column, len(prompt_rows))
        context_values = (
            _column(kwargs, "context", len(prompt_rows))
            if "context" in kwargs
            else prompt_rows
        )
        scores: list[float] = []
        for index, (prompt, completion, rubric_value, context_value) in enumerate(
            zip(prompt_rows, completion_rows, rubric_values, context_values, strict=True)
        ):
            rubric = _rubric(rubric_value, index)
            context = _text(context_value, "context", index)
            candidate = _text(completion, "completion", index)
            try:
                judgment = judge.score(context, candidate, rubric)
            except Exception as exc:
                # Judge exceptions may contain candidate text or provider data.
                # Include only the exception class, never its message.
                raise RuntimeError(
                    f"judge.score failed for batch row {index} ({type(exc).__name__})"
                ) from None
            if not isinstance(judgment, Judgment):
                raise TypeError(
                    f"judge.score returned {type(judgment).__name__} for row {index}; expected Judgment"
                )
            score = judgment.total
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"judge score for batch row {index} must be numeric")
            score = float(score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"judge score for batch row {index} must be finite and in [0, 1]"
                )
            scores.append(score)
        return scores

    return reward


class DemoJudge:
    """Cheap deterministic lexical judge for smoke runs, not evaluation."""

    _word = re.compile(r"[\w'-]+", re.UNICODE)

    def score(self, context: str, candidate: str, rubric: Rubric) -> Judgment:
        """Reward informative, context-related text with a length-shaped signal."""
        candidate_words = {word.casefold() for word in self._word.findall(candidate)}
        context_words = {word.casefold() for word in self._word.findall(context)}
        overlap = len(candidate_words & context_words) / max(1, len(candidate_words))
        # Length provides a smooth signal even when tiny model outputs have no
        # vocabulary overlap; the cap keeps the score in the schema's [0, 1].
        length_score = min(len(candidate.strip()) / 256.0, 1.0)
        value = 0.8 * length_score + 0.2 * overlap
        criterion_scores = {criterion.id: value for criterion in rubric.criteria}
        rationale = "Deterministic smoke score from response length and context word overlap."
        return Judgment(
            criterion_scores=criterion_scores,
            total=sum(c.weight * criterion_scores[c.id] for c in rubric.criteria),
            rationale=rationale,
        )


def _batch(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, Mapping)):
        return [value]
    if not isinstance(value, Sequence):
        try:
            return list(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a batch sequence") from exc
    return list(value)


def _column(kwargs: Mapping[str, Any], name: str, size: int) -> list[Any]:
    if name not in kwargs:
        raise ValueError(f"missing required TRL dataset column {name!r}")
    values = _batch(kwargs[name], name)
    if len(values) != size:
        raise ValueError(
            f"dataset column {name!r} has {len(values)} rows; expected {size}"
        )
    return values


def _rubric(value: Any, index: int) -> Rubric:
    if isinstance(value, Rubric):
        return value
    try:
        if isinstance(value, str):
            return Rubric.model_validate_json(value)
        return Rubric.model_validate(value)
    except Exception as exc:
        raise ValueError(f"invalid rubric in batch row {index}: {exc}") from exc


def _text(value: Any, label: str, index: int) -> str:
    """Normalize plain strings and common Transformers chat-message shapes."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content
        if isinstance(value.get("context"), str):
            return value["context"]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for message in value:
            if isinstance(message, str):
                parts.append(message)
            elif isinstance(message, Mapping) and isinstance(message.get("content"), str):
                parts.append(message["content"])
            else:
                raise TypeError(f"unsupported {label} message in batch row {index}")
        return "\n".join(parts)
    raise TypeError(f"{label} in batch row {index} must be text or chat messages")
