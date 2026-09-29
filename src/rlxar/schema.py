"""Validated data models used by RL-XAR."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _SchemaModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class Example(_SchemaModel):
    """A prompt example and its assigned dataset split."""

    id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    context: str = Field(min_length=1)
    expert: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    metadata: dict[str, object] = Field(default_factory=dict)


class Criterion(_SchemaModel):
    """One weighted criterion in a rubric."""

    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    weight: float = Field(gt=0)


class Rubric(_SchemaModel):
    """A normalized set of criteria and its provenance."""

    id: str = Field(min_length=1)
    criteria: list[Criterion] = Field(min_length=1)
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_criteria(self) -> Rubric:
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique within a rubric")
        total_weight = sum(criterion.weight for criterion in self.criteria)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("criterion weights must sum to 1 (tolerance 1e-6)")
        return self


class Judgment(_SchemaModel):
    """Scores and rationale for applying a rubric."""

    criterion_scores: dict[str, float]
    total: float = Field(ge=0, le=1)
    rationale: str | None = None

    @field_validator("criterion_scores")
    @classmethod
    def validate_scores(cls, scores: dict[str, float]) -> dict[str, float]:
        if any(not criterion_id.strip() for criterion_id in scores):
            raise ValueError("criterion score IDs must be nonempty")
        if any(not 0 <= score <= 1 for score in scores.values()):
            raise ValueError("criterion scores must be in [0, 1]")
        return scores

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, rationale: str | None) -> str | None:
        if rationale is not None and not rationale.strip():
            raise ValueError("rationale must be nonempty when provided")
        return rationale
