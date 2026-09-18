"""Pydantic models for the jev-compatible request and response bodies.

Mirrors https://docs.typesafe.ai/api. Field names are kept identical so the
official TypeSafe SDKs can talk to this server unmodified.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

State = Union[str, dict[str, Any], list[Any]]


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str | dict[str, Any] | list[Any]

    def instructions_text(self) -> str:
        from .state import serialize_state

        return serialize_state(self.instructions)


class NoulQuestion(_Question):
    type: Literal["noul"]
    criteria: dict[str, Any] | str | None = None

    def criterion(self, key: str) -> str | None:
        """Description for the ``true`` / ``false`` side, if given."""
        if isinstance(self.criteria, dict):
            v = self.criteria.get(key)
            return None if v is None else str(v)
        return None


class ChoiceQuestion(_Question):
    type: Literal["choice"]
    # jev documents a mapping option -> description. A bare list of option
    # names is accepted too (descriptions empty).
    criteria: dict[str, Any] | list[str]

    @field_validator("criteria")
    @classmethod
    def _non_empty(cls, v):
        if len(v) < 2:
            raise ValueError("choice needs at least two options")
        if isinstance(v, list) and len(set(v)) != len(v):
            raise ValueError("choice options must be unique")
        return v

    def options(self) -> list[tuple[str, str | None]]:
        if isinstance(self.criteria, list):
            return [(k, None) for k in self.criteria]
        return [(k, None if v is None else str(v) if not isinstance(v, (dict, list)) else _json(v)) for k, v in self.criteria.items()]


class ScoreLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    what: str
    examples: list[str] | None = None


class ScoreQuestion(_Question):
    type: Literal["score"]
    criteria: list[str] | list[ScoreLevel]

    @field_validator("criteria")
    @classmethod
    def _at_least_two(cls, v):
        if len(v) < 2:
            raise ValueError("score needs at least two levels")
        return v

    def levels(self) -> list[tuple[str, list[str] | None]]:
        out = []
        for lv in self.criteria:
            if isinstance(lv, str):
                out.append((lv, None))
            else:
                out.append((lv.what, lv.examples))
        return out

    def legend(self) -> dict[str, Any]:
        """Mirror the criteria back by level number, as jev does."""
        return {
            str(i): (lv if isinstance(lv, str) else lv.model_dump(exclude_none=True))
            for i, lv in enumerate(self.criteria)
        }


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")
]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: State
    model: str | None = None
    # Seen in the score docs example; treated as an alias of ``model``.
    selectedModels: list[str] | None = None
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _resolve_model(self):
        if self.model is None:
            if self.selectedModels:
                self.model = self.selectedModels[0]
            else:
                raise ValueError("model is required")
        if not self.questions:
            raise ValueError("questions must not be empty")
        return self


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Any]
    probabilities: dict[str, float]


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ErrorBody(BaseModel):
    """jev returns ``{"error": {"type": ..., "message": ...}}`` style bodies."""

    error: dict[str, Any]


def _json(v: Any) -> str:
    import json

    return json.dumps(v, ensure_ascii=False)
