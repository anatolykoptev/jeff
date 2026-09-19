"""Backend protocol shared by the GPU (torch) and CPU (onnx) arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Group:
    """One GLiFormer classification group.

    ``name`` and ``description`` both appear in the prompt as free text
    (``[SCHEMA] name description [CLASS] label ... [SEP]``), so they are where
    the jev ``instructions`` text goes.
    """

    key: str
    labels: tuple[str, ...]
    name: str | None = None
    description: str | None = None

    def __post_init__(self):
        if len(self.labels) != len(set(self.labels)):
            raise ValueError(f"group {self.key!r} has duplicate labels")


@dataclass
class ScoredText:
    """Per-text result: raw sigmoid score for every label of every group."""

    scores: dict[str, list[float]]  # group.key -> scores aligned with group.labels
    input_tokens: int
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    name: str

    def score(self, texts: list[str], groups: list[list[Group]]) -> list[ScoredText]:
        """Score ``groups[i]`` against ``texts[i]``. Must return ALL label scores."""
        ...
