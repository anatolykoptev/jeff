"""Decode raw per-label scores into jev answers."""

from __future__ import annotations

import math
from typing import Sequence

from .schemas import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

EPS = 1e-9


def normalize(scores: Sequence[float]) -> list[float]:
    """Independent sigmoids -> a distribution.

    GLiFormer's classification head is not a softmax, so this is a
    renormalization, not a calibrated posterior. See PLAN.md §5.
    """
    s = [max(float(x), 0.0) for x in scores]
    total = sum(s)
    if total <= EPS:
        return [1.0 / len(s)] * len(s)
    return [x / total for x in s]


def confidence(probs: Sequence[float]) -> float:
    """How peaked the distribution is, on 0..1.

    ``(p_max - 1/n) / (1 - 1/n)``: 0 for uniform, 1 for one-hot. This lands
    close to the worked example in the jev docs (0.7/0.3 -> 0.54 there, 0.55
    here). jev does not publish its formula.
    """
    n = len(probs)
    if n < 2:
        return 1.0
    chance = 1.0 / n
    return round((max(probs) - chance) / (1.0 - chance), 4)


def decode(q: Question, scores: Sequence[float]) -> Answer:
    if isinstance(q, NoulQuestion):
        if len(scores) == 1:  # single-label mode: raw sigmoid is P(yes)
            return NoulAnswer(noul=_r(min(max(scores[0], 0.0), 1.0)))
        p = normalize([scores[0], scores[1]])[0]
        return NoulAnswer(noul=_r(p))
    if isinstance(q, ChoiceQuestion):
        keys = [k for k, _ in q.options()]
        probs = normalize(scores)
        best = max(range(len(keys)), key=lambda i: probs[i])
        return ChoiceAnswer(
            choice=keys[best],
            confidence=confidence(probs),
            probabilities={k: _r(p) for k, p in zip(keys, probs)},
        )
    if isinstance(q, ScoreQuestion):
        probs = normalize(scores)
        score = sum(i * p for i, p in enumerate(probs))
        return ScoreAnswer(
            score=_r(score),
            confidence=confidence(probs),
            legend=q.legend(),
            probabilities={str(i): _r(p) for i, p in enumerate(probs)},
        )
    raise TypeError(type(q))


def _r(x: float) -> float:
    return round(float(x), 4)
