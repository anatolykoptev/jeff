import math

import pytest
from pydantic import ValidationError

from jeff.core import Engine, Group, PromptOptions, ScoredText, SystemOneRequest, build_groups
from jeff.core.answers import confidence, normalize
from jeff.core.state import serialize_state
from tests.fixtures import CHOICE_REQ, SCORE_REQ


class FakeBackend:
    name = "fake"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, texts, groups):
        self.calls.append((texts, groups))
        return [ScoredText(scores={g.key: self.scores[g.key] for g in gs}, input_tokens=100) for gs in groups]


def test_state_serialization():
    assert serialize_state("x") == "x"
    assert serialize_state({"a": 1, "b": "two", "c": {"d": [1]}}) == 'a: 1\nb: two\nc: {"d":[1]}'
    assert serialize_state(["a", "b"]) == "a\nb"


def test_request_validation():
    r = SystemOneRequest.model_validate(SCORE_REQ)
    assert r.model == "jev-latest"
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": {}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {"state": "x", "model": "m", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}}
        )
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {"state": "x", "model": "m", "questions": {"q": {"type": "banana", "instructions": "?"}}}
        )


def test_build_groups_default_prompt():
    r = SystemOneRequest.model_validate(CHOICE_REQ)
    gs = build_groups(r.questions)
    by_key = {g.key: g for g in gs}
    assert by_key["team"].name == "Which team should handle this?"
    assert by_key["team"].labels == (
        "billing: Payment or subscription issues",
        "technical: Bugs or integration problems",
        "sales: Pricing or account questions",
    )
    assert by_key["refund"].labels == ("Does the customer request a refund?",)
    assert by_key["refund"].name is None
    assert by_key["urgency"].labels == ("Does this convey urgency?: time-sensitive",)


def test_build_groups_yes_no_noul():
    r = SystemOneRequest.model_validate(CHOICE_REQ)
    gs = build_groups(r.questions, PromptOptions(noul_mode="yes_no"))
    by_key = {g.key: g for g in gs}
    assert by_key["refund"].labels == ("yes", "no")
    assert by_key["urgency"].labels == ("yes: time-sensitive", "no: no urgency")


def test_build_groups_no_fold():
    r = SystemOneRequest.model_validate(CHOICE_REQ)
    gs = build_groups(r.questions, PromptOptions(fold_descriptions=False, instruction_as_name=False))
    team = next(g for g in gs if g.key == "team")
    assert team.name == "team"
    assert team.labels == ("billing", "technical", "sales")


def test_duplicate_levels_stay_aligned():
    r = SystemOneRequest.model_validate(
        {"state": "x", "model": "m", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["low", "low", "high"]}}}
    )
    (g,) = build_groups(r.questions)
    assert g.labels == ("low", "low #2", "high")


def test_group_rejects_duplicate_labels():
    with pytest.raises(ValueError):
        Group(key="k", labels=("a", "a"))


def test_normalize_and_confidence():
    assert normalize([0.0, 0.0]) == [0.5, 0.5]
    p = normalize([0.2, 0.6, 0.2])
    assert math.isclose(sum(p), 1.0)
    assert confidence([1 / 3] * 3) == 0.0
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([0.0, 0.7, 0.3]) == 0.55


def test_score_answer_matches_docs_formula():
    r = SystemOneRequest.model_validate(SCORE_REQ)
    eng = Engine(FakeBackend({"bug_severity": [0.0, 0.7, 0.3]}), "jev-latest")
    resp = eng.run(r).model_dump()
    a = resp["answers"]["bug_severity"]
    assert a["type"] == "score"
    assert a["score"] == 1.3
    assert a["probabilities"] == {"0": 0.0, "1": 0.7, "2": 0.3}
    assert a["legend"] == {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists",
    }
    assert 0 <= a["confidence"] <= 1
    assert resp["usage"] == {"input_tokens": 100, "output_tokens": 6}
    assert resp["model"] == "jev-latest"


def test_choice_and_noul_answers():
    r = SystemOneRequest.model_validate(CHOICE_REQ)
    eng = Engine(FakeBackend({"team": [0.9, 0.05, 0.05], "refund": [0.8], "urgency": [0.1]}), "jev-latest")
    a = eng.run(r).model_dump()["answers"]
    assert a["team"]["choice"] == "billing"
    assert a["team"]["probabilities"]["billing"] == 0.9
    assert set(a["team"]) == {"type", "choice", "confidence", "probabilities"}
    assert a["refund"] == {"type": "noul", "noul": 0.8}
    assert a["urgency"]["noul"] == 0.1


def test_score_levels_with_examples_legend():
    r = SystemOneRequest.model_validate(
        {
            "state": "x",
            "model": "m",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "?",
                    "criteria": [{"what": "calm", "examples": ["ok thanks"]}, {"what": "angry"}],
                }
            },
        }
    )
    eng = Engine(FakeBackend({"q": [0.5, 0.5]}), "m")
    a = eng.run(r).model_dump()["answers"]["q"]
    assert a["legend"] == {"0": {"what": "calm", "examples": ["ok thanks"]}, "1": {"what": "angry"}}
    assert a["score"] == 0.5
