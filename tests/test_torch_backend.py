"""Integration tests against the real base checkpoint (skipped if absent)."""

import pytest

from jeff.core import Engine, SystemOneRequest
from tests.fixtures import CHOICE_REQ, SCORE_REQ


@pytest.fixture(scope="module")
def engine(base_backend):
    return Engine(base_backend, "gliformer-base-v1")


def test_score_example(engine):
    resp = engine.run(SystemOneRequest.model_validate(SCORE_REQ)).model_dump()
    a = resp["answers"]["bug_severity"]
    assert set(a["probabilities"]) == {"0", "1", "2"}
    assert abs(sum(a["probabilities"].values()) - 1) < 1e-3
    # a crash with a workaround (Chrome) should land above "cosmetic"
    assert a["score"] > 0.8
    assert resp["usage"]["input_tokens"] > 20


def test_choice_and_noul(engine):
    a = engine.run(SystemOneRequest.model_validate(CHOICE_REQ)).model_dump()["answers"]
    assert a["team"]["choice"] == "billing"
    assert a["refund"]["noul"] > 0.5


def test_batch_matches_single(engine):
    reqs = [SystemOneRequest.model_validate(SCORE_REQ), SystemOneRequest.model_validate(CHOICE_REQ)]
    batched = [r.model_dump() for r in engine.run_batch(reqs)]
    singles = [engine.run(r).model_dump() for r in reqs]
    for b, s in zip(batched, singles):
        for k in b["answers"]:
            bp = b["answers"][k].get("probabilities") or {"noul": b["answers"][k]["noul"]}
            sp = s["answers"][k].get("probabilities") or {"noul": s["answers"][k]["noul"]}
            for kk in bp:
                assert abs(bp[kk] - sp[kk]) < 2e-2, (k, kk, bp, sp)


def test_group_description_changes_prompt(base_backend):
    from jeff.core import Group

    g1 = [Group(key="q", labels=("yes", "no"), name="Is this urgent?")]
    g2 = [Group(key="q", labels=("yes", "no"), name="Is this urgent?", description="Urgent means the customer needs help today.")]
    t = ["Please respond by end of day, our site is down."]
    s1 = base_backend.score(t, [g1])[0]
    s2 = base_backend.score(t, [g2])[0]
    assert s2.input_tokens > s1.input_tokens
