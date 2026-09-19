"""Request -> groups -> backend -> response. Device-agnostic."""

from __future__ import annotations

from .answers import DEFAULT_TEMPERATURE, decode
from .backend import Backend, Group
from .groups import PromptOptions, build_groups
from .schemas import SystemOneRequest, SystemOneResponse, Usage
from .state import serialize_state

# jev does not meter output tokens; report a nominal count per answer so the field is populated.
OUTPUT_TOKENS_PER_ANSWER = 6


class Engine:
    def __init__(
        self,
        backend: Backend,
        model_name: str,
        opts: PromptOptions | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.backend = backend
        self.model_name = model_name
        self.opts = opts or PromptOptions()
        # Applied when renormalizing scores into probabilities (see answers.normalize).
        self.temperature = temperature

    def run(self, req: SystemOneRequest) -> SystemOneResponse:
        return self.run_batch([req])[0]

    def run_batch(self, reqs: list[SystemOneRequest]) -> list[SystemOneResponse]:
        # One unit is one encoder pass: a text plus the groups that share its
        # prompt. Isolated questions get their own unit; the rest of a request's
        # questions share one.
        texts: list[str] = []
        units: list[list[Group]] = []
        owner: list[int] = []
        for i, r in enumerate(reqs):
            text = serialize_state(r.state, self.opts.state_format)
            shared: list[Group] = []
            for g in build_groups(r.questions, self.opts):
                if self.opts.isolated(r.questions[g.key]):
                    texts.append(text)
                    units.append([g])
                    owner.append(i)
                else:
                    shared.append(g)
            if shared:
                texts.append(text)
                units.append(shared)
                owner.append(i)

        scored = self.backend.score(texts, units)
        raw: list[dict[str, list[float]]] = [{} for _ in reqs]
        tokens = [0] * len(reqs)
        for i, gs, s in zip(owner, units, scored):
            for g in gs:
                raw[i][g.key] = s.scores[g.key]
            tokens[i] += s.input_tokens

        out = []
        for r, scores, n_tok in zip(reqs, raw, tokens):
            answers = {qid: decode(q, scores[qid], self.temperature) for qid, q in r.questions.items()}
            out.append(
                SystemOneResponse(
                    model=self.model_name,
                    answers=answers,
                    usage=Usage(
                        input_tokens=n_tok,
                        output_tokens=OUTPUT_TOKENS_PER_ANSWER * len(answers),
                    ),
                )
            )
        return out
