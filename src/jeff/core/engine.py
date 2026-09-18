"""Request -> groups -> backend -> response. Device-agnostic."""

from __future__ import annotations

from .answers import decode
from .backend import Backend
from .groups import PromptOptions, build_groups
from .schemas import SystemOneRequest, SystemOneResponse, Usage
from .state import serialize_state

# jev bills no output tokens in practice; report a nominal count per answer so
# the field is populated the way clients expect.
OUTPUT_TOKENS_PER_ANSWER = 6


class Engine:
    def __init__(self, backend: Backend, model_name: str, opts: PromptOptions = PromptOptions()):
        self.backend = backend
        self.model_name = model_name
        self.opts = opts

    def run(self, req: SystemOneRequest) -> SystemOneResponse:
        return self.run_batch([req])[0]

    def run_batch(self, reqs: list[SystemOneRequest]) -> list[SystemOneResponse]:
        texts = [serialize_state(r.state) for r in reqs]
        groups = [build_groups(r.questions, self.opts) for r in reqs]
        scored = self.backend.score(texts, groups)
        out = []
        for r, gs, s in zip(reqs, groups, scored):
            answers = {g.key: decode(r.questions[g.key], s.scores[g.key]) for g in gs}
            out.append(
                SystemOneResponse(
                    model=self.model_name,
                    answers=answers,
                    usage=Usage(
                        input_tokens=s.input_tokens,
                        output_tokens=OUTPUT_TOKENS_PER_ANSWER * len(answers),
                    ),
                )
            )
        return out
