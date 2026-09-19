"""Turn jev questions into GLiFormer classification groups."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Group
from .schemas import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

NOUL_YES = "yes"
NOUL_NO = "no"
NOUL_MODES = ("single", "single_named", "yes_no")
ISOLATE_MODES = ("none", "nouls", "all")


@dataclass(frozen=True)
class PromptOptions:
    """How question text is rendered into the prompt.

    Defaults were chosen from the measurements in bench/RESULTS.md.
    """

    # Put the instruction text in the group name slot.
    instruction_as_name: bool = True
    # Append option/level descriptions to the label text ("key: description").
    fold_descriptions: bool = True
    # For score levels given as {what, examples}, append examples to the label.
    fold_examples: bool = False
    # Separator between a label key and its folded description.
    sep: str = ": "
    # noul rendering. "single": one label holding the question, its sigmoid is
    # the answer. "single_named": the question is the group name and the one
    # label is "yes". "yes_no": two labels under the question, renormalized.
    # The base checkpoint only works with "single".
    noul_mode: str = "yes_no"
    # Which questions get their own encoder pass instead of sharing one prompt
    # with the request's other questions: "none", "nouls" or "all". Groups that
    # share a prompt influence each other's scores; each isolated question
    # costs one extra batched encoder pass.
    isolate: str = "nouls"
    # How non-string ``state`` is rendered: "kv" (``key: value`` lines),
    # "json", or "values" (values only, one per line).
    state_format: str = "kv"

    def __post_init__(self):
        if self.noul_mode not in NOUL_MODES:
            raise ValueError(f"noul_mode must be one of {NOUL_MODES}, got {self.noul_mode!r}")
        if self.isolate not in ISOLATE_MODES:
            raise ValueError(f"isolate must be one of {ISOLATE_MODES}, got {self.isolate!r}")

    def isolated(self, q: Question) -> bool:
        return self.isolate == "all" or (self.isolate == "nouls" and isinstance(q, NoulQuestion))


def build_groups(questions: dict[str, Question], opts: PromptOptions | None = None) -> list[Group]:
    opts = opts or PromptOptions()
    return [_group_for(qid, q, opts) for qid, q in questions.items()]


def _group_for(qid: str, q: Question, opts: PromptOptions) -> Group:
    instr = q.instructions_text()
    name = (instr or qid) if opts.instruction_as_name else qid
    if isinstance(q, NoulQuestion):
        if opts.noul_mode == "single":
            label = _fold(instr or qid, q.criterion("true"), opts)
            return Group(key=qid, labels=(label,), name=None)
        if opts.noul_mode == "single_named":
            label = _fold(NOUL_YES, q.criterion("true"), opts)
            return Group(key=qid, labels=(label,), name=name)
        labels = (
            _fold(NOUL_YES, q.criterion("true"), opts),
            _fold(NOUL_NO, q.criterion("false"), opts),
        )
        return Group(key=qid, labels=labels, name=name)
    if isinstance(q, ChoiceQuestion):
        labels = tuple(_fold(k, d, opts) for k, d in q.options())
        return Group(key=qid, labels=_dedupe(labels), name=name)
    if isinstance(q, ScoreQuestion):
        labels = []
        for what, examples in q.levels():
            text = what
            if opts.fold_examples and examples:
                text = f"{what} (e.g. {'; '.join(examples)})"
            labels.append(text)
        return Group(key=qid, labels=_dedupe(tuple(labels)), name=name)
    raise TypeError(type(q))


def _fold(key: str, desc: str | None, opts: PromptOptions) -> str:
    if opts.fold_descriptions and desc:
        return f"{key}{opts.sep}{desc}"
    return key


def _dedupe(labels: tuple[str, ...]) -> tuple[str, ...]:
    """GLiFormer keys labels by string; identical strings would collapse.

    Make them unique with an index suffix so scores stay aligned to levels.
    """
    if len(set(labels)) == len(labels):
        return labels
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return tuple(out)
