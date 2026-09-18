"""Turn jev questions into GLiFormer classification groups."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Group
from .schemas import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

NOUL_YES = "yes"
NOUL_NO = "no"


@dataclass(frozen=True)
class PromptOptions:
    """Knobs for how question text is rendered into the prompt.

    These are the variants §5 of the plan evaluates. Defaults are the current
    best guess, not measured yet.
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
    # the answer (robust on base and large in probes). "yes_no": two labels,
    # renormalized (equally good on large, unreliable on base).
    noul_mode: str = "single"


def build_groups(questions: dict[str, Question], opts: PromptOptions = PromptOptions()) -> list[Group]:
    return [_group_for(qid, q, opts) for qid, q in questions.items()]


def _group_for(qid: str, q: Question, opts: PromptOptions) -> Group:
    name = q.instructions_text() if opts.instruction_as_name else qid
    if isinstance(q, NoulQuestion):
        desc = q.criteria if isinstance(q.criteria, str) else None
        if opts.noul_mode == "single":
            label = _fold(q.instructions_text(), q.criterion("true"), opts)
            return Group(key=qid, labels=(label,), name=None, description=desc)
        labels = (
            _fold(NOUL_YES, q.criterion("true"), opts),
            _fold(NOUL_NO, q.criterion("false"), opts),
        )
        return Group(key=qid, labels=labels, name=name, description=desc)
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
    for l in labels:
        seen[l] = seen.get(l, 0) + 1
        out.append(l if seen[l] == 1 else f"{l} #{seen[l]}")
    return tuple(out)
