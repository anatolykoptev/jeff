"""Serialize jev ``state`` (string | object | array) into plain text.

GLiFormer is text-only, as is jev. The format is deliberately simple and
stable: objects become ``key: value`` lines, arrays become one item per line,
nested containers are rendered as compact JSON.
"""

from __future__ import annotations

import json
from typing import Any


def serialize_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {_scalar(v)}" for k, v in state.items())
    if isinstance(state, list):
        return "\n".join(_scalar(v) for v in state)
    return _scalar(state)


def _scalar(v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
