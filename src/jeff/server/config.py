"""Server settings from environment variables (all prefixed JEFF_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class Settings:
    backend: str = os.environ.get("JEFF_BACKEND", "torch")
    model_path: str = os.environ.get("JEFF_MODEL", "models/gliformer-large-v1")
    # Name reported in responses and GET /v1/models. "jev-latest" is also
    # accepted as an alias in requests so SDK defaults work.
    model_name: str = os.environ.get("JEFF_MODEL_NAME", "gliformer-large-v1")
    model_aliases: list[str] = field(default_factory=lambda: _env_list("JEFF_MODEL_ALIASES") or ["jev-latest", "jev"])
    device: str | None = os.environ.get("JEFF_DEVICE")
    dtype: str | None = os.environ.get("JEFF_DTYPE")
    compile_model: bool = os.environ.get("JEFF_COMPILE", "0") == "1"

    # Auth: empty list disables auth (local dev).
    api_keys: list[str] = field(default_factory=lambda: _env_list("JEFF_API_KEYS"))

    # Dynamic batching.
    max_batch: int = int(os.environ.get("JEFF_MAX_BATCH", "16"))
    max_wait_ms: float = float(os.environ.get("JEFF_MAX_WAIT_MS", "5"))
    max_queue: int = int(os.environ.get("JEFF_MAX_QUEUE", "256"))  # beyond this -> 529

    # Per-key rate limit, requests per second (0 disables).
    rate_limit_rps: float = float(os.environ.get("JEFF_RATE_LIMIT_RPS", "0"))
    rate_limit_burst: int = int(os.environ.get("JEFF_RATE_LIMIT_BURST", "20"))

    # Request limits -> 422.
    max_questions: int = int(os.environ.get("JEFF_MAX_QUESTIONS", "64"))
    max_labels_per_question: int = int(os.environ.get("JEFF_MAX_LABELS", "64"))
    max_state_chars: int = int(os.environ.get("JEFF_MAX_STATE_CHARS", "20000"))

    host: str = os.environ.get("JEFF_HOST", "0.0.0.0")
    port: int = int(os.environ.get("JEFF_PORT", "8000"))
