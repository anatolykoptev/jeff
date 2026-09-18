"""FastAPI app exposing the jev API surface.

Endpoints:  POST /v1/systemone   GET /v1/models   GET /healthz   GET /stats
Errors:     401 / 422 / 429 / 529 with the bodies and headers the TypeSafe SDK parses.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..core.engine import Engine
from ..core.groups import build_groups
from ..core.schemas import ModelMetadata, ModelMetadataList, SystemOneRequest, SystemOneResponse
from ..core.state import serialize_state
from .batcher import Batcher, QueueFull
from .config import Settings

log = logging.getLogger("jeff")
REQUEST_ID_HEADER = "x-typesafe-request-id"


class _Bucket:
    """Token bucket per API key."""

    def __init__(self, rps: float, burst: int):
        self.rps, self.burst = rps, burst
        self.state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last)

    def take(self, key: str) -> float:
        """Return 0 if allowed, else seconds until a token is available."""
        now = time.monotonic()
        tokens, last = self.state.get(key, (float(self.burst), now))
        tokens = min(self.burst, tokens + (now - last) * self.rps)
        if tokens >= 1:
            self.state[key] = (tokens - 1, now)
            return 0.0
        self.state[key] = (tokens, now)
        return (1 - tokens) / self.rps


def _error(status: int, type_: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"type": type_, "message": message}}, status_code=status, headers=headers)


def _validation_error(loc: list[Any], msg: str, type_: str, input_: Any = None) -> JSONResponse:
    return JSONResponse({"detail": [{"loc": loc, "msg": msg, "type": type_, "input": input_}]}, status_code=422)


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine or _load_engine(settings)
        app.state.engine = eng
        app.state.batcher = Batcher(eng, settings.max_batch, settings.max_wait_ms, settings.max_queue)
        await app.state.batcher.start()
        log.info("jeff ready: backend=%s model=%s", eng.backend.name, eng.model_name)
        yield
        await app.state.batcher.stop()

    app = FastAPI(title="jeff", lifespan=lifespan)
    app.state.settings = settings
    bucket = _Bucket(settings.rate_limit_rps, settings.rate_limit_burst) if settings.rate_limit_rps > 0 else None
    accepted_models = {settings.model_name, *settings.model_aliases}

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response

    @app.exception_handler(RequestValidationError)
    async def _on_validation(request: Request, exc: RequestValidationError):
        # Same shape FastAPI emits by default (and the SDK expects); drop
        # non-serializable ctx objects.
        detail = []
        for e in exc.errors():
            d = {k: v for k, v in e.items() if k in ("loc", "msg", "type", "input")}
            detail.append(d)
        return JSONResponse({"detail": detail}, status_code=422)

    def _auth(request: Request) -> str | JSONResponse:
        if not settings.api_keys:
            return "anonymous"
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return _error(401, "authentication_error", "Missing bearer token")
        key = auth[7:].strip()
        if key not in settings.api_keys:
            return _error(401, "authentication_error", "Invalid API key")
        return key

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "model": settings.model_name}

    @app.get("/stats")
    async def stats(request: Request):
        return request.app.state.batcher.stats.snapshot()

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def models(request: Request):
        who = _auth(request)
        if isinstance(who, JSONResponse):
            return who
        desc = f"GLiFormer served by jeff ({settings.model_path})"
        return ModelMetadataList(
            models=[ModelMetadata(name=settings.model_name, description=desc, release_date="2026-09-16")]
            + [ModelMetadata(name=a, description=f"Alias of {settings.model_name}", release_date="2026-09-16") for a in settings.model_aliases]
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def systemone(request: Request, body: SystemOneRequest):
        who = _auth(request)
        if isinstance(who, JSONResponse):
            return who
        if bucket is not None:
            wait = bucket.take(who)
            if wait > 0:
                ms = int(wait * 1000) + 1
                return _error(429, "rate_limit_error", "Rate limit exceeded", {"retry-after-ms": str(ms), "retry-after": str(max(1, ms // 1000))})

        if body.model not in accepted_models:
            return _validation_error(["body", "model"], f"Unknown model {body.model!r}", "value_error", body.model)
        if len(body.questions) > settings.max_questions:
            return _validation_error(["body", "questions"], f"At most {settings.max_questions} questions per request", "too_long")
        if len(serialize_state(body.state)) > settings.max_state_chars:
            return _validation_error(["body", "state"], f"State exceeds {settings.max_state_chars} characters", "string_too_long")
        for g in build_groups(body.questions, request.app.state.engine.opts):
            if len(g.labels) > settings.max_labels_per_question:
                return _validation_error(["body", "questions", g.key, "criteria"], f"At most {settings.max_labels_per_question} options/levels", "too_long")

        try:
            resp = await request.app.state.batcher.submit(body)
        except QueueFull:
            return _error(529, "overloaded_error", "Server is overloaded, retry with backoff", {"retry-after-ms": "500"})
        return resp

    return app


def _load_engine(settings: Settings) -> Engine:
    if settings.backend == "torch":
        from ..backends.torch_backend import TorchBackend

        backend = TorchBackend(
            settings.model_path,
            device=settings.device,
            dtype=settings.dtype,
            compile_model=settings.compile_model,
            batch_size=settings.max_batch,
        )
    elif settings.backend == "onnx":
        from ..backends.onnx_backend import OnnxBackend  # PLAN §4

        backend = OnnxBackend(settings.model_path)
    else:
        raise ValueError(f"unknown JEFF_BACKEND={settings.backend!r}")
    return Engine(backend, settings.model_name)
