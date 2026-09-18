# jeff — a jev-compatible API served by GLiFormer

Serve `knowledgator/gliformer-large-v1` behind an HTTP API that is wire-compatible
with TypeSafe's jev (`POST /v1/systemone`), with two optimized serving arms:

- **GPU arm**: PyTorch bf16 + flashdeberta Triton kernels + torch.compile + dynamic batching. Deployed on Modal.
- **CPU arm**: ONNX Runtime (fp32 / int8 dynamic quant) export of encoder + classification head. Runs on Mac for dev, Modal CPU containers for throughput.

Both arms share one core: jev request -> GLiFormer classification groups -> jev answers.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped

---

## 0. Setup

- [x] `git init`, `.gitignore` (models/, .venv/, *.onnx, bench results)
- [x] `uv init --python 3.12` (gliformer needs >=3.10; torch wheels for 3.14 are not reliable yet). Deps: `gliformer`, `fastapi`, `uvicorn`, `pydantic>=2`, `httpx`, `pytest`.
- [x] Install Modal CLI: `uv tool install modal && modal setup` (account exists, CLI not installed on this machine)
- [x] Download model once to `models/gliformer-large-v1` (also grab `gliformer-base-v1` for fast local iteration and CPU arm comparisons)
- [x] Smoke test: `model.classify(text, {"sentiment": [...], "topic": [...]})` on Mac (MPS or CPU eager attention). Record time-per-call as baseline. **Result (base, 2 groups, bs1): CPU eager 66 ms, MPS 35 ms. Large loads in ~15 s.**

## 1. Core: jev <-> GLiFormer mapping (`jeff/core/`)

Pure functions, no server, no device assumptions. Fully unit-tested.

- [x] **Request models** (`schemas.py`): pydantic models for the jev request body — `state` (str | object | array), `model`, `questions` map; question types `noul`, `choice`, `score`; `criteria` forms (null / dict / list of str / list of `{what, examples}`); also accept `selectedModels` as seen in the score docs. Validation errors -> 422 with jev-style body.
- [x] **State serialization** (`state.py`): string passthrough; object -> `key: value` lines (nested -> JSON); array -> joined lines. Document the format; keep it stable.
- [x] **Prompt builder** (`groups.py`): one classification group per question. Label strings:
  - choice: option key, optionally `key: description` (flag `fold_descriptions`, default on; evaluate in §5)
  - score: level text in order (index = level number)
  - noul: single label = the question text, raw sigmoid = P(yes) (`noul_mode="single"`, default). `yes_no` mode kept as option. **Probe (6 cases, base+large): single-label correct on all 12; yes/no correct on large only.** Base is unreliable for nouls in any mode.
  - Group name = question id (or the instruction text — evaluate both).
- [x] **Decoder** (`answers.py`): raw per-label sigmoid scores -> jev answers.
  - choice: renormalize -> `probabilities`, `choice` = argmax, `confidence = (p_max - 1/n)/(1 - 1/n)` (0 uniform, 1 one-hot; gives 0.55 on the docs example where jev shows 0.54).
  - score: renormalize -> `probabilities` keyed "0".."n-1", `score = sum(i * p_i)`, `legend` mirrors criteria.
  - noul: `noul = p_yes / (p_yes + p_no)`.
  - `usage.input_tokens` = tokenizer count of prompt + text; `output_tokens` = fixed small number per question (jev's are unmetered anyway).
- [x] **Backend protocol** (`backend.py`): `score_groups(text: str, groups: dict[str, list[str]]) -> dict[str, list[float]]` returning *all* label scores (threshold=0). Both arms implement this. Batch variant for dynamic batching.
- [x] Tests: golden fixtures from the typesafe docs examples (bug_severity score, routing choice, refund noul) asserting shape, key order, legend, score formula. Property test: probabilities sum to 1.

## 2. Server (`jeff/server/`)

- [ ] FastAPI app: `POST /v1/systemone`, `GET /v1/models`, `GET /healthz`.
- [ ] Bearer auth via `JEFF_API_KEYS` env (comma-separated); missing/invalid -> 401 jev-style error body.
- [ ] Error parity: 401 / 422 / 429 (token-bucket per key) / 529 (queue full).
- [ ] Request queue + **dynamic batcher**: coalesce requests for up to N ms or B items, one backend call. Configurable `JEFF_MAX_BATCH`, `JEFF_MAX_WAIT_MS`.
- [ ] Backend selected by env: `JEFF_BACKEND=torch|onnx`, `JEFF_MODEL=...`, `JEFF_DEVICE=cuda|mps|cpu`.
- [ ] Limits: max questions per request, max labels per question, max state tokens (long prompts blow up attention cost). Return 422 when exceeded.
- [ ] Verify the official TypeSafe Python SDK works against it unmodified by pointing its base URL at jeff (`sdk/python/api/constants.md` lists the env var).

## 3. GPU arm (`jeff/backends/torch_backend.py`, `deploy/modal_gpu.py`)

- [~] Torch backend: bf16, `attn_kernel=flash` (flashdeberta), `compile_model=True`, `torch.inference_mode`, batch call with left/right padding handled by the library.
- [ ] Modal app: image with CUDA torch + flashdeberta + gliformer; model weights in a Modal Volume; `@modal.web_server` / ASGI app; `keep_warm`/min containers 1; concurrency limits tuned to batcher.
- [ ] Try GPUs: L4 (cheap), A10G, A100/H100. Record p50/p95 latency and req/s per GPU per dollar.
- [ ] CUDA graphs / static shapes: bucket sequence lengths (128/256/512/1024) so compile doesn't recompile per request.
- [ ] Optional: fp16 vs bf16 accuracy/latency check.

## 4. CPU arm (`jeff/backends/onnx_backend.py`, `deploy/modal_cpu.py`, `scripts/export_onnx.py`)

- [ ] Export: trace `encoder + classification head` to ONNX with dynamic axes (batch, seq). Backbone is custom `layout-deberta` (relative attention, c2p/p2c, `position_biased_input=false`), so expect to patch ops that don't trace (gather/relative-position bucketing). Reference: GLiNER's DeBERTa ONNX exports.
- [ ] Reimplement prompt tokenization + group indexing outside the graph (reuse gliformer's processor for tokenization, feed `input_ids`, `attention_mask`, plus the label/group index tensors the head needs).
- [ ] Parity test: ONNX scores match torch scores within 1e-3 on the golden fixtures.
- [ ] ORT optimizations: graph optimization level all, fp32 baseline, then int8 dynamic quantization of MatMuls; measure accuracy delta on §5 eval set. Try `ORT_ENABLE_ALL` + thread tuning.
- [ ] Bench locally on M2 Max (CoreML EP optional) and on Modal CPU containers (vary CPU count). Compare to gliformer-base for the "cheap tier".
- [ ] Stretch: Rust server with `ort` crate if ORT-in-Python overhead dominates at small batch. Only if measurements justify it.

## 5. Evaluation & benchmarks (`bench/`)

- [ ] Small labeled eval set (~200 items) covering the three primitives: sentiment/topic choice, severity/frustration score, yes/no nouls. Source from public classification datasets + hand-written jev-style questions.
- [ ] Accuracy: choice accuracy, score MAE vs ordinal label, noul AUROC. Compare label-string variants (bare key vs `key: description`; group name = id vs instruction).
- [ ] Calibration: reliability diagram; optional temperature scaling fit stored per model.
- [ ] Load test (`bench/load.py` with httpx/asyncio): p50/p95/p99, req/s at concurrency 1/8/32/128 for each arm. Report tokens/sec and $/1M input tokens on Modal pricing next to jev's $0.042/M.
- [ ] Results table in `bench/RESULTS.md`.

## 6. Docs

- [ ] `README.md`: what it is, API parity notes and known differences (calibration, description folding, limits), quickstart local, `curl` example.
- [ ] `deploy/README.md`: Modal deploy instructions for both arms, env vars, GPU choice guidance, cost table.

---

## Known risks / open questions

- **Calibration**: GLiFormer classification is independent sigmoids + threshold, not a softmax. Renormalized "probabilities" won't be calibrated like jev's. Mitigation: §5 temperature scaling.
- **Descriptions**: classification head has no description field (only structuring does). Folding into label text is a guess to validate in §5.
- **ONNX export of layout-deberta**: custom backbone; may need op patches. Fallback for CPU arm: torch fp32 eager with `torch.compile` and int8 dynamic quant via `torch.ao`.
- **Prompt length**: every option/level string goes into the sequence. Many questions × long criteria -> quadratic attention cost. Enforce limits in §2.
- **Local dev has no CUDA**: flashdeberta path only testable on Modal. Keep torch backend runnable with eager attention on MPS/CPU for correctness tests.
- Python 3.14 on this machine: pin 3.12 in the project.

## Progress log

- 2026-09-18: Research done (model card, repo source, jev docs). Plan written.
- 2026-09-18: §0 done. §1 done: `jeff.core` (schemas, state, groups, answers, engine, backend protocol) + `jeff.backends.torch_backend` (reference arm; replicates `GLiFormer.inference` so per-group descriptions and prompt token counts are available). 15 tests green incl. integration on base checkpoint. Found: `threshold=0.0` is treated as unset by the library decoder, use `-1.0` to get all labels.
