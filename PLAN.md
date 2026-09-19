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
- [x] Download model once to `models/gliformer-large-v1` (also download `gliformer-base-v1` for fast local iteration and CPU arm comparisons)
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

- [x] FastAPI app: `POST /v1/systemone`, `GET /v1/models`, `GET /healthz`.
- [x] Bearer auth via `JEFF_API_KEYS` env (comma-separated); missing/invalid -> 401 jev-style error body.
- [x] Error parity: 401 / 422 / 429 (token-bucket per key) / 529 (queue full).
- [x] Request queue + **dynamic batcher**: coalesce requests for up to N ms or B items, one backend call. Configurable `JEFF_MAX_BATCH`, `JEFF_MAX_WAIT_MS`.
- [x] Backend selected by env: `JEFF_BACKEND=torch|onnx`, `JEFF_MODEL=...`, `JEFF_DEVICE=cuda|mps|cpu`.
- [x] Limits: max questions per request, max labels per question, max state tokens (attention cost is quadratic in prompt length). Return 422 when exceeded.
- [x] Verify the official TypeSafe Python SDK works against it unmodified by pointing its base URL at jeff (`sdk/python/api/constants.md` lists the env var). **Done: `typesafe-sdk` 0.7.0 round-trips via `TYPESAFE_BASE_URL`, incl. 401/422 exception mapping and `models.list()`. Wire schema copied from `typesafe_sdk/_schemas/models.py` (instructions optional, noul criteria `{true,false}`, 422 body is FastAPI `{detail:[...]}`, `x-typesafe-request-id` + `retry-after-ms` headers).**

## 3. GPU arm (`jeff/backends/torch_backend.py`, `deploy/modal_gpu.py`) — DONE

**Recommendation: L4 for the HTTP API on Modal (p50 ≈ 270-300 ms intra-Modal / 440 ms from a laptop, $0.80/h idle, ≈ $0.030 per 1M input tokens at the per-container ingress cap). A10G when the backend is called directly or requests are long (28 ms bs=1, 41k tok/s, $0.0075/M).** Full data and reasoning in `bench/RESULTS.md`.

- [x] Torch backend: bf16, `attn_kernel=flash` (flashdeberta) on cuda / eager elsewhere (`JEFF_ATTN`), `torch.inference_mode`, batch call with padding handled by the library. Optional `JEFF_COMPILE` (compiles the DeBERTa encoder in place, falls back to eager if warmup fails), `JEFF_PAD_MULTIPLE` (length bucketing via tokenizer `pad_to_multiple_of`), `JEFF_WARMUP`. `GET /stats` reports the active config; responses carry `x-jeff-server-ms` / `x-jeff-batcher-ms` so client latency can be split into network vs jeff. Note: gliformer 0.1.2 silently drops `compile_torch_model`, and `_attn_implementation="eager"` does not override the checkpoint's `attn_kernel: flash`; both are handled in the backend.
- [x] Modal app (`deploy/modal_gpu.py`): image with CUDA torch + flashdeberta + gliformer; weights in Volume `jeff-models` (`::download`, also fetched lazily by the server); `@modal.asgi_app` Server with `@modal.concurrent(max_inputs=64, target_inputs=16)`, `max_containers=8`; `min_containers` only on `modal deploy`; every `JEFF_*` env forwarded as a Secret; `JEFF_APP_NAME` for side runs so they do not replace the web URL. Helpers: `::bench` (raw backend latency per config), `::profile` (torch.profiler split), `::dtype_check` (fp16 vs bf16 score drift), `::loadtest` / `::ping` (HTTP from inside Modal), `bench/load.py` (HTTP from anywhere), `bench/summarize.py` -> `bench/RESULTS.md`.
- [x] GPUs tried: L4, A10G, A100-40GB, H100. Peak tokens/s flash: 27k / 41k / 71k / 120k; $/M tokens at peak 0.0081 / 0.0075 / 0.0082 / 0.0091 vs jev 0.042. bs=1: 45 / 28 / 49 / 28 ms.
- [x] HTTP through Modal: one container caps at ~50 req/s with inference and ~100 req/s with no model at all (`POST /healthz`), so Modal's web ingress, not the GPU, bounds throughput; intra-Modal RTT ≈ 165 ms + ≈ 50 ms ASGI body delivery. Scale-out with `target_inputs=16` reached 102 req/s on 4 A10G containers. Locally uvicorn fills 27-of-32 batches at 64 clients, so the batcher itself is fine.
- [x] Profiled the bs=1 floor: it is GPU time (54% small-M GEMMs, 21% gliformer's word-level LSTM, 4% attention), not host work; only ~5 ms is pre/post-processing. Not fixable without a smaller checkpoint.
- [x] fp16 vs bf16: no latency gain, raw-score drift up to 0.029. Keep bf16.
- [-] CUDA graphs / static shapes: bucketing implemented but `torch.compile` loses to plain flash (Inductor can't rebuild the disentangled Triton kernel; excluding it leaves a graph break per layer and dynamo hits `recompile_limit`). Defaults: compile off, pad_multiple 0. Revisit with newer gliformer/flashdeberta.
- [ ] Follow-up (only if Modal HTTP throughput matters): expose the backend as a Modal function (`.remote`/`.map`, no web ingress) or host uvicorn on a plain GPU VM; both should recover the raw 80+ req/s per A10G.

## 4. CPU arm (`jeff/backends/onnx_backend.py`, `jeff/backends/onnx_export.py`, `deploy/modal_cpu.py`, `scripts/export_onnx.py`)

- [x] Export: only the encoder (`token_rep_layer(input_ids, attention_mask) -> token_embeds`) goes to ONNX; the legacy tracer handles layout-deberta unmodified (opset 17, dynamic batch/seq, parity 5e-6 on unseen shapes, 7 s for large). Prompt/word splitting, the word RNN and the classification head stay in torch fp32 (they depend on the Python-side group layout and are a few ms), so `OnnxBackend` is `TorchBackend` with the encoder module swapped for an ORT session and every other code path shared.
- [-] Reimplementing tokenization/group indexing outside the graph: unnecessary with the encoder-only split.
- [x] Parity test (`test_onnx_matches_torch`): ONNX-encoder scores match torch within 1e-3 on the fixtures.
- [x] ORT `ORT_ENABLE_ALL`, `JEFF_THREADS`; int8 dynamic quantization of MatMul/Gemm (`scripts/export_onnx.py --int8`; embeddings stay fp32, large int8 file is 1.0 GB vs 1.7 GB). Score drift measured by `bench/cpu_bench.py --drift` (numbers in `bench/RESULTS.md`).
- [x] Bench: `bench/cpu_bench.py` (M2 Max) and `deploy/modal_cpu.py` (8-core Modal containers, large + base). **Verdict: not a cheaper tier.** Large int8 on 8 cores: 296 ms bs=1, 675 tok/s peak, ≈ $0.71/M tokens (17x jev, 90x L4); base ≈ $0.27/M. On a Mac MPS beats ONNX 2-3x. Keep the arm as the no-GPU fallback. int8 needed two fixes (FFN output projection stays fp32; `reduce_range` for x86), after which large drifts 0.02-0.04 mean; base int8 is still broken on x86. Full write-up in `bench/RESULTS.md`.
- [-] Stretch: Rust server with `ort` crate. Not justified: the bs=1 floor is encoder FLOPs (ORT 368 ms vs eager torch 608 ms on 8 cores), not Python overhead.

## 5. Evaluation & benchmarks (`bench/`)

- [x] **Noul context sensitivity** (`bench/noul_probe.py`, 12 texts x 2 nouls x 4 contexts x {str, dict} state, large + base). Cause: all groups share one encoder sequence, so other questions' tokens shift a noul's label representation. Without isolation, large drops from 0.83 (noul alone) to 0.50 (noul last after choice+score); the same noul moves by up to 0.98 across contexts. **Fixes shipped:** `PromptOptions.isolate` (`nouls` default: each noul gets its own encoder pass, batched; spread goes to 0), `noul_mode` (`yes_no` default: ties `single` on accuracy, lowest error, graded probabilities), `state_format` (`kv` default; json/values not better). Base is unusable for nouls in any mode (best 0.67, `single`); keep base for dev only. Details in `bench/RESULTS.md`.

- [x] **Labeled eval set** (`bench/build_evalset.py` -> `bench/data/*.jsonl`, committed): 8 public datasets x 200 label-balanced items, one jev-style question each. choice: AG News, dair-ai/emotion; score: SST-5, Amazon reviews (object state); noul: SMS spam, SST-2, TweetEval irony, BoolQ (per-item instructions). 1,600 items.
- [x] **Accuracy vs jev** (`bench/eval_accuracy.py jeff|http|jev|summarize`, rows in `bench/results/accuracy.jsonl`): jeff-large vs jev-latest/jev-preview on the same items. **jev wins everywhere but binary sentiment** (avg choice acc 0.61 vs 0.69, score MAE 0.61 vs 0.48, noul AUROC 0.84 vs 0.975; BoolQ 0.75 vs 0.95, irony 0.71 vs 0.96, AG News 76% vs 90%; SST-2 0.99 both). Variants: folding descriptions and instruction-as-group-name are both required for nouls (idname -> chance, nodesc drops irony to chance); `yes_no` > `single`; json = kv. Shipped defaults confirmed.
- [x] **Calibration** (`bench/calibrate.py`): jeff over-confident (ECE 0.17-0.41 on choice/score vs jev 0.06-0.38). Global temperature T=3.2 (leave-one-task-out 3.1-3.3) halves ECE and improves noul Brier on hard tasks. Tempering the distribution `score` is taken from costs +0.05-0.09 MAE, so the decoder tempers only `probabilities`/`confidence`/`noul` and computes `score` from the raw distribution. **Shipped as `JEFF_TEMPERATURE=3.2` default**; `large-t1/default` rows in `bench/RESULTS.md` are the T=1 reference. No reliability diagram drawn; the per-task ECE table is the substitute.
- [x] Latency vs jev from a laptop: sequential p50 129 ms (jev) vs 151 ms (jeff L4 on Modal); at 8 clients 131 vs 267 ms (Modal ingress, see §3). Tokens: jev bills 4.3x more input tokens per request than jeff counts (372 vs 86), so per-request cost is ≈ $15.6/M requests (jev) vs ≈ $2.6/M (L4 at the ingress cap) vs ≈ $0.65/M (A10G direct).
- [x] Load test (`bench/load.py`, §3) and results tables in `bench/RESULTS.md` (GPU arm, noul probe, CPU arm, accuracy eval).
- [x] **Multi-question requests** (`eval_accuracy.py --multi`: labeled question last after a choice, a score and a noul distractor). jev: unchanged on every metric. jeff `isolate=nouls` (default): nouls identical, choice -1.5 pt, score +0.01-0.03 MAE. `isolate=none`: nouls move too (BoolQ 0.75 -> 0.73, spam 0.92 -> 0.90 AUROC, ECE doubles). `isolate=all`: single-question numbers reproduced exactly for ~50% more tokens / ~20% more time. Default stays `nouls`; `JEFF_ISOLATE=all` documented for when choice/score must be request-independent.
- [ ] Possible follow-ups: fine-tune GLiFormer on the eval tasks' train splits to close the BoolQ/irony gap (it is a model limit, not serving); a `x-jeff-*` header or `/v1/models` description advertising temperature/isolation so clients can tell configs apart.

## 6. Docs

- [x] `README.md`: quickstart, SDK + curl examples, env vars, GPU + CPU arm deploy commands, "Parity with jev: known differences" (accuracy gap, calibration/temperature, description folding, token accounting, limits).
- [-] `deploy/README.md`: folded into `README.md` + `bench/RESULTS.md` (GPU choice, cost) instead of a third file.

---

## Known risks / open questions

- ~~**Calibration**~~: measured in §5; T=3.2 via `JEFF_TEMPERATURE` fixes most of it and is the default; `score` is left untempered because tempering it raises MAE.
- ~~**Descriptions**~~: folding validated in §5 (required for nouls, neutral for choice).
- **Accuracy gap to jev** on inference-heavy nouls and topic choice (§5) is the main open quality issue; it is a model limit, not a serving one.
- ~~**ONNX export of layout-deberta**~~: exported unmodified with the legacy tracer (§4).
- **Prompt length**: every option/level string goes into the sequence. Many questions × long criteria -> quadratic attention cost. Limits enforced in §2.
- **Local dev has no CUDA**: flashdeberta path only testable on Modal. Keep torch backend runnable with eager attention on MPS/CPU for correctness tests.
- Python 3.14 on this machine: pin 3.12 in the project.

## Progress log

- 2026-09-18: Follow-ups: `score` decoupled from temperature (raw expected level), `JEFF_TEMPERATURE` default 3.2, `--multi`/`--temperature` in `eval_accuracy.py`, multi-question eval on jeff (3 isolation modes) and jev; README rewritten as the shipping doc (config table, benchmark summary, parity notes). 30 tests green.
- 2026-09-18: §5 closed: eval set from 8 public datasets (`bench/build_evalset.py`, `bench/data/`), `bench/eval_accuracy.py` (in-process variants, HTTP, real jev via `TYPESAFE_API_KEY`), `bench/calibrate.py`; ran jeff-large (5 variants), jeff on Modal L4, jev-latest and jev-preview on 1,600 items. Added `JEFF_TEMPERATURE`. README parity section written (§6 closed). Modal image now excludes `bench/results` and `bench/data` (a growing log file broke `modal serve` mid-build).

- 2026-09-18: §5 noul probe done, defaults changed to `isolate=nouls` + `noul_mode=yes_no` (`JEFF_ISOLATE`, `JEFF_NOUL_MODE`, `JEFF_STATE_FORMAT`). §4 CPU arm closed: encoder-only ONNX export (`onnx_export.py`, `scripts/export_onnx.py --int8`), `OnnxBackend`, parity test, `bench/cpu_bench.py`, `deploy/modal_cpu.py`, CPU tables in `bench/summarize.py`; benched on M2 Max and Modal 8-core; verdict "no-GPU fallback only". 30 tests green.

- 2026-09-18: §3 closed: A100/fp16/profile/HTTP scale-out measured; recommendation L4-for-HTTP, A10G-for-direct in `bench/RESULTS.md`. Found Modal web ingress caps one container at ~50 req/s regardless of GPU.
- 2026-09-18: §3 first pass: GPU arm knobs in `TorchBackend`, `deploy/modal_gpu.py` (Volume, ASGI server, per-config bench), `bench/load.py`, `bench/summarize.py`, `bench/RESULTS.md`. Benched L4/A10G/H100 on Modal: flash > eager+compile > eager; compile+flash regresses. 25 tests green.

- 2026-09-18: Research done (model card, repo source, jev docs). Plan written.
- 2026-09-18: §2 done: `jeff.server` (FastAPI, auth, 401/422/429/529, dynamic batcher, limits, `/v1/models`, `/stats`), `uv run jeff` entry point. 24 tests green incl. official SDK live test. Live large model on M2 Max MPS: 148 ms/req sequential (3 questions), 31.9 req/s at 64 concurrent (avg batch 4.5). Severity docs example: ours 0.02/0.81/0.17 vs jev 0/0.7/0.3. Found noul context sensitivity (logged under §5).
- 2026-09-18: §0 done. §1 done: `jeff.core` (schemas, state, groups, answers, engine, backend protocol) + `jeff.backends.torch_backend` (reference arm; replicates `GLiFormer.inference` so per-group descriptions and prompt token counts are available). 15 tests green incl. integration on base checkpoint. Found: `threshold=0.0` is treated as unset by the library decoder, use `-1.0` to get all labels.
