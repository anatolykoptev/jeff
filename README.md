# jeff

A self-hosted, wire-compatible implementation of TypeSafe's [jev](https://docs.typesafe.ai/api)
System One API (`POST /v1/systemone`), served by
[GLiFormer](https://huggingface.co/knowledgator/gliformer-large-v1), a 400M-parameter
zero-shot classifier. The official `typesafe-sdk` works against it unmodified: point
`TYPESAFE_BASE_URL` at jeff.

jev's three primitives map onto GLiFormer classification groups:

| jev question | GLiFormer group | answer |
|---|---|---|
| `choice` (pick one of N options) | one label per option, `key: description` | renormalized probabilities, argmax, confidence |
| `score` (rate on ordered levels) | one label per level, in order | expected level under the raw distribution, per-level probabilities |
| `noul` (yes/no) | `yes` / `no` labels under the question, own encoder pass | P(yes) |

All questions in a request share one encoder pass over the text, except nouls, which get their
own (other questions' tokens otherwise shift a noul's answer by up to 0.98; measured in
`bench/RESULTS.md`). Probabilities are temperature-scaled with T = 3.2, fit on eight public
datasets, so they are roughly calibrated; `score` stays the expected level of the raw
distribution because that had the lower error.

**Accuracy and cost vs jev** (1,600 labeled items, `bench/RESULTS.md`): jev is more accurate on
every task except binary sentiment, with the largest gaps on questions that need inference
(BoolQ 0.75 vs 0.95 AUROC, irony 0.71 vs 0.96) and topic choice (AG News 76% vs 90%). jeff
answers in about the same wall-clock time from a laptop (151 vs 129 ms sequential) and costs
roughly 6x less per request on an L4 behind Modal's web ingress, or 25x less when the GPU is
called directly. Use it where cost, data residency or self-hosting matter more than accuracy on
questions that need inference over the text.

## Quickstart

```sh
uv sync --extra dev
uv run hf download knowledgator/gliformer-large-v1 --local-dir models/gliformer-large-v1
JEFF_API_KEYS=devkey uv run jeff            # http://localhost:8000, cuda > mps > cpu
```

Then use the official SDK unchanged:

```sh
pip install typesafe-sdk
TYPESAFE_API_KEY=devkey TYPESAFE_BASE_URL=http://localhost:8000 python -c '
from typesafe_sdk import TypeSafeClient, Noul, Choice, Score
c = TypeSafeClient()
r = c.system_one("I was charged twice. Please help ASAP.", {
    "billing": Noul(instructions="Is this about billing?"),
    "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": "hostile"}),
    "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
})
print(r.nouls["billing"].noul, r.choices["tone"].choice, r.scores["urgency"].score)'
```

Or curl:

```sh
curl localhost:8000/v1/systemone -H "Authorization: Bearer devkey" -H "Content-Type: application/json" \
  -d '{"state":"The export button crashes in Safari.","model":"jev-latest","questions":{"sev":{"type":"score","instructions":"How severe?","criteria":["cosmetic","degraded","blocking"]}}}'
```

Endpoints: `POST /v1/systemone`, `GET /v1/models`, `GET /healthz`, `GET /stats` (batcher
counters and the active backend/prompt/temperature config). Errors follow jev: 401 (bad key),
422 (FastAPI-style validation body, also for exceeded limits), 429 (per-key token bucket,
`retry-after-ms`), 529 (queue full). Every response carries `x-typesafe-request-id`,
`x-jeff-server-ms` and `x-jeff-batcher-ms`.

For quick local iteration use the base checkpoint (`gliformer-base-v1`, 3x faster) with
`JEFF_NOUL_MODE=single`; it is not accurate enough for nouls otherwise.

## Deploy on Modal (GPU)

L4 is the recommended GPU for the HTTP API; A10G when you call the backend directly from other
Modal functions or requests are long (`bench/RESULTS.md` has the data).

```sh
uv tool install modal && modal setup
modal run deploy/modal_gpu.py::download                       # weights -> Volume "jeff-models" (once)
JEFF_GPU=L4 JEFF_API_KEYS=k1 modal deploy deploy/modal_gpu.py  # keeps one warm container
modal serve deploy/modal_gpu.py                               # ephemeral URL, no warm container
```

Deploy-time env: `JEFF_GPU` (L4), `JEFF_MIN_CONTAINERS` (1 on deploy), `JEFF_MAX_CONTAINERS`
(8), `JEFF_MAX_INPUTS` (64 concurrent per container), `JEFF_TARGET_INPUTS` (16, autoscale
threshold), plus any server `JEFF_*` variable, which is forwarded into the container.

Modal's web ingress caps one container at roughly 50 requests/s regardless of GPU, so
throughput scales with containers, not GPU size. For the raw GPU rate (27k tokens/s on L4,
41k on A10G) call the backend from other Modal functions instead of over HTTP.

## CPU arm (ONNX Runtime)

For hosts without a GPU. The DeBERTa encoder runs in ONNX Runtime (fp32 or dynamic int8);
the word-level RNN and classification head stay in torch. Measured on Modal 8-core containers
the large model takes ~300 ms per request at ~$0.71 per 1M tokens, 17x jev, so it is a
fallback, not a cheaper tier. On a Mac, MPS is 2-3x faster.

```sh
uv sync --extra onnx
uv run python scripts/export_onnx.py models/gliformer-large-v1 --int8   # -> models/.../onnx/encoder{,.int8}.onnx
JEFF_BACKEND=onnx JEFF_QUANT=int8 JEFF_THREADS=8 JEFF_API_KEYS=devkey uv run jeff

modal run deploy/modal_cpu.py::export                                   # weights + ONNX -> Volume (once)
JEFF_CPU=8 JEFF_QUANT=int8 JEFF_API_KEYS=k1 modal deploy deploy/modal_cpu.py
```

## Configuration

All settings are environment variables.

| variable | default | meaning |
|---|---|---|
| `JEFF_MODEL` | `models/gliformer-large-v1` | local checkpoint path |
| `JEFF_MODEL_NAME` | `gliformer-large-v1` | name in responses and `GET /v1/models` |
| `JEFF_MODEL_ALIASES` | `jev-latest,jev` | request `model` values accepted as aliases |
| `JEFF_BACKEND` | `torch` | `torch` or `onnx` |
| `JEFF_DEVICE` | auto | `cuda`, `mps`, `cpu` |
| `JEFF_DTYPE` | bf16 on cuda, fp32 elsewhere | |
| `JEFF_API_KEYS` | empty (auth off) | comma-separated bearer keys |
| `JEFF_MAX_BATCH` / `JEFF_MAX_WAIT_MS` | 16 / 5 | dynamic batcher: coalesce up to N requests or wait this long |
| `JEFF_MAX_QUEUE` | 256 | requests waiting before 529 |
| `JEFF_RATE_LIMIT_RPS` / `JEFF_RATE_LIMIT_BURST` | 0 (off) / 20 | per-key token bucket |
| `JEFF_MAX_QUESTIONS` / `JEFF_MAX_LABELS` / `JEFF_MAX_STATE_CHARS` | 64 / 64 / 20000 | request limits, 422 when exceeded |
| `JEFF_TEMPERATURE` | 3.2 | temperature on `probabilities`, `confidence`, `noul`; 1.0 = raw renormalized sigmoids |
| `JEFF_ISOLATE` | `nouls` | which questions get their own encoder pass: `none`, `nouls`, `all` |
| `JEFF_NOUL_MODE` | `yes_no` | noul rendering: `yes_no`, `single`, `single_named` |
| `JEFF_STATE_FORMAT` | `kv` | how object/array `state` is rendered: `kv`, `json`, `values` |
| `JEFF_ATTN` | `auto` | `flash` (flashdeberta Triton kernels, CUDA) or `eager` |
| `JEFF_COMPILE` / `JEFF_COMPILE_MODE` / `JEFF_PAD_MULTIPLE` | 0 / unset / 0 | torch.compile options; off because compile is slower than the flash kernels |
| `JEFF_WARMUP` | 0 | run warmup shapes at startup (on in the Modal image) |
| `JEFF_QUANT` / `JEFF_THREADS` / `JEFF_ONNX_PATH` | fp32 / auto / auto | CPU arm |
| `JEFF_HOST` / `JEFF_PORT` | 0.0.0.0 / 8000 | |

## Benchmarks

Everything below is from `bench/RESULTS.md`, which has the full tables and how to regenerate them.

**Accuracy** on 200 label-balanced items per dataset, one question each, same requests to both
APIs (`bench/eval_accuracy.py`):

| task | metric | jeff (gliformer-large) | jev-latest |
|---|---|---:|---:|
| AG News (4 topics) | accuracy | 0.755 | 0.905 |
| Emotion (6 classes) | accuracy | 0.470 | 0.470 |
| SST-5 (5 levels) | MAE | 0.611 | 0.489 |
| Amazon stars (5 levels) | MAE | 0.600 | 0.467 |
| SMS spam | AUROC | 0.918 | 0.994 |
| SST-2 positive? | AUROC | 0.991 | 0.996 |
| Tweet irony | AUROC | 0.714 | 0.958 |
| BoolQ (passage + question) | AUROC | 0.749 | 0.954 |

**Latency** from a laptop, p50: jev 129 ms, jeff on an L4 via Modal 151 ms (sequential);
131 vs 267 ms at 8 concurrent clients, where one Modal container's ingress serializes. The
model itself takes 28 ms (A10G) to 45 ms (L4) per request at batch 1, 120k tokens/s on H100.

**Cost per 1M single-question requests** (86 jeff tokens / 372 jev tokens each): jev ≈ $15.6;
jeff on L4 at the ingress cap ≈ $2.6; jeff on A10G called directly ≈ $0.65.

## Parity with jev: known differences

Wire format, SDK behaviour and error codes match (`tests/test_sdk_live.py` drives the official
SDK against a live server). The model behind the API differs:

- **Accuracy.** See the table above. jeff classifies from lexical cues; it is close to jev on
  sentiment and spam and well behind on anything that needs inference over the text.
- **Probabilities.** jeff's are renormalized independent sigmoids, temperature-scaled by 3.2.
  With that, calibration error is within 0.03-0.06 of jev's on most tasks. Because `score` is
  computed from the untempered distribution, `score` is not exactly the probability-weighted
  average of the shown `probabilities` (set `JEFF_TEMPERATURE=1` if you need that identity).
  `confidence` is `(p_max - 1/n)/(1 - 1/n)`; jev does not publish its formula.
- **Descriptions.** Option and noul criteria descriptions are folded into label text
  (`key: description`). Required for nouls; neutral for choice. The instruction text is the
  group name; question ids never reach the model.
- **Context effects.** jev answers each question independently of the others in the request
  (measured: identical metrics with three unrelated questions added). jeff isolates nouls, so
  they are independent too; choice and score questions share one encoder pass and lose about
  1.5 points of accuracy / 0.03 MAE with three unrelated questions in front. `JEFF_ISOLATE=all`
  makes every question independent for roughly 50% more tokens and 20% more latency.
- **Token accounting.** `usage.input_tokens` is the DeBERTa token count of prompt + text, about
  4.3x fewer than jev bills for the same request. `output_tokens` is nominal.
- **Limits.** Question, label and state-size limits above return 422; jev's limits are
  undocumented. Encoder cost is quadratic in state length, and every isolated noul is one more
  encoder pass.

## Development

```sh
uv run pytest -q                                   # 30 tests; base checkpoint needed for the integration ones
```

Layout: `src/jeff/core` (schemas, state rendering, prompt groups, decoder, engine; pure
functions), `src/jeff/backends` (torch and ONNX backends), `src/jeff/server` (FastAPI app,
batcher, settings), `deploy/` (Modal GPU and CPU apps), `bench/` (eval set builder and runner,
calibration fit, load generator, noul probe, results), `PLAN.md` (design decisions and the
log of what was measured).

Bench scripts:

```sh
uv run --group bench python bench/build_evalset.py                 # resample bench/data from HF datasets
uv run python bench/eval_accuracy.py jeff --variants default nodesc # in-process, prompt variants
uv run python bench/eval_accuracy.py jev -c 8                      # real jev; TYPESAFE_API_KEY from .env
uv run python bench/eval_accuracy.py http --url URL --key K --name jeff-l4
uv run python bench/eval_accuracy.py summarize --full              # markdown tables
uv run python bench/calibrate.py --run large/default               # temperature fit
uv run python bench/load.py --url URL --key K -c 1 8 32 -n 200     # HTTP load test
```
