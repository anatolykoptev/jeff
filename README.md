# jeff

A [jev](https://docs.typesafe.ai/api)-compatible System One API served by
[GLiFormer](https://huggingface.co/knowledgator/gliformer-large-v1).

See `PLAN.md` for status. Quick check:

```sh
uv sync --extra dev
uv run hf download knowledgator/gliformer-base-v1 --local-dir models/gliformer-base-v1
uv run pytest -q
```

## Run the server

```sh
JEFF_MODEL=models/gliformer-large-v1 JEFF_API_KEYS=devkey uv run jeff
```

Then use the official SDK unchanged:

```sh
pip install typesafe-sdk
TYPESAFE_API_KEY=devkey TYPESAFE_BASE_URL=http://localhost:8000 python -c '
from typesafe_sdk import TypeSafeClient, Noul, Choice, Score
c = TypeSafeClient()
r = c.system_one("I was charged twice. Please help ASAP.", {
    "billing": Noul(instructions="Is this about billing?"),
    "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
    "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
})
print(r.nouls["billing"].noul, r.choices["tone"].choice, r.scores["urgency"].score)'
```

Or curl:

```sh
curl localhost:8000/v1/systemone -H "Authorization: Bearer devkey" -H "Content-Type: application/json" \
  -d '{"state":"The export button crashes in Safari.","model":"jev-latest","questions":{"sev":{"type":"score","instructions":"How severe?","criteria":["cosmetic","degraded","blocking"]}}}'
```

Env vars: `JEFF_MODEL`, `JEFF_MODEL_NAME`, `JEFF_DEVICE` (cuda/mps/cpu), `JEFF_DTYPE`, `JEFF_COMPILE=1`,
`JEFF_API_KEYS` (comma-separated; empty disables auth), `JEFF_MAX_BATCH`, `JEFF_MAX_WAIT_MS`, `JEFF_MAX_QUEUE`,
`JEFF_RATE_LIMIT_RPS`, `JEFF_MAX_QUESTIONS`, `JEFF_MAX_LABELS`, `JEFF_MAX_STATE_CHARS`, `JEFF_PORT`.
