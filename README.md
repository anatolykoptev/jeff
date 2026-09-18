# jeff

A [jev](https://docs.typesafe.ai/api)-compatible System One API served by
[GLiFormer](https://huggingface.co/knowledgator/gliformer-large-v1).

See `PLAN.md` for status. Quick check:

```sh
uv sync --extra dev
uv run hf download knowledgator/gliformer-base-v1 --local-dir models/gliformer-base-v1
uv run pytest -q
```
