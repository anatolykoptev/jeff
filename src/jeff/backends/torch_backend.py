"""GPU / reference arm: GLiFormer via PyTorch.

Runs on cuda (bf16 + flashdeberta kernels), mps, or cpu (eager attention).
This replicates ``GLiFormer.inference`` closely enough to inject a per-group
``description`` (the public ``classify()`` only accepts names + labels) and
to record prompt token counts for ``usage``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..core.backend import Group, ScoredText

# ``threshold or self.threshold`` inside the decoder treats 0.0 as unset, so a
# negative threshold is the reliable way to get every label back.
ALL_LABELS_THRESHOLD = -1.0


class TorchBackend:
    name = "torch"

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        dtype: str | None = None,
        compile_model: bool = False,
        batch_size: int = 16,
    ):
        from gliformer import GLiFormer

        self.device = device or _default_device()
        kwargs: dict[str, Any] = {}
        if dtype:
            kwargs["dtype"] = dtype
        elif self.device == "cuda":
            kwargs["dtype"] = "bfloat16"
        if compile_model:
            kwargs["compile_torch_model"] = True
        t0 = time.time()
        self.model = GLiFormer.from_pretrained(model_path, map_location=self.device, **kwargs)
        self.model.to(self.device).eval()
        self.load_seconds = time.time() - t0
        self.model_path = model_path
        self.batch_size = batch_size

        collator_cls = self.model.data_collator_class or _resolve_collator(self.model.config)
        self._collator = collator_cls(
            self.model.config,
            data_processor=self.model.data_processor,
            return_tokens=True,
            prepare_labels=False,
        )

    @torch.inference_mode()
    def score(self, texts: list[str], groups: list[list[Group]]) -> list[ScoredText]:
        m = self.model
        m.eval()
        tokens, _, _ = m.prepare_inputs(texts)
        items = []
        for toks, gs in zip(tokens, groups):
            items.append(
                {
                    "tokenized_text": toks,
                    "classification": [
                        {
                            "name": g.name,
                            "description": g.description,
                            "all_labels": list(g.labels),
                            "true_labels": [],
                        }
                        for g in gs
                    ],
                }
            )

        seq_lens: list[int] = []

        def collate(batch):
            out = self._collator(batch)
            seq_lens.extend(int(x) for x in out["attention_mask"].sum(dim=1).tolist())
            return out

        loader = DataLoader(items, batch_size=self.batch_size, shuffle=False, collate_fn=collate)
        decoded, _ = m._process_multitask_batches(
            loader, ALL_LABELS_THRESHOLD, True, True, decoder_kwargs=None
        )
        per_text = decoded.get("classification", [[] for _ in texts])

        results = []
        for gs, groups_out, n_tok in zip(groups, per_text, seq_lens):
            scores: dict[str, list[float]] = {}
            for g, preds in zip(gs, groups_out):
                by_name = {p["class_name"]: float(p["score"]) for p in preds}
                scores[g.key] = [by_name.get(label, 0.0) for label in g.labels]
            results.append(ScoredText(scores=scores, input_tokens=n_tok))
        return results


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_collator(config):
    from gliformer.gliformer import resolve_gliformer_collator_class

    return resolve_gliformer_collator_class(config)


def from_env() -> TorchBackend:
    return TorchBackend(
        model_path=os.environ.get("JEFF_MODEL", "models/gliformer-large-v1"),
        device=os.environ.get("JEFF_DEVICE"),
        dtype=os.environ.get("JEFF_DTYPE"),
        compile_model=os.environ.get("JEFF_COMPILE", "0") == "1",
        batch_size=int(os.environ.get("JEFF_MAX_BATCH", "16")),
    )
