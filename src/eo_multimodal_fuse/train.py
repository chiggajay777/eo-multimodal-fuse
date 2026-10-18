"""Synthetic dual-modality demo.

Offline fallback for the Sentinel benchmark. Metrics use the same non-NaN
scorer as the real run, including both missing-modality conditions.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eo_multimodal_fuse.data import make_loaders
from eo_multimodal_fuse.engine import predict, train_model
from eo_multimodal_fuse.metrics import (
    SEED_COLUMNS,
    SUMMARY_COLUMNS,
    aggregate,
    class_weights,
    markdown_table,
    safe_scores,
    write_csv,
    write_json,
)
from eo_multimodal_fuse.models import CrossAttentionFusion, LateFusion, MLP
from eo_multimodal_fuse.plots import plot_full_modality, plot_missing_modality

DEMO_FULL = [
    "optical_only",
    "sar_only",
    "late_fusion",
    "late_fusion_moddrop",
    "cross_attention_fusion",
    "cross_attention_fusion_moddrop",
]
DEMO_FUSION = [
    "late_fusion",
    "late_fusion_moddrop",
    "cross_attention_fusion",
    "cross_attention_fusion_moddrop",
]


class _UniLoader:
    def __init__(self, base, which: str):
        self.base = base
        self.which = which

    def __iter__(self):
        for opt, sar, y in self.base:
            yield (opt if self.which == "opt" else sar), y


def _score(model, loader, device, multimodal: bool, missing: str | None) -> dict:
    labels, preds, probs, n_bad = predict(
        model, loader, device=device, multimodal=multimodal, missing=missing
    )
    scores = safe_scores(np.concatenate(labels), np.concatenate(preds), np.concatenate(probs))
    if n_bad:
        scores["notes"].append(f"non-finite logits replaced with zeros on {n_bad} samples")
    return scores


def run_demo(
    out_dir: Path,
    *,
    n: int = 2000,
    epochs: int = 8,
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
    modality_dropout: float = 0.5,
    lr: float = 1e-3,
) -> dict:
    """Single-seed synthetic run. Not the reported benchmark."""
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or "cpu")
    train_loader, val_loader = make_loaders(n=n, batch_size=batch_size, seed=seed)
    train_y = []
    for batch in train_loader:
        train_y.append(batch[-1].numpy())
    weight = torch.tensor(class_weights(np.concatenate(train_y)), dtype=torch.float32)

    specs = [
        ("optical_only", MLP(4), False, 0.0, "opt"),
        ("sar_only", MLP(4), False, 0.0, "sar"),
        ("late_fusion", LateFusion(), True, 0.0, None),
        ("late_fusion_moddrop", LateFusion(), True, modality_dropout, None),
        ("cross_attention_fusion", CrossAttentionFusion(), True, 0.0, None),
        ("cross_attention_fusion_moddrop", CrossAttentionFusion(), True, modality_dropout, None),
    ]
    rows = []
    for name, model, multimodal, drop_p, which in specs:
        model = model.to(dev)
        source_train = _UniLoader(train_loader, which) if not multimodal else train_loader
        source_val = _UniLoader(val_loader, which) if not multimodal else val_loader
        train_model(
            model,
            source_train,
            device=dev,
            epochs=epochs,
            lr=lr,
            multimodal=multimodal,
            class_weight=weight,
            modality_dropout=drop_p if multimodal else 0.0,
            augment=False,
        )
        conditions = [("full", None)]
        if multimodal:
            conditions.extend([("optical_missing", "optical"), ("sar_missing", "sar")])
        for condition, missing in conditions:
            scores = _score(model, source_val, dev, multimodal, missing)
            rows.append({"split": "synthetic", "method": name, "condition": condition, "seed": seed, **scores})

    summary = aggregate(rows)
    payload = {
        "dataset": "synthetic optical(4) + SAR-like(4) binary labels",
        "note": "Offline demo. Reported portfolio numbers come from the Sen1Floods11 run.",
        "n_samples": n,
        "epochs": epochs,
        "seed": seed,
        "device": str(dev),
        "modality_dropout": modality_dropout,
        "per_seed": rows,
        "summary": summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "metrics_table.csv", summary, SUMMARY_COLUMNS)
    write_csv(out_dir / "metrics_by_seed.csv", rows, SEED_COLUMNS)
    table = markdown_table(summary, "synthetic", DEMO_FULL, ["full"])
    table += "\n" + markdown_table(summary, "synthetic", DEMO_FUSION, ["full", "optical_missing", "sar_missing"])
    (out_dir / "table.md").write_text(table, encoding="utf-8")
    plot_full_modality(
        summary, out_dir / "figures" / "fusion_accuracy.png", split="synthetic", methods=DEMO_FULL
    )
    plot_missing_modality(
        summary, out_dir / "figures" / "missing_modality.png", split="synthetic", methods=DEMO_FUSION
    )
    payload["artifacts"] = {
        "metrics_json": str(out_dir / "metrics.json"),
        "metrics_csv": str(out_dir / "metrics_table.csv"),
        "figure": str(out_dir / "figures" / "fusion_accuracy.png"),
    }
    return payload
