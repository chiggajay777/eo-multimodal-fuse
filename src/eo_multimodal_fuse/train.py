"""Train unimodal baselines vs fusion; emit metrics table + figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from eo_multimodal_fuse.data import make_loaders
from eo_multimodal_fuse.models import CrossAttentionFusion, LateFusion, MLP


def _train(model, loader, device, epochs, lr, multimodal: bool):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            if multimodal:
                x_opt, x_sar, y = batch
                x_opt, x_sar, y = x_opt.to(device), x_sar.to(device), y.to(device)
                logits = model(x_opt, x_sar)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
            opt.zero_grad()
            loss = crit(logits, y)
            loss.backward()
            opt.step()


def _eval(model, loader, device, multimodal: bool) -> dict:
    model.eval()
    ys, ps, pr = [], [], []
    with torch.no_grad():
        for batch in loader:
            if multimodal:
                x_opt, x_sar, y = batch
                x_opt, x_sar = x_opt.to(device), x_sar.to(device)
                logits = model(x_opt, x_sar)
            else:
                x, y = batch
                x = x.to(device)
                logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
            pr.append(prob)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(pr)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auroc"] = float("nan")
    return out


class _UniLoader:
    """Wrap dual loader to expose one modality."""

    def __init__(self, base, which: str):
        self.base = base
        self.which = which

    def __iter__(self):
        for opt, sar, y in self.base:
            yield (opt if self.which == "opt" else sar), y


def run_demo(
    out_dir: Path,
    *,
    n: int = 2000,
    epochs: int = 8,
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader = make_loaders(n=n, batch_size=batch_size, seed=seed)

    results_methods = []

    # Optical-only
    m_opt = MLP(4).to(dev)
    _train(m_opt, _UniLoader(train_loader, "opt"), dev, epochs, 1e-3, multimodal=False)
    metrics = _eval(m_opt, _UniLoader(val_loader, "opt"), dev, multimodal=False)
    results_methods.append({"name": "optical_only", **metrics})

    # SAR-only
    m_sar = MLP(4).to(dev)
    _train(m_sar, _UniLoader(train_loader, "sar"), dev, epochs, 1e-3, multimodal=False)
    metrics = _eval(m_sar, _UniLoader(val_loader, "sar"), dev, multimodal=False)
    results_methods.append({"name": "sar_only", **metrics})

    # Late fusion
    m_late = LateFusion().to(dev)
    _train(m_late, train_loader, dev, epochs, 1e-3, multimodal=True)
    metrics = _eval(m_late, val_loader, dev, multimodal=True)
    results_methods.append({"name": "late_fusion", **metrics})

    # Cross-attention fusion
    m_xattn = CrossAttentionFusion().to(dev)
    _train(m_xattn, train_loader, dev, epochs, 1e-3, multimodal=True)
    metrics = _eval(m_xattn, val_loader, dev, multimodal=True)
    results_methods.append({"name": "cross_attention_fusion", **metrics})

    # Missing-modality robustness: late fusion with SAR zeros at val
    m_late.eval()
    ys, ps = [], []
    with torch.no_grad():
        for opt, sar, y in val_loader:
            opt = opt.to(dev)
            sar0 = torch.zeros_like(sar).to(dev)
            pred = m_late(opt, sar0).argmax(dim=1).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
    miss_acc = float(accuracy_score(np.concatenate(ys), np.concatenate(ps)))
    results_methods.append(
        {
            "name": "late_fusion_sar_missing",
            "accuracy": miss_acc,
            "f1": float("nan"),
            "auroc": float("nan"),
            "note": "SAR inputs zeroed at eval — robustness check",
        }
    )

    payload = {
        "device": str(dev),
        "dataset": "Synthetic optical(4) + SAR-like(4) binary risk",
        "n_samples": n,
        "epochs": epochs,
        "methods": results_methods,
    }

    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_lines = ["method,accuracy,f1,auroc"]
    for m in results_methods:
        csv_lines.append(
            f"{m['name']},{m.get('accuracy', '')},{m.get('f1', '')},{m.get('auroc', '')}"
        )
    csv_path = out_dir / "metrics_table.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # Figure — accuracy bars for main methods
    main = [m for m in results_methods if m["name"] != "late_fusion_sar_missing"]
    names = [m["name"] for m in main]
    accs = [m["accuracy"] for m in main]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#90be6d", "#f9c74f", "#577590", "#43aa8b"]
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("eo-multimodal-fuse — baseline vs fusion")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig_path = fig_dir / "fusion_accuracy.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    payload["artifacts"] = {
        "metrics_json": str(out_dir / "metrics.json"),
        "metrics_csv": str(csv_path),
        "figure": str(fig_path),
    }
    return payload
