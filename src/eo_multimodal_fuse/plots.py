"""Bar charts with sample-standard-deviation error bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_LABELS = {
    "optical_indices": "opt\nindices",
    "optical_only": "opt\nCNN",
    "sar_indices": "SAR\nindices",
    "sar_only": "SAR\nCNN",
    "late_fusion": "late",
    "late_fusion_moddrop": "late\n+drop",
    "cross_attention_fusion": "xattn",
    "cross_attention_fusion_moddrop": "xattn\n+drop",
}
METHOD_COLORS = {
    "optical_indices": "#4C78A8",
    "optical_only": "#72B7B2",
    "sar_indices": "#F58518",
    "sar_only": "#E45756",
    "late_fusion": "#54A24B",
    "late_fusion_moddrop": "#1B7F4E",
    "cross_attention_fusion": "#B279A2",
    "cross_attention_fusion_moddrop": "#725193",
}
CONDITION_COLORS = {
    "full": "#0072B2",
    "optical_missing": "#E69F00",
    "sar_missing": "#009E73",
}
CONDITION_LABELS = {
    "full": "Both modalities",
    "optical_missing": "Optical missing",
    "sar_missing": "SAR missing",
}


def _lookup(summary: list[dict], split: str, method: str, condition: str) -> dict | None:
    for row in summary:
        if row["split"] == split and row["method"] == method and row["condition"] == condition:
            return row
    return None


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_full_modality(
    summary: list[dict],
    path: Path,
    *,
    split: str,
    methods: list[str],
    reference: dict | None = None,
) -> None:
    """Grouped accuracy and macro-F1 bars for the full-modality condition."""
    rows = []
    for method in methods:
        row = _lookup(summary, split, method, "full")
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no full-modality rows for split {split}")
    labels = [METHOD_LABELS.get(row["method"], row["method"]) for row in rows]
    colors = [METHOD_COLORS.get(row["method"], "#4C78A8") for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    specs = (
        (axes[0], "accuracy_mean", "accuracy_std", "Accuracy"),
        (axes[1], "macro_f1_mean", "macro_f1_std", "Macro-F1"),
    )
    for ax, mean_key, std_key, title in specs:
        means = [row[mean_key] for row in rows]
        stds = [row[std_key] for row in rows]
        yerr = None if any(value is None for value in stds) else stds
        ax.bar(x, means, color=colors, yerr=yerr, capsize=3, ecolor="#222222", width=0.78)
        if reference and reference.get(mean_key.replace("_mean", "")) is not None:
            ref_key = "accuracy" if "accuracy" in mean_key else "macro_f1"
            ax.axhline(reference[ref_key], color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    axes[0].set_ylabel("Score on held-out chips")
    fig.suptitle(f"Full modality ({split}), mean ± sample std across seeds", fontsize=12)
    if reference:
        axes[1].plot([], [], color="#333333", linestyle="--", label="majority class")
        fig.legend(frameon=False, fontsize=8, loc="outside lower center", ncol=1)
    fig.tight_layout()
    _save(fig, path)


def plot_missing_modality(
    summary: list[dict],
    path: Path,
    *,
    split: str,
    methods: list[str],
) -> None:
    """Macro-F1 and AUROC when a fusion modality is replaced by zeros."""
    conditions = ["full", "optical_missing", "sar_missing"]
    present = [method for method in methods if _lookup(summary, split, method, "full")]
    if not present:
        raise RuntimeError(f"no fusion rows for split {split}")
    x = np.arange(len(present))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    specs = (
        (axes[0], "macro_f1_mean", "macro_f1_std", "Macro-F1"),
        (axes[1], "auroc_mean", "auroc_std", "AUROC"),
    )
    for ax, mean_key, std_key, title in specs:
        for i, condition in enumerate(conditions):
            means = []
            stds = []
            for method in present:
                row = _lookup(summary, split, method, condition)
                means.append(0.0 if row is None or row[mean_key] is None else row[mean_key])
                stds.append(0.0 if row is None or row[std_key] is None else row[std_key])
            bar_kwargs = {}
            if ax is axes[0]:
                bar_kwargs["label"] = CONDITION_LABELS[condition]
            ax.bar(
                x + (i - 1) * width,
                means,
                width=width,
                color=CONDITION_COLORS[condition],
                yerr=stds,
                capsize=2,
                ecolor="#222222",
                **bar_kwargs,
            )
        ax.set_xticks(x, [METHOD_LABELS.get(method, method) for method in present], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    axes[0].set_ylabel("Score")
    fig.suptitle(
        f"Missing-modality robustness ({split}), mean ± sample std",
        fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=3)
    fig.tight_layout()
    _save(fig, path)


def plot_geographic_shift(
    summary: list[dict],
    path: Path,
    *,
    methods: list[str],
    source_split: str = "test",
    shift_split: str = "bolivia",
) -> None:
    """Macro-F1 and AUROC on the in-domain chip hold-out versus Bolivia."""
    present = [
        method
        for method in methods
        if _lookup(summary, source_split, method, "full") and _lookup(summary, shift_split, method, "full")
    ]
    if not present:
        return
    x = np.arange(len(present))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    series = (
        (source_split, "In-domain test", "#0072B2"),
        (shift_split, "Bolivia hold-out", "#D55E00"),
    )
    specs = (
        (axes[0], "macro_f1_mean", "macro_f1_std", "Macro-F1"),
        (axes[1], "auroc_mean", "auroc_std", "AUROC"),
    )
    for ax, mean_key, std_key, title in specs:
        for i, (split, label, color) in enumerate(series):
            means, stds = [], []
            for method in present:
                row = _lookup(summary, split, method, "full")
                means.append(0.0 if row[mean_key] is None else row[mean_key])
                stds.append(0.0 if row[std_key] is None else row[std_key])
            bar_kwargs = {}
            if ax is axes[0]:
                bar_kwargs["label"] = label
            ax.bar(
                x + (i - 0.5) * width,
                means,
                width=width,
                color=color,
                yerr=stds,
                capsize=2,
                ecolor="#222222",
                **bar_kwargs,
            )
        ax.set_xticks(x, [METHOD_LABELS.get(method, method) for method in present], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    axes[0].set_ylabel("Full-modality score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=2)
    fig.suptitle("Same-event chip hold-out vs Bolivia geographic hold-out", fontsize=12)
    fig.tight_layout()
    _save(fig, path)
