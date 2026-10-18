"""Classification metrics that never emit NaN.

Degenerate predictions (one predicted class, constant scores, or a one-class
label set) stay numeric where a definition exists. AUROC is JSON null with a
short explanation when it is undefined.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def safe_scores(y_true, y_pred, y_prob, *, note_collapse: bool = True) -> dict:
    """Accuracy, macro-F1, and AUROC.

    Macro-F1 uses ``zero_division=0`` so a missing predicted class contributes
    0 instead of NaN. AUROC is ``None`` when the labels have a single class or
    the probabilities are non-finite.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.size == 0:
        raise RuntimeError("empty evaluation set")
    if y_true.shape != y_pred.shape or y_true.shape != y_prob.shape:
        raise ValueError("y_true, y_pred, and y_prob must share a shape")

    notes: list[str] = []
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if not math.isfinite(accuracy) or not math.isfinite(macro_f1):
        raise RuntimeError("non-finite accuracy or macro-F1")

    auroc: float | None = None
    auroc_note: str | None = None
    if not np.isfinite(y_prob).all():
        auroc_note = "undefined: predicted probabilities were non-finite"
        notes.append(auroc_note)
    elif len(np.unique(y_true)) < 2:
        auroc_note = "undefined: labels contain a single class, so AUROC is not defined"
        notes.append(auroc_note)
    else:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError as exc:
            auroc_note = f"undefined: {exc}"
            notes.append(auroc_note)
        else:
            if not math.isfinite(auroc):
                auroc = None
                auroc_note = "undefined: AUROC was non-finite"
                notes.append(auroc_note)

    if note_collapse and len(np.unique(y_pred)) < 2:
        notes.append(
            "predictions collapsed to a single class; "
            "macro-F1 uses zero_division=0 for the absent class"
        )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "auroc_note": auroc_note,
        "pred_positive_rate": float(np.mean(y_pred == 1)),
        "positive_rate": float(np.mean(y_true == 1)),
        "n": int(y_true.size),
        "notes": notes,
    }


def class_weights(y) -> np.ndarray:
    """Inverse-frequency weights for two classes, shape (2,)."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(np.float64)
    if counts.shape[0] != 2 or np.any(counts == 0):
        raise RuntimeError(f"training labels must contain both classes, counts={counts.tolist()}")
    return counts.sum() / (2.0 * counts)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if not math.isfinite(mean):
        raise RuntimeError("non-finite mean")
    if len(arr) < 2:
        return mean, None
    std = float(arr.std(ddof=1))
    if not math.isfinite(std):
        raise RuntimeError("non-finite standard deviation")
    return mean, std


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean and sample standard deviation (ddof=1) across seeds.

    Rows that share ``(split, method, condition)`` are grouped. AUROC mean/std
    use only seeds where AUROC is defined.
    """
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["split"], row["method"], row["condition"])
        groups.setdefault(key, []).append(row)

    summary: list[dict] = []
    for key in sorted(groups):
        members = groups[key]
        acc_m, acc_s = _mean_std([m["accuracy"] for m in members])
        f1_m, f1_s = _mean_std([m["macro_f1"] for m in members])
        auroc_vals = [m["auroc"] for m in members if m["auroc"] is not None]
        auroc_m, auroc_s = _mean_std(auroc_vals)
        pred_m, _ = _mean_std([m["pred_positive_rate"] for m in members])
        notes: list[str] = []
        if len(members) < 2:
            notes.append("standard deviation is undefined with fewer than 2 seeds")
        n_defined = len(auroc_vals)
        if n_defined < len(members):
            reasons = sorted({m["auroc_note"] for m in members if m["auroc"] is None and m.get("auroc_note")})
            reason = "; ".join(reasons) if reasons else "undefined"
            notes.append(f"AUROC defined for {n_defined}/{len(members)} seeds ({reason})")
        collapsed = sum(
            1
            for m in members
            if any("collapsed" in note for note in m.get("notes", []))
        )
        if collapsed:
            notes.append(f"predictions collapsed to one class in {collapsed}/{len(members)} seeds")
        # Preserve per-seed explanatory notes that are not already covered.
        extra = []
        for m in members:
            for note in m.get("notes", []):
                if note not in extra and "collapsed" not in note:
                    extra.append(note)
        summary.append(
            {
                "split": key[0],
                "method": key[1],
                "condition": key[2],
                "n_seeds": len(members),
                "n": members[0]["n"],
                "positive_rate": members[0]["positive_rate"],
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "macro_f1_mean": f1_m,
                "macro_f1_std": f1_s,
                "auroc_mean": auroc_m,
                "auroc_std": auroc_s,
                "auroc_n_defined": n_defined,
                "pred_positive_rate_mean": pred_m,
                "notes": notes,
            }
        )
    return summary


def format_mean_std(mean: float | None, std: float | None, digits: int = 3) -> str:
    if mean is None:
        return "undefined"
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_table(summary: list[dict], split: str, methods: list[str], conditions: list[str]) -> str:
    header = "| Method | Condition | Accuracy | Macro-F1 | AUROC |"
    rule = "|---|---|---|---|---|"
    lines = [header, rule]
    by_key = {(r["split"], r["method"], r["condition"]): r for r in summary}
    for method in methods:
        for condition in conditions:
            row = by_key.get((split, method, condition))
            if row is None:
                continue
            lines.append(
                "| {method} | {condition} | {acc} | {f1} | {auroc} |".format(
                    method=method,
                    condition=condition,
                    acc=format_mean_std(row["accuracy_mean"], row["accuracy_std"]),
                    f1=format_mean_std(row["macro_f1_mean"], row["macro_f1_std"]),
                    auroc=format_mean_std(row["auroc_mean"], row["auroc_std"]),
                )
            )
    return "\n".join(lines) + "\n"


_METRIC_COLUMNS = {
    "accuracy",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1",
    "macro_f1_mean",
    "macro_f1_std",
    "auroc",
    "auroc_mean",
    "auroc_std",
}


def _cell(value, column: str) -> str:
    if value is None:
        # Missing metrics are explicit. Optional notes stay blank.
        return "undefined" if column in _METRIC_COLUMNS else ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _cell(row.get(col), col) for col in columns})


def jsonable(obj):
    """Convert numpy values and reject non-finite floats."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("refusing to serialize a non-finite float")
        return obj
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(payload), indent=2, allow_nan=False)
    if "NaN" in text or "Infinity" in text:
        raise ValueError("JSON payload contained a non-finite token")
    path.write_text(text + "\n", encoding="utf-8")


SUMMARY_COLUMNS = [
    "split",
    "method",
    "condition",
    "n_seeds",
    "n",
    "positive_rate",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "auroc_mean",
    "auroc_std",
    "auroc_n_defined",
    "pred_positive_rate_mean",
    "notes",
]

SEED_COLUMNS = [
    "split",
    "method",
    "condition",
    "seed",
    "accuracy",
    "macro_f1",
    "auroc",
    "auroc_note",
    "pred_positive_rate",
    "positive_rate",
    "n",
    "notes",
]
