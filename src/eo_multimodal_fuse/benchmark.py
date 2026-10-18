"""Multi-seed Sentinel-1 / Sentinel-2 benchmark."""

from __future__ import annotations

import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

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
from eo_multimodal_fuse.models import ImageCrossAttentionFusion, ImageLateFusion, ImageUnimodal
from eo_multimodal_fuse.plots import plot_full_modality, plot_geographic_shift, plot_missing_modality
from eo_multimodal_fuse.realdata import (
    OPTICAL_FEATURE_NAMES,
    SAR_FEATURE_NAMES,
    prepare_dataset,
)

FULL_METHODS = [
    "majority_class",
    "optical_indices",
    "optical_only",
    "sar_indices",
    "sar_only",
    "late_fusion",
    "late_fusion_moddrop",
    "cross_attention_fusion",
    "cross_attention_fusion_moddrop",
]
PLOT_METHODS = [name for name in FULL_METHODS if name != "majority_class"]
FUSION_METHODS = [
    "late_fusion",
    "late_fusion_moddrop",
    "cross_attention_fusion",
    "cross_attention_fusion_moddrop",
]
FUSION_CONDITIONS = ["full", "optical_missing", "sar_missing"]


class PairDataset(Dataset):
    def __init__(self, optical, sar, y):
        self.optical = optical
        self.sar = sar
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        opt = torch.from_numpy(np.asarray(self.optical[idx], dtype=np.float32))
        sar = torch.from_numpy(np.asarray(self.sar[idx], dtype=np.float32))
        return opt, sar, int(self.y[idx])


class SingleDataset(Dataset):
    def __init__(self, array, y):
        self.array = array
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.asarray(self.array[idx], dtype=np.float32))
        return x, int(self.y[idx])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(dataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _concat(chunks) -> np.ndarray:
    return np.concatenate(chunks, axis=0)


def _score_loader(model, loader, device, multimodal: bool, missing: str | None) -> dict:
    labels, preds, probs, n_bad = predict(
        model, loader, device=device, multimodal=multimodal, missing=missing
    )
    scores = safe_scores(_concat(labels), _concat(preds), _concat(probs))
    if n_bad:
        scores["notes"].append(f"non-finite logits replaced with zeros on {n_bad} samples")
    return scores


def _row(split: str, method: str, condition: str, seed: int, scores: dict) -> dict:
    return {
        "split": split,
        "method": method,
        "condition": condition,
        "seed": seed,
        **scores,
    }


def _majority(train_y: np.ndarray, eval_y: np.ndarray) -> dict:
    counts = np.bincount(train_y.astype(np.int64), minlength=2)
    majority = int(np.argmax(counts))
    pred = np.full(eval_y.shape, majority, dtype=np.int64)
    prob = np.full(eval_y.shape, float(np.mean(train_y == 1)), dtype=np.float64)
    scores = safe_scores(eval_y, pred, prob, note_collapse=False)
    scores["notes"].append(
        "deterministic majority class of the training labels; "
        "AUROC uses a constant score equal to the training positive rate"
    )
    return scores


def _logreg(train_x, train_y, eval_x, eval_y, seed: int) -> dict:
    scaler = StandardScaler()
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(scaler.fit_transform(train_x), train_y)
    transformed = scaler.transform(eval_x)
    classes = list(model.classes_)
    if 1 not in classes:
        prob = np.zeros(eval_y.shape[0], dtype=np.float64)
        pred = model.predict(transformed)
    else:
        prob = model.predict_proba(transformed)[:, classes.index(1)]
        pred = model.predict(transformed)
    return safe_scores(eval_y, pred, prob)


def _neural_specs(modality_dropout: float):
    return [
        ("optical_only", lambda: ImageUnimodal(6), False, 0.0, "opt"),
        ("sar_only", lambda: ImageUnimodal(2), False, 0.0, "sar"),
        ("late_fusion", lambda: ImageLateFusion(), True, 0.0, None),
        ("late_fusion_moddrop", lambda: ImageLateFusion(), True, modality_dropout, None),
        ("cross_attention_fusion", lambda: ImageCrossAttentionFusion(), True, 0.0, None),
        (
            "cross_attention_fusion_moddrop",
            lambda: ImageCrossAttentionFusion(),
            True,
            modality_dropout,
            None,
        ),
    ]


def _split_names(bundle: dict) -> list[str]:
    names = ["test"]
    if "bolivia_y" in bundle["arrays"]:
        names.append("bolivia")
    return names


def run_benchmark(
    out_dir: str | Path,
    data_dir: str | Path,
    *,
    seeds: list[int],
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    train_chips_per_event: int = 4,
    test_chips_per_event: int = 2,
    modality_dropout: float = 0.5,
    patch_size: int = 64,
    water_fraction: float = 0.05,
    include_bolivia: bool = True,
    limit_events: int | None = None,
    device: str | None = None,
    num_threads: int | None = None,
    publish_figures: str | Path | None = None,
    argv: list[str] | None = None,
) -> dict:
    """Train every method on each seed and write metrics, tables, and figures."""
    if len(seeds) < 1:
        raise ValueError("at least one seed is required")
    started = time.perf_counter()
    threads = num_threads or os.cpu_count() or 1
    torch.set_num_threads(threads)
    dev = torch.device(device or "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_started = time.perf_counter()
    bundle = prepare_dataset(
        data_dir,
        train_chips_per_event=train_chips_per_event,
        test_chips_per_event=test_chips_per_event,
        patch_size=patch_size,
        water_fraction=water_fraction,
        include_bolivia=include_bolivia,
        limit_events=limit_events,
    )
    data_seconds = time.perf_counter() - data_started
    arrays = bundle["arrays"]
    train_y = arrays["train_y"]
    weight = torch.tensor(class_weights(train_y), dtype=torch.float32)
    splits = _split_names(bundle)
    pair = {
        name: PairDataset(arrays[f"{name}_optical"], arrays[f"{name}_sar"], arrays[f"{name}_y"])
        for name in ["train", *splits]
    }
    rows: list[dict] = []
    train_full: list[dict] = []

    print(
        f"training on {dev} with {threads} threads, {len(train_y)} train patches, "
        f"seeds {seeds}, epochs {epochs}",
        flush=True,
    )
    for seed in seeds:
        for split in splits:
            scores = _majority(train_y, arrays[f"{split}_y"])
            rows.append(_row(split, "majority_class", "full", seed, scores))
            rows.append(
                _row(
                    split,
                    "optical_indices",
                    "full",
                    seed,
                    _logreg(
                        arrays["train_optical_features"],
                        train_y,
                        arrays[f"{split}_optical_features"],
                        arrays[f"{split}_y"],
                        seed,
                    ),
                )
            )
            rows.append(
                _row(
                    split,
                    "sar_indices",
                    "full",
                    seed,
                    _logreg(
                        arrays["train_sar_features"],
                        train_y,
                        arrays[f"{split}_sar_features"],
                        arrays[f"{split}_y"],
                        seed,
                    ),
                )
            )

        for name, builder, multimodal, drop_p, which in _neural_specs(modality_dropout):
            set_seed(seed)
            model = builder()
            if multimodal:
                train_ds = pair["train"]
            else:
                source = arrays["train_optical"] if which == "opt" else arrays["train_sar"]
                train_ds = SingleDataset(source, train_y)
            train_loader = _loader(train_ds, batch_size, seed, shuffle=True)
            losses = train_model(
                model,
                train_loader,
                device=dev,
                epochs=epochs,
                lr=lr,
                multimodal=multimodal,
                class_weight=weight,
                modality_dropout=drop_p if multimodal else 0.0,
                augment=True,
            )
            # Training-set score for the full-modality condition only.
            train_eval = _loader(train_ds, batch_size, seed, shuffle=False)
            train_scores = _score_loader(model, train_eval, dev, multimodal, None)
            train_full.append(
                {
                    "method": name,
                    "seed": seed,
                    "train_loss_per_epoch": losses,
                    "train_loss_last": losses[-1],
                    "train_accuracy": train_scores["accuracy"],
                    "train_macro_f1": train_scores["macro_f1"],
                    "train_auroc": train_scores["auroc"],
                }
            )
            bits = []
            for split in splits:
                if multimodal:
                    eval_ds = pair[split]
                else:
                    source = arrays[f"{split}_optical"] if which == "opt" else arrays[f"{split}_sar"]
                    eval_ds = SingleDataset(source, arrays[f"{split}_y"])
                eval_loader = _loader(eval_ds, batch_size, seed, shuffle=False)
                conditions = [("full", None)]
                if multimodal:
                    conditions.extend([("optical_missing", "optical"), ("sar_missing", "sar")])
                for condition, missing in conditions:
                    scores = _score_loader(model, eval_loader, dev, multimodal, missing)
                    rows.append(_row(split, name, condition, seed, scores))
                    if split == "test":
                        auroc = "undefined" if scores["auroc"] is None else f"{scores['auroc']:.3f}"
                        bits.append(
                            f"{condition} acc {scores['accuracy']:.3f} "
                            f"f1 {scores['macro_f1']:.3f} auroc {auroc}"
                        )
            loss_text = ",".join(f"{value:.3f}" for value in losses)
            print(
                f"[seed {seed}] {name} loss[{loss_text}] | " + " | ".join(bits),
                flush=True,
            )

    summary = aggregate(rows)
    payload = _payload(
        bundle=bundle,
        rows=rows,
        summary=summary,
        train_full=train_full,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        modality_dropout=modality_dropout,
        patch_size=patch_size,
        water_fraction=water_fraction,
        device=str(dev),
        threads=threads,
        data_seconds=data_seconds,
        argv=argv,
    )
    _write_outputs(out_dir, payload, summary, publish_figures)
    payload["runtime_seconds"] = time.perf_counter() - started
    payload["artifacts"]["metrics_json"] = str(out_dir / "metrics.json")
    # Rewrite so the file includes the final runtime.
    write_json(out_dir / "metrics.json", payload)
    print(f"wrote {out_dir / 'metrics.json'}", flush=True)
    return payload


def _payload(
    *,
    bundle,
    rows,
    summary,
    train_full,
    seeds,
    epochs,
    batch_size,
    lr,
    modality_dropout,
    patch_size,
    water_fraction,
    device,
    threads,
    data_seconds,
    argv,
) -> dict:
    splits_meta = {}
    for name, meta in bundle["splits"].items():
        splits_meta[name] = {
            "n_chips": meta["n_chips"],
            "n_patches": meta["n_patches"],
            "positive_rate": meta["positive_rate"],
            "chips": [chip["event"] + "_" + str(chip["chip_id"]) for chip in meta["chips"]],
        }
    return {
        "dataset": {
            "name": "Sen1Floods11 v1.1 hand-labeled subset",
            "citation": (
                "Bonafilia et al., Sen1Floods11: a georeferenced dataset to train and "
                "test deep learning flood algorithms for Sentinel-1, CVPR Workshops 2020."
            ),
            "source": "https://storage.googleapis.com/sen1floods11/v1.1/",
            "download_bytes": bundle.get("download_bytes"),
            "patch_size": patch_size,
            "water_fraction_threshold": water_fraction,
            "label": (
                "patch is positive when at least this fraction of valid hand-labeled "
                "pixels are water; patches with <90% valid pixels are dropped"
            ),
            "optical_bands": list(OPTICAL_FEATURE_NAMES),
            "sar_index_features": list(SAR_FEATURE_NAMES),
            "splits": splits_meta,
            "protocol": bundle.get("protocol"),
        },
        "training": {
            "seeds": list(seeds),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "class_weight": "inverse frequency on the training patches",
            "modality_dropout": modality_dropout,
            "modality_dropout_rule": (
                "on fusion models whose name ends in _moddrop, each training sample "
                "zeros optical or SAR with equal probability, with total probability "
                "equal to modality_dropout; both are never zeroed together"
            ),
            "missing_eval": (
                "optical_missing / sar_missing replace that input with zeros in the "
                "normalized tensor, the same code used by modality dropout"
            ),
            "std": "sample standard deviation across seeds with ddof=1",
            "seed_scope": (
                "each method is constructed and trained from the same seed "
                "independently of the other methods"
            ),
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "torch_threads": threads,
            "device": device,
        },
        "data_seconds": data_seconds,
        "per_seed": rows,
        "train_full": train_full,
        "summary": summary,
        "argv": argv or [],
    }


def _write_outputs(out_dir: Path, payload: dict, summary: list[dict], publish_figures) -> None:
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "metrics_table.csv", summary, SUMMARY_COLUMNS)
    write_csv(out_dir / "metrics_by_seed.csv", payload["per_seed"], SEED_COLUMNS)
    sections = [
        "# Sen1Floods11 subset results",
        "",
        "Values are mean ± sample standard deviation across seeds. "
        "`undefined` means AUROC was not defined.",
        "",
        "## In-domain test, both modalities",
        "",
        markdown_table(summary, "test", FULL_METHODS, ["full"]),
        "",
        "## In-domain test, missing modality",
        "",
        markdown_table(summary, "test", FUSION_METHODS, FUSION_CONDITIONS),
    ]
    if any(row["split"] == "bolivia" for row in summary):
        sections.extend(
            [
                "",
                "## Bolivia geographic hold-out, both modalities",
                "",
                markdown_table(summary, "bolivia", FULL_METHODS, ["full"]),
                "",
                "## Bolivia geographic hold-out, missing modality",
                "",
                markdown_table(summary, "bolivia", FUSION_METHODS, FUSION_CONDITIONS),
            ]
        )
    (out_dir / "table.md").write_text("\n".join(sections) + "\n", encoding="utf-8")

    majority = next(
        row
        for row in summary
        if row["split"] == "test" and row["method"] == "majority_class" and row["condition"] == "full"
    )
    reference = {"accuracy": majority["accuracy_mean"], "macro_f1": majority["macro_f1_mean"]}
    figure_dir = out_dir / "figures"
    plot_full_modality(
        summary,
        figure_dir / "fusion_accuracy.png",
        split="test",
        methods=PLOT_METHODS,
        reference=reference,
    )
    plot_missing_modality(
        summary,
        figure_dir / "missing_modality.png",
        split="test",
        methods=FUSION_METHODS,
    )
    if any(row["split"] == "bolivia" for row in summary):
        plot_geographic_shift(
            summary,
            figure_dir / "geographic_shift.png",
            methods=PLOT_METHODS,
        )
    if publish_figures:
        publish = Path(publish_figures)
        publish.mkdir(parents=True, exist_ok=True)
        for path in figure_dir.glob("*.png"):
            target = publish / path.name
            target.write_bytes(path.read_bytes())
    payload["artifacts"] = {
        "metrics_json": str(out_dir / "metrics.json"),
        "metrics_csv": str(out_dir / "metrics_table.csv"),
        "metrics_by_seed": str(out_dir / "metrics_by_seed.csv"),
        "table": str(out_dir / "table.md"),
        "figures": [str(path) for path in sorted(figure_dir.glob("*.png"))],
    }
