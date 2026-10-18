"""Metric edge cases: no NaNs, collapse, and one-class labels."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eo_multimodal_fuse.metrics import SEED_COLUMNS, aggregate, safe_scores, write_csv, write_json


class SafeScoreTests(unittest.TestCase):
    def test_collapsed_predictions_stay_finite(self):
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.zeros_like(y_true)
        y_prob = np.full(y_true.shape, 0.2)
        scores = safe_scores(y_true, y_pred, y_prob)
        self.assertTrue(math.isfinite(scores["accuracy"]))
        self.assertTrue(math.isfinite(scores["macro_f1"]))
        self.assertTrue(math.isfinite(scores["auroc"]))
        self.assertTrue(any("collapsed" in note for note in scores["notes"]))
        self.assertLess(scores["macro_f1"], 0.8)

    def test_single_class_labels_leave_auroc_undefined(self):
        y_true = np.ones(6, dtype=np.int64)
        y_pred = np.ones(6, dtype=np.int64)
        y_prob = np.linspace(0.1, 0.9, 6)
        scores = safe_scores(y_true, y_pred, y_prob)
        self.assertIsNone(scores["auroc"])
        self.assertIn("single class", scores["auroc_note"])
        self.assertTrue(math.isfinite(scores["accuracy"]))
        self.assertTrue(math.isfinite(scores["macro_f1"]))

    def test_json_rejects_nan_and_writes_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            write_json(path, {"auroc": None, "accuracy": 0.5})
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertIsNone(json.loads(text)["auroc"])
            with self.assertRaises(ValueError):
                write_json(path, {"auroc": float("nan")})

    def test_csv_blank_note_is_not_undefined(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics_by_seed.csv"
            write_csv(
                path,
                [
                    {
                        "split": "test",
                        "method": "optical_only",
                        "condition": "full",
                        "seed": 0,
                        "accuracy": 0.5,
                        "macro_f1": 0.4,
                        "auroc": 0.7,
                        "auroc_note": None,
                        "pred_positive_rate": 0.2,
                        "positive_rate": 0.2,
                        "n": 10,
                        "notes": [],
                    }
                ],
                SEED_COLUMNS,
            )
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            row = text.strip().splitlines()[1]
            self.assertIn("0.700000", row)
            self.assertNotIn("undefined", row)

    def test_aggregate_uses_sample_std(self):
        rows = []
        for seed, acc in enumerate((0.5, 0.7, 0.9)):
            rows.append(
                {
                    "split": "test",
                    "method": "late_fusion",
                    "condition": "full",
                    "seed": seed,
                    "accuracy": acc,
                    "macro_f1": acc,
                    "auroc": None,
                    "auroc_note": "undefined: labels contain a single class",
                    "pred_positive_rate": 0.0,
                    "positive_rate": 0.4,
                    "n": 10,
                    "notes": [],
                }
            )
        summary = aggregate(rows)[0]
        self.assertAlmostEqual(summary["accuracy_mean"], 0.7)
        self.assertAlmostEqual(summary["accuracy_std"], float(np.std([0.5, 0.7, 0.9], ddof=1)))
        self.assertIsNone(summary["auroc_mean"])
        self.assertEqual(summary["auroc_n_defined"], 0)


if __name__ == "__main__":
    unittest.main()
