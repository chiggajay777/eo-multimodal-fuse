"""Modality dropout and a one-epoch image-model smoke test."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eo_multimodal_fuse.engine import apply_modality_dropout, predict, train_model
from eo_multimodal_fuse.metrics import safe_scores
from eo_multimodal_fuse.models import ImageCrossAttentionFusion, ImageLateFusion


class EngineTests(unittest.TestCase):
    def test_dropout_never_zeros_both_modalities(self):
        torch.manual_seed(0)
        opt = torch.ones(80, 2, 8, 8)
        sar = torch.full((80, 2, 8, 8), 2.0)
        dropped_opt, dropped_sar = apply_modality_dropout(opt, sar, 1.0)
        opt_zero = dropped_opt.view(80, -1).abs().sum(1) == 0
        sar_zero = dropped_sar.view(80, -1).abs().sum(1) == 0
        self.assertTrue(torch.equal(opt_zero, ~sar_zero))
        self.assertGreater(int(opt_zero.sum()), 0)
        self.assertGreater(int(sar_zero.sum()), 0)
        same_opt, same_sar = apply_modality_dropout(opt, sar, 0.0)
        self.assertTrue(torch.equal(same_opt, opt))
        self.assertTrue(torch.equal(same_sar, sar))

    def test_missing_modality_scores_are_finite(self):
        torch.manual_seed(0)
        opt = torch.rand(32, 6, 64, 64)
        sar = torch.rand(32, 2, 64, 64)
        y = torch.randint(0, 2, (32,))
        ds = TensorDataset(opt, sar, y)
        loader = DataLoader(ds, batch_size=8)
        model = ImageLateFusion()
        weight = torch.tensor([1.0, 1.0])
        train_model(
            model,
            loader,
            device=torch.device("cpu"),
            epochs=1,
            lr=1e-3,
            multimodal=True,
            class_weight=weight,
            modality_dropout=0.5,
            augment=True,
        )
        for missing in (None, "optical", "sar"):
            labels, preds, probs, n_bad = predict(
                model, loader, device=torch.device("cpu"), multimodal=True, missing=missing
            )
            scores = safe_scores(
                np.concatenate(labels), np.concatenate(preds), np.concatenate(probs)
            )
            self.assertEqual(n_bad, 0)
            self.assertTrue(np.isfinite(scores["accuracy"]))
            self.assertTrue(np.isfinite(scores["macro_f1"]))
            self.assertTrue(scores["auroc"] is None or np.isfinite(scores["auroc"]))

        attn = ImageCrossAttentionFusion()
        train_model(
            attn,
            loader,
            device=torch.device("cpu"),
            epochs=1,
            lr=1e-3,
            multimodal=True,
            class_weight=weight,
            modality_dropout=0.5,
            augment=False,
        )
        labels, preds, probs, _ = predict(
            attn, loader, device=torch.device("cpu"), multimodal=True, missing="sar"
        )
        scores = safe_scores(np.concatenate(labels), np.concatenate(preds), np.concatenate(probs))
        self.assertTrue(np.isfinite(scores["macro_f1"]))


if __name__ == "__main__":
    unittest.main()
