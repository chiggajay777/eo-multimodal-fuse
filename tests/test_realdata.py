"""Subset selection and patch labels, without downloading imagery."""

from __future__ import annotations

import unittest

import numpy as np

from eo_multimodal_fuse.realdata import (
    Chip,
    normalize_optical,
    normalize_sar,
    optical_features,
    parse_split,
    patch_label,
    sar_features,
    select_chips,
    systematic_indices,
)


class RealDataTests(unittest.TestCase):
    def test_systematic_indices_are_unique_and_stable(self):
        self.assertEqual(systematic_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(systematic_indices(5, 5), [0, 1, 2, 3, 4])
        self.assertEqual(systematic_indices(7, 1), [3])
        self.assertEqual(len(set(systematic_indices(6, 4))), 4)

    def test_select_chips_spreads_events_and_does_not_overlap_splits(self):
        text = "\n".join(
            f"Ghana_{i}_S1Hand.tif,Ghana_{i}_LabelHand.tif" for i in (10, 30, 20, 40)
        )
        text += "\n" + "\n".join(
            f"India_{i}_S1Hand.tif,India_{i}_LabelHand.tif" for i in (3, 1, 2)
        )
        chips = parse_split(text)
        chosen = select_chips(chips, per_event=2)
        self.assertEqual([chip.base for chip in chosen], ["Ghana_10", "Ghana_40", "India_1", "India_3"])
        held = {Chip("Ghana", 10, "Ghana_10_S1Hand.tif", "Ghana_10_LabelHand.tif").base}
        self.assertTrue(held.isdisjoint({chip.base for chip in select_chips(chips, 1, limit_events=1)}))

    def test_patch_label_threshold_and_nodata(self):
        water = np.zeros((8, 8), dtype=np.int16)
        water[:2, :] = 1  # 16/64 = 0.25
        self.assertEqual(patch_label(water, 0.05), 1)
        self.assertEqual(patch_label(water, 0.5), 0)
        nodata = water.copy()
        nodata[:5, :] = -1
        self.assertIsNone(patch_label(nodata, 0.05))

    def test_normalization_and_indices(self):
        self.assertAlmostEqual(float(normalize_optical(np.array([0, 10000, 20000]))[1]), 1.0)
        self.assertAlmostEqual(float(normalize_optical(np.array([0, 10000, 20000]))[2]), 1.0)
        sar = normalize_sar(np.array([-30.0, 5.0, -12.5]))
        self.assertAlmostEqual(float(sar[0]), 0.0)
        self.assertAlmostEqual(float(sar[1]), 1.0)
        refl = np.zeros((6, 2, 2), dtype=np.float32)
        refl[2] = 0.1  # red
        refl[3] = 0.3  # nir
        refl[1] = 0.2  # green
        refl[4] = 0.05  # swir1
        feats = optical_features(refl)
        self.assertAlmostEqual(float(feats[8]), 0.5, places=5)  # NDVI
        db = np.zeros((2, 2, 2), dtype=np.float32)
        db[0] = -12
        db[1] = -18
        sar_feats = sar_features(db)
        self.assertAlmostEqual(float(sar_feats[2]), -6.0)


if __name__ == "__main__":
    unittest.main()
