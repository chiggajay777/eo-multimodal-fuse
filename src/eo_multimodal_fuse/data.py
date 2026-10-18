"""Synthetic dual-modality samples for the offline ``--demo`` path.

Modality A: optical-like RGB stats (NDVI proxy, brightness, greenness).
Modality B: SAR-like proxies (backscatter mean/std, texture).
Label: binary hazard / change-risk flag with multimodal dependence.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def _generate(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Latent factors
    wetness = rng.normal(0, 1, size=n)
    roughness = rng.normal(0, 1, size=n)
    urban = rng.normal(0, 1, size=n)

    # Optical-like features (4): ndvi, brightness, green_ratio, blue_ratio
    opt = np.stack(
        [
            0.2 + 0.4 * wetness - 0.2 * urban + rng.normal(0, 0.15, n),  # ndvi-ish
            0.5 + 0.3 * urban - 0.1 * wetness + rng.normal(0, 0.1, n),  # brightness
            0.3 + 0.35 * wetness + rng.normal(0, 0.1, n),  # green
            0.25 + 0.2 * wetness - 0.1 * urban + rng.normal(0, 0.1, n),  # blue
        ],
        axis=1,
    ).astype(np.float32)

    # SAR-like features (4): vv_mean, vh_mean, vv_std, texture
    sar = np.stack(
        [
            -12 + 3 * roughness - 2 * wetness + rng.normal(0, 0.8, n),
            -18 + 2.5 * roughness - 1.5 * wetness + rng.normal(0, 0.8, n),
            1.5 + 0.8 * roughness + rng.normal(0, 0.3, n),
            0.4 + 0.5 * urban + 0.3 * roughness + rng.normal(0, 0.2, n),
        ],
        axis=1,
    ).astype(np.float32)

    # Label depends on BOTH modalities (true multimodal signal)
    logit = 1.2 * wetness + 0.9 * roughness + 0.5 * urban - 0.3
    # Soften with optical ndvi + SAR vv
    logit = logit - 0.8 * opt[:, 0] + 0.05 * (-sar[:, 0] - 12)
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(np.int64)
    return opt, sar, y


class DualModalityDataset(Dataset):
    def __init__(self, n: int = 2000, seed: int = 42):
        self.opt, self.sar, self.y = _generate(n, seed=seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.opt[idx]),
            torch.from_numpy(self.sar[idx]),
            int(self.y[idx]),
        )


def make_loaders(n: int = 2000, batch_size: int = 64, seed: int = 42):
    ds = DualModalityDataset(n=n, seed=seed)
    n_val = max(1, int(0.25 * len(ds)))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
