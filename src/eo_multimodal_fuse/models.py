"""Unimodal baselines, late fusion, and cross-attention fusion.

Vector MLPs are the synthetic ``--demo`` models. Image CNNs are the Sentinel
benchmark. Fusion models read a missing modality as an all-zero tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusion(nn.Module):
    """Encode each feature vector, concatenate, classify."""

    def __init__(self, opt_dim: int = 4, sar_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.opt_enc = nn.Sequential(nn.Linear(opt_dim, hidden), nn.ReLU())
        self.sar_enc = nn.Sequential(nn.Linear(sar_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden * 2, 2)

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.opt_enc(opt), self.sar_enc(sar)], dim=-1))


class CrossAttentionFusion(nn.Module):
    """One-step cross-attention: optical queries SAR, then classify."""

    def __init__(self, opt_dim: int = 4, sar_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.opt_proj = nn.Linear(opt_dim, hidden)
        self.sar_proj = nn.Linear(sar_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        q = self.opt_proj(opt).unsqueeze(1)
        kv = self.sar_proj(sar).unsqueeze(1)
        fused, _ = self.attn(q, kv, kv, need_weights=False)
        h = torch.cat([fused.squeeze(1), q.squeeze(1)], dim=-1)
        return self.head(h)


class ConvTower(nn.Module):
    """Three 3x3 convolutions with 2x pooling. 64px input -> 8x8 map."""

    def __init__(self, in_ch: int, width: int = 16):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_ch
        for _ in range(3):
            layers.extend(
                [
                    nn.Conv2d(channels, width, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            channels = width
        self.net = nn.Sequential(*layers)
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pool(feat: torch.Tensor) -> torch.Tensor:
    return feat.mean(dim=(2, 3))


def _tokens(feat: torch.Tensor) -> torch.Tensor:
    # (B, C, H, W) -> (B, H*W, C)
    return feat.flatten(2).transpose(1, 2)


class ImageUnimodal(nn.Module):
    def __init__(self, in_ch: int, width: int = 16):
        super().__init__()
        self.tower = ConvTower(in_ch, width)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(width, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(_pool(self.tower(x)))


class ImageLateFusion(nn.Module):
    def __init__(self, opt_ch: int = 6, sar_ch: int = 2, width: int = 16):
        super().__init__()
        self.opt_enc = ConvTower(opt_ch, width)
        self.sar_enc = ConvTower(sar_ch, width)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(width * 2, 2))

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        z = torch.cat([_pool(self.opt_enc(opt)), _pool(self.sar_enc(sar))], dim=-1)
        return self.head(z)


class ImageCrossAttentionFusion(nn.Module):
    """Optical spatial tokens query SAR tokens, then a two-layer head."""

    def __init__(self, opt_ch: int = 6, sar_ch: int = 2, width: int = 16, heads: int = 4):
        super().__init__()
        self.opt_enc = ConvTower(opt_ch, width)
        self.sar_enc = ConvTower(sar_ch, width)
        self.attn = nn.MultiheadAttention(width, num_heads=heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(width * 2, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, 2),
        )

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        query = _tokens(self.opt_enc(opt))
        kv = _tokens(self.sar_enc(sar))
        fused, _ = self.attn(query, kv, kv, need_weights=False)
        z = torch.cat([fused.mean(dim=1), query.mean(dim=1)], dim=-1)
        return self.head(z)
