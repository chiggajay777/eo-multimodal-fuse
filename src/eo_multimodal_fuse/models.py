"""Unimodal MLPs + late fusion + simple cross-attention fusion."""

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
    """Encode each modality, concat, classify."""

    def __init__(self, opt_dim: int = 4, sar_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.opt_enc = nn.Sequential(nn.Linear(opt_dim, hidden), nn.ReLU())
        self.sar_enc = nn.Sequential(nn.Linear(sar_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden * 2, 2)

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.opt_enc(opt), self.sar_enc(sar)], dim=-1))


class CrossAttentionFusion(nn.Module):
    """One-step cross-attn: optical queries SAR keys/values, then classify."""

    def __init__(self, opt_dim: int = 4, sar_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.opt_proj = nn.Linear(opt_dim, hidden)
        self.sar_proj = nn.Linear(sar_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        q = self.opt_proj(opt).unsqueeze(1)  # [B,1,H]
        kv = self.sar_proj(sar).unsqueeze(1)
        fused, _ = self.attn(q, kv, kv)
        h = torch.cat([fused.squeeze(1), q.squeeze(1)], dim=-1)
        return self.head(h)
