"""Shared training and prediction loop for vector and image models."""

from __future__ import annotations

import torch
import torch.nn as nn


def _as_tensor_batch(batch, device: torch.device, multimodal: bool):
    if multimodal:
        opt, sar, y = batch
        return opt.to(device), sar.to(device), y.to(device)
    x, y = batch
    return x.to(device), y.to(device)


def _flip_pair(opt: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if opt.ndim != 4:
        return opt, sar
    if torch.rand(()) < 0.5:
        opt = torch.flip(opt, [2])
        sar = torch.flip(sar, [2])
    if torch.rand(()) < 0.5:
        opt = torch.flip(opt, [3])
        sar = torch.flip(sar, [3])
    return opt, sar


def _flip_single(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        return x
    if torch.rand(()) < 0.5:
        x = torch.flip(x, [2])
    if torch.rand(()) < 0.5:
        x = torch.flip(x, [3])
    return x


def apply_modality_dropout(
    opt: torch.Tensor, sar: torch.Tensor, p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero one modality on a random subset of the batch.

    Each sample is dropped with probability ``p``. The dropped modality is
    optical or SAR with equal probability. Both modalities are never dropped
    on the same sample. ``p <= 0`` leaves the batch unchanged.
    """
    if p <= 0:
        return opt, sar
    batch = opt.shape[0]
    drop = torch.rand(batch, device=opt.device) < p
    drop_opt = drop & (torch.rand(batch, device=opt.device) < 0.5)
    drop_sar = drop & ~drop_opt
    if bool(drop_opt.any()):
        opt = opt.clone()
        opt[drop_opt] = 0
    if bool(drop_sar.any()):
        sar = sar.clone()
        sar[drop_sar] = 0
    return opt, sar


def train_model(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    multimodal: bool,
    class_weight: torch.Tensor | None = None,
    modality_dropout: float = 0.0,
    augment: bool = False,
    weight_decay: float = 1e-4,
) -> list[float]:
    """AdamW training. Returns mean cross-entropy per epoch."""
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weight = None if class_weight is None else class_weight.to(device)
    crit = nn.CrossEntropyLoss(weight=weight)
    losses: list[float] = []
    for _ in range(epochs):
        total = 0.0
        seen = 0
        for batch in loader:
            if multimodal:
                x_opt, x_sar, y = _as_tensor_batch(batch, device, True)
                if augment:
                    x_opt, x_sar = _flip_pair(x_opt, x_sar)
                x_opt, x_sar = apply_modality_dropout(x_opt, x_sar, modality_dropout)
                logits = model(x_opt, x_sar)
            else:
                x, y = _as_tensor_batch(batch, device, False)
                if augment:
                    x = _flip_single(x)
                logits = model(x)
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits during training")
            opt.zero_grad(set_to_none=True)
            loss = crit(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * int(y.shape[0])
            seen += int(y.shape[0])
        if seen == 0:
            raise RuntimeError("training loader was empty")
        losses.append(total / seen)
    return losses


def predict(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    multimodal: bool,
    missing: str | None = None,
) -> tuple[list, list, list, int]:
    """Return label, prediction, and positive-class probability lists.

    ``missing`` is ``None``, ``"optical"``, or ``"sar"``. A missing modality is
    an all-zero tensor in the same normalized space used by modality dropout.
    Non-finite logits are replaced with zeros and counted.
    """
    if missing not in (None, "optical", "sar"):
        raise ValueError(f"unknown missing modality {missing!r}")
    if missing and not multimodal:
        raise ValueError("missing-modality evaluation requires a fusion model")
    model.eval()
    labels: list = []
    preds: list = []
    probs: list = []
    n_nonfinite = 0
    with torch.inference_mode():
        for batch in loader:
            if multimodal:
                opt, sar, y = _as_tensor_batch(batch, device, True)
                if missing == "optical":
                    opt = torch.zeros_like(opt)
                elif missing == "sar":
                    sar = torch.zeros_like(sar)
                logits = model(opt, sar)
            else:
                x, y = _as_tensor_batch(batch, device, False)
                logits = model(x)
            bad = ~torch.isfinite(logits)
            n_nonfinite += int(bad.any(dim=1).sum())
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            labels.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())
    return labels, preds, probs, n_nonfinite
