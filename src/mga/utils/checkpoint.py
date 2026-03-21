"""Checkpoint utilities for saving and loading models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        path: Path to save checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        metadata: Optional additional metadata
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        path: Path to checkpoint
        optimizer: Optional optimizer to load state into
        device: Device to load weights to
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Checkpoint dictionary (for accessing metadata, epoch, etc.)
    """
    checkpoint = torch.load(path, map_location=torch.device(device), weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
