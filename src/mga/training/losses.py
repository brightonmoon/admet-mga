"""
Loss functions for MGA training.

This module provides loss functions with mask support for
multi-task learning with missing labels.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


def get_loss_function(
    task_type: Literal["classification", "regression"],
    pos_weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Get appropriate loss function for task type.

    Args:
        task_type: Type of task ("classification" or "regression")
        pos_weight: Optional positive class weights for classification

    Returns:
        Loss function module
    """
    if task_type == "classification":
        return nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    else:
        return nn.L1Loss(reduction="none")


def compute_masked_loss(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    task_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute loss with mask for missing labels.

    Args:
        loss_fn: Loss function
        logits: Model predictions [B, T]
        labels: Ground truth labels [B, T]
        mask: Mask for valid labels [B, T] (0=missing, 1=valid)
        task_weight: Optional per-task weights [T]

    Returns:
        Scalar loss value
    """
    # Compute raw loss
    raw_loss = loss_fn(logits, labels)

    # Apply mask (ignore missing labels)
    masked_loss = raw_loss * (mask != 0).float()

    if task_weight is not None:
        # Apply task-specific weights
        weighted_loss = torch.mean(masked_loss, dim=0) * task_weight
        return weighted_loss.mean()
    else:
        return masked_loss.mean()


def compute_pos_weight(
    train_set,
    n_classification_tasks: int,
) -> torch.Tensor:
    """
    Compute positive class weights for imbalanced classification.

    Args:
        train_set: Training dataset (list of [smiles, graph, labels, mask])
        n_classification_tasks: Number of classification tasks

    Returns:
        Tensor of positive class weights [n_tasks]
    """
    import numpy as np

    _, _, labels, _ = map(list, zip(*train_set))
    labels = np.array(labels)

    weights = []
    for task in range(n_classification_tasks):
        task_labels = labels[:, task]
        # Filter out missing values (typically marked as 123456)
        valid_mask = (task_labels != 123456) & (~np.isnan(task_labels))
        valid_labels = task_labels[valid_mask]

        num_pos = np.sum(valid_labels == 1)
        num_neg = np.sum(valid_labels == 0)

        if num_pos > 0:
            weight = num_neg / (num_pos + 1e-8)
        else:
            weight = 1.0

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
