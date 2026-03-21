"""
Metrics tracking for MGA training.

This module provides the Meter class for tracking and computing
evaluation metrics during training and validation.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)


class Meter:
    """
    Track and summarize model performance for multi-task learning.

    Supports both classification and regression metrics with mask handling
    for missing labels.

    Example:
        >>> meter = Meter()
        >>> for batch in dataloader:
        ...     predictions = model(batch)
        ...     meter.update(predictions, labels, mask)
        >>> auc_scores = meter.roc_auc_score()
    """

    def __init__(self, max_samples: int = 10000):
        """
        Initialize Meter.

        Args:
            max_samples: Maximum samples to accumulate before compacting.
                        Lower values reduce peak RAM usage but may have
                        slight overhead from more frequent compaction.
                        Default: 10000 (good balance for most datasets)
        """
        self.mask: List[torch.Tensor] = []
        self.y_pred: List[torch.Tensor] = []
        self.y_true: List[torch.Tensor] = []
        # Cache for concatenated data to avoid repeated concatenation
        self._cached_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        # Memory optimization: compact data when accumulated samples exceed threshold
        self.max_samples = max_samples
        self._accumulated_count = 0

    def update(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Update meter with batch predictions.

        Args:
            y_pred: Predicted labels [B, T]
            y_true: Ground truth labels [B, T]
            mask: Mask for valid labels [B, T] (0=missing, 1=valid)
        """
        # Invalidate cache when new data arrives
        self._cached_data = None
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())
        self._accumulated_count += y_pred.shape[0]

        # Memory optimization: compact when exceeding threshold
        if self._accumulated_count > self.max_samples and len(self.y_pred) > 1:
            self._compact_data()

    def _compact_data(self) -> None:
        """
        Compact accumulated data into single tensors to reduce RAM usage.

        This merges multiple small tensors into one, reducing Python object
        overhead and memory fragmentation.
        """
        if len(self.y_pred) <= 1:
            return

        # Concatenate all tensors into single tensors
        compacted_mask = torch.cat(self.mask, dim=0)
        compacted_pred = torch.cat(self.y_pred, dim=0)
        compacted_true = torch.cat(self.y_true, dim=0)

        # Replace lists with single-element lists containing compacted tensors
        self.mask = [compacted_mask]
        self.y_pred = [compacted_pred]
        self.y_true = [compacted_true]

        # Update cache
        self._cached_data = (compacted_mask, compacted_pred, compacted_true)

    def reset(self) -> None:
        """Reset meter state and clear cache."""
        # Use clear() instead of reassignment to reuse list objects
        self.mask.clear()
        self.y_pred.clear()
        self.y_true.clear()
        self._cached_data = None
        self._accumulated_count = 0

    def _get_concatenated_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate all batches with caching for efficiency."""
        if self._cached_data is None:
            self._cached_data = (
                torch.cat(self.mask, dim=0),
                torch.cat(self.y_pred, dim=0),
                torch.cat(self.y_true, dim=0),
            )
        return self._cached_data

    def roc_auc_score(self) -> List[float]:
        """
        Compute ROC-AUC score for each task.

        Returns:
            List of ROC-AUC scores for all tasks
        """
        mask, y_pred, y_true = self._get_concatenated_data()
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]

        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            if len(np.unique(task_y_true)) < 2:
                # Skip if only one class present
                scores.append(0.5)
            else:
                scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))

        return scores

    def f1_score(self, threshold: float = 0.5) -> List[float]:
        """
        Compute F1 score for each task.

        Args:
            threshold: Classification threshold

        Returns:
            List of F1 scores for all tasks
        """
        mask, y_pred, y_true = self._get_concatenated_data()
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]

        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = (y_pred[:, task][task_w != 0].numpy() > threshold).astype(int)
            scores.append(round(f1_score(task_y_true, task_y_pred, zero_division=0), 4))

        return scores

    def accuracy(self, threshold: float = 0.5) -> List[float]:
        """
        Compute accuracy for each task.

        Args:
            threshold: Classification threshold

        Returns:
            List of accuracy scores for all tasks
        """
        mask, y_pred, y_true = self._get_concatenated_data()
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]

        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = (y_pred[:, task][task_w != 0].numpy() > threshold).astype(int)
            scores.append(round(accuracy_score(task_y_true, task_y_pred), 4))

        return scores

    def mcc(self, threshold: float = 0.5) -> List[float]:
        """
        Compute Matthews Correlation Coefficient for each task.

        Args:
            threshold: Classification threshold

        Returns:
            List of MCC scores for all tasks
        """
        mask, y_pred, y_true = self._get_concatenated_data()
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]

        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = (y_pred[:, task][task_w != 0].numpy() > threshold).astype(int)

            if len(np.unique(task_y_true)) < 2:
                scores.append(0.0)
            else:
                scores.append(round(matthews_corrcoef(task_y_true, task_y_pred), 4))

        return scores

    def roc_precision_recall_score(self) -> List[float]:
        """
        Compute AUC-PRC (Area Under Precision-Recall Curve) for each task.

        Returns:
            List of AUC-PRC scores for all tasks
        """
        mask, y_pred, y_true = self._get_concatenated_data()
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]

        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            if len(np.unique(task_y_true)) < 2:
                scores.append(0.5)
            else:
                precision, recall, _ = precision_recall_curve(task_y_true, task_y_pred)
                scores.append(round(auc(recall, precision), 4))

        return scores

    def rmse(self) -> List[float]:
        """
        Compute RMSE for each task.

        Returns:
            List of RMSE values for all tasks
        """
        mask, y_pred, y_true = self._get_concatenated_data()
        n_tasks = y_true.shape[1]

        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(np.sqrt(mean_squared_error(task_y_true, task_y_pred)), 4))

        return scores

    def mae(self) -> List[float]:
        """
        Compute MAE for each task.

        Returns:
            List of MAE values for all tasks
        """
        mask, y_pred, y_true = self._get_concatenated_data()
        n_tasks = y_true.shape[1]

        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(np.mean(np.abs(task_y_true - task_y_pred)), 4))

        return scores

    def r2(self) -> List[float]:
        """
        Compute R2 score for each task.

        Returns:
            List of R2 scores for all tasks
        """
        mask, y_pred, y_true = self._get_concatenated_data()
        n_tasks = y_true.shape[1]

        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            if len(task_y_true) < 2:
                scores.append(0.0)
            else:
                scores.append(round(r2_score(task_y_true, task_y_pred), 4))

        return scores

    def compute_classification_metrics(
        self,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, List[float]]:
        """
        Compute multiple classification metrics efficiently.

        Performs data concatenation and sigmoid only once for all metrics,
        optimizing CPU/RAM usage.

        Args:
            metrics: List of metric names. Default: ["roc_auc", "accuracy", "mcc"]
            threshold: Classification threshold for binary predictions

        Returns:
            Dictionary mapping metric names to per-task score lists
        """
        if metrics is None:
            metrics = ["roc_auc", "accuracy", "mcc"]

        mask, y_pred, y_true = self._get_concatenated_data()
        y_pred_proba = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]

        results: Dict[str, List[float]] = {m: [] for m in metrics}

        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_proba = y_pred_proba[:, task][task_w != 0].numpy()
            task_y_pred_binary = (task_y_proba > threshold).astype(int)

            has_both_classes = len(np.unique(task_y_true)) >= 2

            for metric in metrics:
                if metric == "roc_auc":
                    score = roc_auc_score(task_y_true, task_y_proba) if has_both_classes else 0.5
                elif metric == "accuracy":
                    score = accuracy_score(task_y_true, task_y_pred_binary)
                elif metric == "mcc":
                    score = matthews_corrcoef(task_y_true, task_y_pred_binary) if has_both_classes else 0.0
                elif metric == "f1":
                    score = f1_score(task_y_true, task_y_pred_binary, zero_division=0)
                elif metric == "roc_prc":
                    if has_both_classes:
                        precision, recall, _ = precision_recall_curve(task_y_true, task_y_proba)
                        score = auc(recall, precision)
                    else:
                        score = 0.5
                else:
                    raise ValueError(f"Unknown classification metric: {metric}")

                results[metric].append(round(float(score), 4))

        return results

    def compute_regression_metrics(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        """
        Compute multiple regression metrics efficiently.

        Performs data concatenation only once for all metrics,
        optimizing CPU/RAM usage.

        Args:
            metrics: List of metric names. Default: ["r2", "rmse", "mae"]

        Returns:
            Dictionary mapping metric names to per-task score lists
        """
        if metrics is None:
            metrics = ["r2", "rmse", "mae"]

        mask, y_pred, y_true = self._get_concatenated_data()
        n_tasks = y_true.shape[1]

        results: Dict[str, List[float]] = {m: [] for m in metrics}

        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            for metric in metrics:
                if metric == "r2":
                    score = r2_score(task_y_true, task_y_pred) if len(task_y_true) >= 2 else 0.0
                elif metric == "rmse":
                    score = np.sqrt(mean_squared_error(task_y_true, task_y_pred))
                elif metric == "mae":
                    score = np.mean(np.abs(task_y_true - task_y_pred))
                else:
                    raise ValueError(f"Unknown regression metric: {metric}")

                results[metric].append(round(float(score), 4))

        return results

    def return_pred_true(
        self,
        apply_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return raw predictions and ground truth.

        Args:
            apply_sigmoid: Whether to apply sigmoid to predictions

        Returns:
            Tuple of (predictions, ground_truth)
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)

        return y_pred, y_true

    def compute_metric(
        self,
        metric_name: Literal[
            "roc_auc", "l1", "rmse", "mae", "roc_prc", "r2",
            "return_clas_pred_true", "return_reg_pred_true"
        ],
        reduction: Literal["mean", "sum"] = "mean",
    ) -> List[float] | Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute metric by name.

        Args:
            metric_name: Name of the metric to compute
            reduction: Reduction method for l1 loss

        Returns:
            Metric values for each task or (predictions, ground_truth) tuple
        """
        metric_map = {
            "roc_auc": self.roc_auc_score,
            "accuracy": self.accuracy,
            "mcc": self.mcc,
            "f1": self.f1_score,
            "rmse": self.rmse,
            "mae": self.mae,
            "roc_prc": self.roc_precision_recall_score,
            "r2": self.r2,
            "return_clas_pred_true": lambda: self.return_pred_true(apply_sigmoid=True),
            "return_reg_pred_true": lambda: self.return_pred_true(apply_sigmoid=False),
        }

        if metric_name not in metric_map:
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available: {list(metric_map.keys())}"
            )

        return metric_map[metric_name]()
