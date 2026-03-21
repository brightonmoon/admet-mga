"""
Training callbacks for MGA.

This module provides callbacks for training control:
- EarlyStopping: Stop training when validation metric stops improving
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping callback with model checkpointing.

    Monitors a validation metric and stops training when it stops improving.
    Saves the best model checkpoint.

    Args:
        patience: Number of epochs to wait before stopping
        mode: "higher" if higher metric is better, "lower" if lower is better
        filename: Path to save checkpoint
        task_name: Name of the task (used for default filename)
        pretrained_model: Path to pretrained model (for delta weight extraction)

    Example:
        >>> stopper = EarlyStopping(patience=10, mode="higher")
        >>> for epoch in range(num_epochs):
        ...     val_score = validate(model)
        ...     if stopper.step(val_score, model):
        ...         print("Early stopping triggered")
        ...         break
        >>> stopper.load_checkpoint(model)  # Load best model
    """

    def __init__(
        self,
        patience: int = 10,
        mode: Literal["higher", "lower"] = "higher",
        filename: Optional[str] = None,
        task_name: str = "model",
        pretrained_model: Optional[str] = None,
    ):
        if filename is None:
            filename = f"./models/{task_name}_early_stop.pth"

        if mode not in ["higher", "lower"]:
            raise ValueError(f"mode must be 'higher' or 'lower', got '{mode}'")
        self.mode = mode
        self._check = self._check_higher if mode == "higher" else self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.pretrained_model = pretrained_model

    def _check_higher(self, score: float, prev_best: float) -> bool:
        return score > prev_best

    def _check_lower(self, score: float, prev_best: float) -> bool:
        return score < prev_best

    def step(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop and save checkpoint if improved.

        Args:
            score: Current validation score
            model: Model to checkpoint

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def step_no_save(self, score: float) -> bool:
        """
        Check if training should stop without saving checkpoint.

        Args:
            score: Current validation score

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, self.filename)

    def load_checkpoint(self, model: nn.Module, device: str = "cpu") -> None:
        """
        Load saved checkpoint into model.

        Args:
            model: Model to load weights into
            device: Device to load weights to
        """
        checkpoint = torch.load(
            self.filename, map_location=torch.device(device), weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    def load_pretrained_rgcn(self, model: nn.Module) -> None:
        """
        Load only RGCN layers from pretrained model.

        Useful for transfer learning where only encoder is pretrained.

        Args:
            model: Model to load weights into
        """
        pretrained_parameters = [
            "rgcn_layer1.graph_conv_layer.h_bias",
            "rgcn_layer1.graph_conv_layer.loop_weight",
            "rgcn_layer1.graph_conv_layer.linear_r.W",
            "rgcn_layer1.graph_conv_layer.linear_r.coeff",
            "rgcn_layer1.res_connection.weight",
            "rgcn_layer1.res_connection.bias",
            "rgcn_layer1.bn_layer.weight",
            "rgcn_layer1.bn_layer.bias",
            "rgcn_layer1.bn_layer.running_mean",
            "rgcn_layer1.bn_layer.running_var",
            "rgcn_layer1.bn_layer.num_batches_tracked",
            "rgcn_layer2.graph_conv_layer.h_bias",
            "rgcn_layer2.graph_conv_layer.loop_weight",
            "rgcn_layer2.graph_conv_layer.linear_r.W",
            "rgcn_layer2.graph_conv_layer.linear_r.coeff",
            "rgcn_layer2.res_connection.weight",
            "rgcn_layer2.res_connection.bias",
            "rgcn_layer2.bn_layer.weight",
            "rgcn_layer2.bn_layer.bias",
            "rgcn_layer2.bn_layer.running_mean",
            "rgcn_layer2.bn_layer.running_var",
            "rgcn_layer2.bn_layer.num_batches_tracked",
        ]

        if self.pretrained_model is None:
            raise ValueError("pretrained_model path not set")

        pretrained_path = Path("./models") / self.pretrained_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(
            pretrained_path, map_location=device, weights_only=True
        )

        pretrained_dict = {
            k: v
            for k, v in checkpoint["model_state_dict"].items()
            if k in pretrained_parameters
        }

        model.load_state_dict(pretrained_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} pretrained RGCN parameters")

    def load_pretrained_with_attention(
        self,
        model: nn.Module,
        select_task_index: list[int],
    ) -> None:
        """
        Load RGCN layers and task-specific attention from pretrained model.

        Args:
            model: Model to load weights into
            select_task_index: List of task indices to load attention for
        """
        pretrained_parameters = [
            "rgcn_layer1.graph_conv_layer.h_bias",
            "rgcn_layer1.graph_conv_layer.loop_weight",
            "rgcn_layer1.graph_conv_layer.linear_r.W",
            "rgcn_layer1.graph_conv_layer.linear_r.coeff",
            "rgcn_layer1.res_connection.weight",
            "rgcn_layer1.res_connection.bias",
            "rgcn_layer1.bn_layer.weight",
            "rgcn_layer1.bn_layer.bias",
            "rgcn_layer1.bn_layer.running_mean",
            "rgcn_layer1.bn_layer.running_var",
            "rgcn_layer1.bn_layer.num_batches_tracked",
            "rgcn_layer2.graph_conv_layer.h_bias",
            "rgcn_layer2.graph_conv_layer.loop_weight",
            "rgcn_layer2.graph_conv_layer.linear_r.W",
            "rgcn_layer2.graph_conv_layer.linear_r.coeff",
            "rgcn_layer2.res_connection.weight",
            "rgcn_layer2.res_connection.bias",
            "rgcn_layer2.bn_layer.weight",
            "rgcn_layer2.bn_layer.bias",
            "rgcn_layer2.bn_layer.running_mean",
            "rgcn_layer2.bn_layer.running_var",
            "rgcn_layer2.bn_layer.num_batches_tracked",
        ]

        # Add task-specific attention parameters
        for task_idx in select_task_index:
            pretrained_parameters.extend([
                f"weighted_sum_readout.atom_weighting_specific.{task_idx}.0.weight",
                f"weighted_sum_readout.atom_weighting_specific.{task_idx}.0.bias",
            ])

        if self.pretrained_model is None:
            raise ValueError("pretrained_model path not set")

        pretrained_path = Path("./models") / self.pretrained_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(
            pretrained_path, map_location=device, weights_only=True
        )

        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in checkpoint["model_state_dict"].items()
            if k in pretrained_parameters
        }

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} pretrained parameters with attention")
