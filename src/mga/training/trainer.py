"""
Trainer class for MGA models.

This module provides the MGATrainer class for training and evaluating
MGA models with wandb integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from mga.config import MGAConfig
from mga.metrics import Meter
from mga.training.callbacks import EarlyStopping
from mga.training.losses import compute_masked_loss, compute_pos_weight, get_loss_function


class MGATrainer:
    """
    Trainer for MGA models with wandb integration.

    Handles training loop, validation, and logging with support for
    multi-task learning (classification + regression).

    Args:
        model: MGA model to train
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Optional test data loader

    Example:
        >>> trainer = MGATrainer(model, config, train_loader, val_loader)
        >>> trainer.train()
        >>> test_results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: MGAConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = config.training.device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.min_lr,
        )

        # Setup loss functions
        self.loss_fn_classification = get_loss_function("classification")
        self.loss_fn_regression = get_loss_function("regression")

        # Setup early stopping
        # Model save path: use config.paths.model_dir if set, otherwise default to ./models/
        model_dir = config.paths.model_dir if config.paths.model_dir else Path("./models")
        checkpoint_filename = str(model_dir / f"{config.task.task_name}_early_stop.pth")

        self.stopper = EarlyStopping(
            patience=config.training.patience,
            mode="higher",
            filename=checkpoint_filename,
            task_name=config.task.task_name,
            pretrained_model=str(config.paths.pretrained_model_path) if config.paths.pretrained_model_path else None,
        )

        # wandb
        self.use_wandb = config.training.wandb.enabled
        self.wandb_run = None

    def _init_wandb(self) -> None:
        """Initialize wandb logging."""
        if not self.use_wandb:
            return

        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.training.wandb.project,
                entity=self.config.training.wandb.entity,
                name=self.config.training.wandb.name or f"{self.config.task.task_name}_run",
                config=self.config.to_wandb_config(),
                tags=self.config.training.wandb.tags,
                notes=self.config.training.wandb.notes,
            )
        except ImportError:
            print("wandb not installed, disabling logging")
            self.use_wandb = False

    def _log_wandb(self, metrics: Dict, step: int) -> None:
        """Log metrics to wandb."""
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def train_epoch(self, epoch: int) -> Tuple[List[float], float]:
        """
        Run one training epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (task_scores, average_loss)
        """
        self.model.train()
        total_loss = 0.0

        meter_c = Meter()
        meter_r = Meter()

        task_class = self.config.task.task_class
        n_classification = len(self.config.task.classification_list)

        for batch_data in self.train_loader:
            smiles, bg, labels, mask = batch_data

            # Move to device with non_blocking for async transfer when using pin_memory
            bg = bg.to(self.device)
            mask = mask.float().to(self.device, non_blocking=True)
            atom_feats = bg.ndata.pop(self.config.task.atom_data_field).float().to(self.device, non_blocking=True)
            bond_feats = bg.edata.pop(self.config.task.bond_data_field).long().to(self.device, non_blocking=True)

            # Forward pass
            logits = self.model(bg, atom_feats, bond_feats)
            labels = labels.type_as(logits).to(self.device)

            # Compute loss based on task type
            if task_class == "classification_regression":
                logits_c = logits[:, :n_classification]
                labels_c = labels[:, :n_classification]
                mask_c = mask[:, :n_classification]

                logits_r = logits[:, n_classification:]
                labels_r = labels[:, n_classification:]
                mask_r = mask[:, n_classification:]

                loss = (
                    compute_masked_loss(self.loss_fn_classification, logits_c, labels_c, mask_c)
                    + compute_masked_loss(self.loss_fn_regression, logits_r, labels_r, mask_r)
                )

                meter_c.update(logits_c, labels_c, mask_c)
                meter_r.update(logits_r, labels_r, mask_r)

            elif task_class == "classification":
                loss = compute_masked_loss(self.loss_fn_classification, logits, labels, mask)
                meter_c.update(logits, labels, mask)

            else:  # regression
                loss = compute_masked_loss(self.loss_fn_regression, logits, labels, mask)
                meter_r.update(logits, labels, mask)

            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)

        if task_class == "classification_regression":
            scores = (
                meter_c.compute_metric(self.config.task.classification_metric_name)
                + meter_r.compute_metric(self.config.task.regression_metric_name)
            )
        elif task_class == "classification":
            scores = meter_c.compute_metric(self.config.task.classification_metric_name)
        else:
            scores = meter_r.compute_metric(self.config.task.regression_metric_name)

        avg_score = np.mean(scores)
        print(f"Epoch {epoch + 1}/{self.config.training.num_epochs}, "
              f"Train Loss: {avg_loss:.4f}, Train Score: {avg_score:.4f}")

        # Clear meter state to free memory
        meter_c.reset()
        meter_r.reset()

        return scores, avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        return_all_metrics: bool = False,
    ) -> Union[List[float], Dict[str, Any]]:
        """
        Evaluate model on data loader.

        Args:
            data_loader: Data loader to evaluate on
            return_all_metrics: If True, return all metrics for wandb logging

        Returns:
            List of primary scores OR dict of all metrics if return_all_metrics=True
        """
        self.model.eval()

        meter_c = Meter()
        meter_r = Meter()

        task_class = self.config.task.task_class
        n_classification = len(self.config.task.classification_list)

        for batch_data in data_loader:
            smiles, bg, labels, mask = batch_data

            # Move to device with non_blocking for async transfer when using pin_memory
            bg = bg.to(self.device)
            labels = labels.float().to(self.device, non_blocking=True)
            mask = mask.float().to(self.device, non_blocking=True)
            atom_feats = bg.ndata.pop(self.config.task.atom_data_field).float().to(self.device, non_blocking=True)
            bond_feats = bg.edata.pop(self.config.task.bond_data_field).long().to(self.device, non_blocking=True)

            # Forward pass
            logits = self.model(bg, atom_feats, bond_feats)
            labels = labels.type_as(logits)

            # Update meters
            if task_class == "classification_regression":
                logits_c = logits[:, :n_classification]
                labels_c = labels[:, :n_classification]
                mask_c = mask[:, :n_classification]

                logits_r = logits[:, n_classification:]
                labels_r = labels[:, n_classification:]
                mask_r = mask[:, n_classification:]

                meter_c.update(logits_c, labels_c, mask_c)
                meter_r.update(logits_r, labels_r, mask_r)

            elif task_class == "classification":
                meter_c.update(logits, labels, mask)
            else:
                meter_r.update(logits, labels, mask)

        # Compute metrics
        if not return_all_metrics:
            # Original behavior: return primary metric scores only
            if task_class == "classification_regression":
                return (
                    meter_c.compute_metric(self.config.task.classification_metric_name)
                    + meter_r.compute_metric(self.config.task.regression_metric_name)
                )
            elif task_class == "classification":
                return meter_c.compute_metric(self.config.task.classification_metric_name)
            else:
                return meter_r.compute_metric(self.config.task.regression_metric_name)

        # New behavior: return all metrics for wandb logging
        all_metrics: Dict[str, Any] = {}

        if task_class in ["classification", "classification_regression"]:
            all_metrics["classification"] = meter_c.compute_classification_metrics(
                metrics=["roc_auc", "accuracy", "mcc"]
            )

        if task_class in ["regression", "classification_regression"]:
            all_metrics["regression"] = meter_r.compute_regression_metrics(
                metrics=["r2", "rmse", "mae"]
            )

        # Compute primary scores (for valid score calculation)
        if task_class == "classification_regression":
            all_metrics["primary_scores"] = (
                all_metrics["classification"]["roc_auc"]
                + all_metrics["regression"]["r2"]
            )
        elif task_class == "classification":
            all_metrics["primary_scores"] = all_metrics["classification"]["roc_auc"]
        else:
            all_metrics["primary_scores"] = all_metrics["regression"]["r2"]

        return all_metrics

    def train(self) -> Dict:
        """
        Run full training loop.

        Returns:
            Dictionary with training history and best scores
        """
        self._init_wandb()

        history = {
            "train_scores": [],
            "val_scores": [],
            "train_losses": [],
        }

        # Get task lists (constant throughout training)
        task_class = self.config.task.task_class
        classification_tasks = list(self.config.task.classification_list)
        regression_tasks = list(self.config.task.regression_list)
        task_list = classification_tasks + regression_tasks
        if not task_list:
            task_list = list(self.config.task.select_task_list)
            if task_class == "classification":
                classification_tasks = task_list
            elif task_class == "regression":
                regression_tasks = task_list

        for epoch in range(self.config.training.num_epochs):
            # Train
            train_scores, train_loss = self.train_epoch(epoch)
            history["train_scores"].append(train_scores)
            history["train_losses"].append(train_loss)

            # Validate - get all metrics for wandb logging
            val_all_metrics = self.evaluate(self.val_loader, return_all_metrics=True)
            val_scores = val_all_metrics["primary_scores"]
            history["val_scores"].append(val_scores)
            val_score = np.mean(val_scores)

            print(f"Validation Score: {val_score:.4f}")

            # Log to wandb
            log_dict = {
                "train_loss": train_loss,
                "train_score": np.mean(train_scores),
                "val_score": val_score,
                "best_val_score": self.stopper.best_score or val_score,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            # Log classification metrics (mean and per-task)
            if "classification" in val_all_metrics:
                clas_metrics = val_all_metrics["classification"]
                log_dict["val_auroc_mean"] = float(np.mean(clas_metrics["roc_auc"]))
                log_dict["val_accuracy_mean"] = float(np.mean(clas_metrics["accuracy"]))
                log_dict["val_mcc_mean"] = float(np.mean(clas_metrics["mcc"]))

                for i, task_name in enumerate(classification_tasks):
                    if i < len(clas_metrics["roc_auc"]):
                        log_dict[f"{task_name}_val_auroc"] = clas_metrics["roc_auc"][i]
                        log_dict[f"{task_name}_val_accuracy"] = clas_metrics["accuracy"][i]
                        log_dict[f"{task_name}_val_mcc"] = clas_metrics["mcc"][i]

            # Log regression metrics (mean and per-task)
            if "regression" in val_all_metrics:
                reg_metrics = val_all_metrics["regression"]
                log_dict["val_r2_mean"] = float(np.mean(reg_metrics["r2"]))
                log_dict["val_rmse_mean"] = float(np.mean(reg_metrics["rmse"]))
                log_dict["val_mae_mean"] = float(np.mean(reg_metrics["mae"]))

                for i, task_name in enumerate(regression_tasks):
                    if i < len(reg_metrics["r2"]):
                        log_dict[f"{task_name}_val_r2"] = reg_metrics["r2"][i]
                        log_dict[f"{task_name}_val_rmse"] = reg_metrics["rmse"][i]
                        log_dict[f"{task_name}_val_mae"] = reg_metrics["mae"][i]

            # Log train scores per task (primary metric only)
            for i, task_name in enumerate(task_list):
                if i < len(train_scores):
                    log_dict[f"{task_name}_train"] = train_scores[i]

            self._log_wandb(log_dict, epoch)

            # Update scheduler
            self.scheduler.step(val_score)

            # Early stopping
            if self.stopper.step(val_score, self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        self.stopper.load_checkpoint(self.model, device=self.device)

        # Final test evaluation
        if self.test_loader:
            test_all_metrics = self.evaluate(self.test_loader, return_all_metrics=True)
            test_scores = test_all_metrics["primary_scores"]
            history["test_scores"] = test_scores
            print(f"Test Score: {np.mean(test_scores):.4f}")

            if self.use_wandb:
                test_log = {"test_score": float(np.mean(test_scores))}

                # Log classification metrics
                if "classification" in test_all_metrics:
                    clas_metrics = test_all_metrics["classification"]
                    test_log["test_auroc_mean"] = float(np.mean(clas_metrics["roc_auc"]))
                    test_log["test_accuracy_mean"] = float(np.mean(clas_metrics["accuracy"]))
                    test_log["test_mcc_mean"] = float(np.mean(clas_metrics["mcc"]))

                    for i, task_name in enumerate(classification_tasks):
                        if i < len(clas_metrics["roc_auc"]):
                            test_log[f"{task_name}_test_auroc"] = clas_metrics["roc_auc"][i]
                            test_log[f"{task_name}_test_accuracy"] = clas_metrics["accuracy"][i]
                            test_log[f"{task_name}_test_mcc"] = clas_metrics["mcc"][i]

                # Log regression metrics
                if "regression" in test_all_metrics:
                    reg_metrics = test_all_metrics["regression"]
                    test_log["test_r2_mean"] = float(np.mean(reg_metrics["r2"]))
                    test_log["test_rmse_mean"] = float(np.mean(reg_metrics["rmse"]))
                    test_log["test_mae_mean"] = float(np.mean(reg_metrics["mae"]))

                    for i, task_name in enumerate(regression_tasks):
                        if i < len(reg_metrics["r2"]):
                            test_log[f"{task_name}_test_r2"] = reg_metrics["r2"][i]
                            test_log[f"{task_name}_test_rmse"] = reg_metrics["rmse"][i]
                            test_log[f"{task_name}_test_mae"] = reg_metrics["mae"][i]

                self._log_wandb(test_log, self.config.training.num_epochs)

        # Finish wandb
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()

        return history
