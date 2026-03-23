"""
Pydantic-based configuration classes for MGA.

This module provides type-safe configuration management using Pydantic,
replacing the previous dictionary-based configuration approach.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    in_feats: int = Field(default=40, description="Input feature dimension (atom features)")
    rgcn_hidden_feats: List[int] = Field(
        default=[64, 64],
        description="Hidden dimensions for RGCN layers"
    )
    classifier_hidden_feats: int = Field(
        default=64,
        description="Hidden dimension for classifier MLP"
    )
    n_tasks: int = Field(default=1, description="Number of prediction tasks")
    loop: bool = Field(default=True, description="Whether to use self-loop in RGCN")
    rgcn_drop_out: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout rate for RGCN")
    drop_out: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout rate for classifier")
    num_rels: int = Field(default=64*21, description="Number of relation types for RGCN")
    return_weight: bool = Field(default=False, description="Whether to return attention weights")
    return_mol_embedding: bool = Field(default=False, description="Whether to return molecule embeddings")


class WandbConfig(BaseModel):
    """Weights & Biases configuration."""

    enabled: bool = Field(default=True, description="Whether to use wandb logging")
    project: str = Field(default="mga-training", description="wandb project name")
    entity: Optional[str] = Field(default=None, description="wandb team/entity name")
    name: Optional[str] = Field(default=None, description="Run name (auto-generated if None)")
    tags: List[str] = Field(default_factory=list, description="Tags for the run")
    notes: Optional[str] = Field(default=None, description="Notes for the run")

    @field_validator("project")
    @classmethod
    def project_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("wandb project name cannot be empty")
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""

    num_epochs: int = Field(default=500, gt=0, description="Number of training epochs")
    batch_size: int = Field(default=512, gt=0, description="Batch size")
    lr: float = Field(default=0.001, gt=0, description="Learning rate")
    weight_decay: float = Field(default=1e-5, ge=0, description="Weight decay for optimizer")
    patience: int = Field(default=50, gt=0, description="Early stopping patience")
    device: str = Field(default="cuda", description="Device to use (cuda/cpu)")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Scheduler settings
    scheduler_factor: float = Field(default=0.5, gt=0, lt=1, description="LR scheduler factor")
    scheduler_patience: int = Field(default=10, gt=0, description="LR scheduler patience")
    min_lr: float = Field(default=1e-7, gt=0, description="Minimum learning rate")

    # DataLoader optimization settings
    # Defaults are optimized for small datasets (minimizes RAM usage)
    # For large datasets (>50,000), recommend num_workers=4, pin_memory=True
    num_workers: int = Field(default=0, ge=0, description="Number of DataLoader workers (0 for small datasets to reduce RAM)")
    pin_memory: bool = Field(default=False, description="Pin memory for faster CPU-to-GPU transfer (enable for large datasets)")
    persistent_workers: bool = Field(default=False, description="Keep DataLoader workers alive between epochs (enable for large datasets)")
    prefetch_factor: int = Field(default=2, gt=0, description="Number of batches to prefetch per worker")

    # wandb
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ["cuda", "cpu", "mps"]:
            raise ValueError(f"Invalid device: {v}. Must be 'cuda', 'cpu', or 'mps'")
        return v


class TaskConfig(BaseModel):
    """Task configuration for multi-task learning."""

    task_name: str = Field(description="Name of the task/experiment")
    task_class: Literal["classification", "regression", "classification_regression"] = Field(
        default="classification",
        description="Type of task"
    )
    classification_list: List[str] = Field(
        default_factory=list,
        description="List of classification task names"
    )
    regression_list: List[str] = Field(
        default_factory=list,
        description="List of regression task names"
    )
    select_task_list: List[str] = Field(
        default_factory=list,
        description="Selected tasks for training"
    )
    select_task_index: Optional[List[int]] = Field(
        default=None,
        description="Indices of selected tasks"
    )

    # Metric settings
    classification_metric_name: str = Field(default="roc_auc", description="Metric for classification")
    regression_metric_name: str = Field(default="r2", description="Metric for regression")

    # Data field names
    atom_data_field: str = Field(default="atom", description="Node feature field name")
    bond_data_field: str = Field(default="etype", description="Edge type field name")


class PathConfig(BaseModel):
    """Path configuration."""

    data_dir: Path = Field(default=Path("data"), description="Data directory")
    model_dir: Path = Field(default=Path("models"), description="Model checkpoint directory")
    result_dir: Path = Field(default=Path("results"), description="Results directory")
    config_dir: Path = Field(default=Path("config"), description="Config directory")

    # Specific file paths
    train_data_path: Optional[Path] = Field(default=None, description="Training data path")
    pretrained_model_path: Optional[Path] = Field(default=None, description="Pretrained model path")

    @field_validator("data_dir", "model_dir", "result_dir", "config_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class TransferConfig(BaseModel):
    """Transfer learning configuration.

    Supports four strategies:
    1. full_finetune: Load encoder weights, train all layers with low LR
    2. feature_extraction: Freeze encoder, train only new task head
    3. selective_layer: Freeze early RGCN layers, finetune later layers + head
    4. attention_transfer: Transfer attention weights from similar pretrained task
    """

    # Strategy selection
    strategy: Literal[
        "full_finetune",
        "feature_extraction",
        "selective_layer",
        "attention_transfer"
    ] = Field(
        default="selective_layer",
        description="Transfer learning strategy to use"
    )

    # Pretrained model settings
    pretrained_model_path: Optional[Path] = Field(
        default=None,
        description="Path to pretrained model checkpoint"
    )
    source_n_tasks: int = Field(
        default=52,
        description="Number of tasks in pretrained model"
    )

    # Layer freezing settings
    freeze_encoder: bool = Field(
        default=False,
        description="Freeze entire RGCN encoder (for feature_extraction strategy)"
    )
    freeze_layers: List[int] = Field(
        default_factory=list,
        description="Indices of RGCN layers to freeze (e.g., [0] freezes first layer)"
    )

    # Differential learning rate
    encoder_lr_multiplier: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Multiplier for encoder LR relative to head LR (encoder_lr = base_lr * multiplier)"
    )

    # Attention transfer settings (for attention_transfer strategy)
    source_task_indices: Optional[List[int]] = Field(
        default=None,
        description="Indices of source tasks for attention transfer"
    )

    # Gradual unfreezing
    unfreeze_epoch: Optional[int] = Field(
        default=None,
        description="Epoch at which to unfreeze frozen layers (None = never unfreeze)"
    )

    @field_validator("pretrained_model_path", mode="before")
    @classmethod
    def ensure_pretrained_path(cls, v):
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class MGAConfig(BaseModel):
    """Complete MGA configuration combining all sub-configs."""

    # YAML 파일에서 불필요한 키(data_name, bin_path, group_path 등)가 전달될 수 있으므로 명시적으로 무시
    model_config = ConfigDict(extra="ignore")

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    task: TaskConfig = Field(default_factory=lambda: TaskConfig(task_name="mga-default"))
    paths: PathConfig = Field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MGAConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Config path is not a file: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}") from e
        if data is None:
            data = {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "MGAConfig":
        """Create configuration from dictionary (legacy support)."""
        # Map legacy keys to new structure
        model_data = {}
        training_data = {}
        task_data = {}
        path_data = {}

        # Model config mappings
        model_keys = [
            "in_feats", "rgcn_hidden_feats", "classifier_hidden_feats",
            "n_tasks", "loop", "rgcn_drop_out", "drop_out", "num_rels",
            "return_weight", "return_mol_embedding"
        ]

        # Training config mappings
        training_keys = [
            "num_epochs", "batch_size", "lr", "weight_decay", "patience",
            "device", "seed"
        ]

        # Task config mappings
        task_keys = [
            "task_name", "task_class", "classification_list", "regression_list",
            "select_task_list", "select_task_index", "classification_metric_name",
            "regression_metric_name", "atom_data_field", "bond_data_field"
        ]

        for key, value in data.items():
            if key in model_keys:
                model_data[key] = value
            elif key in training_keys:
                training_data[key] = value
            elif key in task_keys:
                task_data[key] = value
            elif key == "model":
                model_data.update(value)
            elif key == "training":
                training_data.update(value)
            elif key == "task":
                task_data.update(value)
            elif key == "paths":
                path_data.update(value)

        # Ensure task_name exists
        if "task_name" not in task_data:
            task_data["task_name"] = "mga-default"

        return cls(
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
            training=TrainingConfig(**training_data) if training_data else TrainingConfig(),
            task=TaskConfig(**task_data),
            paths=PathConfig(**path_data) if path_data else PathConfig(),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            # Use mode='json' to serialize Path objects as strings
            yaml.dump(self.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to flat dictionary (legacy format support)."""
        result = {}
        result.update(self.model.model_dump())
        result.update(self.training.model_dump())
        result.update(self.task.model_dump())
        return result

    def to_wandb_config(self) -> dict:
        """Convert to wandb-compatible config dictionary."""
        return {
            **self.model.model_dump(),
            **{k: v for k, v in self.training.model_dump().items() if k != "wandb"},
            **self.task.model_dump(),
        }


def load_config(path: str | Path) -> MGAConfig:
    """Load configuration from YAML file."""
    return MGAConfig.from_yaml(path)


def save_config(config: MGAConfig, path: str | Path) -> None:
    """Save configuration to YAML file."""
    config.to_yaml(path)
