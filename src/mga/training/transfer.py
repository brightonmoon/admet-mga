"""
Transfer learning utilities for MGA models.

This module provides TransferLearningManager for:
- Loading pretrained encoder weights
- Freezing/unfreezing model layers
- Creating differential learning rate parameter groups
- Attention transfer between tasks
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TransferLearningManager:
    """
    Manager for transfer learning weight loading and layer freezing.

    Supports four strategies:
    1. full_finetune: Load encoder, train all with differential LR
    2. feature_extraction: Freeze encoder, train head only
    3. selective_layer: Freeze early layers, train later layers + head
    4. attention_transfer: Copy attention from similar task

    Example:
        >>> manager = TransferLearningManager(
        ...     pretrained_path="models/toxicity_early_stop.pth",
        ...     strategy="selective_layer",
        ...     freeze_layers=[0],
        ... )
        >>> manager.setup(model)
        >>> param_groups = manager.get_parameter_groups(model, base_lr=0.001)
        >>> optimizer = torch.optim.Adam(param_groups)
    """

    # RGCN layer parameter names for weight loading
    RGCN_LAYER1_PARAMS = [
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
    ]

    RGCN_LAYER2_PARAMS = [
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

    # Dynamic GNN layer parameter pattern (for MGA class with gnn_layers)
    GNN_LAYER_PATTERN = "gnn_layers.{idx}."

    def __init__(
        self,
        pretrained_path: Optional[str | Path] = None,
        strategy: str = "selective_layer",
        source_n_tasks: int = 52,
        freeze_layers: Optional[List[int]] = None,
        source_task_indices: Optional[List[int]] = None,
        encoder_lr_multiplier: float = 0.1,
        unfreeze_epoch: Optional[int] = None,
    ):
        """
        Initialize TransferLearningManager.

        Args:
            pretrained_path: Path to pretrained model checkpoint
            strategy: One of "full_finetune", "feature_extraction",
                     "selective_layer", "attention_transfer"
            source_n_tasks: Number of tasks in pretrained model
            freeze_layers: List of RGCN layer indices to freeze (e.g., [0])
            source_task_indices: Task indices for attention transfer
            encoder_lr_multiplier: LR multiplier for encoder (encoder_lr = base_lr * multiplier)
            unfreeze_epoch: Epoch to unfreeze frozen layers (None = never)
        """
        self.pretrained_path = Path(pretrained_path) if pretrained_path else None
        self.strategy = strategy
        self.source_n_tasks = source_n_tasks
        self.freeze_layers = freeze_layers or [0]
        self.source_task_indices = source_task_indices
        self.encoder_lr_multiplier = encoder_lr_multiplier
        self.unfreeze_epoch = unfreeze_epoch

        self._frozen_params: List[str] = []
        self._loaded_params_count = 0

    def setup(self, model: nn.Module) -> int:
        """
        Apply transfer learning strategy to model.

        Args:
            model: Model to setup for transfer learning

        Returns:
            Number of parameters loaded from pretrained model
        """
        if self.pretrained_path is None:
            print("No pretrained model path specified, skipping weight loading")
            return 0

        if self.strategy == "full_finetune":
            self._loaded_params_count = self.load_pretrained_encoder(model)
            # No freezing, all layers train with differential LR

        elif self.strategy == "feature_extraction":
            self._loaded_params_count = self.load_pretrained_encoder(model)
            self.freeze_encoder(model)

        elif self.strategy == "selective_layer":
            self._loaded_params_count = self.load_pretrained_encoder(model)
            self.freeze_encoder_layers(model, self.freeze_layers)

        elif self.strategy == "attention_transfer":
            if self.source_task_indices is None:
                raise ValueError("source_task_indices required for attention_transfer strategy")
            self._loaded_params_count = self.load_pretrained_with_attention(
                model, self.source_task_indices
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self._loaded_params_count

    def load_pretrained_encoder(self, model: nn.Module) -> int:
        """
        Load only RGCN encoder weights from pretrained model.

        Works with both MGATest (rgcn_layer1/2) and MGA (gnn_layers) architectures.

        Args:
            model: Model to load weights into

        Returns:
            Number of parameters loaded
        """
        if self.pretrained_path is None:
            raise ValueError("pretrained_path not set")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(
            self.pretrained_path, map_location=device, weights_only=True
        )

        pretrained_state = checkpoint.get("model_state_dict", checkpoint)
        model_state = model.state_dict()

        # Determine which parameters to load based on model architecture
        encoder_params = self.RGCN_LAYER1_PARAMS + self.RGCN_LAYER2_PARAMS

        # Check if model uses dynamic gnn_layers (MGA class)
        has_gnn_layers = any(k.startswith("gnn_layers.") for k in model_state.keys())

        if has_gnn_layers:
            # Map from rgcn_layer1/2 to gnn_layers.0/1
            param_mapping = self._create_gnn_layers_mapping()
            loaded_dict = {}
            for old_key, new_key in param_mapping.items():
                if old_key in pretrained_state and new_key in model_state:
                    if pretrained_state[old_key].shape == model_state[new_key].shape:
                        loaded_dict[new_key] = pretrained_state[old_key]
        else:
            # Direct loading for MGATest with rgcn_layer1/2
            loaded_dict = {
                k: v for k, v in pretrained_state.items()
                if k in encoder_params and k in model_state
            }

        model.load_state_dict(loaded_dict, strict=False)
        print(f"Loaded {len(loaded_dict)} pretrained encoder parameters")

        return len(loaded_dict)

    def load_pretrained_with_attention(
        self,
        model: nn.Module,
        source_task_indices: List[int],
    ) -> int:
        """
        Load encoder + attention weights from specific source tasks.

        Args:
            model: Model to load weights into
            source_task_indices: Which source task's attention to copy

        Returns:
            Number of parameters loaded
        """
        # First load encoder
        loaded_count = self.load_pretrained_encoder(model)

        if self.pretrained_path is None:
            return loaded_count

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(
            self.pretrained_path, map_location=device, weights_only=True
        )
        pretrained_state = checkpoint.get("model_state_dict", checkpoint)
        model_state = model.state_dict()

        # Load attention weights from source tasks
        attention_dict = {}
        for target_idx, source_idx in enumerate(source_task_indices):
            source_weight_key = f"weighted_sum_readout.atom_weighting_specific.{source_idx}.0.weight"
            source_bias_key = f"weighted_sum_readout.atom_weighting_specific.{source_idx}.0.bias"
            target_weight_key = f"weighted_sum_readout.atom_weighting_specific.{target_idx}.0.weight"
            target_bias_key = f"weighted_sum_readout.atom_weighting_specific.{target_idx}.0.bias"

            if source_weight_key in pretrained_state and target_weight_key in model_state:
                attention_dict[target_weight_key] = pretrained_state[source_weight_key]
            if source_bias_key in pretrained_state and target_bias_key in model_state:
                attention_dict[target_bias_key] = pretrained_state[source_bias_key]

        if attention_dict:
            model.load_state_dict(attention_dict, strict=False)
            print(f"Loaded {len(attention_dict)} attention parameters from source tasks {source_task_indices}")

        return loaded_count + len(attention_dict)

    def freeze_encoder(self, model: nn.Module) -> None:
        """
        Freeze entire RGCN encoder.

        Args:
            model: Model to freeze encoder layers
        """
        frozen_count = 0
        for name, param in model.named_parameters():
            if self._is_encoder_param(name):
                param.requires_grad = False
                self._frozen_params.append(name)
                frozen_count += 1

        print(f"Frozen {frozen_count} encoder parameters")

    def freeze_encoder_layers(
        self,
        model: nn.Module,
        layer_indices: List[int],
    ) -> None:
        """
        Freeze specific RGCN layers.

        Args:
            model: Model to freeze layers in
            layer_indices: List of layer indices to freeze (0-based)
        """
        frozen_count = 0
        for name, param in model.named_parameters():
            for layer_idx in layer_indices:
                if self._is_layer_param(name, layer_idx):
                    param.requires_grad = False
                    self._frozen_params.append(name)
                    frozen_count += 1
                    break

        print(f"Frozen {frozen_count} parameters in layers {layer_indices}")

    def unfreeze_all(self, model: nn.Module) -> None:
        """
        Unfreeze all previously frozen parameters.

        Args:
            model: Model to unfreeze
        """
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if name in self._frozen_params:
                param.requires_grad = True
                unfrozen_count += 1

        self._frozen_params.clear()
        print(f"Unfrozen {unfrozen_count} parameters")

    def maybe_unfreeze(self, model: nn.Module, current_epoch: int) -> bool:
        """
        Unfreeze layers if current epoch matches unfreeze_epoch.

        Args:
            model: Model to potentially unfreeze
            current_epoch: Current training epoch

        Returns:
            True if unfreezing was performed
        """
        if self.unfreeze_epoch is not None and current_epoch >= self.unfreeze_epoch:
            if self._frozen_params:  # Only unfreeze if there are frozen params
                self.unfreeze_all(model)
                return True
        return False

    def get_parameter_groups(
        self,
        model: nn.Module,
        base_lr: float,
        weight_decay: float = 1e-5,
    ) -> List[Dict]:
        """
        Create parameter groups with differential learning rates.

        Encoder parameters get base_lr * encoder_lr_multiplier.
        Head parameters get base_lr.
        Frozen parameters are excluded.

        Args:
            model: Model to create parameter groups for
            base_lr: Base learning rate for head parameters
            weight_decay: Weight decay for all parameters

        Returns:
            List of parameter group dicts for optimizer
        """
        encoder_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters

            if self._is_encoder_param(name):
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = []

        if encoder_params:
            encoder_lr = base_lr * self.encoder_lr_multiplier
            param_groups.append({
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "encoder",
            })
            print(f"Encoder params: {len(encoder_params)}, lr={encoder_lr:.2e}")

        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "head",
            })
            print(f"Head params: {len(head_params)}, lr={base_lr:.2e}")

        return param_groups

    def _is_encoder_param(self, name: str) -> bool:
        """Check if parameter belongs to encoder (RGCN layers)."""
        encoder_prefixes = [
            "rgcn_layer1.",
            "rgcn_layer2.",
            "gnn_layers.",
        ]
        return any(name.startswith(prefix) for prefix in encoder_prefixes)

    def _is_layer_param(self, name: str, layer_idx: int) -> bool:
        """Check if parameter belongs to specific RGCN layer index."""
        # For MGATest style (rgcn_layer1, rgcn_layer2)
        if name.startswith(f"rgcn_layer{layer_idx + 1}."):
            return True
        # For MGA style (gnn_layers.0, gnn_layers.1)
        if name.startswith(f"gnn_layers.{layer_idx}."):
            return True
        return False

    def _create_gnn_layers_mapping(self) -> Dict[str, str]:
        """Create mapping from rgcn_layer1/2 names to gnn_layers.0/1 names."""
        mapping = {}

        for old_key in self.RGCN_LAYER1_PARAMS:
            new_key = old_key.replace("rgcn_layer1.", "gnn_layers.0.")
            mapping[old_key] = new_key

        for old_key in self.RGCN_LAYER2_PARAMS:
            new_key = old_key.replace("rgcn_layer2.", "gnn_layers.1.")
            mapping[old_key] = new_key

        return mapping

    def get_strategy_description(self) -> str:
        """Get human-readable description of current strategy."""
        descriptions = {
            "full_finetune": "Full Finetuning: Load encoder, train all layers with differential LR",
            "feature_extraction": "Feature Extraction: Freeze encoder, train only task head",
            "selective_layer": f"Selective Layer: Freeze layers {self.freeze_layers}, train rest",
            "attention_transfer": f"Attention Transfer: Copy attention from tasks {self.source_task_indices}",
        }
        return descriptions.get(self.strategy, f"Unknown strategy: {self.strategy}")
