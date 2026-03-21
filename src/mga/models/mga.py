"""
MGA (Molecular Graph Attention) model implementations.

This module contains the main model architectures:
- BaseGNN: Base class with task-specific prediction heads
- MGA: Dynamic RGCN architecture with configurable layers
- MGATest: Fixed 2-layer RGCN architecture for inference
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import dgl

import torch
import torch.nn as nn

from mga.models.layers import RGCNLayer, WeightAndSum
from mga.models.heads import create_fc_layer, create_output_layer


class BaseGNN(nn.Module):
    """
    Base GNN model with task-specific attention and prediction heads.

    This class provides the shared architecture for MGA variants:
    - RGCN layers for graph encoding (defined in subclasses)
    - Task-specific attention readout
    - Multi-layer classifier heads for each task

    Args:
        gnn_out_feats: Output dimension of GNN layers
        n_tasks: Number of prediction tasks
        rgcn_drop_out: Dropout rate for RGCN (unused in base, for compatibility)
        return_mol_embedding: Whether to return molecule embeddings instead of predictions
        return_weight: Whether to return attention weights
        classifier_hidden_feats: Hidden dimension for classifier MLPs
        dropout: Dropout rate for classifiers

    Attributes:
        gnn_layers: ModuleList of RGCN layers (populated by subclasses)
        weighted_sum_readout: Task-specific attention readout
        fc_layers1/2/3: Classifier hidden layers for each task
        output_layer1: Output layers for each task
    """

    def __init__(
        self,
        gnn_out_feats: int,
        n_tasks: int,
        rgcn_drop_out: float = 0.5,
        return_mol_embedding: bool = False,
        return_weight: bool = False,
        classifier_hidden_feats: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()

        # Attention readout
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(
            gnn_out_feats,
            self.task_num,
            return_weight=return_weight,
        )

        self.fc_in_feats = gnn_out_feats
        self.return_mol_embedding = return_mol_embedding

        # Task-specific classifiers (3 hidden layers each)
        self.fc_layers1 = nn.ModuleList([
            create_fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])
        self.fc_layers2 = nn.ModuleList([
            create_fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])
        self.fc_layers3 = nn.ModuleList([
            create_fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])

        # Output layers
        self.output_layer1 = nn.ModuleList([
            create_output_layer(classifier_hidden_feats, 1)
            for _ in range(self.task_num)
        ])

    def forward(
        self,
        bg: dgl.DGLGraph,
        node_feats: torch.Tensor,
        etype: torch.Tensor,
        norm: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            bg: Batched DGL graph
            node_feats: Node (atom) features [N, in_feats]
            etype: Edge types [E]
            norm: Optional edge normalizer

        Returns:
            If return_mol_embedding: molecule embeddings [B, gnn_out_feats]
            If return_weight: (predictions, attention_weights, node_feats)
            Otherwise: predictions [B, n_tasks]
        """
        # Apply GNN layers
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, etype, norm)

        # Compute molecule features from atom features
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        # Task-specific predictions - collect in list then concatenate once
        # (More efficient than incremental concatenation which creates N-1 intermediate tensors)
        predictions = []
        for i in range(self.task_num):
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predictions.append(self.output_layer1[i](h3))

        prediction_all = torch.cat(predictions, dim=1)

        # Return based on mode
        if self.return_mol_embedding:
            return feats_list[0]

        if self.return_weight:
            return prediction_all, atom_weight_list, node_feats

        return prediction_all


class MGA(BaseGNN):
    """
    Molecular Graph Attention model with dynamic RGCN architecture.

    This model uses configurable RGCN layers for molecular encoding
    and task-specific attention for multi-task learning.

    Args:
        in_feats: Input atom feature dimension (default: 40)
        rgcn_hidden_feats: List of hidden dimensions for RGCN layers
        n_tasks: Number of prediction tasks
        return_weight: Whether to return attention weights
        classifier_hidden_feats: Hidden dimension for classifier MLPs
        loop: Whether to use self-loops in RGCN
        return_mol_embedding: Whether to return molecule embeddings
        rgcn_drop_out: Dropout rate for RGCN layers
        dropout: Dropout rate for classifiers

    Example:
        >>> model = MGA(
        ...     in_feats=40,
        ...     rgcn_hidden_feats=[64, 64],
        ...     n_tasks=5,
        ...     classifier_hidden_feats=128,
        ... )
        >>> predictions = model(batched_graph, atom_feats, edge_types)
    """

    def __init__(
        self,
        in_feats: int = 40,
        rgcn_hidden_feats: Optional[List[int]] = None,
        n_tasks: int = 1,
        return_weight: bool = False,
        classifier_hidden_feats: int = 128,
        loop: bool = False,
        return_mol_embedding: bool = False,
        rgcn_drop_out: float = 0.5,
        dropout: float = 0.0,
    ):
        if rgcn_hidden_feats is None:
            rgcn_hidden_feats = [64, 64]

        super().__init__(
            gnn_out_feats=rgcn_hidden_feats[-1],
            n_tasks=n_tasks,
            classifier_hidden_feats=classifier_hidden_feats,
            return_mol_embedding=return_mol_embedding,
            return_weight=return_weight,
            rgcn_drop_out=rgcn_drop_out,
            dropout=dropout,
        )

        # Build RGCN layers dynamically
        current_in_feats = in_feats
        for out_feats in rgcn_hidden_feats:
            self.gnn_layers.append(
                RGCNLayer(
                    current_in_feats,
                    out_feats,
                    loop=loop,
                    rgcn_drop_out=rgcn_drop_out,
                )
            )
            current_in_feats = out_feats


class MGATest(nn.Module):
    """
    MGA model with fixed 2-layer architecture for inference.

    This model has a fixed structure optimized for inference,
    matching the pretrained model checkpoints.

    Args:
        in_feats: Input atom feature dimension
        rgcn_hidden_feats: Hidden dimension for first RGCN layer
        gnn_out_feats: Output dimension for second RGCN layer
        n_tasks: Number of prediction tasks
        return_weight: Whether to return attention weights
        return_mol_embedding: Whether to return molecule embeddings
        loop: Whether to use self-loops in RGCN
        rgcn_drop_out: Dropout rate for RGCN layers
        classifier_hidden_feats: Hidden dimension for classifier MLPs
        dropout: Dropout rate for classifiers

    Example:
        >>> model = MGATest(
        ...     in_feats=40,
        ...     rgcn_hidden_feats=64,
        ...     gnn_out_feats=64,
        ...     n_tasks=5,
        ... )
        >>> predictions = model(batched_graph, atom_feats, edge_types)
    """

    def __init__(
        self,
        in_feats: int,
        rgcn_hidden_feats: int,
        gnn_out_feats: int,
        n_tasks: int,
        return_weight: bool = False,
        return_mol_embedding: bool = False,
        loop: bool = False,
        rgcn_drop_out: float = 0.5,
        classifier_hidden_feats: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.task_num = n_tasks

        # Fixed 2-layer RGCN
        self.rgcn_layer1 = RGCNLayer(
            in_feats,
            rgcn_hidden_feats,
            loop=loop,
            rgcn_drop_out=rgcn_drop_out,
        )
        self.rgcn_layer2 = RGCNLayer(
            rgcn_hidden_feats,
            gnn_out_feats,
            loop=loop,
            rgcn_drop_out=rgcn_drop_out,
        )

        # Attention readout
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(
            gnn_out_feats,
            self.task_num,
            return_weight=return_weight,
        )

        self.fc_in_feats = gnn_out_feats
        self.return_mol_embedding = return_mol_embedding

        # Task-specific classifiers
        self.fc_layers1 = nn.ModuleList([
            self._fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])
        self.fc_layers2 = nn.ModuleList([
            self._fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])
        self.fc_layers3 = nn.ModuleList([
            self._fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])

        # Output layers
        self.output_layer1 = nn.ModuleList([
            self._output_layer(classifier_hidden_feats, 1)
            for _ in range(self.task_num)
        ])

    def _fc_layer(self, dropout: float, in_feats: int, hidden_feats: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
        )

    def _output_layer(self, hidden_feats: int, out_feats: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_feats, out_feats),
        )

    def forward(
        self,
        bg: dgl.DGLGraph,
        node_feats: torch.Tensor,
        etype: torch.Tensor,
        norm: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            bg: Batched DGL graph
            node_feats: Node (atom) features [N, in_feats]
            etype: Edge types [E]
            norm: Optional edge normalizer

        Returns:
            If return_mol_embedding: molecule embeddings [B, gnn_out_feats]
            If return_weight: (predictions, attention_weights, node_feats)
            Otherwise: predictions [B, n_tasks]
        """
        # Apply fixed RGCN layers
        node_feats = self.rgcn_layer1(bg, node_feats, etype, norm)
        node_feats = self.rgcn_layer2(bg, node_feats, etype, norm)

        # Compute molecule features
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        # Task-specific predictions - collect in list then concatenate once
        # (More efficient than incremental concatenation which creates N-1 intermediate tensors)
        predictions = []
        for i in range(self.task_num):
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predictions.append(self.output_layer1[i](h3))

        prediction_all = torch.cat(predictions, dim=1)

        # Return based on mode
        if self.return_mol_embedding:
            return feats_list[0]

        if self.return_weight:
            return prediction_all, atom_weight_list, node_feats

        return prediction_all
