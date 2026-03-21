"""
Neural network layers for MGA.

This module contains the core building blocks:
- RGCNLayer: Relational Graph Convolutional Network layer
- WeightAndSum: Attention-based graph readout with task-specific weights
"""

from __future__ import annotations

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.readout import sum_nodes

# Number of edge relation types for RGCN
# Calculated as: bond_features (64) * atom_pair_types (21) = 1344
# bond_features = 4 bond types * 2 conjugated * 2 ring * 4 stereo = 64
# atom_pair_types = 12 common atom pairs + others = ~21 categories
DEFAULT_NUM_RELS = 64 * 21


class WeightAndSum(nn.Module):
    """
    Attention-weighted sum readout for graph neural networks.

    Computes task-specific attention weights for atoms and performs
    weighted sum pooling to generate molecule-level representations.

    Args:
        in_feats: Input feature dimension
        task_num: Number of tasks (each gets separate attention)
        attention: Whether to use attention mechanism
        return_weight: Whether to return attention weights

    Example:
        >>> readout = WeightAndSum(64, task_num=5)
        >>> mol_feats = readout(batched_graph, node_features)
        >>> len(mol_feats)  # Returns list of 5 task-specific features
        5
    """

    def __init__(
        self,
        in_feats: int,
        task_num: int = 1,
        attention: bool = True,
        return_weight: bool = False,
    ):
        super().__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight = return_weight

        # Task-specific attention weights
        self.atom_weighting_specific = nn.ModuleList([
            self._atom_weight_layer(in_feats) for _ in range(task_num)
        ])

        # Shared attention weights (optional fallback)
        self.shared_weighting = self._atom_weight_layer(in_feats)

    def _atom_weight_layer(self, in_feats: int) -> nn.Sequential:
        """Create attention weight layer."""
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, bg: dgl.DGLGraph, feats: torch.Tensor):
        """
        Compute weighted sum of atom features.

        Args:
            bg: Batched DGL graph
            feats: Node features [N, in_feats]

        Returns:
            If attention=True:
                feat_list: List of task-specific molecule features
                atom_list: (optional) List of attention weights if return_weight=True
            If attention=False:
                shared_feats_sum: Shared molecule features
        """
        feat_list = []
        atom_list = []

        # Calculate task-specific features
        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        # Calculate shared features (fallback)
        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')

        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            return feat_list
        return shared_feats_sum


class RGCNLayer(nn.Module):
    """
    Relational Graph Convolutional Network layer.

    Applies relation-aware graph convolution with optional residual
    connections and batch normalization.

    Args:
        in_feats: Input feature dimension
        out_feats: Output feature dimension
        num_rels: Number of relation types
        activation: Activation function
        loop: Whether to include self-loops
        residual: Whether to use residual connections
        batchnorm: Whether to use batch normalization
        rgcn_drop_out: Dropout rate

    Example:
        >>> layer = RGCNLayer(40, 64, num_rels=336, loop=True)
        >>> out_feats = layer(graph, node_feats, edge_types)
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_rels: int = DEFAULT_NUM_RELS,
        activation=F.relu,
        loop: bool = False,
        residual: bool = True,
        batchnorm: bool = True,
        rgcn_drop_out: float = 0.5,
    ):
        super().__init__()

        self.activation = activation

        # Relational graph convolution
        self.graph_conv_layer = RelGraphConv(
            in_feats,
            out_feats,
            num_rels=num_rels,
            regularizer='basis',
            num_bases=None,
            bias=True,
            activation=activation,
            self_loop=loop,
            dropout=rgcn_drop_out,
        )

        # Residual connection
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        # Batch normalization
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(
        self,
        bg: dgl.DGLGraph,
        node_feats: torch.Tensor,
        etype: torch.Tensor,
        norm: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Update atom representations.

        Args:
            bg: Batched DGL graph
            node_feats: Node features [N, in_feats]
            etype: Edge types [E]
            norm: Optional edge normalizer [E, 1]

        Returns:
            new_feats: Updated node features [N, out_feats]
        """
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)

        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats
