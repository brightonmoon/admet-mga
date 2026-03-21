"""
Collate functions for batching molecular graphs.

This module provides collate functions for use with PyTorch DataLoader.
"""

from __future__ import annotations

from typing import List, Tuple

import dgl
import torch


def collate_molgraphs(data: List) -> Tuple:
    """
    Collate function for training/validation.

    Batches molecular graphs and their labels for model training.

    Args:
        data: List of [smiles, graph, labels, mask] tuples

    Returns:
        Tuple of (smiles_list, batched_graph, labels_tensor, mask_tensor)
    """
    smiles, graphs, labels, mask = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    # Use torch.as_tensor for efficient conversion (avoids intermediate numpy copy)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)

    return smiles, bg, labels, mask


def collate_molgraphs_inference(data: List) -> Tuple:
    """
    Collate function for inference (no labels).

    Args:
        data: List of [smiles, graph] tuples

    Returns:
        Tuple of (smiles_list, batched_graph)
    """
    smiles, graphs = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return smiles, bg
