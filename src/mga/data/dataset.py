"""
Dataset utilities for MGA.

This module provides functions for building and loading molecular
graph datasets for training and inference.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.graph_serialize import load_graphs, save_graphs
from rdkit.Chem import MolFromSmiles

from mga.data.features import atom_features, etype_features

# Constant for missing label values (used for masking)
MISSING_LABEL_VALUE = 123456


def construct_graph_from_smiles(smiles: str) -> dgl.DGLGraph:
    """
    Convert SMILES string to DGL graph.

    Creates a bidirectional graph with atom features as node features
    and edge types (etype) for relational graph convolution.

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        DGL graph with 'atom' node features and 'etype'/'normal' edge features
    """
    g = dgl.graph(([], []))

    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    # Add nodes (atoms)
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    atoms_feature_all = []
    for atom in mol.GetAtoms():
        atom_feature = atom_features(atom).tolist()
        atoms_feature_all.append(atom_feature)

    g.ndata["atom"] = torch.tensor(atoms_feature_all)

    # Add edges (bonds) - bidirectional
    src_list = []
    dst_list = []
    etype_feature_all = []

    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = etype_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        # Add both directions
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.extend([etype_feature, etype_feature])

    g.add_edges(src_list, dst_list)

    # Compute edge normalization - O(n) using Counter instead of O(n²) .count()
    etype_counts = Counter(etype_feature_all)
    total_edges = len(etype_feature_all)
    if total_edges > 0:
        normal_all = [
            round(etype_counts[etype] / total_edges, 1)
            for etype in etype_feature_all
        ]
    else:
        normal_all = []

    g.edata["etype"] = torch.tensor(etype_feature_all)
    g.edata["normal"] = torch.tensor(normal_all)

    return g


def build_mask(labels_list, mask_value: float = MISSING_LABEL_VALUE) -> List[int]:
    """
    Build mask for missing labels.

    Args:
        labels_list: List of label values
        mask_value: Value indicating missing label

    Returns:
        List of mask values (0=missing, 1=valid)
    """
    mask = []
    for label in labels_list:
        if label == mask_value or (isinstance(label, float) and np.isnan(label)):
            mask.append(0)
        else:
            mask.append(1)
    return mask


def build_dataset(
    df: pd.DataFrame,
    labels_list: List[str],
    smiles_col: str = "smiles",
    verbose: bool = True,
    verbose_mode: str = "failures",
) -> List:
    """
    Build graph dataset from DataFrame.

    Args:
        df: DataFrame with SMILES and labels
        labels_list: List of label column names
        smiles_col: Name of SMILES column
        verbose: Whether to print progress (deprecated, use verbose_mode)
        verbose_mode: Output mode - "all" (progress+failures), "failures" (failures only), "none" (silent)

    Returns:
        List of [smiles, graph, labels, mask, group] tuples
    """
    dataset = []
    failed_molecules = []
    labels = df[labels_list]
    split_index = df.get("group", pd.Series(["training"] * len(df)))
    smiles_list = df[smiles_col]
    total = len(smiles_list)

    # Reset index to ensure continuous 0-based indexing
    if not labels.index.equals(pd.RangeIndex(len(labels))):
        labels = labels.reset_index(drop=True)
    if not split_index.index.equals(pd.RangeIndex(len(split_index))):
        split_index = split_index.reset_index(drop=True)

    # Determine output mode
    show_progress = verbose and verbose_mode == "all"
    show_failures = verbose and verbose_mode in ["all", "failures"]

    for i, smiles in enumerate(smiles_list):
        try:
            g = construct_graph_from_smiles(smiles)
            mask = build_mask(labels.iloc[i], mask_value=MISSING_LABEL_VALUE)
            molecule = [smiles, g, labels.iloc[i].tolist(), mask, split_index.iloc[i]]
            dataset.append(molecule)

            if show_progress and (i + 1) % 100 == 0:
                print(f"{i + 1}/{total} molecules processed")

        except (ValueError, RuntimeError) as e:
            if show_failures:
                print(f"Failed to process {smiles}: {e}")
            failed_molecules.append(smiles)

    if show_failures and failed_molecules:
        print(f"Failed molecules ({len(failed_molecules)}): {failed_molecules[:5]}...")

    if not dataset:
        raise ValueError("No valid molecules were processed from the input data")

    return dataset


def inference_build_dataset(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
) -> List:
    """
    Build dataset for inference (no labels).

    Args:
        df: DataFrame with SMILES
        smiles_col: Name of SMILES column

    Returns:
        List of [smiles, graph] tuples
    """
    dataset = []
    failed_molecules = []
    smiles_list = df[smiles_col]

    for smiles in smiles_list:
        try:
            g = construct_graph_from_smiles(smiles)
            dataset.append([smiles, g])
        except (ValueError, RuntimeError):
            failed_molecules.append(smiles)

    return dataset


def save_graph_dataset(
    origin_path: str,
    save_path: str,
    group_path: str,
    task_list: Optional[List[str]] = None,
) -> None:
    """
    Build and save graph dataset from CSV.

    Args:
        origin_path: Path to source CSV
        save_path: Path to save binary graphs
        group_path: Path to save group information
        task_list: Optional list of task columns (uses all non-smiles/group if None)
    """
    data_origin = pd.read_csv(origin_path)
    data_origin = data_origin.fillna(MISSING_LABEL_VALUE)

    labels_list = [x for x in data_origin.columns if x not in ["smiles", "group"]]
    if task_list is not None:
        labels_list = task_list

    dataset = build_dataset(data_origin, labels_list, smiles_col="smiles")

    smiles, graphs, labels, mask, split_index = map(list, zip(*dataset))

    graph_labels = {
        "labels": torch.tensor(labels),
        "mask": torch.tensor(mask),
    }

    split_index_pd = pd.DataFrame({"smiles": smiles, "group": split_index})
    split_index_pd.to_csv(group_path, index=None)

    save_graphs(save_path, graphs, graph_labels)
    print(f"Saved {len(graphs)} graphs to {save_path}")


def load_graph_dataset(
    bin_path: str,
    group_path: str,
    select_task_index: Optional[List[int]] = None,
) -> Tuple[List, List, List, int]:
    """
    Load graph dataset with train/val/test split.

    Args:
        bin_path: Path to binary graph file
        group_path: Path to group CSV
        select_task_index: Optional indices of tasks to select

    Returns:
        Tuple of (train_set, val_set, test_set, n_tasks)
    """
    smiles = pd.read_csv(group_path, index_col=None).smiles.values
    group = pd.read_csv(group_path, index_col=None).group.to_list()
    graphs, detailed_info = load_graphs(bin_path)

    labels = detailed_info["labels"]
    mask = detailed_info["mask"]

    # Select specific tasks if requested
    if select_task_index is not None:
        if isinstance(select_task_index, int):
            select_task_index = [select_task_index]

        num_tasks = labels.shape[1]
        if any(idx >= num_tasks for idx in select_task_index):
            print(f"Warning: task index out of bounds, using [0]")
            select_task_index = [0]

        labels = labels[:, select_task_index]
        mask = mask[:, select_task_index]

    # Find samples with no valid labels
    notuse_mask = torch.mean(mask.float(), 1).numpy().tolist()
    not_use_index = [i for i, m in enumerate(notuse_mask) if m == 0]

    # Split by group
    train_index = []
    val_index = []
    test_index = []

    for i, g in enumerate(group):
        if i in not_use_index:
            continue
        if g == "training":
            train_index.append(i)
        elif g in ["valid", "val"]:
            val_index.append(i)
        elif g == "test":
            test_index.append(i)

    # Build datasets
    graphs_np = np.array(graphs)

    def build_split(indices):
        dataset = []
        for i in indices:
            molecule = [smiles[i], graphs_np[i], labels[i].numpy(), mask[i].numpy()]
            dataset.append(molecule)
        return dataset

    train_set = build_split(train_index)
    val_set = build_split(val_index)
    test_set = build_split(test_index)
    n_tasks = labels.shape[1]

    print(f"Loaded: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}, tasks={n_tasks}")

    return train_set, val_set, test_set, n_tasks
