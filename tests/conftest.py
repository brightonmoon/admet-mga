"""
Shared pytest fixtures for MGA test suite.

All fixtures use CPU and minimal model dimensions for fast execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch


# ── Config fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mini_config():
    """Minimal MGAConfig for CPU testing (n_tasks=2, 2 epochs)."""
    from mga.config import MGAConfig
    from mga.config.config import ModelConfig, TrainingConfig, TaskConfig, PathConfig

    return MGAConfig(
        model=ModelConfig(
            in_feats=40,
            rgcn_hidden_feats=[16, 16],
            classifier_hidden_feats=16,
            n_tasks=2,
            rgcn_drop_out=0.0,
            drop_out=0.0,
        ),
        training=TrainingConfig(
            num_epochs=2,
            batch_size=4,
            lr=0.001,
            patience=10,
            device="cpu",
            seed=42,
        ),
        task=TaskConfig(
            task_name="test-task",
            task_class="classification",
            classification_list=["task_0", "task_1"],
            regression_list=[],
        ),
        paths=PathConfig(
            data_dir=Path("data"),
            model_dir=Path("models"),
        ),
    )


@pytest.fixture
def mini_config_regression():
    """MGAConfig for regression task testing."""
    from mga.config import MGAConfig
    from mga.config.config import ModelConfig, TrainingConfig, TaskConfig, PathConfig

    return MGAConfig(
        model=ModelConfig(
            in_feats=40,
            rgcn_hidden_feats=[16, 16],
            classifier_hidden_feats=16,
            n_tasks=2,
            rgcn_drop_out=0.0,
            drop_out=0.0,
        ),
        training=TrainingConfig(
            num_epochs=2,
            batch_size=4,
            lr=0.001,
            patience=10,
            device="cpu",
            seed=42,
        ),
        task=TaskConfig(
            task_name="test-regression",
            task_class="regression",
            classification_list=[],
            regression_list=["task_0", "task_1"],
        ),
        paths=PathConfig(),
    )


# ── Model fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mini_model():
    """Small MGA model for CPU testing (2 tasks, hidden=[16,16])."""
    from mga.models import MGA
    return MGA(
        in_feats=40,
        rgcn_hidden_feats=[16, 16],
        n_tasks=2,
        classifier_hidden_feats=16,
        loop=True,
        rgcn_drop_out=0.0,
        dropout=0.0,
    )


@pytest.fixture
def mini_model_test():
    """Small MGATest model (fixed 2-layer RGCN, 2 tasks)."""
    from mga.models.mga import MGATest
    return MGATest(
        in_feats=40,
        rgcn_hidden_feats=[16, 16],
        n_tasks=2,
        classifier_hidden_feats=16,
        loop=True,
        rgcn_drop_out=0.0,
        dropout=0.0,
    )


# ── Graph / data fixtures ────────────────────────────────────────────────────

@pytest.fixture
def mini_graph():
    """Single DGL graph for ethanol (CCO) for quick graph tests."""
    from mga.data.dataset import construct_graph_from_smiles
    return construct_graph_from_smiles("CCO")


_TEST_SMILES: List[str] = [
    "CCO",            # ethanol
    "CC(=O)O",        # acetic acid
    "c1ccccc1",       # benzene
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
    "C1CCCCC1",       # cyclohexane
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccc2c(c1)ccc3cccc4ccc5cccc1ccccc12345",  # coronene (large)
    "O=C(O)c1ccccc1",  # benzoic acid
    "CC(N)C(=O)O",    # alanine
]


@pytest.fixture
def mini_dataset(tmp_path):
    """10-molecule dataset with 2 classification tasks (some labels missing)."""
    import pandas as pd
    from mga.data.dataset import build_dataset

    rng = np.random.default_rng(42)
    n = len(_TEST_SMILES)
    labels_0 = rng.integers(0, 2, n).tolist()
    labels_1 = rng.integers(0, 2, n).tolist()
    # Make a few labels missing
    labels_0[2] = 123456
    labels_1[5] = 123456

    df = pd.DataFrame({
        "smiles": _TEST_SMILES,
        "task_0": labels_0,
        "task_1": labels_1,
        "group": ["training"] * 7 + ["valid"] * 2 + ["test"] * 1,
    })

    return build_dataset(df, task_list=["task_0", "task_1"])


@pytest.fixture
def mini_loaders(mini_dataset):
    """Train/val/test DataLoaders from mini_dataset (batch_size=4).

    build_dataset returns 5-element items: (smiles, graph, labels, mask, split_index).
    collate_molgraphs expects 4-element items, so we strip split_index here.
    """
    from torch.utils.data import DataLoader
    from mga.data.collate import collate_molgraphs

    # strip split_index (5th element) to match collate_molgraphs expectation
    train_set = [(m[0], m[1], m[2], m[3]) for m in mini_dataset if m[4] == "training"]
    val_set   = [(m[0], m[1], m[2], m[3]) for m in mini_dataset if m[4] in ("valid", "val")]
    test_set  = [(m[0], m[1], m[2], m[3]) for m in mini_dataset if m[4] == "test"]

    kwargs = {"batch_size": 4, "collate_fn": collate_molgraphs, "shuffle": False}
    return (
        DataLoader(train_set, **kwargs),
        DataLoader(val_set, **kwargs),
        DataLoader(test_set, **kwargs),
    )


# ── Checkpoint fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def tmp_checkpoint(tmp_path, mini_model):
    """Save mini_model to a temp path and return the path."""
    ckpt_path = tmp_path / "test_model.pth"
    torch.save({"model_state_dict": mini_model.state_dict()}, ckpt_path)
    return ckpt_path
