"""Data processing utilities for MGA."""

from mga.data.features import atom_features, bond_features, etype_features
from mga.data.dataset import (
    construct_graph_from_smiles,
    build_dataset,
    load_graph_dataset,
    save_graph_dataset,
    inference_build_dataset,
    MISSING_LABEL_VALUE,
)
from mga.data.collate import collate_molgraphs, collate_molgraphs_inference
from mga.data.tdc_fetcher import (
    fetch_tdc_dataset,
    fetch_multiple_datasets,
    fetch_preset,
    list_available_datasets,
    list_presets,
    save_tdc_to_csv,
    ALL_DATASETS,
    DATASET_PRESETS,
)

__all__ = [
    # Features
    "atom_features",
    "bond_features",
    "etype_features",
    # Dataset
    "construct_graph_from_smiles",
    "build_dataset",
    "load_graph_dataset",
    "save_graph_dataset",
    "inference_build_dataset",
    "MISSING_LABEL_VALUE",
    # Collate
    "collate_molgraphs",
    "collate_molgraphs_inference",
    # TDC Fetcher
    "fetch_tdc_dataset",
    "fetch_multiple_datasets",
    "fetch_preset",
    "list_available_datasets",
    "list_presets",
    "save_tdc_to_csv",
    "ALL_DATASETS",
    "DATASET_PRESETS",
]
