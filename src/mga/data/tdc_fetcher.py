"""
PyTDC Data Fetcher for MGA.

This module provides utilities to fetch ADMET datasets from
Therapeutics Data Commons (TDC) and prepare them for MGA training.

Reference: https://tdcommons.ai/start/
GitHub: https://github.com/mims-harvard/TDC
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# Available ADMET datasets in TDC
# Format: dataset_name -> (category, tdc_name, task_type)
ADME_DATASETS = {
    # Absorption
    "Caco2_Wang": ("ADME", "Caco2_Wang", "regression"),
    "HIA_Hou": ("ADME", "HIA_Hou", "classification"),
    "Pgp_Broccatelli": ("ADME", "Pgp_Broccatelli", "classification"),
    "Bioavailability_Ma": ("ADME", "Bioavailability_Ma", "classification"),
    # Distribution
    "Lipophilicity_AstraZeneca": ("ADME", "Lipophilicity_AstraZeneca", "regression"),
    "Solubility_AqSolDB": ("ADME", "Solubility_AqSolDB", "regression"),
    "HydrationFreeEnergy_FreeSolv": ("ADME", "HydrationFreeEnergy_FreeSolv", "regression"),
    "BBB_Martins": ("ADME", "BBB_Martins", "classification"),
    "PPBR_AZ": ("ADME", "PPBR_AZ", "regression"),
    "VDss_Lombardo": ("ADME", "VDss_Lombardo", "regression"),
    # Metabolism (CYP Inhibition)
    "CYP1A2_Veith": ("ADME", "CYP1A2_Veith", "classification"),
    "CYP2C9_Veith": ("ADME", "CYP2C9_Veith", "classification"),
    "CYP2C19_Veith": ("ADME", "CYP2C19_Veith", "classification"),
    "CYP2D6_Veith": ("ADME", "CYP2D6_Veith", "classification"),
    "CYP3A4_Veith": ("ADME", "CYP3A4_Veith", "classification"),
    # Metabolism (CYP Substrate)
    "CYP2C9_Substrate_CarbonMangels": ("ADME", "CYP2C9_Substrate_CarbonMangels", "classification"),
    "CYP2D6_Substrate_CarbonMangels": ("ADME", "CYP2D6_Substrate_CarbonMangels", "classification"),
    "CYP3A4_Substrate_CarbonMangels": ("ADME", "CYP3A4_Substrate_CarbonMangels", "classification"),
    # Excretion
    "Half_Life_Obach": ("ADME", "Half_Life_Obach", "regression"),
    "Clearance_Hepatocyte_AZ": ("ADME", "Clearance_Hepatocyte_AZ", "regression"),
    "Clearance_Microsome_AZ": ("ADME", "Clearance_Microsome_AZ", "regression"),
}

TOX_DATASETS = {
    "hERG": ("Tox", "hERG", "classification"),
    "AMES": ("Tox", "AMES", "classification"),
    "DILI": ("Tox", "DILI", "classification"),
    "Skin_Reaction": ("Tox", "Skin_Reaction", "classification"),
    "Carcinogens_Lagunin": ("Tox", "Carcinogens_Lagunin", "classification"),
    "LD50_Zhu": ("Tox", "LD50_Zhu", "regression"),
    "ClinTox": ("Tox", "ClinTox", "classification"),
}

ALL_DATASETS = {**ADME_DATASETS, **TOX_DATASETS}

# Preset groups for convenience
DATASET_PRESETS = {
    "cyp_inhibition": [
        "CYP1A2_Veith", "CYP2C9_Veith", "CYP2C19_Veith",
        "CYP2D6_Veith", "CYP3A4_Veith",
    ],
    "cyp_substrate": [
        "CYP2C9_Substrate_CarbonMangels",
        "CYP2D6_Substrate_CarbonMangels",
        "CYP3A4_Substrate_CarbonMangels",
    ],
    "absorption": [
        "Caco2_Wang", "HIA_Hou", "Pgp_Broccatelli", "Bioavailability_Ma",
    ],
    "distribution": [
        "Lipophilicity_AstraZeneca", "Solubility_AqSolDB", "BBB_Martins",
        "PPBR_AZ", "VDss_Lombardo",
    ],
    "excretion": [
        "Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ",
    ],
    "toxicity": list(TOX_DATASETS.keys()),
    "all_classification": [
        name for name, (_, _, task_type) in ALL_DATASETS.items()
        if task_type == "classification"
    ],
    "all_regression": [
        name for name, (_, _, task_type) in ALL_DATASETS.items()
        if task_type == "regression"
    ],
}


def list_available_datasets() -> Dict[str, str]:
    """
    List all available TDC ADMET datasets with their task types.

    Returns:
        Dict mapping dataset name to task type ('classification' or 'regression')
    """
    return {name: info[2] for name, info in ALL_DATASETS.items()}


def list_presets() -> Dict[str, List[str]]:
    """
    List available dataset presets.

    Returns:
        Dict mapping preset name to list of dataset names
    """
    return DATASET_PRESETS


def fetch_tdc_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = None,
    split_method: str = "scaffold",
    split_seed: int = 42,
    split_frac: List[float] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch a single TDC dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'CYP2C9_Veith')
        cache_dir: Directory to cache downloaded data
        split_method: Split method ('scaffold', 'random', 'cold_drug')
        split_seed: Random seed for splitting
        split_frac: Split fractions [train, valid, test], default [0.7, 0.1, 0.2]

    Returns:
        Tuple of (DataFrame with smiles/label/group, task_type)

    Example:
        >>> df, task_type = fetch_tdc_dataset('CYP2C9_Veith')
        >>> print(df.columns)  # ['smiles', 'CYP2C9_Veith', 'group']
    """
    try:
        from tdc.single_pred import ADME, Tox
    except ImportError:
        raise ImportError(
            "PyTDC is required. Install with: pip install PyTDC"
        )

    if dataset_name not in ALL_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(ALL_DATASETS.keys())}"
        )

    if split_frac is None:
        split_frac = [0.7, 0.1, 0.2]

    category, tdc_name, task_type = ALL_DATASETS[dataset_name]

    # Fetch data from TDC
    if category == "ADME":
        data = ADME(name=tdc_name, path=cache_dir)
    else:  # Tox
        data = Tox(name=tdc_name, path=cache_dir)

    # Get split data
    # TDC returns: {'train': df, 'valid': df, 'test': df}
    split = data.get_split(
        method=split_method,
        seed=split_seed,
        frac=split_frac,
    )

    # Combine splits with group labels
    dfs = []
    group_mapping = {
        "train": "training",
        "valid": "valid",
        "test": "test",
    }

    for split_name, group_name in group_mapping.items():
        split_df = split[split_name].copy()
        split_df["group"] = group_name
        dfs.append(split_df)

    df = pd.concat(dfs, ignore_index=True)

    # Rename columns to MGA standard format
    # TDC format: Drug_ID, Drug, Y
    df = df.rename(columns={
        "Drug": "smiles",
        "Y": dataset_name,
    })

    # Keep only needed columns
    keep_cols = ["smiles", dataset_name, "group"]
    if "Drug_ID" in df.columns:
        keep_cols = ["Drug_ID"] + keep_cols
    df = df[[c for c in keep_cols if c in df.columns]]

    return df, task_type


def fetch_multiple_datasets(
    dataset_names: List[str],
    cache_dir: Optional[str] = None,
    split_method: str = "scaffold",
    split_seed: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch and merge multiple TDC datasets for multi-task learning.

    Molecules are merged by SMILES. Missing labels are marked as NaN.

    Args:
        dataset_names: List of dataset names
        cache_dir: Directory to cache downloaded data
        split_method: Split method for all datasets
        split_seed: Random seed for splitting
        verbose: Print progress

    Returns:
        Tuple of (merged DataFrame, dict mapping task name to task type)

    Example:
        >>> df, task_types = fetch_multiple_datasets(['CYP2C9_Veith', 'CYP3A4_Veith'])
        >>> print(df.columns)  # ['smiles', 'CYP2C9_Veith', 'CYP3A4_Veith', 'group']
    """
    if not dataset_names:
        raise ValueError("At least one dataset name is required")

    task_types = {}
    all_dfs = []

    for i, name in enumerate(dataset_names):
        if verbose:
            print(f"[{i+1}/{len(dataset_names)}] Fetching {name}...")

        df, task_type = fetch_tdc_dataset(
            name,
            cache_dir=cache_dir,
            split_method=split_method,
            split_seed=split_seed,
        )
        task_types[name] = task_type
        all_dfs.append(df)

    # Single dataset case
    if len(all_dfs) == 1:
        return all_dfs[0], task_types

    # Merge multiple datasets on SMILES
    merged = all_dfs[0][["smiles", dataset_names[0], "group"]].copy()

    for df, name in zip(all_dfs[1:], dataset_names[1:]):
        # Merge on smiles, keeping all molecules (outer join)
        task_df = df[["smiles", name]].copy()
        merged = pd.merge(
            merged,
            task_df,
            on="smiles",
            how="outer",
        )

    # Fill missing group values
    # For molecules that appear in multiple datasets with different groups,
    # prioritize: test > valid > training
    group_priority = {"test": 3, "valid": 2, "training": 1}

    for df in all_dfs[1:]:
        for idx, row in df.iterrows():
            smiles = row["smiles"]
            new_group = row["group"]

            mask = merged["smiles"] == smiles
            if mask.any():
                current_group = merged.loc[mask, "group"].iloc[0]
                if pd.isna(current_group):
                    merged.loc[mask, "group"] = new_group
                elif group_priority.get(new_group, 0) > group_priority.get(current_group, 0):
                    merged.loc[mask, "group"] = new_group

    # Fill any remaining NaN groups with 'training'
    merged["group"] = merged["group"].fillna("training")

    # Reorder columns: smiles, tasks, group
    task_cols = [c for c in merged.columns if c not in ["smiles", "group"]]
    merged = merged[["smiles"] + task_cols + ["group"]]

    if verbose:
        print(f"Merged dataset: {len(merged)} molecules, {len(task_cols)} tasks")
        for task in task_cols:
            valid_count = merged[task].notna().sum()
            print(f"  - {task}: {valid_count} valid labels ({task_types[task]})")

    return merged, task_types


def fetch_preset(
    preset_name: str,
    cache_dir: Optional[str] = None,
    split_method: str = "scaffold",
    split_seed: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch a preset group of datasets.

    Args:
        preset_name: Name of the preset (e.g., 'cyp_inhibition')
        cache_dir: Directory to cache downloaded data
        split_method: Split method
        split_seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (merged DataFrame, dict of task_types)

    Available presets:
        - cyp_inhibition: CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4 inhibition
        - cyp_substrate: CYP2C9, CYP2D6, CYP3A4 substrate
        - absorption: Caco2, HIA, Pgp, Bioavailability
        - distribution: Lipophilicity, Solubility, BBB, PPBR, VDss
        - excretion: Half-life, Hepatocyte clearance, Microsome clearance
        - toxicity: hERG, AMES, DILI, etc.
        - all_classification: All classification tasks
        - all_regression: All regression tasks
    """
    if preset_name not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(DATASET_PRESETS.keys())}"
        )

    dataset_names = DATASET_PRESETS[preset_name]
    if verbose:
        print(f"Preset '{preset_name}' contains {len(dataset_names)} datasets:")
        for name in dataset_names:
            print(f"  - {name}")

    return fetch_multiple_datasets(
        dataset_names,
        cache_dir=cache_dir,
        split_method=split_method,
        split_seed=split_seed,
        verbose=verbose,
    )


def save_tdc_to_csv(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    task_types: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save TDC data to CSV format compatible with MGA prepare_data.py.

    Args:
        df: DataFrame with smiles, task columns, and group
        output_path: Output CSV path
        task_types: Optional dict of task types for metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} molecules to {output_path}")

    # Save task types metadata
    if task_types:
        meta_path = output_path.with_suffix(".meta.csv")
        meta_df = pd.DataFrame([
            {"task": k, "type": v} for k, v in task_types.items()
        ])
        meta_df.to_csv(meta_path, index=False)
        print(f"Saved task metadata to {meta_path}")


def print_dataset_info():
    """Print information about all available datasets."""
    print("=" * 70)
    print("Available TDC ADMET Datasets")
    print("=" * 70)

    print("\n[ADME Datasets]")
    for name, (_, _, task_type) in ADME_DATASETS.items():
        print(f"  {name:40s} ({task_type})")

    print("\n[Toxicity Datasets]")
    for name, (_, _, task_type) in TOX_DATASETS.items():
        print(f"  {name:40s} ({task_type})")

    print("\n[Presets]")
    for preset_name, datasets in DATASET_PRESETS.items():
        print(f"  {preset_name:20s}: {len(datasets)} datasets")

    print("=" * 70)
