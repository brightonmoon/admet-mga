"""
MGA Data Preparation CLI entry point.

Usage:
    mga-data from-csv -i data/raw.csv -o data/processed.bin
    mga-data from-tdc -d CYP2C9_Veith -o data/cyp2c9.bin
    mga-data from-tdc --preset cyp_inhibition -o data/cyp_all.bin
    mga-data list-tdc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_from_csv(args):
    """Convert CSV file to binary graph format."""
    from mga.data import save_graph_dataset

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if args.group_output:
        group_path = Path(args.group_output)
    else:
        group_path = output_path.with_name(output_path.stem + "_group.csv")

    task_list = None
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",")]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MGA Data Preparation (from CSV)")
    print("=" * 60)
    print(f"Input CSV:    {input_path}")
    print(f"Output BIN:   {output_path}")
    print(f"Output Group: {group_path}")
    if task_list:
        print(f"Tasks:        {', '.join(task_list)}")
    else:
        print("Tasks:        All columns (except smiles, group)")
    print("=" * 60)

    save_graph_dataset(
        origin_path=str(input_path),
        save_path=str(output_path),
        group_path=str(group_path),
        task_list=task_list,
    )

    print("=" * 60)
    print("Completed!")
    print(f"  Binary: {output_path}")
    print(f"  Group:  {group_path}")
    print("=" * 60)


def cmd_from_tdc(args):
    """Fetch TDC dataset and convert to binary graph format."""
    from mga.data import save_graph_dataset
    from mga.data.tdc_fetcher import (
        fetch_tdc_dataset,
        fetch_multiple_datasets,
        fetch_preset,
        save_tdc_to_csv,
        list_presets,
        ALL_DATASETS,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.group_output:
        group_path = Path(args.group_output)
    else:
        group_path = output_path.with_name(output_path.stem + "_group.csv")

    csv_path = output_path.with_suffix(".csv")

    cache_dir = args.cache_dir
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MGA Data Preparation (from TDC)")
    print("=" * 60)

    if args.preset:
        available_presets = list_presets()
        if args.preset not in available_presets:
            print(f"Error: Unknown preset '{args.preset}'")
            print(f"Available presets: {list(available_presets.keys())}")
            sys.exit(1)

        print(f"Preset: {args.preset}")
        df, task_types = fetch_preset(
            args.preset,
            cache_dir=cache_dir,
            split_method=args.split_method,
            split_seed=args.seed,
            verbose=True,
        )
    elif args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]

        for name in dataset_names:
            if name not in ALL_DATASETS:
                print(f"Error: Unknown dataset '{name}'")
                print("Use 'mga-data list-tdc' to see available datasets")
                sys.exit(1)

        if len(dataset_names) == 1:
            print(f"Dataset: {dataset_names[0]} (single-task)")
            df, task_type = fetch_tdc_dataset(
                dataset_names[0],
                cache_dir=cache_dir,
                split_method=args.split_method,
                split_seed=args.seed,
            )
            task_types = {dataset_names[0]: task_type}
        else:
            print(f"Datasets: {', '.join(dataset_names)} (multi-task)")
            df, task_types = fetch_multiple_datasets(
                dataset_names,
                cache_dir=cache_dir,
                split_method=args.split_method,
                split_seed=args.seed,
                verbose=True,
            )
    else:
        print("Error: Specify --datasets or --preset")
        sys.exit(1)

    print("-" * 60)
    print(f"Split method: {args.split_method}")
    print(f"Split seed:   {args.seed}")
    print(f"Total molecules: {len(df)}")
    print(f"Tasks: {list(task_types.keys())}")
    print("-" * 60)

    if args.save_csv:
        save_tdc_to_csv(df, csv_path, task_types)

    df = df.fillna(123456)

    temp_csv = output_path.with_name(output_path.stem + "_temp.csv")
    df.to_csv(temp_csv, index=False)

    task_cols = [c for c in df.columns if c not in ["smiles", "group", "Drug_ID"]]

    print("\nBuilding graph dataset...")

    save_graph_dataset(
        origin_path=str(temp_csv),
        save_path=str(output_path),
        group_path=str(group_path),
        task_list=task_cols,
    )

    if not args.save_csv:
        temp_csv.unlink()
    else:
        temp_csv.rename(csv_path)

    meta_path = output_path.with_name(output_path.stem + "_meta.csv")
    import pandas as pd
    meta_df = pd.DataFrame([
        {"task": k, "type": v, "index": i}
        for i, (k, v) in enumerate(task_types.items())
    ])
    meta_df.to_csv(meta_path, index=False)

    print("=" * 60)
    print("Completed!")
    print(f"  Binary: {output_path}")
    print(f"  Group:  {group_path}")
    print(f"  Meta:   {meta_path}")
    if args.save_csv:
        print(f"  CSV:    {csv_path}")
    print("=" * 60)


def cmd_list_tdc(args):
    """List available TDC datasets."""
    from mga.data.tdc_fetcher import print_dataset_info
    print_dataset_info()


def main():
    parser = argparse.ArgumentParser(
        description="MGA Data Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # from-csv command
    csv_parser = subparsers.add_parser(
        "from-csv",
        help="Convert CSV file to binary graph format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input CSV format:
  - 'smiles' column: SMILES strings
  - 'group' column: 'training', 'valid'/'val', or 'test'
  - Other columns: task labels (NaN for missing)

Example:
  mga-data from-csv -i data/raw.csv -o data/processed.bin
  mga-data from-csv -i data/raw.csv -o data/processed.bin --tasks CYP2C9,CYP3A4
        """,
    )
    csv_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    csv_parser.add_argument("-o", "--output", required=True, help="Output binary file (.bin)")
    csv_parser.add_argument("-g", "--group-output", help="Output group CSV (default: <output>_group.csv)")
    csv_parser.add_argument("-t", "--tasks", help="Comma-separated task columns (default: all)")

    # from-tdc command
    tdc_parser = subparsers.add_parser(
        "from-tdc",
        help="Fetch TDC dataset and convert to binary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mga-data from-tdc -d CYP2C9_Veith -o data/cyp2c9.bin
  mga-data from-tdc -d CYP2C9_Veith,CYP3A4_Veith -o data/cyp.bin
  mga-data from-tdc --preset cyp_inhibition -o data/cyp_all.bin
  mga-data from-tdc -d BBB_Martins -o data/bbb.bin --split scaffold

Available presets:
  cyp_inhibition, cyp_substrate, absorption, distribution,
  excretion, toxicity, all_classification, all_regression
        """,
    )
    tdc_parser.add_argument("-d", "--datasets", help="Comma-separated dataset names")
    tdc_parser.add_argument("--preset", help="Use a preset group of datasets")
    tdc_parser.add_argument("-o", "--output", required=True, help="Output binary file (.bin)")
    tdc_parser.add_argument("-g", "--group-output", help="Output group CSV")
    tdc_parser.add_argument("--cache-dir", help="TDC cache directory")
    tdc_parser.add_argument("--split-method", default="scaffold",
                           choices=["scaffold", "random", "cold_drug"],
                           help="Split method (default: scaffold)")
    tdc_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    tdc_parser.add_argument("--save-csv", action="store_true",
                           help="Also save intermediate CSV file")

    # list-tdc command
    subparsers.add_parser(
        "list-tdc",
        help="List available TDC ADMET datasets",
    )

    args = parser.parse_args()

    if args.command == "from-csv":
        cmd_from_csv(args)
    elif args.command == "from-tdc":
        cmd_from_tdc(args)
    elif args.command == "list-tdc":
        cmd_list_tdc(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
