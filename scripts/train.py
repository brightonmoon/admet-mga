#!/usr/bin/env python3
"""
MGA Training Script

Unified training script for MGA models with wandb integration.

Usage:
    python src/scripts/train.py --config config/config_admet.yaml
    python src/scripts/train.py --config config/config_admet.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from mga.config import MGAConfig, load_config
from mga.data import load_graph_dataset, collate_molgraphs
from mga.models import MGA
from mga.training import MGATrainer
from mga.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MGA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data-bin",
        type=str,
        default=None,
        help="Path to binary graph data (overrides config)",
    )
    parser.add_argument(
        "--data-group",
        type=str,
        default=None,
        help="Path to group CSV (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI arguments
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.lr = args.lr
    if args.no_wandb:
        config.training.wandb.enabled = False
    if args.seed:
        config.training.seed = args.seed
    if args.device:
        config.training.device = args.device

    # Set device
    if config.training.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.training.device = "cpu"

    # Set random seed
    set_random_seed(config.training.seed)

    print("=" * 60)
    print(f"MGA Training: {config.task.task_name}")
    print("=" * 60)
    print(f"Device: {config.training.device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.lr}")
    print(f"wandb: {'enabled' if config.training.wandb.enabled else 'disabled'}")
    print("=" * 60)

    # Load data
    data_bin = args.data_bin or str(config.paths.data_dir / "admet_52.bin")
    data_group = args.data_group or str(config.paths.data_dir / "admet_52_group.csv")

    print(f"Loading data from {data_bin}...")
    train_set, val_set, test_set, n_tasks = load_graph_dataset(
        bin_path=data_bin,
        group_path=data_group,
        select_task_index=config.task.select_task_index,
    )

    # Update n_tasks in config
    config.model.n_tasks = n_tasks

    # Create data loaders with optimized settings from config
    num_workers = min(config.training.num_workers, multiprocessing.cpu_count() or 1)
    use_cuda = config.training.device == "cuda"

    # Common DataLoader kwargs for resource optimization
    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "collate_fn": collate_molgraphs,
        "num_workers": num_workers,
        "pin_memory": config.training.pin_memory and use_cuda,  # Only pin memory for GPU
        "persistent_workers": config.training.persistent_workers and num_workers > 0,
        "prefetch_factor": config.training.prefetch_factor if num_workers > 0 else None,
    }

    print(f"DataLoader: num_workers={num_workers}, pin_memory={loader_kwargs['pin_memory']}")

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    # Create model
    model = MGA(
        in_feats=config.model.in_feats,
        rgcn_hidden_feats=config.model.rgcn_hidden_feats,
        n_tasks=config.model.n_tasks,
        classifier_hidden_feats=config.model.classifier_hidden_feats,
        loop=config.model.loop,
        rgcn_drop_out=config.model.rgcn_drop_out,
        dropout=config.model.drop_out,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer and train
    trainer = MGATrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    history = trainer.train()

    print("=" * 60)
    print("Training completed!")
    print(f"Best validation score: {trainer.stopper.best_score:.4f}")
    if "test_scores" in history:
        import numpy as np
        print(f"Test score: {np.mean(history['test_scores']):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
