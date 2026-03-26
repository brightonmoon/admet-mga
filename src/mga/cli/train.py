"""
MGA Training CLI entry point.

Usage:
    mga-train --config config/config_admet.yaml
    mga-train --config config/config_admet.yaml --no-wandb --device cpu
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing

import torch
from torch.utils.data import DataLoader

from mga.config import MGAConfig, load_config
from mga.data import load_graph_dataset, collate_molgraphs
from mga.models import MGA
from mga.training import MGATrainer
from mga.utils import set_random_seed
from mga.utils.logging import configure_logging


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
    # Transfer learning arguments
    parser.add_argument(
        "--transfer-strategy",
        type=str,
        choices=["full_finetune", "feature_extraction", "selective_layer", "attention_transfer"],
        default=None,
        help="Transfer learning strategy",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint for transfer learning",
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        nargs="+",
        default=None,
        help="RGCN layer indices to freeze (e.g. --freeze-layers 0)",
    )
    parser.add_argument(
        "--encoder-lr-multiplier",
        type=float,
        default=None,
        help="LR multiplier for encoder relative to head (e.g. 0.1)",
    )
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logs (WARNING and above only)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write logs to file in addition to stdout",
    )
    # Data validation
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Run data validation before training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    configure_logging(level=log_level, log_file=args.log_file)

    import logging as _logging
    logger = _logging.getLogger("mga.cli.train")

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

    # Apply transfer learning CLI overrides
    if args.transfer_strategy or args.pretrained_model:
        from mga.config.config import TransferConfig
        if config.transfer is None:
            config.transfer = TransferConfig()
        if args.transfer_strategy:
            config.transfer.strategy = args.transfer_strategy
        if args.pretrained_model:
            from pathlib import Path
            config.transfer.pretrained_model_path = Path(args.pretrained_model)
        if args.freeze_layers is not None:
            config.transfer.freeze_layers = args.freeze_layers
        if args.encoder_lr_multiplier is not None:
            config.transfer.encoder_lr_multiplier = args.encoder_lr_multiplier

    # Set device
    if config.training.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.training.device = "cpu"

    # Set random seed
    set_random_seed(config.training.seed)

    logger.info("=" * 60)
    logger.info(f"MGA Training: {config.task.task_name}")
    logger.info("=" * 60)
    logger.info(f"Device:     {config.training.device}")
    logger.info(f"Epochs:     {config.training.num_epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"LR:         {config.training.lr}")
    logger.info(f"wandb:      {'enabled' if config.training.wandb.enabled else 'disabled'}")
    if config.transfer and config.transfer.pretrained_model_path:
        logger.info(f"Transfer:   {config.transfer.strategy} from {config.transfer.pretrained_model_path}")
    logger.info("=" * 60)

    # Load data
    data_bin = args.data_bin or str(config.paths.data_dir / "admet_52.bin")
    data_group = args.data_group or str(config.paths.data_dir / "admet_52_group.csv")

    logger.info(f"Loading data from {data_bin}...")
    train_set, val_set, test_set, n_tasks = load_graph_dataset(
        bin_path=data_bin,
        group_path=data_group,
        select_task_index=config.task.select_task_index,
    )

    # Update n_tasks in config
    config.model.n_tasks = n_tasks

    # Create data loaders
    num_workers = min(config.training.num_workers, multiprocessing.cpu_count() or 1)
    use_cuda = config.training.device == "cuda"

    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "collate_fn": collate_molgraphs,
        "num_workers": num_workers,
        "pin_memory": config.training.pin_memory and use_cuda,
        "persistent_workers": config.training.persistent_workers and num_workers > 0,
        "prefetch_factor": config.training.prefetch_factor if num_workers > 0 else None,
    }

    logger.info(f"DataLoader: num_workers={num_workers}, pin_memory={loader_kwargs['pin_memory']}")

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

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    # Create trainer and train
    trainer = MGATrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    history = trainer.train()

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation score: {trainer.stopper.best_score:.4f}")
    if "test_scores" in history:
        import numpy as np
        logger.info(f"Test score: {np.mean(history['test_scores']):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
