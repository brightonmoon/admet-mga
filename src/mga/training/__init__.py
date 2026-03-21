"""Training utilities for MGA."""

from mga.training.trainer import MGATrainer
from mga.training.callbacks import EarlyStopping
from mga.training.losses import get_loss_function, compute_masked_loss
from mga.training.transfer import TransferLearningManager

__all__ = [
    "MGATrainer",
    "EarlyStopping",
    "get_loss_function",
    "compute_masked_loss",
    "TransferLearningManager",
]
