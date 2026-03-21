"""
MGA (Molecular Graph Attention) - ADMET Property Prediction

A PyTorch-based framework for molecular property prediction using
graph neural networks with multi-task learning capabilities.
"""

__version__ = "0.1.0"
__author__ = "PharosiBio"

from mga.models import MGA, MGATest, RGCNLayer, WeightAndSum
from mga.config import ModelConfig, TrainingConfig, TaskConfig

__all__ = [
    "MGA",
    "MGATest",
    "RGCNLayer",
    "WeightAndSum",
    "ModelConfig",
    "TrainingConfig",
    "TaskConfig",
]
