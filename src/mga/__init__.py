"""
MGA (Molecular Graph Attention) - ADMET Property Prediction

A PyTorch-based framework for molecular property prediction using
graph neural networks with multi-task learning capabilities.
"""

__version__ = "0.2.0"
__author__ = "PharosiBio"

# DGL/PyTorch 의존 모듈은 환경에 따라 없을 수 있으므로 조건부 import
try:
    from mga.models import MGA, MGATest, RGCNLayer, WeightAndSum
    from mga.config import ModelConfig, TrainingConfig, TaskConfig
    from mga.inference import ADMETPredictor
    __all__ = [
        "MGA",
        "MGATest",
        "RGCNLayer",
        "WeightAndSum",
        "ModelConfig",
        "TrainingConfig",
        "TaskConfig",
        "ADMETPredictor",
    ]
except ImportError:
    __all__ = []
