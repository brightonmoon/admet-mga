"""Model definitions for MGA."""

from mga.models.layers import RGCNLayer, WeightAndSum
from mga.models.mga import MGA, MGATest, BaseGNN
from mga.models.heads import MLPClassifier

__all__ = [
    "RGCNLayer",
    "WeightAndSum",
    "MGA",
    "MGATest",
    "BaseGNN",
    "MLPClassifier",
]
