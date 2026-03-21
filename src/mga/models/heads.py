"""
Prediction heads for MGA models.

This module contains task-specific prediction layers:
- MLPClassifier: Multi-layer perceptron for classification/regression
"""

from __future__ import annotations

import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier/regressor.

    A simple MLP head with dropout, ReLU activation, and batch normalization.

    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden layer dimension
        n_tasks: Number of output tasks/classes
        dropout: Dropout rate

    Example:
        >>> head = MLPClassifier(64, 128, n_tasks=1, dropout=0.2)
        >>> predictions = head(molecule_features)
    """

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        n_tasks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks),
        )

    def forward(self, h):
        """
        Forward pass.

        Args:
            h: Input features [B, in_feats]

        Returns:
            predictions: Output predictions [B, n_tasks]
        """
        return self.predict(h)


def create_fc_layer(dropout: float, in_feats: int, hidden_feats: int) -> nn.Sequential:
    """
    Create a fully connected layer with dropout, ReLU, and batch normalization.

    Args:
        dropout: Dropout rate
        in_feats: Input feature dimension
        hidden_feats: Output feature dimension

    Returns:
        nn.Sequential module
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feats, hidden_feats),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_feats),
    )


def create_output_layer(hidden_feats: int, out_feats: int) -> nn.Sequential:
    """
    Create an output layer (simple linear transformation).

    Args:
        hidden_feats: Input feature dimension
        out_feats: Output feature dimension

    Returns:
        nn.Sequential module
    """
    return nn.Sequential(
        nn.Linear(hidden_feats, out_feats),
    )
