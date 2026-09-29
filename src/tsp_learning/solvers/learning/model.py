"""A compact neural edge scorer."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from tsp_learning.solvers.learning.features import EDGE_FEATURE_DIM


class EdgeScoringModel(nn.Module):
    """Score whether an undirected edge is useful in a high-quality tour."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized edge logit per feature row."""

        return cast(torch.Tensor, self.network(features).squeeze(-1))
