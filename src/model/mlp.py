"""Feed-forward network used inside each transformer block."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")
        if expansion_factor < 1:
            raise ValueError("expansion_factor must be at least 1.")

        hidden_dim = embedding_dim * expansion_factor
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_states)
