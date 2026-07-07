"""Positional embedding layer for the GPT model."""

from __future__ import annotations

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Learn one position vector for each token position in the context window."""

    def __init__(self, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        if context_length < 1:
            raise ValueError("context_length must be at least 1.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")

        self.context_length = context_length
        self.embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence_length={sequence_length} is larger than "
                f"context_length={self.context_length}."
            )

        positions = torch.arange(sequence_length, device=device)
        return self.embedding(positions).unsqueeze(0)
