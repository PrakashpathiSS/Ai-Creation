"""Token embedding layer for the GPT model."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Convert token IDs into dense vectors."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)
