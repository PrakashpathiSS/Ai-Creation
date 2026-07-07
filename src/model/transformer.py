"""Transformer block for the GPT model."""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .mlp import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention followed by feed-forward MLP."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = FeedForward(embedding_dim=embedding_dim, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states
