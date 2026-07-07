"""Causal self-attention for a small GPT model."""

from __future__ import annotations

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if context_length < 1:
            raise ValueError("context_length must be at least 1.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv_projection = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, context_length, context_length))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = hidden_states.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got {embedding_dim}."
            )

        query, key, value = self.qkv_projection(hidden_states).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(
            ~causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            attention_scores = attention_scores.masked_fill(
                ~key_mask,
                torch.finfo(attention_scores.dtype).min,
            )

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        context = attention_weights @ value
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, sequence_length, embedding_dim)
        output = self.output_projection(context)
        output = self.output_dropout(output)

        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(dtype=output.dtype)

        return output

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)
