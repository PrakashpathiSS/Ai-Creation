"""Small GPT-style language model for next-token prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .embeddings import TokenEmbedding
from .positional import PositionalEmbedding
from .transformer import TransformerBlock


@dataclass(frozen=True)
class GPTConfig:
    """Configuration for the small GPT model."""

    vocab_size: int
    context_length: int = 128
    embedding_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    pad_token_id: int = 0
    label_ignore_index: int = -100


class GPTLanguageModel(nn.Module):
    """A compact GPT model that predicts the next token at every position."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.vocab_size < 1:
            raise ValueError("vocab_size must be at least 1.")
        if config.context_length < 1:
            raise ValueError("context_length must be at least 1.")
        if config.embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")
        if config.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if config.num_heads < 1:
            raise ValueError("num_heads must be at least 1.")
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.config = config
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = PositionalEmbedding(
            context_length=config.context_length,
            embedding_dim=config.embedding_dim,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    context_length=config.context_length,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Input sequence length {sequence_length} is larger than "
                f"context_length={self.config.context_length}."
            )

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(
            sequence_length=sequence_length,
            device=input_ids.device,
        )
        hidden_states = self.dropout(token_embeddings + position_embeddings)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, self.config.vocab_size),
                labels.reshape(batch_size * sequence_length),
                ignore_index=self.config.label_ignore_index,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate token IDs autoregressively from a prompt."""
        if max_new_tokens < 1:
            return input_ids
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.context_length :]
            output = self(context)
            next_token_logits = output["logits"][:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, k=top_k)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < values[:, [-1]],
                    torch.finfo(next_token_logits.dtype).min,
                )

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return generated

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()


def load_vocab_size(tokenizer_model_path: str | Path) -> int:
    """Read vocabulary size from the saved tokenizer model."""
    path = Path(tokenizer_model_path)
    data: Any = json.loads(path.read_text(encoding="utf-8"))

    vocabulary = data.get("vocabulary") if isinstance(data, dict) else None
    if not isinstance(vocabulary, dict):
        raise ValueError(f"Tokenizer model is missing a vocabulary object: {path}")

    return len(vocabulary)
