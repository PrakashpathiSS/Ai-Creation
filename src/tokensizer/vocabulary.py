"""Vocabulary helpers for the tokenizer package."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .constants import (
    BYTE_TOKEN_PATTERN,
    DIGITS,
    LOWERCASE_LETTERS,
    SPECIAL_CHARACTERS,
    SPECIAL_TOKENS,
    SKIPPED_SPECIAL_TOKENS,
    UNK_TOKEN,
    UPPERCASE_LETTERS,
)


def base_vocabulary_tokens() -> list[str]:
    """Return the default seed vocabulary."""
    return [
        *LOWERCASE_LETTERS,
        *UPPERCASE_LETTERS,
        *DIGITS,
        *SPECIAL_CHARACTERS,
    ]


def validate_vocabulary(
    vocabulary: Mapping[str, int],
    *,
    required_special_tokens: Sequence[str] = SPECIAL_TOKENS,
    require_contiguous_ids: bool = True,
) -> None:
    """Validate that a vocabulary can be safely used for encoding and decoding."""
    if not isinstance(vocabulary, Mapping):
        raise TypeError("Vocabulary must be a mapping of token strings to integer IDs.")

    ids: set[int] = set()
    for token, token_id in vocabulary.items():
        if not isinstance(token, str):
            raise TypeError("Vocabulary tokens must be strings.")
        if not isinstance(token_id, int):
            raise TypeError("Vocabulary IDs must be integers.")
        if token_id < 0:
            raise ValueError("Vocabulary IDs must be non-negative.")
        if token_id in ids:
            raise ValueError("Vocabulary IDs must be unique.")
        ids.add(token_id)

    missing_special_tokens = [token for token in required_special_tokens if token not in vocabulary]
    if missing_special_tokens:
        raise ValueError(
            "Vocabulary is missing required special tokens: "
            + ", ".join(repr(token) for token in missing_special_tokens)
        )

    if require_contiguous_ids:
        sorted_ids = sorted(ids)
        expected_ids = list(range(len(sorted_ids)))
        if sorted_ids != expected_ids:
            raise ValueError("Vocabulary IDs must form a contiguous zero-based range.")


def build_vocabulary(
    tokens: list[str],
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
) -> dict[str, int]:
    """Create a token-to-id vocabulary."""
    vocabulary: dict[str, int] = {}

    for token in special_tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    validate_vocabulary(vocabulary, required_special_tokens=special_tokens)
    return vocabulary


def extend_vocabulary(tokens: list[str], vocabulary: dict[str, int]) -> dict[str, int]:
    """Add new tokens to an existing vocabulary without changing old IDs."""
    validate_vocabulary(vocabulary)
    next_id = max(vocabulary.values(), default=-1) + 1

    for token in (*SPECIAL_TOKENS, *tokens):
        if token not in vocabulary:
            vocabulary[token] = next_id
            next_id += 1

    validate_vocabulary(vocabulary)
    return vocabulary


def save_vocabulary(vocabulary: dict[str, int], file_path: str | Path) -> None:
    """Save a vocabulary as JSON."""
    path = Path(file_path)
    validate_vocabulary(vocabulary)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_vocabulary = dict(sorted(vocabulary.items(), key=lambda item: item[1]))
    path.write_text(
        json.dumps(ordered_vocabulary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_vocabulary(file_path: str | Path) -> dict[str, int]:
    """Load a vocabulary from JSON."""
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Vocabulary file must contain a JSON object.")

    vocabulary: dict[str, int] = {}
    for token, token_id in data.items():
        if not isinstance(token_id, int):
            raise TypeError("Vocabulary IDs must be integers.")
        vocabulary[str(token)] = token_id

    validate_vocabulary(vocabulary)
    return vocabulary


def tokens_to_ids(
    tokens: list[str],
    vocabulary: Mapping[str, int],
    unknown_token: str = UNK_TOKEN,
) -> list[int]:
    """Convert tokens into numbers using a vocabulary."""
    validate_vocabulary(vocabulary)
    if unknown_token not in vocabulary:
        raise ValueError(f"Vocabulary is missing the unknown token {unknown_token!r}.")

    unknown_id = vocabulary[unknown_token]
    return [vocabulary.get(token, unknown_id) for token in tokens]


def ids_to_tokens(
    token_ids: list[int],
    vocabulary: Mapping[str, int],
    unknown_token: str = UNK_TOKEN,
) -> list[str]:
    """Convert token IDs back into tokens."""
    validate_vocabulary(vocabulary)
    id_to_token = {token_id: token for token, token_id in vocabulary.items()}
    return [id_to_token.get(token_id, unknown_token) for token_id in token_ids]


def detokenize(tokens: list[str]) -> str:
    """Join tokens into text and clean spacing around punctuation."""
    filtered_tokens = [token for token in tokens if token not in SKIPPED_SPECIAL_TOKENS]
    if not filtered_tokens:
        return ""

    byte_matches = [BYTE_TOKEN_PATTERN.fullmatch(token) for token in filtered_tokens]
    if all(byte_matches):
        byte_values = bytes(int(match.group(1), 16) for match in byte_matches if match is not None)
        return byte_values.decode("utf-8", errors="replace")

    text = " ".join(filtered_tokens)
    text = text.replace(" ##", "")
    text = text.replace("▁", " ")
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r'([\(\[\{"\'])\s+', r"\1", text)
    text = re.sub(r"\s+([)\]\}])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


__all__ = [
    "base_vocabulary_tokens",
    "validate_vocabulary",
    "build_vocabulary",
    "extend_vocabulary",
    "save_vocabulary",
    "load_vocabulary",
    "tokens_to_ids",
    "ids_to_tokens",
    "detokenize",
]
