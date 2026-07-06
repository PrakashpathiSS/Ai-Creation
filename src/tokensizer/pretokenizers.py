"""Basic text pre-tokenizers used by the tokenizer package."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .constants import UNK_TOKEN


def character_tokenizer(text: str) -> list[str]:
    """Split text into single characters."""
    return list(text)


def whitespace_tokenizer(text: str) -> list[str]:
    """Split text wherever there is whitespace."""
    return text.split()


def word_tokenizer(text: str) -> list[str]:
    """Split text into words and punctuation marks."""
    return re.findall(r"\w+|[^\w\s]", text)


def sentence_tokenizer(text: str) -> list[str]:
    """Split text into sentences using common ending punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def regex_tokenizer(text: str, pattern: str) -> list[str]:
    """Split text using a custom regular expression pattern."""
    return re.findall(pattern, text)


def byte_level_tokenizer(text: str) -> list[str]:
    """Split text into UTF-8 byte tokens."""
    return [f"<0x{byte:02X}>" for byte in text.encode("utf-8")]


def subword_level_tokenizer(
    text: str,
    vocabulary: Mapping[str, int] | set[str] | None = None,
    continuation_prefix: str = "##",
) -> list[str]:
    """Split text into greedy subword pieces.

    When a learned vocabulary is provided, the tokenizer performs a
    longest-match search over each pre-tokenized word. Without a vocabulary,
    it falls back to character pieces so the helper still behaves predictably
    during smoke tests.
    """
    tokens: list[str] = []

    for word in word_tokenizer(text):
        if not word:
            continue

        if vocabulary is None:
            tokens.append(word[0])
            tokens.extend(f"{continuation_prefix}{char}" for char in word[1:])
            continue

        start = 0
        word_tokens: list[str] = []

        while start < len(word):
            end = len(word)
            current_token = None

            while start < end:
                piece = word[start:end]
                candidate = piece if start == 0 else f"{continuation_prefix}{piece}"
                if candidate in vocabulary:
                    current_token = candidate
                    break
                end -= 1

            if current_token is None:
                fallback_piece = word[start]
                candidate = fallback_piece if start == 0 else f"{continuation_prefix}{fallback_piece}"
                if candidate in vocabulary:
                    current_token = candidate
                    end = start + 1
                else:
                    word_tokens = [UNK_TOKEN]
                    break

            word_tokens.append(current_token)
            start = end

        tokens.extend(word_tokens)

    return tokens


def sentencepiece_tokenizer(text: str) -> list[str]:
    """SentencePiece-style tokenizer that marks spaces with ▁."""
    normalized_text = "▁" + re.sub(r"\s+", " ▁", text.strip())
    return re.findall(r"▁?\w+|[^\w\s]", normalized_text)


__all__ = [
    "character_tokenizer",
    "whitespace_tokenizer",
    "word_tokenizer",
    "sentence_tokenizer",
    "regex_tokenizer",
    "byte_level_tokenizer",
    "subword_level_tokenizer",
    "sentencepiece_tokenizer",
]
