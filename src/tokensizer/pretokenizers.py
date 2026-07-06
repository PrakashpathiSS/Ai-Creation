"""Basic text pre-tokenizers used by the learning tokenizer package."""

from __future__ import annotations

import re


def character_tokenizer(text: str) -> list[str]:
    """Split text into single characters."""
    return list[str](text)


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


def subword_level_tokenizer(text: str, chunk_size: int = 3) -> list[str]:
    """Split words into small chunks."""
    tokens: list[str] = []

    for word in whitespace_tokenizer(text):
        if len(word) <= chunk_size:
            tokens.append(word)
            continue

        tokens.append(word[:chunk_size])
        for index in range(chunk_size, len(word), chunk_size):
            tokens.append(f"##{word[index:index + chunk_size]}")

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
