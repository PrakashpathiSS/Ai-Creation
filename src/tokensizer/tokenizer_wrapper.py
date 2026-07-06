"""Trainable tokenizer wrapper."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .constants import BOS_TOKEN, EOS_TOKEN, SKIPPED_SPECIAL_TOKENS
from .pretokenizers import subword_level_tokenizer, word_tokenizer
from .vocabulary import (
    base_vocabulary_tokens,
    build_vocabulary,
    detokenize,
    extend_vocabulary,
    ids_to_tokens,
    load_vocabulary,
    save_vocabulary,
    tokens_to_ids,
    validate_vocabulary,
)

TokenFunction = Callable[..., list[str]]


def _tokenizer_supports_vocabulary(tokenizer: TokenFunction) -> bool:
    """Return True when a tokenizer can accept a vocabulary argument."""
    try:
        signature = inspect.signature(tokenizer)
    except (TypeError, ValueError):
        return False

    if "vocabulary" in signature.parameters:
        return True

    return any(
        parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )


def _call_tokenizer(
    tokenizer: TokenFunction,
    text: str,
    vocabulary: Mapping[str, int] | set[str] | None = None,
) -> list[str]:
    """Call a tokenizer with or without a vocabulary argument."""
    if vocabulary is not None and _tokenizer_supports_vocabulary(tokenizer):
        return tokenizer(text, vocabulary)
    return tokenizer(text)


def _normalize_texts(texts: str | Iterable[str]) -> list[str]:
    """Normalize a single string or iterable of strings into a list of texts."""
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _tokenize_corpus(
    tokenizer: TokenFunction,
    texts: str | Iterable[str],
    vocabulary: Mapping[str, int] | set[str] | None = None,
) -> list[str]:
    """Tokenize one text or many texts while keeping corpus boundaries intact."""
    tokens: list[str] = []
    for text in _normalize_texts(texts):
        tokens.extend(_call_tokenizer(tokenizer, text, vocabulary))
    return tokens


@dataclass
class TokenizerWrapper:
    """A small trainable tokenizer for beginner AI experiments."""

    tokenize: TokenFunction = word_tokenizer
    training_tokenize: TokenFunction | None = None
    vocabulary: dict[str, int] = field(default_factory=dict)
    _tokenize_with_vocabulary: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokenize_with_vocabulary = _tokenizer_supports_vocabulary(self.tokenize)
        if self.training_tokenize is None:
            self.training_tokenize = (
                subword_level_tokenizer if self._tokenize_with_vocabulary else self.tokenize
            )

    @property
    def is_trained(self) -> bool:
        """Return True when a vocabulary has been loaded or trained."""
        return bool(self.vocabulary)

    @property
    def vocabulary_size(self) -> int:
        """Return the number of entries in the vocabulary."""
        return len(self.vocabulary)

    def fit(self, texts: str | Iterable[str], reset: bool = False) -> dict[str, int]:
        """Learn a vocabulary from one text or a corpus of texts."""
        tokenizer = self.training_tokenize or self.tokenize
        vocabulary_hint: Mapping[str, int] | set[str] | None = None
        if self._tokenize_with_vocabulary and self.vocabulary:
            vocabulary_hint = self.vocabulary

        tokens = _tokenize_corpus(tokenizer, texts, vocabulary_hint)

        if reset or not self.vocabulary:
            self.vocabulary = build_vocabulary(tokens)
        else:
            self.vocabulary = extend_vocabulary(tokens, self.vocabulary)
        return self.vocabulary

    def train(self, text: str | Iterable[str], reset: bool = False) -> dict[str, int]:
        """Backward-compatible wrapper around ``fit``."""
        return self.fit(text, reset=reset)

    def fit_from_files(
        self,
        file_paths: Iterable[str | Path],
        reset: bool = False,
        encoding: str = "utf-8",
    ) -> dict[str, int]:
        """Train the tokenizer from a list of text files."""
        texts = [Path(file_path).read_text(encoding=encoding) for file_path in file_paths]
        return self.fit(texts, reset=reset)

    def pretrain(self) -> dict[str, int]:
        """Seed the vocabulary with a-z, A-Z, 0-9 and special characters."""
        if not self.vocabulary:
            self.vocabulary = build_vocabulary([])
        self.vocabulary = extend_vocabulary(base_vocabulary_tokens(), self.vocabulary)
        return self.vocabulary

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Convert text into token IDs."""
        self._ensure_trained()
        tokens = self._tokenize(text)
        if add_special_tokens:
            if BOS_TOKEN not in self.vocabulary or EOS_TOKEN not in self.vocabulary:
                raise ValueError(
                    "Vocabulary must contain BOS and EOS tokens before adding special tokens."
                )
            tokens = [BOS_TOKEN, *tokens, EOS_TOKEN]
        return tokens_to_ids(tokens, self.vocabulary)

    def encode_batch(
        self,
        texts: Iterable[str],
        add_special_tokens: bool = False,
    ) -> list[list[int]]:
        """Encode multiple texts in one call."""
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back into readable text."""
        self._ensure_trained()
        tokens = ids_to_tokens(token_ids, self.vocabulary)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in SKIPPED_SPECIAL_TOKENS]
        return detokenize(tokens)

    def decode_batch(
        self,
        batch_token_ids: Iterable[Iterable[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode multiple token ID sequences in one call."""
        return [
            self.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]

    def save(self, file_path: str | Path) -> None:
        """Save this tokenizer vocabulary."""
        self._ensure_trained()
        save_vocabulary(self.vocabulary, file_path)

    def load(self, file_path: str | Path) -> dict[str, int]:
        """Load this tokenizer vocabulary."""
        self.vocabulary = load_vocabulary(file_path)
        return self.vocabulary

    def _ensure_trained(self) -> None:
        if not self.vocabulary:
            raise ValueError("Train the tokenizer before calling encode or decode.")
        validate_vocabulary(self.vocabulary)

    def _tokenize(self, text: str) -> list[str]:
        if self._tokenize_with_vocabulary:
            return _call_tokenizer(self.tokenize, text, self.vocabulary)
        return _call_tokenizer(self.tokenize, text)


__all__ = ["TokenFunction", "TokenizerWrapper"]
