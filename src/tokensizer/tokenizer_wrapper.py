"""Production-oriented subword tokenizer wrapper."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .constants import BOS_TOKEN, EOS_TOKEN, SKIPPED_SPECIAL_TOKENS, SPECIAL_TOKENS, UNK_TOKEN
from .pretokenizers import word_tokenizer
from .vocabulary import base_vocabulary_tokens, build_vocabulary, detokenize, ids_to_tokens, tokens_to_ids, validate_vocabulary

TokenFunction = Callable[..., list[str]]

MODEL_VERSION = 1
CONTINUATION_PREFIX = "##"
BOUNDARY_MARKER = "</w>"


def _normalize_text(text: str, lowercase: bool) -> str:
    """Normalize text before training or encoding."""
    normalized = unicodedata.normalize("NFKC", text)
    if lowercase:
        normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_texts(texts: str | Iterable[str]) -> list[str]:
    """Normalize a single string or iterable of strings into a list of texts."""
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Merge one pair inside a symbol sequence."""
    merged: list[str] = []
    index = 0

    while index < len(symbols):
        if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1

    return tuple(merged)


@dataclass
class TokenizerWrapper:
    """A trainable subword tokenizer based on BPE-style learning."""

    tokenize: TokenFunction = word_tokenizer
    training_tokenize: TokenFunction | None = None
    target_vocab_size: int = 2000
    min_frequency: int = 2
    lowercase: bool = False
    vocabulary: dict[str, int] = field(default_factory=dict)
    merges: list[tuple[str, str]] = field(default_factory=list)
    _is_trained: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.training_tokenize is None:
            self.training_tokenize = self.tokenize
        if self.target_vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("target_vocab_size must be at least large enough for special tokens.")
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be at least 1.")

    @property
    def is_trained(self) -> bool:
        """Return True when a model has been trained or loaded."""
        return self._is_trained

    @property
    def vocabulary_size(self) -> int:
        """Return the number of entries in the vocabulary."""
        return len(self.vocabulary)

    def fit(self, texts: str | Iterable[str], reset: bool = False) -> dict[str, int]:
        """Learn a subword vocabulary from one text or a corpus."""
        normalized_texts = [
            _normalize_text(text, self.lowercase)
            for text in _normalize_texts(texts)
        ]
        normalized_texts = [text for text in normalized_texts if text]
        if not normalized_texts:
            raise ValueError("Training text cannot be empty.")

        if reset:
            self.vocabulary = {}
            self.merges = []
            self._is_trained = False

        word_frequencies = self._collect_word_frequencies(normalized_texts)
        raw_tokens, merges = self._learn_bpe(word_frequencies)

        vocab_tokens = self._build_vocabulary_tokens(raw_tokens)
        self.vocabulary = build_vocabulary(vocab_tokens)
        self.merges = merges
        self._is_trained = True
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
        """Seed the vocabulary with the base character set."""
        if not self.vocabulary:
            self.vocabulary = build_vocabulary(self._seed_raw_tokens(Counter()))
        return self.vocabulary

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Convert text into token IDs."""
        self._ensure_trained()
        normalized_text = _normalize_text(text, self.lowercase)
        tokens = self._tokenize(normalized_text)
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
        """Save the trained tokenizer model."""
        self._ensure_trained()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": MODEL_VERSION,
            "algorithm": "bpe",
            "config": {
                "target_vocab_size": self.target_vocab_size,
                "min_frequency": self.min_frequency,
                "lowercase": self.lowercase,
                "continuation_prefix": CONTINUATION_PREFIX,
            },
            "vocabulary": dict(sorted(self.vocabulary.items(), key=lambda item: item[1])),
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, file_path: str | Path) -> dict[str, int]:
        """Load a tokenizer model."""
        path = Path(file_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, dict) and "vocabulary" in data:
            vocabulary = data["vocabulary"]
            if not isinstance(vocabulary, dict):
                raise ValueError("Tokenizer model vocabulary must be a JSON object.")

            self.vocabulary = {str(token): int(token_id) for token, token_id in vocabulary.items()}
            merges_data = data.get("merges") or []
            self.merges = [
                (str(pair[0]), str(pair[1]))
                for pair in merges_data
                if isinstance(pair, list) and len(pair) == 2
            ]

            config = data.get("config", {})
            if isinstance(config, dict):
                self.target_vocab_size = int(config.get("target_vocab_size", self.target_vocab_size))
                self.min_frequency = int(config.get("min_frequency", self.min_frequency))
                self.lowercase = bool(config.get("lowercase", self.lowercase))

            validate_vocabulary(self.vocabulary)
            self._is_trained = True
            return self.vocabulary

        raise ValueError("Tokenizer file must contain a tokenizer model object.")

    def _ensure_trained(self) -> None:
        if not self._is_trained or not self.vocabulary:
            raise ValueError("Train the tokenizer before calling encode or decode.")
        validate_vocabulary(self.vocabulary)

    def _collect_word_frequencies(self, texts: list[str]) -> Counter[tuple[str, ...]]:
        """Count the words that the subword trainer should learn from."""
        tokenizer = self.training_tokenize or self.tokenize
        frequencies: Counter[tuple[str, ...]] = Counter()

        for text in texts:
            for token in tokenizer(text):
                if not token:
                    continue
                frequencies[(*token, BOUNDARY_MARKER)] += 1

        return frequencies

    def _seed_raw_tokens(self, word_frequencies: Counter[tuple[str, ...]]) -> list[str]:
        """Collect the base character vocabulary used before merges."""
        corpus_chars = sorted(
            {
                symbol
                for word in word_frequencies
                for symbol in word
                if symbol != BOUNDARY_MARKER
            }
        )

        seed_tokens = list(base_vocabulary_tokens())
        if self.lowercase:
            seed_tokens = [
                token
                for token in seed_tokens
                if not (token.isalpha() and token.isupper())
            ]

        ordered_tokens: list[str] = []
        for token in (*seed_tokens, *corpus_chars):
            if token and token not in ordered_tokens:
                ordered_tokens.append(token)

        return ordered_tokens

    def _learn_bpe(self, word_frequencies: Counter[tuple[str, ...]]) -> tuple[list[str], list[tuple[str, str]]]:
        """Learn merges and return the resulting raw tokens."""
        sequences = Counter(word_frequencies)
        raw_tokens = self._seed_raw_tokens(word_frequencies)
        seen_tokens = set(raw_tokens)
        merges: list[tuple[str, str]] = []

        max_raw_tokens = max((self.target_vocab_size - len(SPECIAL_TOKENS)) // 2, len(raw_tokens))
        max_new_tokens = max_raw_tokens - len(raw_tokens)

        for _ in range(max_new_tokens):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, frequency in sequences.items():
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += frequency

            if not pair_counts:
                break

            best_pair, best_frequency = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            if best_frequency < self.min_frequency:
                break

            merges.append(best_pair)
            merged_symbol = (best_pair[0] + best_pair[1]).replace(BOUNDARY_MARKER, "")
            if merged_symbol and merged_symbol not in seen_tokens:
                seen_tokens.add(merged_symbol)
                raw_tokens.append(merged_symbol)

            next_sequences: Counter[tuple[str, ...]] = Counter()
            for symbols, frequency in sequences.items():
                next_sequences[_merge_pair(symbols, best_pair)] += frequency
            sequences = next_sequences

        return raw_tokens, merges

    def _build_vocabulary_tokens(self, raw_tokens: list[str]) -> list[str]:
        """Expand raw tokens into the actual vocabulary entries."""
        ordered_tokens: list[str] = []
        for token in raw_tokens:
            if token not in ordered_tokens:
                ordered_tokens.append(token)

        continuation_tokens = [f"{CONTINUATION_PREFIX}{token}" for token in ordered_tokens]
        return [*ordered_tokens, *continuation_tokens]

    def _segment_token(self, token: str) -> list[str]:
        """Segment one pre-tokenized token into vocabulary pieces."""
        if token in self.vocabulary:
            return [token]

        pieces: list[str] = []
        start = 0

        while start < len(token):
            end = len(token)
            matched_piece = None

            while start < end:
                piece = token[start:end]
                candidate = piece if start == 0 else f"{CONTINUATION_PREFIX}{piece}"
                if candidate in self.vocabulary:
                    matched_piece = candidate
                    break
                end -= 1

            if matched_piece is None:
                return [UNK_TOKEN]

            pieces.append(matched_piece)
            start = end

        return pieces

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using the trained vocabulary."""
        tokens: list[str] = []
        for token in self.tokenize(text):
            if not token:
                continue
            tokens.extend(self._segment_token(token))
        return tokens


__all__ = ["TokenFunction", "TokenizerWrapper"]
