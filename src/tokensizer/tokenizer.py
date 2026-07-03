"""Simple tokenizer examples for learning how text becomes token IDs."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
SKIPPED_SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)

LOWERCASE_LETTERS = tuple(string.ascii_lowercase)
UPPERCASE_LETTERS = tuple(string.ascii_uppercase)
DIGITS = tuple(string.digits)
SPECIAL_CHARACTERS = tuple(string.punctuation)

TokenFunction = Callable[[str], list[str]]


def base_vocabulary_tokens() -> list[str]:
    """Return the pre-training tokens: a-z, A-Z, 0-9 and special characters."""
    return [
        *LOWERCASE_LETTERS,
        *UPPERCASE_LETTERS,
        *DIGITS,
        *SPECIAL_CHARACTERS,
    ]


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


def subword_level_tokenizer(text: str, chunk_size: int = 3) -> list[str]:
    """Split words into small chunks.

    The first chunk is plain. Later chunks start with ## to show they continue
    the same word, similar to beginner explanations of subword tokenizers.
    """
    tokens: list[str] = []

    for word in whitespace_tokenizer(text):
        if len(word) <= chunk_size:
            tokens.append(word)
            continue

        tokens.append(word[:chunk_size])
        for index in range(chunk_size, len(word), chunk_size):
            tokens.append(f"##{word[index:index + chunk_size]}")

    return tokens


def bpe_tokenizer(text: str, merges: int = 10) -> list[str]:
    """Educational Byte Pair Encoding tokenizer.

    This starts with characters and repeatedly merges the most common adjacent
    pair. It is intentionally small so the algorithm is easy to read.
    """
    words = [list(word) + ["</w>"] for word in whitespace_tokenizer(text)]

    for _ in range(merges):
        pair_counts: Counter[tuple[str, str]] = Counter()

        for word in words:
            pair_counts.update(zip(word, word[1:]))

        if not pair_counts:
            break

        best_pair, count = pair_counts.most_common(1)[0]
        if count < 2:
            break

        merged = "".join(best_pair)
        words = [_merge_pair(word, best_pair, merged) for word in words]

    return [token for word in words for token in word if token != "</w>"]


def wordpiece_tokenizer(text: str, vocabulary: set[str] | None = None) -> list[str]:
    """Greedy WordPiece-style tokenizer.

    If no vocabulary is provided, this falls back to the simple subword example.
    """
    if vocabulary is None:
        return subword_level_tokenizer(text)

    output_tokens: list[str] = []

    for word in whitespace_tokenizer(text):
        start = 0
        word_tokens: list[str] = []

        while start < len(word):
            end = len(word)
            current_token = None

            while start < end:
                piece = word[start:end]
                candidate = piece if start == 0 else f"##{piece}"
                if candidate in vocabulary:
                    current_token = candidate
                    break
                end -= 1

            if current_token is None:
                word_tokens = [UNK_TOKEN]
                break

            word_tokens.append(current_token)
            start = end

        output_tokens.extend(word_tokens)

    return output_tokens


def sentencepiece_tokenizer(text: str) -> list[str]:
    """SentencePiece-style tokenizer that marks spaces with ▁."""
    normalized_text = "▁" + re.sub(r"\s+", " ▁", text.strip())
    return re.findall(r"▁?\w+|[^\w\s]", normalized_text)


def unigram_tokenizer(text: str, vocabulary: set[str] | None = None) -> list[str]:
    """Simple Unigram-style tokenizer that chooses the longest known pieces."""
    if vocabulary is None:
        vocabulary = set(subword_level_tokenizer(text))

    output_tokens: list[str] = []

    for word in whitespace_tokenizer(text):
        start = 0
        while start < len(word):
            matches = [
                piece
                for piece in vocabulary
                if word.startswith(piece.replace("##", ""), start)
            ]

            if not matches:
                output_tokens.append(UNK_TOKEN)
                start += 1
                continue

            best_piece = max(matches, key=lambda piece: len(piece.replace("##", "")))
            output_tokens.append(best_piece)
            start += len(best_piece.replace("##", ""))

    return output_tokens


def build_vocabulary(
    tokens: list[str],
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
) -> dict[str, int]:
    """Create a token-to-id vocabulary.

    Special tokens are added first so their IDs stay predictable.
    """
    vocabulary: dict[str, int] = {}

    for token in special_tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def extend_vocabulary(tokens: list[str], vocabulary: dict[str, int]) -> dict[str, int]:
    """Add new tokens to an existing vocabulary without changing old IDs.

    New IDs are based on the highest existing ID, so gaps in the vocabulary
    never cause two tokens to share the same ID.
    """
    next_id = max(vocabulary.values(), default=-1) + 1

    for token in (*SPECIAL_TOKENS, *tokens):
        if token not in vocabulary:
            vocabulary[token] = next_id
            next_id += 1

    return vocabulary


def save_vocabulary(vocabulary: dict[str, int], file_path: str | Path) -> None:
    """Save a vocabulary as JSON."""
    path = Path(file_path)
    path.write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")


def load_vocabulary(file_path: str | Path) -> dict[str, int]:
    """Load a vocabulary from JSON."""
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(token): int(token_id) for token, token_id in data.items()}


def tokens_to_ids(
    tokens: list[str],
    vocabulary: dict[str, int],
    unknown_token: str = UNK_TOKEN,
) -> list[int]:
    """Convert tokens into numbers using a vocabulary."""
    unknown_id = vocabulary[unknown_token]
    return [vocabulary.get(token, unknown_id) for token in tokens]


def ids_to_tokens(
    token_ids: list[int],
    vocabulary: dict[str, int],
    unknown_token: str = UNK_TOKEN,
) -> list[str]:
    """Convert token IDs back into tokens."""
    id_to_token = {token_id: token for token, token_id in vocabulary.items()}
    return [id_to_token.get(token_id, unknown_token) for token_id in token_ids]


def detokenize(tokens: list[str]) -> str:
    """Join tokens into text and clean spacing around punctuation."""
    text = " ".join(tokens)
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


@dataclass
class SimpleTokenizer:
    """A small trainable tokenizer for beginner AI experiments."""

    tokenize: TokenFunction = word_tokenizer
    vocabulary: dict[str, int] = field(default_factory=dict)

    def train(self, text: str, reset: bool = False) -> dict[str, int]:
        """Learn a vocabulary from text.

        By default this keeps existing tokens and appends new ones. Use
        reset=True when you want to start a fresh vocabulary.
        """
        tokens = self.tokenize(text)
        if reset or not self.vocabulary:
            self.vocabulary = build_vocabulary(tokens)
        else:
            self.vocabulary = extend_vocabulary(tokens, self.vocabulary)
        return self.vocabulary

    def pretrain(self) -> dict[str, int]:
        """Seed the vocabulary with a-z, A-Z, 0-9 and special characters."""
        if not self.vocabulary:
            self.vocabulary = build_vocabulary([])
        self.vocabulary = extend_vocabulary(base_vocabulary_tokens(), self.vocabulary)
        return self.vocabulary

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Convert text into token IDs."""
        self._ensure_trained()
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [BOS_TOKEN, *tokens, EOS_TOKEN]
        return tokens_to_ids(tokens, self.vocabulary)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back into readable text."""
        self._ensure_trained()
        tokens = ids_to_tokens(token_ids, self.vocabulary)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in SKIPPED_SPECIAL_TOKENS]
        return detokenize(tokens)

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


def _merge_pair(
    tokens: list[str],
    pair: tuple[str, str],
    merged_token: str,
) -> list[str]:
    merged_tokens: list[str] = []
    index = 0

    while index < len(tokens):
        if index < len(tokens) - 1 and (tokens[index], tokens[index + 1]) == pair:
            merged_tokens.append(merged_token)
            index += 2
        else:
            merged_tokens.append(tokens[index])
            index += 1

    return merged_tokens
