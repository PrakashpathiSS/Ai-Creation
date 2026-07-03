"""Simple tokenizer examples for learning how text becomes token IDs."""

from __future__ import annotations

import inspect
import json
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping
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

TokenFunction = Callable[..., list[str]]
BYTE_TOKEN_PATTERN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def base_vocabulary_tokens() -> list[str]:
    """Return the pre-training tokens: a-z, A-Z, 0-9 and special characters."""
    return [
        *LOWERCASE_LETTERS,
        *UPPERCASE_LETTERS,
        *DIGITS,
        *SPECIAL_CHARACTERS,
    ]


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


def _combine_texts(texts: str | Iterable[str]) -> str:
    """Normalize single strings and iterables of strings into one corpus string."""
    if isinstance(texts, str):
        return texts
    return "\n".join(texts)


def validate_vocabulary(vocabulary: Mapping[str, int]) -> None:
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


def wordpiece_tokenizer(
    text: str,
    vocabulary: Mapping[str, int] | set[str] | None = None,
) -> list[str]:
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


def unigram_tokenizer(
    text: str,
    vocabulary: Mapping[str, int] | set[str] | None = None,
) -> list[str]:
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

    validate_vocabulary(vocabulary)
    return vocabulary


def extend_vocabulary(tokens: list[str], vocabulary: dict[str, int]) -> dict[str, int]:
    """Add new tokens to an existing vocabulary without changing old IDs.

    New IDs are based on the highest existing ID, so gaps in the vocabulary
    never cause two tokens to share the same ID.
    """
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


@dataclass
class SimpleTokenizer:
    """A small trainable tokenizer for beginner AI experiments.

    ``tokenize`` is used for encoding. ``training_tokenize`` is used only when
    building a vocabulary, which makes it easier to use a simple corpus tokenizer
    for training and a more advanced tokenizer for inference.
    """

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
        corpus = _combine_texts(texts)
        tokens = _call_tokenizer(self.training_tokenize or self.tokenize, corpus)

        if reset or not self.vocabulary:
            self.vocabulary = build_vocabulary(tokens)
        else:
            self.vocabulary = extend_vocabulary(tokens, self.vocabulary)
        return self.vocabulary

    def train(self, text: str, reset: bool = False) -> dict[str, int]:
        """Backward-compatible wrapper around ``fit``."""
        return self.fit(text, reset=reset)

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
            return _call_tokenizer(self.tokenize, text, set(self.vocabulary))
        return _call_tokenizer(self.tokenize, text)


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
