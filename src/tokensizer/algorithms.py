"""Educational subword tokenizer algorithms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from .constants import UNK_TOKEN
from .pretokenizers import subword_level_tokenizer, whitespace_tokenizer


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


def bpe_tokenizer(text: str, merges: int = 10) -> list[str]:
    """Educational Byte Pair Encoding tokenizer."""
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
    """Greedy WordPiece-style tokenizer."""
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


__all__ = [
    "bpe_tokenizer",
    "wordpiece_tokenizer",
    "unigram_tokenizer",
]
