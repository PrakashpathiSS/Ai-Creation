"""Shared tokenizer constants and regex helpers."""

from __future__ import annotations

import re
import string

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

BYTE_TOKEN_PATTERN = re.compile(r"<0x([0-9A-Fa-f]{2})>")

__all__ = [
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "SPECIAL_TOKENS",
    "SKIPPED_SPECIAL_TOKENS",
    "LOWERCASE_LETTERS",
    "UPPERCASE_LETTERS",
    "DIGITS",
    "SPECIAL_CHARACTERS",
    "BYTE_TOKEN_PATTERN",
]
