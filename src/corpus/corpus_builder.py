"""Build a training corpus from scraped text files."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "inventory_corpus"
DEFAULT_OUTPUT_FILENAME = "inventory_corpus.txt"
DEFAULT_SEPARATOR = "\n\n"


def build_inventory_corpus(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    separator: str = DEFAULT_SEPARATOR,
) -> Path:
    """Combine scraped .txt files into one normalized corpus file.

    The corpus is written as plain UTF-8 text so it can be used for tokenizer
    training or later model pretraining.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    text_files = _collect_text_files(source_path)
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in: {source_path}")

    documents = _load_documents(text_files)
    if not documents:
        raise ValueError(f"No readable text was found in: {source_path}")

    corpus_text = separator.join(documents).strip() + "\n"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus_filename = Path(output_filename)
    if corpus_filename.suffix.lower() != ".txt":
        corpus_filename = corpus_filename.with_suffix(".txt")

    corpus_file = output_path / corpus_filename.name
    corpus_file.write_text(corpus_text, encoding="utf-8")
    return corpus_file


def _collect_text_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path] if source_path.suffix.lower() == ".txt" else []
    return sorted(
        file_path
        for file_path in source_path.rglob("*.txt")
        if file_path.is_file()
    )


def _load_documents(text_files: list[Path]) -> list[str]:
    documents: list[str] = []
    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        normalized_text = _normalize_whitespace(text)
        if normalized_text:
            documents.append(normalized_text)
    return documents


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
