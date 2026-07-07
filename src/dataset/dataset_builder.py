"""Build a tokenized training dataset from the inventory corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tokensizer import TokenizerWrapper

DEFAULT_CORPUS_FILE = Path(__file__).resolve().parents[1] / "data" / "inventory_corpus" / "inventory_corpus.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dataset"
DEFAULT_OUTPUT_FILENAME = "inventory_tokenized_dataset.jsonl"
DEFAULT_TOKENIZER_MODEL = Path(__file__).resolve().parents[1] / "tokensizer" / "tokenizer_model.json"

#They control how long each training chunk is, how much the next chunk overlaps, 
# and when to skip chunks that are too small to learn from.


DEFAULT_SEQUENCE_LENGTH = 128 # 128 +1 stores tokens for the input_ids & target_ids how long each chunk is
DEFAULT_STRIDE = 64 #number of tokens to shift forward for the next chunk and start from the previous chunk
DEFAULT_MIN_SEQUENCE_LENGTH = 2 # if sequence is less than 2, we skip it because it's too short no use


def build_tokenized_dataset(
    corpus_file: str | Path = DEFAULT_CORPUS_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    tokenizer_model_path: str | Path = DEFAULT_TOKENIZER_MODEL,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    stride: int = DEFAULT_STRIDE,
    min_sequence_length: int = DEFAULT_MIN_SEQUENCE_LENGTH,
    train_tokenizer_if_missing: bool = True,
) -> Path:
    """Convert the corpus into token ID input/target pairs.

    Each dataset record stores:
    - `input_ids`: token IDs for the model input
    - `target_ids`: the same sequence shifted by one token

    This matches the training style used for next-token prediction models.
    """
    corpus_path = Path(corpus_file)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file does not exist: {corpus_path}")

    tokenizer = _load_or_train_tokenizer(
        tokenizer_model_path=tokenizer_model_path,
        corpus_path=corpus_path,
        train_tokenizer_if_missing=train_tokenizer_if_missing,
    )

    documents = _load_documents(corpus_path)
    if not documents:
        raise ValueError(f"No readable text was found in: {corpus_path}")

    records = _build_records(
        documents,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=stride,
        min_sequence_length=min_sequence_length,
    )
    if not records:
        raise ValueError(f"No tokenized samples were produced from: {corpus_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_filename = Path(output_filename)
    if dataset_filename.suffix.lower() != ".jsonl":
        dataset_filename = dataset_filename.with_suffix(".jsonl")

    dataset_file = output_path / dataset_filename.name
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    dataset_file.write_text(payload, encoding="utf-8")
    return dataset_file


def _load_or_train_tokenizer(
    *,
    tokenizer_model_path: str | Path,
    corpus_path: Path,
    train_tokenizer_if_missing: bool,
) -> TokenizerWrapper:
    tokenizer = TokenizerWrapper(target_vocab_size=2000, min_frequency=2)
    model_path = Path(tokenizer_model_path)

    if model_path.exists():
        tokenizer.load(model_path)
        return tokenizer

    if not train_tokenizer_if_missing:
        raise FileNotFoundError(f"Tokenizer model does not exist: {model_path}")

    tokenizer.fit_from_files([corpus_path], reset=True)
    tokenizer.save(model_path)
    return tokenizer


def _load_documents(corpus_path: Path) -> list[str]:
    corpus_text = corpus_path.read_text(encoding="utf-8", errors="replace")
    documents: list[str] = []
    for chunk in re.split(r"\n{2,}", corpus_text):
        normalized = _normalize_whitespace(chunk)
        if normalized:
            documents.append(normalized)
    return documents


def _build_records(
    documents: list[str],
    *,
    tokenizer: TokenizerWrapper,
    sequence_length: int,
    stride: int,
    min_sequence_length: int,
) -> list[dict[str, object]]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if min_sequence_length < 2:
        raise ValueError("min_sequence_length must be at least 2.")

    records: list[dict[str, object]] = []
    max_window_size = sequence_length + 1

    for document_index, document in enumerate(documents):
        token_ids = tokenizer.encode(document)
        if len(token_ids) < min_sequence_length:
            continue

        windows = _build_windows(token_ids, max_window_size=max_window_size, stride=stride)
        for window_index, window in enumerate(windows):
            if len(window) < 2:
                continue
            records.append(
                {
                    "id": len(records),
                    "input_ids": window[:-1],
                    "target_ids": window[1:],
                }
            )

    return records


def _build_windows(
    token_ids: list[int],
    *,
    max_window_size: int,
    stride: int,
) -> list[list[int]]:
    if len(token_ids) <= max_window_size:
        return [token_ids]

    windows: list[list[int]] = []
    last_start = len(token_ids) - max_window_size

    for start in range(0, last_start + 1, stride):
        windows.append(token_ids[start : start + max_window_size])

    if windows and (last_start % stride) != 0:
        windows.append(token_ids[last_start : last_start + max_window_size])

    return windows


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
