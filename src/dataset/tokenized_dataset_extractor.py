"""Decode tokenized dataset rows back into readable text."""

from __future__ import annotations

import json
from pathlib import Path

from tokensizer import TokenizerWrapper

DEFAULT_TOKENIZED_DATASET_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "dataset" / "inventory_tokenized_dataset.jsonl"
)
DEFAULT_TOKENIZER_MODEL = Path(__file__).resolve().parents[1] / "tokensizer" / "tokenizer_model.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "decoded"
DEFAULT_OUTPUT_FILENAME = "inventory_tokenized_dataset_decoded.txt"


def extract_input_ids_to_text(
    tokenized_dataset_file: str | Path = DEFAULT_TOKENIZED_DATASET_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    tokenizer_model_path: str | Path = DEFAULT_TOKENIZER_MODEL,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    token_field: str = "input_ids",
    include_metadata: bool = True,
) -> Path:
    """Decode token IDs from a JSONL dataset into plain text lines.

    Each JSONL row is expected to contain a list of token IDs in ``token_field``.
    The decoded output keeps one line per row so overlapping windows remain easy
    to inspect during debugging.
    """
    dataset_path = Path(tokenized_dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Tokenized dataset does not exist: {dataset_path}")

    tokenizer = TokenizerWrapper()
    tokenizer.load(tokenizer_model_path)

    decoded_lines: list[str] = []
    for line_number, row in _read_jsonl(dataset_path):
        token_ids = row.get(token_field)
        if token_ids is None and token_field == "input_ids":
            token_ids = row.get("input_id")
        if not isinstance(token_ids, list):
            raise ValueError(
                f"Row {line_number} is missing a list field named {token_field!r}."
            )

        decoded_text = tokenizer.decode([int(token_id) for token_id in token_ids])
        if include_metadata:
            row_id = row.get("_id", row.get("id", line_number))
            decoded_lines.append(f"[id={row_id}] {decoded_text}")
        else:
            decoded_lines.append(decoded_text)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = Path(output_filename)
    if output_file.suffix.lower() != ".txt":
        output_file = output_file.with_suffix(".txt")

    decoded_file = output_path / output_file.name
    decoded_file.write_text("\n".join(decoded_lines) + "\n", encoding="utf-8")
    return decoded_file


def _read_jsonl(file_path: Path) -> list[tuple[int, dict[str, object]]]:
    rows: list[tuple[int, dict[str, object]]] = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be a JSON object.")
        rows.append((line_number, row))
    return rows
