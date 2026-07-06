"""Helpers for creating plain text files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_txt_file(
    text: str,
    output_path: str | Path,
    *,
    filename: str | None = None,
) -> Path:
    """Write text to a UTF-8 .txt file and return the saved path.

    If ``output_path`` points to a folder, a filename is created automatically.
    """
    path = Path(output_path)

    if path.suffix.lower() != ".txt":
        path.mkdir(parents=True, exist_ok=True)
        generated_name = filename or f"scraped_text_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
        path = path / generated_name

    if path.suffix.lower() != ".txt":
        path = path.with_suffix(".txt")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
