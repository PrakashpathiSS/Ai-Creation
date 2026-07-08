"""PDF scraper that extracts readable text into plain text files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from components import create_txt_file

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_RAW_PDF_DIR = DEFAULT_RAW_DIR / "pdf"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def scrape_pdf_text(
    pdf_file: str | Path,
    *,
    max_pages: int | None = None,
) -> str:
    """Extract normalized text from a PDF file."""
    PdfReader = _require_pypdf_reader()
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path}")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be at least 1 when provided.")

    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"PDF is encrypted and cannot be read: {pdf_path}") from exc

    page_texts: list[str] = []
    pages = reader.pages[:max_pages] if max_pages is not None else reader.pages
    for page in pages:
        text = page.extract_text() or ""
        normalized_text = _normalize_whitespace(text)
        if normalized_text:
            page_texts.append(normalized_text)

    return "\n\n".join(page_texts).strip()


def scrape_pdf_to_file(
    pdf_file: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    filename: str | None = None,
    max_pages: int | None = None,
) -> Path:
    """Extract PDF text and save it to a UTF-8 .txt file."""
    pdf_path = Path(pdf_file)
    text = scrape_pdf_text(pdf_path, max_pages=max_pages)
    if not text:
        raise ValueError(f"No readable text was found in PDF: {pdf_path}")

    output_filename = filename or _build_output_filename(pdf_path)
    return create_txt_file(text, output_path, filename=output_filename)


def scrape_pdf_directory_to_files(
    source_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    max_pages: int | None = None,
) -> list[Path]:
    """Extract text from every PDF under a file or directory."""
    source = Path(source_path) if source_path is not None else _default_pdf_source()
    pdf_files = collect_pdf_files(source)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {source}")

    root = source if source.is_dir() else source.parent
    output_files: list[Path] = []
    for pdf_file in pdf_files:
        output_files.append(
            scrape_pdf_to_file(
                pdf_file,
                output_path,
                filename=_build_output_filename(pdf_file, root=root),
                max_pages=max_pages,
            )
        )

    return output_files


def collect_pdf_files(source_path: str | Path) -> list[Path]:
    """Collect PDF files from a single file or recursively from a folder."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"PDF source path does not exist: {source}")

    if source.is_file():
        return [source] if source.suffix.lower() == ".pdf" else []

    return sorted(
        file_path
        for file_path in source.rglob("*.pdf")
        if file_path.is_file()
    )


def _default_pdf_source() -> Path:
    if DEFAULT_RAW_PDF_DIR.exists():
        return DEFAULT_RAW_PDF_DIR
    return DEFAULT_RAW_DIR


def _build_output_filename(pdf_path: Path, *, root: Path | None = None) -> str:
    if root is not None:
        try:
            relative_name = pdf_path.relative_to(root).with_suffix("")
        except ValueError:
            relative_name = Path(pdf_path.stem)
    else:
        relative_name = Path(pdf_path.stem)

    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(relative_name)).strip("_").lower()
    if not slug:
        slug = "pdf_document"
    return f"pdf_{slug}.txt"


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def _require_pypdf_reader() -> Any:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pypdf is required for PDF scraping. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return PdfReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract PDF text into .txt files.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="PDF file or folder. Defaults to data/raw/pdf, then data/raw.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination folder for extracted text files.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum pages to extract from each PDF.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_files = scrape_pdf_directory_to_files(
        args.source,
        args.output,
        max_pages=args.max_pages,
    )
    for output_file in output_files:
        print(f"Saved PDF text to: {output_file}")


if __name__ == "__main__":
    main()
