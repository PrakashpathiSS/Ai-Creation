"""CSV and Google Sheets scraper that saves readable rows as plain text."""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

try:
    from components import create_txt_file
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from components import create_txt_file

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_RAW_CSV_DIR = DEFAULT_RAW_DIR / "csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; AiCreationSpreadsheetScraper/1.0)"
SUPPORTED_SPREADSHEET_SUFFIXES = {".csv", ".tsv"}


def scrape_csv_text(
    csv_file: str | Path,
    *,
    delimiter: str | None = None,
    max_rows: int | None = None,
    include_header: bool = True,
    encoding: str = "utf-8",
) -> str:
    """Convert a local CSV/TSV file into readable plain text."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if csv_path.suffix.lower() not in SUPPORTED_SPREADSHEET_SUFFIXES:
        raise ValueError(f"Expected a .csv or .tsv file, got: {csv_path}")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be at least 1 when provided.")

    raw_text = csv_path.read_text(encoding=encoding, errors="replace")
    return _csv_to_readable_text(
        raw_text,
        source_name=csv_path.stem,
        delimiter=delimiter or _default_delimiter_for_file(csv_path),
        max_rows=max_rows,
        include_header=include_header,
    )


def scrape_csv_to_file(
    csv_file: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    filename: str | None = None,
    delimiter: str | None = None,
    max_rows: int | None = None,
    include_header: bool = True,
) -> Path:
    """Extract readable text from one CSV/TSV file and save it to .txt."""
    csv_path = Path(csv_file)
    text = scrape_csv_text(
        csv_path,
        delimiter=delimiter,
        max_rows=max_rows,
        include_header=include_header,
    )
    if not text:
        raise ValueError(f"No readable rows were found in CSV file: {csv_path}")

    output_filename = filename or _build_output_filename(csv_path)
    return create_txt_file(text, output_path, filename=output_filename)


def scrape_csv_directory_to_files(
    source_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    delimiter: str | None = None,
    max_rows: int | None = None,
    include_header: bool = True,
) -> list[Path]:
    """Extract text from every CSV/TSV under a file or directory."""
    source = Path(source_path) if source_path is not None else _default_csv_source()
    csv_files = collect_csv_files(source)
    if not csv_files:
        raise FileNotFoundError(f"No CSV or TSV files found in: {source}")

    root = source if source.is_dir() else source.parent
    output_files: list[Path] = []
    for csv_file in csv_files:
        output_files.append(
            scrape_csv_to_file(
                csv_file,
                output_path,
                filename=_build_output_filename(csv_file, root=root),
                delimiter=delimiter,
                max_rows=max_rows,
                include_header=include_header,
            )
        )
    return output_files


def scrape_google_sheet_text(
    sheet_url: str,
    *,
    gid: str | int | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    max_rows: int | None = None,
    include_header: bool = True,
) -> str:
    """Download a public Google Sheet as CSV and convert it to readable text."""
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be at least 1 when provided.")

    csv_url = build_google_sheet_csv_url(sheet_url, gid=gid)
    request = Request(
        csv_url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw_text = response.read().decode(charset, errors="replace")
    except URLError as exc:
        raise ConnectionError(f"Failed to fetch Google Sheet: {sheet_url}") from exc

    return _csv_to_readable_text(
        raw_text,
        source_name="google_sheet",
        delimiter=",",
        max_rows=max_rows,
        include_header=include_header,
    )


def scrape_google_sheet_to_file(
    sheet_url: str,
    output_path: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    filename: str | None = None,
    gid: str | int | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    max_rows: int | None = None,
    include_header: bool = True,
) -> Path:
    """Download a public Google Sheet and save readable rows to .txt."""
    text = scrape_google_sheet_text(
        sheet_url,
        gid=gid,
        timeout=timeout,
        user_agent=user_agent,
        max_rows=max_rows,
        include_header=include_header,
    )
    if not text:
        raise ValueError(f"No readable rows were found in Google Sheet: {sheet_url}")

    output_filename = filename or _build_google_sheet_filename(sheet_url, gid=gid)
    return create_txt_file(text, output_path, filename=output_filename)


def build_google_sheet_csv_url(sheet_url: str, *, gid: str | int | None = None) -> str:
    """Convert a Google Sheets URL into a CSV export URL."""
    parsed = urlparse(sheet_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Google Sheet URL must start with http:// or https://.")
    if "docs.google.com" not in parsed.netloc:
        raise ValueError("Expected a docs.google.com Google Sheets URL.")

    query = _parse_url_params(parsed)
    if _query_is_csv_export(query):
        return sheet_url

    gid_value = str(gid) if gid is not None else _first_query_value(query, "gid")
    if "/pub" in parsed.path:
        query["output"] = ["csv"]
        if gid_value is not None:
            query["gid"] = [gid_value]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

    sheet_id = _extract_google_sheet_id(parsed.path, query)
    params = {"format": "csv"}
    if gid_value is not None:
        params["gid"] = gid_value
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?{urlencode(params)}"


def collect_csv_files(source_path: str | Path) -> list[Path]:
    """Collect CSV/TSV files from a single file or recursively from a folder."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"CSV source path does not exist: {source}")

    if source.is_file():
        return [source] if source.suffix.lower() in SUPPORTED_SPREADSHEET_SUFFIXES else []

    return sorted(
        file_path
        for file_path in source.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SPREADSHEET_SUFFIXES
    )


def _csv_to_readable_text(
    raw_text: str,
    *,
    source_name: str,
    delimiter: str | None,
    max_rows: int | None,
    include_header: bool,
) -> str:
    normalized_text = raw_text.replace("\x00", " ").strip()
    if not normalized_text:
        return ""

    delimiter = delimiter or _detect_delimiter(normalized_text)
    reader = csv.reader(io.StringIO(normalized_text), delimiter=delimiter)
    rows = [
        [_normalize_cell(cell) for cell in row]
        for row in reader
        if any(_normalize_cell(cell) for cell in row)
    ]
    if not rows:
        return ""

    headers: list[str] = []
    data_rows = rows
    if include_header and len(rows) > 1:
        headers = [_fallback_header(cell, index) for index, cell in enumerate(rows[0])]
        data_rows = rows[1:]

    if max_rows is not None:
        data_rows = data_rows[:max_rows]

    lines = [f"Source: {_normalize_cell(source_name)}."]
    for row_index, row in enumerate(data_rows, start=1):
        line = _format_row(row, headers=headers, row_index=row_index)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _format_row(row: list[str], *, headers: list[str], row_index: int) -> str:
    if headers:
        parts = []
        for column_index, cell in enumerate(row):
            if not cell:
                continue
            header = headers[column_index] if column_index < len(headers) else f"Column {column_index + 1}"
            parts.append(f"{header}: {cell}")
        return ". ".join(parts) + "." if parts else ""

    cells = [cell for cell in row if cell]
    if not cells:
        return ""
    return f"Row {row_index}: " + ". ".join(cells) + "."


def _normalize_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _fallback_header(value: str, index: int) -> str:
    normalized = _normalize_cell(value)
    return normalized if normalized else f"Column {index + 1}"


def _default_delimiter_for_file(file_path: Path) -> str | None:
    if file_path.suffix.lower() == ".tsv":
        return "\t"
    return None


def _detect_delimiter(raw_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(raw_text[:4096], delimiters=",\t;|")
    except csv.Error:
        return ","
    return dialect.delimiter


def _default_csv_source() -> Path:
    if DEFAULT_RAW_CSV_DIR.exists():
        return DEFAULT_RAW_CSV_DIR
    return DEFAULT_RAW_DIR


def _build_output_filename(csv_path: Path, *, root: Path | None = None) -> str:
    if root is not None:
        try:
            relative_name = csv_path.relative_to(root).with_suffix("")
        except ValueError:
            relative_name = Path(csv_path.stem)
    else:
        relative_name = Path(csv_path.stem)

    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(relative_name)).strip("_").lower()
    if not slug:
        slug = "spreadsheet"
    return f"csv_{slug}.txt"


def _build_google_sheet_filename(sheet_url: str, *, gid: str | int | None = None) -> str:
    parsed = urlparse(sheet_url)
    query = _parse_url_params(parsed)
    sheet_id = _extract_google_sheet_id(parsed.path, query, required=False) or "google_sheet"
    gid_value = str(gid) if gid is not None else _first_query_value(query, "gid")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", sheet_id).strip("_").lower()
    if gid_value is not None:
        slug = f"{slug}_gid_{re.sub(r'[^A-Za-z0-9]+', '_', gid_value).strip('_').lower()}"
    return f"google_sheet_{slug}.txt"


def _query_is_csv_export(query: dict[str, list[str]]) -> bool:
    return "csv" in {value.lower() for values in query.values() for value in values}


def _parse_url_params(parsed_url: object) -> dict[str, list[str]]:
    query = parse_qs(getattr(parsed_url, "query", ""))
    fragment_params = parse_qs(getattr(parsed_url, "fragment", ""))
    for key, values in fragment_params.items():
        query.setdefault(key, values)
    return query


def _extract_google_sheet_id(
    path: str,
    query: dict[str, list[str]],
    *,
    required: bool = True,
) -> str | None:
    match = re.search(r"/spreadsheets/d/([^/]+)", path)
    if match:
        return match.group(1)

    key = _first_query_value(query, "key")
    if key is not None:
        return key

    if required:
        raise ValueError("Could not find a Google Sheet ID in the URL.")
    return None


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CSV or Google Sheet rows into .txt files.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="CSV/TSV file or folder. Defaults to data/raw/csv, then data/raw.",
    )
    parser.add_argument(
        "--google-sheet-url",
        default=None,
        help="Public Google Sheets URL to export as CSV.",
    )
    parser.add_argument("--gid", default=None, help="Optional Google Sheets tab gid.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination folder for extracted text files.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional maximum rows to convert.")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the first row as data instead of column headers.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.google_sheet_url:
        output_file = scrape_google_sheet_to_file(
            args.google_sheet_url,
            args.output,
            gid=args.gid,
            max_rows=args.max_rows,
            include_header=not args.no_header,
        )
        print(f"Saved Google Sheet text to: {output_file}")
        return

    output_files = scrape_csv_directory_to_files(
        args.source,
        args.output,
        max_rows=args.max_rows,
        include_header=not args.no_header,
    )
    for output_file in output_files:
        print(f"Saved CSV text to: {output_file}")


if __name__ == "__main__":
    main()
