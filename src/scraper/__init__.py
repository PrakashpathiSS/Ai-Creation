"""Website, PDF, CSV, and Google Sheets scraping helpers."""

from .pdf_scraper import (
    DEFAULT_RAW_DIR,
    DEFAULT_RAW_PDF_DIR,
    collect_pdf_files,
    scrape_pdf_directory_to_files,
    scrape_pdf_text,
    scrape_pdf_to_file,
)

from .website_scraper import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    build_parser,
    main,
    scrape_website_text,
    scrape_website_to_file,
)

from .spreadsheet_scraper import (
    DEFAULT_RAW_CSV_DIR,
    build_google_sheet_csv_url,
    collect_csv_files,
    scrape_csv_directory_to_files,
    scrape_csv_text,
    scrape_csv_to_file,
    scrape_google_sheet_text,
    scrape_google_sheet_to_file,
)

__all__ = [
    "DEFAULT_RAW_DIR",
    "DEFAULT_RAW_CSV_DIR",
    "DEFAULT_RAW_PDF_DIR",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "build_google_sheet_csv_url",
    "build_parser",
    "collect_csv_files",
    "collect_pdf_files",
    "main",
    "scrape_csv_directory_to_files",
    "scrape_csv_text",
    "scrape_csv_to_file",
    "scrape_google_sheet_text",
    "scrape_google_sheet_to_file",
    "scrape_pdf_directory_to_files",
    "scrape_pdf_text",
    "scrape_pdf_to_file",
    "scrape_website_text",
    "scrape_website_to_file",
]
