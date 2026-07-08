"""Website and PDF scraping helpers."""

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

__all__ = [
    "DEFAULT_RAW_DIR",
    "DEFAULT_RAW_PDF_DIR",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "build_parser",
    "collect_pdf_files",
    "main",
    "scrape_pdf_directory_to_files",
    "scrape_pdf_text",
    "scrape_pdf_to_file",
    "scrape_website_text",
    "scrape_website_to_file",
]
