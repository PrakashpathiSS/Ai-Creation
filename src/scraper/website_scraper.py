"""Standalone website scraper that saves readable text to a file."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from components import create_txt_file

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; AiCreationScraper/1.0)"
DEFAULT_TIMEOUT_SECONDS = 20


class _VisibleTextExtractor(HTMLParser):
    """Extract visible text while skipping script and style content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return

        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return unescape(" ".join(self._parts))


def scrape_website_text(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    """Fetch a URL and return cleaned visible text."""
    try:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            with urlopen(url, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
                raw_text = response.read().decode(charset, errors="replace")
        else:
            request = Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
                raw_text = response.read().decode(charset, errors="replace")
    except URLError as exc:
        raise ConnectionError(f"Failed to fetch website: {url}") from exc

    if content_type in {"text/html", "application/xhtml+xml", "application/xml", "text/xml"}:
        return _extract_html_text(raw_text)

    return _normalize_whitespace(raw_text)


def scrape_website_to_file(
    url: str,
    output_path: str | Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    filename: str | None = None,
) -> Path:
    """Scrape a website and write the cleaned text to a UTF-8 text file."""
    text = scrape_website_text(url, timeout=timeout, user_agent=user_agent)
    if not text:
        raise ValueError(f"No readable text was found at: {url}")
    dynamic_filename = filename or _build_filename_from_url(url)
    return create_txt_file(text, output_path, filename=dynamic_filename)


def _extract_html_text(markup: str) -> str:
    markup = _best_effort_main_markup(markup)
    parser = _VisibleTextExtractor()
    parser.feed(markup)
    parser.close()
    return _normalize_whitespace(parser.get_text())


def _best_effort_main_markup(markup: str) -> str:
    for tag in ("main", "article", "body"):
        match = re.search(
            rf"<{tag}\b[^>]*>(.*?)</{tag}>",
            markup,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1)
    return markup


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_filename_from_url(url: str) -> str:
    """Build a stable, URL-based filename with a timestamp."""
    parsed = urlparse(url)
    path_bits = [part for part in parsed.path.split("/") if part]
    base_name = path_bits[-1] if path_bits else parsed.netloc
    base_name = re.sub(r"[^A-Za-z0-9]+", "_", base_name).strip("_").lower()
    if not base_name:
        base_name = "scraped_text"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{base_name}_{timestamp}.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape website text into a file.")
    parser.add_argument("--url", required=True, help="Website URL to scrape.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination folder or text file for the scraped content.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Network timeout in seconds.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header for HTTP requests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = scrape_website_to_file(
        args.url,
        args.output,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )
    print(f"Saved scraped text to: {output_path}")


if __name__ == "__main__":
    main()
