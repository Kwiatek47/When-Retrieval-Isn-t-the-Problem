from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser


SKIP_HEADINGS = {
    "authors",
    "affiliations",
    "references",
    "review questions",
    "copyright",
    "disclaimer",
}


@dataclass(frozen=True)
class SectionBlock:
    heading: str
    paragraphs: list[str]


class PrintableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._capture_title = False
        self._in_heading = False
        self._in_paragraph = False
        self._heading_level = 0
        self._buffer: list[str] = []
        self._current_heading = ""
        self.sections: list[SectionBlock] = []
        self._paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "meta":
            meta = {key: value for key, value in attrs if key and value}
            if meta.get("name") == "citation_title":
                self.title = (meta.get("content") or "").strip()
        if tag in {"h1", "h2", "h3"}:
            self._flush_paragraph()
            self._in_heading = True
            self._heading_level = int(tag[1])
        elif tag == "p":
            self._in_paragraph = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3"} and self._in_heading:
            self._in_heading = False
            heading = " ".join(self._buffer).strip()
            self._buffer.clear()
            if self._heading_level <= 2:
                self._flush_section(heading)
            return

        if tag == "p" and self._in_paragraph:
            self._in_paragraph = False
            paragraph = " ".join(self._buffer).strip()
            self._buffer.clear()
            if len(paragraph) >= 40:
                self._paragraphs.append(paragraph)

    def handle_data(self, data: str) -> None:
        if self._in_heading or self._in_paragraph:
            text = data.strip()
            if text:
                self._buffer.append(text)

    def _flush_paragraph(self) -> None:
        if self._in_paragraph:
            self._in_paragraph = False
            paragraph = " ".join(self._buffer).strip()
            self._buffer.clear()
            if len(paragraph) >= 40:
                self._paragraphs.append(paragraph)

    def _flush_section(self, heading: str) -> None:
        self._flush_paragraph()
        if self._paragraphs and self._current_heading:
            self.sections.append(SectionBlock(self._current_heading, list(self._paragraphs)))
        self._paragraphs.clear()
        cleaned_heading = heading.strip()
        if cleaned_heading and cleaned_heading.lower() not in SKIP_HEADINGS:
            self._current_heading = cleaned_heading
        else:
            self._current_heading = ""

    def close(self) -> None:
        if self._current_heading and self._paragraphs:
            self.sections.append(SectionBlock(self._current_heading, list(self._paragraphs)))
        super().close()


def parse_printable_html(html: str) -> tuple[str, list[SectionBlock]]:
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
    parser = PrintableParser()
    parser.feed(cleaned)
    parser.close()
    return parser.title, parser.sections
