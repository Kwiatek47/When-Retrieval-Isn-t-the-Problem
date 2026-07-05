from __future__ import annotations

import hashlib
import re
from typing import Any


def slugify(value: str, *, max_length: int = 48) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not text:
        return "section"
    return text[:max_length].strip("-")


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_paragraphs(text: str, *, max_words: int = 420) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        cleaned = text.strip()
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = word_count(paragraph)
        if current and current_words + paragraph_words > max_words:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_words = paragraph_words
            continue
        current.append(paragraph)
        current_words += paragraph_words

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_chunk_row(
    *,
    chunk_id: str,
    title: str,
    text: str,
    source: str,
    url: str,
    section: str,
    publication_types: list[str],
    year: int | None = None,
    journal: str = "StatPearls",
    doc_id: str | None = None,
    chunk_index: int = 0,
    parent_chunk_id: str | None = None,
    is_review: bool = True,
    is_systematic: bool = False,
) -> dict[str, Any]:
    body = text.strip()
    if not chunk_id or not body:
        raise ValueError("chunk_id and text are required for chunk rows.")

    heading = title.strip() or "Untitled medical chunk"
    full_text = f"Title: {heading}\n\nSection: {section}\n\n{body}"

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id or chunk_id.rsplit(":", 1)[0],
        "title": heading,
        "text": full_text,
        "source": source,
        "url": url,
        "journal": journal,
        "year": year,
        "publication_types": publication_types,
        "section": section,
        "is_review": is_review,
        "is_systematic": is_systematic,
        "word_count": word_count(body),
        "parent_chunk_id": parent_chunk_id,
        "text_hash": text_hash(full_text),
        "chunk_index": chunk_index,
    }
