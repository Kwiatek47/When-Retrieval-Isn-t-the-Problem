"""Fetch StatPearls chapter printable HTML and write a native chunks parquet.

Output columns (StatPearls-native, mapped to canonical by
``scripts.data.corpora.adapters.statpearls.StatPearlsAdapter``):

- ``chunk_id`` - ``statpearls:{nbk_id_lower}:s{section_idx}:{slug}:{split_idx}``
- ``doc_id`` - ``statpearls:{nbk_id_lower}``
- ``nbk_id`` - ``NBK123456``
- ``title`` - chapter title
- ``section`` - section heading
- ``section_index`` - int
- ``chunk_index`` - int (split index within section)
- ``parent_chunk_id`` - first split of the section, or None for the first split
- ``text`` - fully formatted chunk body ("Title: ...\n\nSection: ...\n\n{body}")
- ``url`` - https://www.ncbi.nlm.nih.gov/books/{NBK_ID}/
- ``year`` - int extracted from publisher line
- ``word_count`` - int of body-only word count
- ``text_hash`` - sha256(text)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq


STATPEARLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STATPEARLS_DIR))

from parse_printable import parse_printable_html  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/raw/statpearls/chapter_manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/interim/statpearls/chunks.parquet"
DEFAULT_STATS = PROJECT_ROOT / "data/interim/statpearls/build_stats.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build StatPearls chunks.parquet from chapter manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N manifest rows (0 = all).")
    parser.add_argument("--max-words", type=int, default=420)
    parser.add_argument("--min-words", type=int, default=50, help="Drop chunks shorter than this many words.")
    parser.add_argument("--request-delay", type=float, default=0.2)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise RuntimeError(
            f"Manifest not found: {args.manifest}. "
            "Run scripts/data/statpearls/discover_chapters.py first."
        )

    records = _load_manifest(args.manifest, limit=args.limit)
    if not records:
        raise RuntimeError(f"No chapter records found in {args.manifest}.")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    with httpx.Client(
        timeout=60.0,
        headers={"User-Agent": "MedChat-StatPearls/1.0"},
        follow_redirects=True,
    ) as client:
        for index, record in enumerate(records, start=1):
            nbk_id = str(record.get("nbk_id") or "").strip().upper()
            title = str(record.get("title") or "").strip()
            if not nbk_id:
                failures.append({"record": str(record), "error": "missing nbk_id"})
                continue

            try:
                chapter_rows = _fetch_chapter_rows(
                    client=client,
                    nbk_id=nbk_id,
                    fallback_title=title,
                    max_words=args.max_words,
                    min_words=args.min_words,
                )
                rows.extend(chapter_rows)
                print(f"[{index}/{len(records)}] {nbk_id} -> {len(chapter_rows)} chunks", flush=True)
            except Exception as exc:  # noqa: BLE001 - collect per-chapter failures for stats
                failures.append({"nbk_id": nbk_id, "error": str(exc)})
                print(f"[{index}/{len(records)}] {nbk_id} FAILED: {exc}", flush=True)

            time.sleep(args.request_delay)

    if not rows:
        raise RuntimeError("No chunks were built. Check manifest and network access.")

    rows, duplicate_count = _dedupe_rows(rows)
    if duplicate_count:
        print(f"Dropped {duplicate_count} duplicate chunk_id rows.", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output, compression="zstd")

    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "output": str(args.output),
        "chapters_requested": len(records),
        "chunks_written": len(rows),
        "chapter_failures": len(failures),
        "failures": failures[:20],
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} chunks to {args.output}")


def _dedupe_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id and chunk_id not in unique:
            unique[chunk_id] = row
    duplicate_count = len(rows) - len(unique)
    return list(unique.values()), duplicate_count


def _load_manifest(path: Path, *, limit: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def _fetch_chapter_rows(
    *,
    client: httpx.Client,
    nbk_id: str,
    fallback_title: str,
    max_words: int,
    min_words: int,
) -> list[dict[str, object]]:
    url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/?report=printable"
    response = client.get(url)
    response.raise_for_status()

    title, sections = parse_printable_html(response.text)
    if not title:
        title = fallback_title or nbk_id
    if not sections:
        raise RuntimeError("no sections parsed from printable report")

    year = _extract_year(response.text)
    doc_id = f"statpearls:{nbk_id.lower()}"
    chapter_url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/"
    rows: list[dict[str, object]] = []

    for section_idx, section in enumerate(sections):
        section_slug = _slugify(section.heading)
        body = "\n\n".join(section.paragraphs).strip()
        if not body:
            continue

        split_bodies = _split_paragraphs(body, max_words=max_words)
        section_base_id = f"statpearls:{nbk_id.lower()}:s{section_idx}"
        split_idx = 0
        parent_chunk_id: str | None = None
        for chunk_body in split_bodies:
            if _word_count(chunk_body) < min_words:
                continue
            chunk_id = f"{section_base_id}:{section_slug}:{split_idx}"
            text = f"Title: {title}\n\nSection: {section.heading}\n\n{chunk_body}"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "nbk_id": nbk_id,
                    "title": title,
                    "section": section.heading,
                    "section_index": section_idx,
                    "chunk_index": split_idx,
                    "parent_chunk_id": parent_chunk_id,
                    "text": text,
                    "url": chapter_url,
                    "year": year,
                    "word_count": _word_count(chunk_body),
                    "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
            if split_idx == 0:
                parent_chunk_id = chunk_id
            split_idx += 1

    return rows


def _extract_year(html: str) -> int | None:
    match = re.search(r"StatPearls Publishing;\s*(\d{4})", html)
    if not match:
        return None
    return int(match.group(1))


_WORD_RE = re.compile(r"\b\w+\b")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _slugify(value: str, *, max_length: int = 48) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    if not text:
        return "section"
    return text[:max_length].strip("-")


def _split_paragraphs(text: str, *, max_words: int = 420) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        cleaned = text.strip()
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = _word_count(paragraph)
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


if __name__ == "__main__":
    main()
