#!/usr/bin/env python3
"""
Convert cleaned NICE markdown guidance into adaptive chunks stored in Parquet.

Use this at the corpus chunking stage, after preprocessing into
`data/interim/nice/markdown_clean/`. Do not write directly to the global
`data/processed/chunks.parquet` until the cross-corpus merge step is ready.

The output keeps the RAG index contract (`chunk_id`, `doc_id`, `source`, `text`)
and also preserves NICE-specific metadata for citations and filtering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "nice" / "markdown_clean"
DEFAULT_DOCUMENTS = PROJECT_ROOT / "data" / "interim" / "nice" / "documents.parquet"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "nice" / "chunks.parquet"

TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_REGEX = re.compile(r"\w+", re.UNICODE)
HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*)$")
NICE_PREFIX_REGEX = re.compile(r"^([a-z]+)")
NUMERIC_SUBHEADING_REGEX = re.compile(r"^\d+\.\d+(?:\.\d+)*$")

DROP_SECTION_HINTS = (
    "your responsibility",
    "contents",
    "commercial arrangement",
    "implementation",
    "recommendations for research",
    "recommendations for data collection",
    "antimicrobial surveillance",
    "committee members",
    "nice project team",
    "equality impact assessment",
    "resource impact",
    "economic evidence",
    "cost effectiveness",
    "cost-effectiveness",
    "incremental net health benefit",
)

CLINICAL_CORE_SECTION_HINTS = (
    "recommendation",
    "rationale and impact",
    "why the committee",
    "what this means in practice",
    "information about",
    "marketing authorisation",
    "dosage",
    "indication",
    "contraindication",
    "assessment",
    "diagnosis",
    "diagnostic",
    "symptom",
    "signs",
    "risk factor",
    "management",
    "treatment",
    "therapy",
    "referral",
    "monitoring",
    "follow-up",
    "prevention",
    "clinical need",
    "clinical evidence",
    "safety",
    "adverse",
    "complication",
    "stewardship",
    "patient information",
    "information and support",
)


@dataclass
class DocumentMeta:
    document_id: str
    external_id: str
    title: str
    source_url: str
    guidance_type: str
    publication_date: str


@dataclass
class ChunkCandidate:
    text: str
    header_path: str
    section_name: str
    metadata: dict[str, Any]

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)

    @property
    def word_count(self) -> int:
        return count_words(self.text)


def count_tokens(text: str) -> int:
    # Approximation good enough for adaptive thresholding before embedding.
    return len(TOKEN_REGEX.findall(text))


def count_words(text: str) -> int:
    return len(WORD_REGEX.findall(text))


def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = TOKEN_REGEX.findall(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, max_tokens - overlap_tokens)

    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        chunk_text = _detokenize(window)
        if chunk_text:
            chunks.append(chunk_text)
        if start + max_tokens >= len(tokens):
            break
        start += step

    return chunks


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+[-–—]\s+", "–", text)
    return text.strip()


def parse_markdown_sections(md_text: str) -> list[tuple[list[str], str]]:
    """Return list of (header_path_list, text_block)."""
    lines = md_text.splitlines()
    header_stack: list[str] = []
    buffer: list[str] = []
    parsed: list[tuple[list[str], str]] = []

    def flush_buffer() -> None:
        block = "\n".join(buffer).strip()
        if block:
            parsed.append((header_stack.copy(), block))

    for line in lines:
        heading_match = HEADING_REGEX.match(line.strip())
        if heading_match:
            flush_buffer()
            buffer.clear()

            level = len(heading_match.group(1))
            title = normalize_heading_title(heading_match.group(2))
            if NUMERIC_SUBHEADING_REGEX.match(title) and header_stack and level <= len(header_stack):
                if NUMERIC_SUBHEADING_REGEX.match(header_stack[-1]):
                    level = len(header_stack)
                else:
                    level = min(len(header_stack) + 1, 6)
            header_stack[:] = header_stack[: level - 1]
            header_stack.append(title)
            continue

        buffer.append(line)

    flush_buffer()
    return parsed


def normalize_heading_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().strip("*").strip())


def section_allowed(header_list: list[str], policy: str) -> bool:
    if policy == "all":
        return True
    if not header_list:
        return False

    normalized_path = " > ".join(_normalize_for_match(h) for h in header_list)
    if any(hint in normalized_path for hint in DROP_SECTION_HINTS):
        return False
    return any(hint in normalized_path for hint in CLINICAL_CORE_SECTION_HINTS)


def _normalize_for_match(value: str) -> str:
    value = re.sub(r"^\d+(?:\.\d+)*\s+", "", value.lower())
    value = value.replace("*", "")
    return re.sub(r"\s+", " ", value).strip()


def to_candidates(
    md_text: str,
    max_tokens: int,
    overlap_tokens: int,
    section_policy: str,
) -> list[ChunkCandidate]:
    sections = parse_markdown_sections(md_text)
    out: list[ChunkCandidate] = []

    for header_list, section_text in sections:
        if not section_allowed(header_list, section_policy):
            continue

        header_path = " -> ".join(header_list) if header_list else "ROOT"
        section_name = header_list[-1] if header_list else "ROOT"
        sub_chunks = split_text_by_tokens(section_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        for idx, sub_chunk in enumerate(sub_chunks):
            out.append(
                ChunkCandidate(
                    text=sub_chunk,
                    header_path=header_path,
                    section_name=section_name,
                    metadata={"subchunk_index": idx, "subchunk_total": len(sub_chunks)},
                )
            )

    return out


def merge_small_neighbors(candidates: list[ChunkCandidate], min_tokens: int) -> list[ChunkCandidate]:
    if not candidates:
        return []

    merged: list[ChunkCandidate] = []
    i = 0

    while i < len(candidates):
        current = candidates[i]
        if current.token_count >= min_tokens or i == len(candidates) - 1:
            merged.append(current)
            i += 1
            continue

        nxt = candidates[i + 1]
        if not merge_compatible(current, nxt):
            merged.append(current)
            i += 1
            continue

        merged_text = f"{current.text}\n\n{nxt.text}".strip()
        merged_header_path = current.header_path if current.header_path == nxt.header_path else f"{current.header_path} || {nxt.header_path}"
        merged_section_name = current.section_name if current.section_name == nxt.section_name else "MIXED_SECTION"
        merged_metadata = {
            "merged_from_small_chunks": True,
            "left_tokens": current.token_count,
            "right_tokens": nxt.token_count,
            "left_metadata": current.metadata,
            "right_metadata": nxt.metadata,
        }

        merged.append(
            ChunkCandidate(
                text=merged_text,
                header_path=merged_header_path,
                section_name=merged_section_name,
                metadata=merged_metadata,
            )
        )
        i += 2

    return merged


def merge_compatible(left: ChunkCandidate, right: ChunkCandidate) -> bool:
    if left.header_path == right.header_path:
        return True
    return top_level_section(left.header_path) == top_level_section(right.header_path)


def top_level_section(header_path: str) -> str:
    parts = [part.strip() for part in header_path.split("->")]
    return parts[1] if len(parts) > 1 else header_path


def iter_markdown_files(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.rglob("*.md") if p.is_file())


def load_document_lookup(documents_path: Path | None) -> dict[str, dict[str, Any]]:
    if documents_path is None or not documents_path.exists():
        return {}

    df = pd.read_parquet(documents_path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        keys = {
            _clean_key(row.get("external_id")),
            _clean_key(row.get("document_id")),
            _clean_key(str(row.get("document_id") or "").removeprefix("nice-")),
            _clean_key(Path(str(row.get("markdown_relpath") or "")).stem),
        }
        for key in keys:
            if key:
                lookup[key] = row
    return lookup


def _clean_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def document_meta_for(file_path: Path, md_text: str, lookup: dict[str, dict[str, Any]]) -> DocumentMeta:
    stem = file_path.stem.lower()
    row = lookup.get(stem) or lookup.get(f"nice-{stem}") or {}

    external_id = str(row.get("external_id") or stem.upper()).strip()
    document_id = str(row.get("document_id") or f"nice-{external_id.lower()}").strip()
    title = str(row.get("title") or infer_title(md_text) or external_id).strip()
    source_url = str(row.get("source_url") or f"https://www.nice.org.uk/guidance/{external_id.lower()}").strip()
    guidance_type = str(row.get("guidance_type") or prefix_for_id(external_id)).strip()
    publication_date = str(row.get("published_at") or row.get("publication_date") or "").strip()

    return DocumentMeta(
        document_id=document_id,
        external_id=external_id,
        title=title,
        source_url=source_url,
        guidance_type=guidance_type,
        publication_date=publication_date,
    )


def infer_title(md_text: str) -> str:
    for line in md_text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("#"):
            return normalize_heading_title(stripped.lstrip("#").strip())
    return ""


def prefix_for_id(value: str) -> str:
    match = NICE_PREFIX_REGEX.match(value.strip().lower())
    return match.group(1) if match else ""


def document_allowed(meta: DocumentMeta, include_prefixes: set[str], exclude_prefixes: set[str]) -> bool:
    prefix = prefix_for_id(meta.external_id)
    if include_prefixes and prefix not in include_prefixes:
        return False
    return prefix not in exclude_prefixes


def build_rows(candidates: list[ChunkCandidate], meta: DocumentMeta, prepend_context: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        text = text_with_context(candidate, meta) if prepend_context else candidate.text
        metadata = {
            **candidate.metadata,
            "external_id": meta.external_id,
            "guidance_type": meta.guidance_type,
        }
        rows.append(
            {
                "chunk_id": f"{meta.document_id}:chunk:{index:04d}",
                "doc_id": meta.document_id,
                "document_id": meta.document_id,
                "source": "nice",
                "source_type": "guideline",
                "source_name": "NICE",
                "title": meta.title,
                "external_id": meta.external_id,
                "url": meta.source_url,
                "source_url": meta.source_url,
                "publication_date": meta.publication_date,
                "guidance_type": meta.guidance_type,
                "section": candidate.section_name,
                "section_name": candidate.section_name,
                "header_path": candidate.header_path,
                "text": text,
                "word_count": count_words(text),
                "chunk_index": index,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata": json.dumps(metadata, ensure_ascii=True, sort_keys=True),
            }
        )
    return rows


def text_with_context(candidate: ChunkCandidate, meta: DocumentMeta) -> str:
    return (
        f"Title: {meta.title}\n"
        f"NICE guidance: {meta.external_id}\n"
        f"Section: {candidate.header_path}\n\n"
        f"{candidate.text}"
    ).strip()


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    md_files = list(iter_markdown_files(input_dir))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in: {input_dir}")

    document_lookup = load_document_lookup(Path(args.documents).resolve() if args.documents else None)
    include_prefixes = {p.lower() for p in args.include_prefix}
    exclude_prefixes = {p.lower() for p in args.exclude_prefix}

    all_rows: list[dict[str, Any]] = []
    skipped_by_prefix = 0
    skipped_empty = 0

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        meta = document_meta_for(file_path, text, document_lookup)
        if not document_allowed(meta, include_prefixes, exclude_prefixes):
            skipped_by_prefix += 1
            continue

        candidates = to_candidates(
            md_text=text,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            section_policy=args.section_policy,
        )
        candidates = merge_small_neighbors(candidates, min_tokens=args.min_tokens)
        if not candidates:
            skipped_empty += 1
            continue
        all_rows.extend(build_rows(candidates, meta, prepend_context=not args.no_prepend_context))

    if not all_rows:
        raise RuntimeError("No chunks generated. Relax --include-prefix/--exclude-prefix or --section-policy.")

    df = pd.DataFrame(all_rows)
    df.to_parquet(output_file, index=False)

    print(f"Input markdown files: {len(md_files)}")
    print(f"Skipped by prefix: {skipped_by_prefix}")
    print(f"Skipped without selected sections: {skipped_empty}")
    print(f"Output chunks: {len(df)}")
    print(f"Saved parquet: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert cleaned NICE markdown files to chunks.parquet")
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR), help="Directory with cleaned NICE markdown files")
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT_FILE), help="Target parquet path")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS), help="NICE documents.parquet with metadata")
    parser.add_argument(
        "--section-policy",
        choices=["clinical-core", "all"],
        default="clinical-core",
        help="clinical-core keeps recommendations, diagnosis/management/treatment, rationale, safety, dosage and clinical evidence sections",
    )
    parser.add_argument("--include-prefix", nargs="*", default=[], help="Only include NICE IDs with these prefixes, e.g. ng cg amr ta")
    parser.add_argument("--exclude-prefix", nargs="*", default=[], help="Exclude NICE IDs with these prefixes, e.g. qs mib es")
    parser.add_argument("--max_tokens", type=int, default=600, help="Split threshold")
    parser.add_argument("--min_tokens", type=int, default=200, help="Merge threshold")
    parser.add_argument("--overlap_tokens", type=int, default=50, help="Overlap during splitting")
    parser.add_argument("--no-prepend-context", action="store_true", help="Do not prepend title / NICE id / section to chunk text")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
