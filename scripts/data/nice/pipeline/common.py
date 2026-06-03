"""Shared paths and metadata helpers for the NICE preprocessing pipeline."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_PDF_DIR = REPO_ROOT / "data" / "raw" / "nice" / "pdf"
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "nice"
INTERIM_MARKDOWN_DIR = INTERIM_DIR / "markdown"
INTERIM_MARKDOWN_CLEAN_DIR = INTERIM_DIR / "markdown_clean"

NICE_ID_RE = re.compile(r"\b([A-Z]{2,5})\s*(\d+)\b")
NICE_URL_RE = re.compile(r"https?://www\.nice\.org\.uk/guidance/([a-z0-9]+)", re.IGNORECASE)
PUBLISHED_RE = re.compile(
    r"(?:Technology appraisal guidance|Clinical guideline|Published)[:\s]*(\d{1,2}\s+\w+\s+\d{4})",
    re.IGNORECASE,
)


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def slugify_nice_id(external_id: str) -> str:
    token = external_id.strip().lower().replace(" ", "")
    return f"nice-{token}"


def extract_metadata_from_markdown(md_text: str, fallback_stem: str) -> dict:
    head = "\n".join(md_text.splitlines()[:40])
    external_id = ""
    source_url = ""

    url_match = NICE_URL_RE.search(head)
    if url_match:
        source_url = url_match.group(0)
        external_id = url_match.group(1).upper()

    id_match = NICE_ID_RE.search(head)
    if id_match and not external_id:
        external_id = f"{id_match.group(1).upper()}{id_match.group(2)}"
        source_url = source_url or f"https://www.nice.org.uk/guidance/{external_id.lower()}"

    if not external_id:
        external_id = fallback_stem.upper()
        source_url = source_url or ""

    title = ""
    for line in md_text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("**") and stripped.endswith("**") and "picture" not in stripped:
            title = stripped.strip("*").strip()
            if len(title) > 20:
                break

    published_at = ""
    pub_match = PUBLISHED_RE.search(head)
    if pub_match:
        published_at = pub_match.group(1).strip()

    guidance_type = "technology_appraisal" if external_id.startswith("TA") else "clinical_guideline"

    return {
        "document_id": slugify_nice_id(external_id),
        "title": title or fallback_stem,
        "external_id": external_id,
        "guidance_type": guidance_type,
        "source_url": source_url,
        "published_at": published_at,
        "language": "en",
    }


def default_metadata_extra(md_text: str) -> dict:
    """Light signals for later boilerplate filtering / routing."""
    lower = md_text.lower()
    return {
        "has_recommendations_section": "recommendations" in lower,
        "has_committee_rationale": "why the committee" in lower,
        "has_dosage_tables": "dosage" in lower or "dosing" in lower,
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dumps_metadata(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True)
