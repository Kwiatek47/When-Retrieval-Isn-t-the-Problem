from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CHAPTER_SEARCH_TERM = "statpearls[book] AND chapter[bookpart]"


@dataclass(frozen=True)
class NCBIConfig:
    email: str
    api_key: str | None
    tool: str = "medchat-statpearls"
    delay_seconds: float = 0.34
    timeout_seconds: float = 60.0


def load_config() -> NCBIConfig:
    email = os.getenv("NCBI_EMAIL", "").strip()
    if not email:
        raise RuntimeError("Set NCBI_EMAIL before calling NCBI E-utilities (required by NCBI policy).")
    api_key = os.getenv("NCBI_API_KEY", "").strip() or None
    delay = 0.11 if api_key else 0.34
    return NCBIConfig(email=email, api_key=api_key, delay_seconds=delay)


def eutils_get(endpoint: str, params: dict[str, str | int], config: NCBIConfig) -> httpx.Response:
    request_params: dict[str, str | int] = {
        **params,
        "tool": config.tool,
        "email": config.email,
    }
    if config.api_key:
        request_params["api_key"] = config.api_key

    last_error: Exception | None = None
    for attempt in range(5):
        time.sleep(config.delay_seconds)
        try:
            with httpx.Client(timeout=config.timeout_seconds, headers={"User-Agent": config.tool}) as client:
                response = client.get(f"{EUTILS_BASE}/{endpoint}", params=request_params)
                response.raise_for_status()
                return response
        except Exception as exc:  # noqa: BLE001 - retry transient NCBI/network failures
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"NCBI request failed after retries: {endpoint}") from last_error


def esearch_chapter_uids(
    *,
    config: NCBIConfig,
    retstart: int,
    retmax: int,
) -> tuple[list[str], int]:
    response = eutils_get(
        "esearch.fcgi",
        {
            "db": "books",
            "term": CHAPTER_SEARCH_TERM,
            "retstart": retstart,
            "retmax": retmax,
            "retmode": "xml",
        },
        config,
    )
    root = ET.fromstring(response.content)
    ids = [element.text for element in root.findall(".//Id") if element.text]
    count_element = root.find(".//Count")
    total = int(count_element.text) if count_element is not None and count_element.text else len(ids)
    return ids, total


def esummary_chapters(uids: list[str], *, config: NCBIConfig) -> list[dict[str, str]]:
    if not uids:
        return []

    response = eutils_get(
        "esummary.fcgi",
        {"db": "books", "id": ",".join(uids), "retmode": "xml"},
        config,
    )
    root = ET.fromstring(response.content)
    rows: list[dict[str, str]] = []
    for doc in root.findall(".//DocSum"):
        row = {item.get("Name", ""): (item.text or "").strip() for item in doc.findall("Item")}
        row["uid"] = (doc.findtext("Id") or "").strip()
        rows.append(row)
    return rows


def extract_nbk_id(rid: str) -> str | None:
    rid = rid.strip()
    if not rid:
        return None
    if rid.upper().startswith("NBK"):
        return rid.split("/", 1)[0].upper()
    return None
