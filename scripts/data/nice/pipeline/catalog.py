"""Fetch published NICE guidance catalog from the public website."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

PUBLISHED_URL = "https://www.nice.org.uk/guidance/published"
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
USER_AGENT = "Medical-RAG-NICE-preprocess/1.0 (+research; respectful)"


@dataclass
class GuidanceEntry:
    guidance_ref: str
    guidance_id: str
    title: str
    url: str
    publication_date: str
    guidance_types: list[str]
    path_and_query: str


def fetch_published_page(page: int, page_size: int, timeout: int) -> dict:
    query = f"?pa={page}&ps={page_size}"
    url = f"{PUBLISHED_URL}{query}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    match = NEXT_DATA_RE.search(html)
    if not match:
        raise RuntimeError(f"Could not parse __NEXT_DATA__ from {url}")
    payload = json.loads(match.group(1))
    return payload["props"]["pageProps"]["results"]


def iter_published_guidance(
    page_size: int = 500,
    timeout: int = 60,
    delay_seconds: float = 0.3,
    max_pages: int | None = None,
) -> Iterator[GuidanceEntry]:
    page = 1
    total: int | None = None

    while True:
        if max_pages is not None and page > max_pages:
            break

        results = fetch_published_page(page, page_size, timeout)
        if total is None:
            total = int(results["resultCount"])

        documents = results.get("documents") or []
        if not documents:
            break

        for doc in documents:
            ref = (doc.get("guidanceRef") or "").strip()
            if not ref:
                continue
            path = doc.get("pathAndQuery") or f"/guidance/{ref.lower()}"
            yield GuidanceEntry(
                guidance_ref=ref,
                guidance_id=ref.lower(),
                title=(doc.get("title") or "").strip(),
                url=(doc.get("url") or f"https://www.nice.org.uk{path}").strip(),
                publication_date=(doc.get("publicationDate") or "")[:10],
                guidance_types=list(doc.get("niceGuidanceType") or []),
                path_and_query=path,
            )

        last_result = int(results.get("lastResult") or 0)
        if last_result >= total:
            break

        page += 1
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def fetch_all_published(
    page_size: int = 500,
    timeout: int = 60,
    delay_seconds: float = 0.3,
    max_pages: int | None = None,
) -> list[GuidanceEntry]:
    seen: set[str] = set()
    out: list[GuidanceEntry] = []
    for entry in iter_published_guidance(page_size, timeout, delay_seconds, max_pages):
        if entry.guidance_id in seen:
            continue
        seen.add(entry.guidance_id)
        out.append(entry)
    return out
