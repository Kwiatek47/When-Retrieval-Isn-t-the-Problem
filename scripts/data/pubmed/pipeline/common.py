import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass(frozen=True)
class NCBIConfig:
    email: str = os.getenv("NCBI_EMAIL", "your.email@example.com")
    api_key: str | None = os.getenv("NCBI_API_KEY") or None
    tool: str = "medical-rag-pubmed-pipeline"
    delay_seconds: float = 0.34
    timeout_seconds: float = 60.0


def read_pmids(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_hash(value: str) -> str:
    return hashlib.sha256(clean_text(value).lower().encode("utf-8")).hexdigest()


def json_loads_lenient(content: bytes) -> dict:
    try:
        return orjson.loads(content)
    except orjson.JSONDecodeError:
        return orjson.loads(content.decode("utf-8", errors="ignore"), option=orjson.OPT_STRICT_INTEGER)


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
def ncbi_get(endpoint: str, params: dict, config: NCBIConfig) -> httpx.Response:
    request_params = {
        **params,
        "tool": config.tool,
        "email": config.email,
    }
    if config.api_key:
        request_params["api_key"] = config.api_key

    time.sleep(config.delay_seconds)
    with httpx.Client(timeout=config.timeout_seconds) as client:
        response = client.get(f"{EUTILS_BASE}/{endpoint}", params=request_params)
        response.raise_for_status()
        return response

