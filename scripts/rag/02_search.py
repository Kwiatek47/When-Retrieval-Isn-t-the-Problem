from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "reports" / "search_smoke_result.json"
REQUIRED_RESULT_FIELDS = ("chunk_id", "score", "title", "text", "source")


def main() -> None:
    args = _parse_args()
    response = _search(
        api_url=args.api_url,
        search_path=args.search_path,
        query=args.query,
        top_k=args.top_k,
    )
    _validate_response(response, require_results=args.require_results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(response, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the public /search endpoint.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--api-url", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--search-path", default=os.getenv("RAG_SEARCH_PATH", "/search"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-results", action="store_true")
    return parser.parse_args()


def _search(*, api_url: str, search_path: str, query: str, top_k: int) -> dict[str, Any]:
    normalized_path = "/" + search_path.strip("/")
    url = f"{api_url.rstrip('/')}{normalized_path}?{urlencode({'q': query, 'top_k': top_k})}"
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def _validate_response(response: dict[str, Any], *, require_results: bool) -> None:
    results = response.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Search response does not contain a `results` list.")
    if require_results and not results:
        raise RuntimeError("Search returned no results.")

    for index, item in enumerate(results):
        missing = [field for field in REQUIRED_RESULT_FIELDS if field not in item]
        if missing:
            raise RuntimeError(f"Result {index} is missing fields: {', '.join(missing)}")
        if not str(item.get("chunk_id") or "").strip():
            raise RuntimeError(f"Result {index} has an empty chunk_id.")
        if not str(item.get("text") or "").strip():
            raise RuntimeError(f"Result {index} has empty text.")


if __name__ == "__main__":
    main()
