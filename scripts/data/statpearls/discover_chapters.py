from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

STATPEARLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STATPEARLS_DIR))

from ncbi_client import esearch_chapter_uids, esummary_chapters, extract_nbk_id, load_config  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/raw/statpearls/chapter_manifest.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover StatPearls chapter NBK IDs via NCBI Books E-utilities.")
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=200, help="Maximum chapter records to write.")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    config = load_config()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen_nbk: set[str] = set()
    written = 0
    retstart = 0

    with args.out.open("w", encoding="utf-8") as manifest_file:
        while written < args.limit:
            page_size = min(args.batch_size, args.limit - written)
            uids, total = esearch_chapter_uids(config=config, retstart=retstart, retmax=page_size)
            if not uids:
                break

            for row in esummary_chapters(uids, config=config):
                if row.get("RType", "").lower() != "chapter":
                    continue
                nbk_id = extract_nbk_id(row.get("RID", ""))
                if not nbk_id or nbk_id in seen_nbk:
                    continue
                seen_nbk.add(nbk_id)
                record = {
                    "nbk_id": nbk_id,
                    "uid": row.get("uid", ""),
                    "title": row.get("Title", ""),
                    "pub_date": row.get("PubDate", ""),
                    "book": row.get("Book", "statpearls"),
                }
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.limit:
                    break

            retstart += len(uids)
            if retstart >= total:
                break

    print(f"Wrote {written} chapter records to {args.out} (unique NBK IDs).")


if __name__ == "__main__":
    main()
