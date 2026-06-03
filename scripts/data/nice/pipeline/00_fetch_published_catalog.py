#!/usr/bin/env python3
"""Download the full list of published NICE guidance (no PDFs yet)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from catalog import fetch_all_published
from common import INTERIM_DIR, repo_relative


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch all published NICE guidance metadata")
    parser.add_argument(
        "--out",
        type=Path,
        default=INTERIM_DIR / "guidance_catalog.jsonl",
        help="Output JSONL (one guidance entry per line)",
    )
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between catalog pages")
    parser.add_argument("--max-pages", type=int, default=None, help="For testing only")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    print("Fetching published guidance catalog from nice.org.uk ...")
    entries = fetch_all_published(
        page_size=args.page_size,
        timeout=args.timeout,
        delay_seconds=args.delay,
        max_pages=args.max_pages,
    )

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    try:
        shown = repo_relative(out)
    except ValueError:
        shown = str(out)
    print(f"Wrote {len(entries)} entries -> {shown}")


if __name__ == "__main__":
    main()
