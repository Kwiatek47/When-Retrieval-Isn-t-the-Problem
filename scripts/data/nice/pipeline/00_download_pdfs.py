#!/usr/bin/env python3
"""
Download NICE guidance PDFs.

Modes:
  --all-published     fetch catalog + download every entry that has a PDF
  --from-catalog FILE use ids from guidance_catalog.jsonl
  --ids TA1158        manual subset
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dataclasses import asdict

from catalog import fetch_all_published
from common import INTERIM_DIR, RAW_PDF_DIR, repo_relative

GUIDANCE_PAGE = "https://www.nice.org.uk/guidance/{guidance_id}"
PDF_HREF_RE = re.compile(
    r'href="(/guidance/[^"]+?-pdf-\d+)"[^>]*>\s*Download guidance \(PDF\)',
    re.IGNORECASE,
)
PDF_HREF_FALLBACK_RE = re.compile(r'href="(/guidance/[^"]+?-pdf-\d+)"', re.IGNORECASE)
USER_AGENT = "Medical-RAG-NICE-preprocess/1.0 (+research; respectful)"
DEFAULT_CATALOG = INTERIM_DIR / "guidance_catalog.jsonl"
DEFAULT_REPORT = INTERIM_DIR / "pdf_download_report.json"


def normalize_guidance_id(raw: str) -> str:
    token = raw.strip().lower().replace(" ", "")
    if not re.fullmatch(r"[a-z][a-z0-9]*\d+", token):
        raise ValueError(f"Unrecognized NICE id format: {raw!r}")
    return token


def fetch_html(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def resolve_pdf_url(guidance_id: str, timeout: int) -> str:
    page_url = GUIDANCE_PAGE.format(guidance_id=guidance_id)
    html = fetch_html(page_url, timeout=timeout)
    match = PDF_HREF_RE.search(html) or PDF_HREF_FALLBACK_RE.search(html)
    if not match:
        raise RuntimeError(f"No PDF link on {page_url}")
    return f"https://www.nice.org.uk{match.group(1)}"


def download_file(url: str, dest: Path, timeout: int, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"Downloaded file is not a PDF: {url}")
            dest.write_bytes(data)
            return
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 6.0))
    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from last_error


def load_ids(ids: list[str], ids_file: Path | None) -> list[str]:
    out: list[str] = []
    for item in ids:
        out.append(normalize_guidance_id(item))
    if ids_file:
        for line in ids_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(normalize_guidance_id(line))
    seen: set[str] = set()
    unique: list[str] = []
    for gid in out:
        if gid not in seen:
            seen.add(gid)
            unique.append(gid)
    return unique


def load_catalog(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gid = (row.get("guidance_id") or row.get("guidance_ref", "")).strip().lower()
            if gid and gid not in seen:
                seen.add(gid)
                ids.append(gid)
    return ids


def ensure_catalog(
    catalog_path: Path,
    page_size: int,
    catalog_delay: float,
    timeout: int,
    max_pages: int | None,
) -> Path:
    if catalog_path.exists():
        return catalog_path
    print(f"Catalog missing; fetching into {repo_relative(catalog_path)} ...")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    entries = fetch_all_published(
        page_size=page_size,
        timeout=timeout,
        delay_seconds=catalog_delay,
        max_pages=max_pages,
    )
    with catalog_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    print(f"Catalog: {len(entries)} published guidance entries")
    return catalog_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NICE guidance PDFs into data/raw/nice/pdf/")
    parser.add_argument("--ids", nargs="*", default=[], help="NICE ids, e.g. TA1158 NG28")
    parser.add_argument("--ids-file", type=Path, help="Text file with one id per line")
    parser.add_argument(
        "--all-published",
        action="store_true",
        help="Download PDFs for all published guidance (uses guidance_catalog.jsonl)",
    )
    parser.add_argument(
        "--from-catalog",
        type=Path,
        default=None,
        help=f"JSONL catalog path (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Catalog path for --all-published")
    parser.add_argument("--fetch-catalog-only", action="store_true", help="Only refresh catalog, no PDFs")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--catalog-delay", type=float, default=0.3)
    parser.add_argument("--max-catalog-pages", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=RAW_PDF_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between PDF downloads (seconds)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if PDF already exists")
    parser.add_argument("--limit", type=int, default=None, help="Max PDFs to attempt (testing)")
    parser.add_argument("--dry-run", action="store_true", help="List targets only")
    args = parser.parse_args()

    catalog_path = args.catalog.resolve()

    if args.all_published or args.fetch_catalog_only:
        ensure_catalog(
            catalog_path,
            page_size=args.page_size,
            catalog_delay=args.catalog_delay,
            timeout=args.timeout,
            max_pages=args.max_catalog_pages,
        )
        if args.fetch_catalog_only:
            print("Catalog ready.")
            return

    guidance_ids: list[str] = []
    if args.all_published:
        guidance_ids = load_catalog(catalog_path)
    elif args.from_catalog:
        guidance_ids = load_catalog(args.from_catalog.resolve())
    else:
        try:
            guidance_ids = load_ids(args.ids, args.ids_file)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    if not guidance_ids:
        print(
            "Provide --all-published, --from-catalog, --ids, or --ids-file",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.limit is not None:
        guidance_ids = guidance_ids[: args.limit]

    out_dir = args.out_dir.resolve()
    print(f"Targets: {len(guidance_ids)} guidance document(s)")

    if args.dry_run:
        for gid in guidance_ids[:20]:
            print(f"  {gid}")
        if len(guidance_ids) > 20:
            print(f"  ... and {len(guidance_ids) - 20} more")
        return

    ok = 0
    skipped = 0
    no_pdf = 0
    failed = 0
    failures: list[dict] = []

    for i, gid in enumerate(guidance_ids, start=1):
        dest = out_dir / f"{gid}.pdf"
        if args.skip_existing and dest.exists():
            skipped += 1
            if i % 100 == 0 or i == len(guidance_ids):
                print(f"[{i}/{len(guidance_ids)}] skipped={skipped} ok={ok} no_pdf={no_pdf} failed={failed}")
            continue

        try:
            pdf_url = resolve_pdf_url(gid, timeout=args.timeout)
            download_file(pdf_url, dest, timeout=args.timeout)
            ok += 1
            if i % 25 == 0 or i == len(guidance_ids):
                print(f"[{i}/{len(guidance_ids)}] ok={ok} skipped={skipped} no_pdf={no_pdf} failed={failed}")
        except RuntimeError as exc:
            if "No PDF link" in str(exc):
                no_pdf += 1
                failures.append({"guidance_id": gid, "error": str(exc), "kind": "no_pdf"})
            else:
                failed += 1
                failures.append({"guidance_id": gid, "error": str(exc), "kind": "error"})
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead) as exc:
            failed += 1
            failures.append({"guidance_id": gid, "error": str(exc), "kind": "network"})

        if args.delay > 0 and i < len(guidance_ids):
            time.sleep(args.delay)

    report = {
        "total_targets": len(guidance_ids),
        "downloaded": ok,
        "skipped_existing": skipped,
        "no_pdf": no_pdf,
        "failed": failed,
        "failures_sample": failures[:200],
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"Done. downloaded={ok} skipped={skipped} no_pdf={no_pdf} failed={failed} "
        f"report={repo_relative(args.report.resolve())}"
    )


if __name__ == "__main__":
    main()
