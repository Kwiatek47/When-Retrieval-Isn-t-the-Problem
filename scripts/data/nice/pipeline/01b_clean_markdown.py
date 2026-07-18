#!/usr/bin/env python3
"""
Remove NICE boilerplate from parsed markdown before chunking.

Reads:  data/interim/nice/markdown/*.md
Writes: data/interim/nice/markdown_clean/*.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import INTERIM_DIR, INTERIM_MARKDOWN_DIR, repo_relative

INTERIM_MARKDOWN_CLEAN_DIR = INTERIM_DIR / "markdown_clean"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
NICE_REPEATED_TITLE_RE = re.compile(r"^[^#|]{25,}\(([A-Z]{1,5}\d+)\)\s*$")

# Sections dropped entirely (case-insensitive substring match on heading title).
DROP_SECTION_HEADINGS = (
    "your responsibility",
    "contents",
    "evaluation committee members",
    "nice project team",
    "antimicrobials evaluation committee",
    "committee members",
    "implementation",
    "recommendations for research",
    "recommendations for data collection",
    "antimicrobial surveillance",
)

# Sections always kept (clinical core); if empty, only DROP list applies.
KEEP_SECTION_HINTS = (
    "recommendation",
    "what this means in practice",
    "why the committee made",
    "information about",
    "marketing authorisation",
    "dosage",
    "commercial arrangement",
)

LINE_DROP_PATTERNS = [
    re.compile(r"^©\s*NICE\b", re.I),
    re.compile(r"^Page\s+\d+\s+of", re.I),
    re.compile(r"^www\.nice\.org\.uk/", re.I),
    re.compile(r"^\*\*==>\s*picture\b", re.I),
    re.compile(r"^Subject to Notice of rights", re.I),
    re.compile(r"^All problems \(adverse events\)", re.I),
    re.compile(r"^Commissioners and/or providers", re.I),
    re.compile(r"^Yellow Card Scheme", re.I),
    re.compile(r"^environmentally sustainable", re.I),
    re.compile(r"^\d{1,4}$"),
]


def heading_level(line: str) -> int | None:
    m = HEADING_RE.match(line.strip())
    return len(m.group(1)) if m else None


def heading_title(line: str) -> str:
    m = HEADING_RE.match(line.strip())
    if not m:
        return ""
    return m.group(2).strip().strip("*").strip()


def should_drop_section(title: str) -> bool:
    t = title.lower()
    if any(k in t for k in KEEP_SECTION_HINTS):
        return False
    return any(k in t for k in DROP_SECTION_HEADINGS)


def drop_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if any(p.search(s) for p in LINE_DROP_PATTERNS):
        return True
    return bool(NICE_REPEATED_TITLE_RE.match(s))


def normalize_heading(line: str) -> str:
    m = HEADING_RE.match(line.strip())
    if not m:
        return line
    title = m.group(2).strip().strip("*").strip()
    return f"{m.group(1)} {title}"


def clean_markdown(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    skip_until_level: int | None = None

    for line in lines:
        if drop_line(line):
            continue

        level = heading_level(line)
        if level is not None:
            title = heading_title(line)
            if skip_until_level is not None:
                if level <= skip_until_level:
                    skip_until_level = None
                else:
                    continue
            if should_drop_section(title):
                skip_until_level = level
                continue

        if skip_until_level is not None:
            continue

        # Drop markdown table-of-contents rows (dotted leaders).
        if re.match(r"^\|.*\.{3,}.*\|", line):
            continue

        out.append(normalize_heading(line))

    # Collapse excessive blank lines.
    cleaned: list[str] = []
    blank_run = 0
    for line in out:
        if not line.strip():
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip() + "\n"


def display_path(path: Path) -> str:
    try:
        return repo_relative(path)
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean NICE markdown boilerplate")
    parser.add_argument("--in-dir", type=Path, default=INTERIM_MARKDOWN_DIR)
    parser.add_argument("--out-dir", type=Path, default=INTERIM_MARKDOWN_CLEAN_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown in {in_dir}")

    ok = 0
    skipped = 0
    for path in files:
        dest = out_dir / path.name
        if args.skip_existing and dest.exists():
            skipped += 1
            continue
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_markdown(raw)
        dest.write_text(cleaned, encoding="utf-8")
        ok += 1

    print(
        f"Cleaned {ok} files -> {display_path(out_dir)} "
        f"(skipped={skipped}, input={len(files)})"
    )


if __name__ == "__main__":
    main()
