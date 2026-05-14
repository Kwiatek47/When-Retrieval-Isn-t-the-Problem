from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "embeddings" / "chunks_inspection_report.md"

REQUIRED_COLUMNS = {"chunk_id", "text"}


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.chunks)
    report = _inspect(rows=rows, chunks_path=args.chunks)
    _write_report(args.out, report)
    print(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect chunks.parquet before embedding.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Chunks parquet file does not exist: {path}")
    table = pq.read_table(path)
    missing = sorted(REQUIRED_COLUMNS - set(table.column_names))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")
    return table.to_pylist()


def _inspect(*, rows: list[dict[str, Any]], chunks_path: Path) -> str:
    chunk_ids = [_clean_str(row.get("chunk_id")) for row in rows]
    texts = [_clean_str(row.get("text")) for row in rows]
    non_empty_texts = [text for text in texts if text]
    duplicate_count = len(chunk_ids) - len(set(chunk_ids))
    empty_chunk_id_count = sum(1 for chunk_id in chunk_ids if not chunk_id)
    empty_text_count = sum(1 for text in texts if not text)
    word_counts = [len(text.split()) for text in non_empty_texts]

    lines = [
        "# Chunks Inspection Report",
        "",
        f"- Chunks file: `{chunks_path}`",
        f"- Rows: {len(rows)}",
        f"- Unique `chunk_id`: {len(set(chunk_ids))}",
        f"- Duplicate `chunk_id`: {duplicate_count}",
        f"- Empty `chunk_id`: {empty_chunk_id_count}",
        f"- Empty `text`: {empty_text_count}",
    ]
    if word_counts:
        lines.extend(
            [
                f"- Mean text words: {mean(word_counts):.1f}",
                f"- Median text words: {median(word_counts):.1f}",
                f"- Min text words: {min(word_counts)}",
                f"- Max text words: {max(word_counts)}",
            ]
        )

    status = "PASS" if duplicate_count == 0 and empty_chunk_id_count == 0 and empty_text_count == 0 else "FAIL"
    lines.extend(["", f"Status: **{status}**", ""])
    return "\n".join(lines)


def _write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
