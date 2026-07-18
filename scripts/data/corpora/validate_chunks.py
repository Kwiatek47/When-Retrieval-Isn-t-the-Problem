#!/usr/bin/env python3
"""Quick validation of a corpus chunk parquet (or shard directory).

Prints row count, schema and duplicate ``chunk_id`` count, plus empty-text
diagnostics.  Exits with non-zero when required columns are missing or empty
counts are non-zero.  Intended as a pre-merge smoke check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {"chunk_id", "text"}


def load_table(path: Path):
    if path.is_dir():
        dataset = ds.dataset(path, format="parquet")
        return dataset.to_table()
    return pq.read_table(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a corpus chunk parquet.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: path does not exist: {args.path}", file=sys.stderr)
        return 2

    table = load_table(args.path)
    print(f"path: {args.path}")
    print(f"rows: {table.num_rows}")
    print(f"columns: {table.column_names}")

    missing = REQUIRED_COLUMNS - set(table.column_names)
    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}", file=sys.stderr)
        return 3

    chunk_ids = table.column("chunk_id").to_pylist()
    empty_ids = sum(1 for value in chunk_ids if not str(value or "").strip())
    duplicates = len(chunk_ids) - len(set(chunk_ids))

    texts = table.column("text").to_pylist()
    empty_texts = sum(1 for value in texts if not str(value or "").strip())

    print(f"empty_chunk_id: {empty_ids}")
    print(f"duplicate_chunk_id: {duplicates}")
    print(f"empty_text: {empty_texts}")

    if empty_ids or duplicates or empty_texts:
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
