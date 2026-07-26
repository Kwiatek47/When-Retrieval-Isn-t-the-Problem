from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_TERMS = (
    "headache",
    "migraine",
    "cephalalgia",
    "tension-type",
    "computer vision syndrome",
    "eye strain",
    "eyestrain",
    "screen",
    "visual display",
    "neck pain",
)


def main() -> None:
    args = _parse_args()
    terms = tuple(term.lower() for term in args.term if term.strip())
    if not terms:
        raise RuntimeError("At least one search term is required.")

    source = pq.ParquetFile(args.chunks)
    writer: pq.ParquetWriter | None = None
    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    try:
        for row_group_index in range(source.num_row_groups):
            table = source.read_row_group(row_group_index)
            rows = []
            for row in table.to_pylist():
                haystack = f"{row.get('title') or ''} {row.get('text') or ''}".lower()
                if any(term in haystack for term in terms):
                    rows.append(row)
                    written += 1
                    if args.limit and written >= args.limit:
                        break

            if rows:
                out_table = pa.Table.from_pylist(rows, schema=table.schema)
                if writer is None:
                    writer = pq.ParquetWriter(args.out, out_table.schema)
                writer.write_table(out_table)
                print(f"Matched {written} chunks after row group {row_group_index}.")

            if args.limit and written >= args.limit:
                break
    finally:
        if writer is not None:
            writer.close()

    if written == 0:
        raise RuntimeError(f"No chunks matched terms: {', '.join(terms)}")
    print(f"Wrote {written} chunks to {args.out}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a topic-specific chunks parquet subset.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/topic_chunks.parquet"))
    parser.add_argument("--term", action="append", default=list(DEFAULT_TERMS))
    parser.add_argument("--limit", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
