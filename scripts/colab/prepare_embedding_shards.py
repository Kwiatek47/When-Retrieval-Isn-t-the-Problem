from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    args = _parse_args()
    if not args.chunks.exists():
        raise RuntimeError(
            f"Missing source chunks parquet: {args.chunks}. "
            "Upload the full PubMed chunks file to data/processed/chunks.parquet before running the benchmark."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [args.out_dir / f"chunks_shard_{index}.parquet" for index in range(args.num_shards)]
    if all(path.exists() and path.stat().st_size > 0 for path in shard_paths) and not args.force:
        print("Embedding benchmark shards already exist:")
        for path in shard_paths:
            print(f"  {path}")
        return

    for path in shard_paths:
        if path.exists():
            path.unlink()

    parquet = pq.ParquetFile(args.chunks)
    total_rows = int(parquet.metadata.num_rows)
    if total_rows <= 0:
        raise RuntimeError(f"No rows in chunks parquet: {args.chunks}")

    shard_ranges = _shard_ranges(total_rows, args.num_shards)
    writers: list[pq.ParquetWriter | None] = [None for _ in range(args.num_shards)]
    written = [0 for _ in range(args.num_shards)]
    row_offset = 0

    try:
        for batch in parquet.iter_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch])
            batch_start = row_offset
            batch_end = row_offset + table.num_rows
            for shard_index, (start, end) in enumerate(shard_ranges):
                overlap_start = max(batch_start, start)
                overlap_end = min(batch_end, end)
                if overlap_start >= overlap_end:
                    continue
                sliced = table.slice(overlap_start - batch_start, overlap_end - overlap_start)
                if writers[shard_index] is None:
                    writers[shard_index] = pq.ParquetWriter(shard_paths[shard_index], sliced.schema, compression=args.compression)
                writers[shard_index].write_table(sliced)
                written[shard_index] += sliced.num_rows
            row_offset = batch_end
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()

    for shard_index, path in enumerate(shard_paths):
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Shard {shard_index} was not written: {path}")
        print(f"Wrote {path} rows={written[shard_index]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split data/processed/chunks.parquet into benchmark shard parquet files.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _shard_ranges(total_rows: int, num_shards: int) -> list[tuple[int, int]]:
    if num_shards <= 0:
        raise RuntimeError("--num-shards must be > 0.")
    base = total_rows // num_shards
    remainder = total_rows % num_shards
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(num_shards):
        length = base + (1 if index < remainder else 0)
        end = start + length
        ranges.append((start, end))
        start = end
    return ranges


if __name__ == "__main__":
    main()
