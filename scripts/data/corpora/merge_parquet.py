from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple chunk parquet files into one.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input parquet paths.")
    parser.add_argument("--output", required=True, help="Merged parquet output path.")
    args = parser.parse_args()

    tables: list[pa.Table] = []
    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            raise RuntimeError(f"Input parquet does not exist: {path}")
        tables.append(pq.read_table(path))

    if not tables:
        raise RuntimeError("No input parquet files provided.")

    merged = pa.concat_tables(tables, promote_options="default")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, output, compression="zstd")
    print(f"Wrote {merged.num_rows} rows to {output}")


if __name__ == "__main__":
    main()
