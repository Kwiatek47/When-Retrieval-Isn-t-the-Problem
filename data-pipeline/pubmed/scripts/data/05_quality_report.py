import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--pmids", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    con = duckdb.connect()
    pmid_count = sum(1 for line in Path(args.pmids).read_text().splitlines() if line.strip())
    document_count = con.execute(f"SELECT count(*) FROM read_parquet('{args.documents}')").fetchone()[0]
    chunk_count = con.execute(f"SELECT count(*) FROM read_parquet('{args.chunks}')").fetchone()[0]
    duplicate_chunk_ids = con.execute(
        f"SELECT count(*) FROM (SELECT chunk_id FROM read_parquet('{args.chunks}') GROUP BY chunk_id HAVING count(*) > 1)"
    ).fetchone()[0]
    empty_text = con.execute(
        f"SELECT count(*) FROM read_parquet('{args.chunks}') WHERE text IS NULL OR length(trim(text)) = 0"
    ).fetchone()[0]
    avg_text_chars = con.execute(
        f"SELECT avg(length(text)) FROM read_parquet('{args.chunks}')"
    ).fetchone()[0]

    summary = {
        "pmid_count": pmid_count,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "empty_text": empty_text,
        "avg_text_chars": round(float(avg_text_chars or 0), 2),
    }

    report = f"""# Data Quality Report

## Summary

```text
PMIDs: {pmid_count}
documents: {document_count}
chunks: {chunk_count}
duplicate chunk_id: {duplicate_chunk_ids}
empty text: {empty_text}
average text chars: {summary["avg_text_chars"]}
```

## Interpretation

The dataset is ready for embedding when `empty text` and `duplicate chunk_id` are both zero.
"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote report to {out}")
    print(f"Wrote summary JSON to {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()

