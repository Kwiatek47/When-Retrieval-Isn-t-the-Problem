import argparse
import json
from datetime import date
from pathlib import Path


QUERY = '((review[pt] OR systematic[sb]) AND hasabstract AND english[la] AND ("2021/05/10"[dp] : "2026/05/10"[dp]) NOT (retracted publication[pt] OR retraction notice[pt]))'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-name", default="pubmed_reviews_v1")
    parser.add_argument("--region", default="eu-central-1")
    args = parser.parse_args()

    manifest = {
        "dataset_name": args.dataset_name,
        "created_at": date.today().isoformat(),
        "cloud": {
            "provider": "aws_s3",
            "region": args.region,
            "bucket": args.bucket,
            "prefix": f"processed/{args.dataset_name}/",
        },
        "query": QUERY,
        "files": {
            "documents": "documents.parquet",
            "chunks": "chunks.parquet",
            "pmids": "pmids.txt",
            "quality_report": "data_quality_report.md",
            "cleaning_stats": "cleaning_stats.json",
        },
        "schema_version": "chunks_schema_v1",
        "chunking": "one_pubmed_abstract_equals_one_chunk",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote manifest to {out}")


if __name__ == "__main__":
    main()

