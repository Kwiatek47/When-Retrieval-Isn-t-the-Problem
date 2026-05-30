from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from embedding_benchmark.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_PATH,
    load_json,
    load_registry,
    model_output_dir,
    model_slugs,
    now_iso,
    write_json_atomic,
)


def main() -> None:
    args = _parse_args()
    registry = load_registry(args.registry)
    selected = model_slugs(registry, args.models)
    rows = []
    for model_slug in selected:
        summary_path = model_output_dir(args.output_root, model_slug, args.precision) / "evaluation" / "summary.json"
        if not summary_path.exists():
            rows.append({"model": model_slug, "status": "missing_summary", "summary_path": str(summary_path)})
            continue
        summary = load_json(summary_path)
        metrics = summary.get("metrics_overall") or {}
        rows.append(
            {
                "model": model_slug,
                "status": "ok",
                "query_count": summary.get("query_count"),
                "pmid_hit_at_1": metrics.get("pmid_hit_at_1"),
                "pmid_hit_at_5": metrics.get("pmid_hit_at_5"),
                "pmid_hit_at_10": metrics.get("pmid_hit_at_10"),
                "pmid_hit_at_20": metrics.get("pmid_hit_at_20"),
                "chunk_hit_at_10": metrics.get("chunk_hit_at_10"),
                "mrr_at_10": metrics.get("mrr_at_10"),
                "ndcg_at_10": metrics.get("ndcg_at_10"),
                "failure_count_at_10": summary.get("failure_count_at_10"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "summary_path": str(summary_path),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "model_comparison.csv"
    json_path = args.out_dir / "model_comparison.json"
    markdown_path = args.out_dir / "model_comparison.md"
    _write_csv(csv_path, rows)
    write_json_atomic(json_path, {"created_at": now_iso(), "rows": rows})
    _write_markdown(markdown_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-model benchmark summaries into comparison files.")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "reports")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "status",
        "query_count",
        "pmid_hit_at_1",
        "pmid_hit_at_5",
        "pmid_hit_at_10",
        "pmid_hit_at_20",
        "chunk_hit_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "failure_count_at_10",
        "elapsed_seconds",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Embedding Model Comparison",
        "",
        "| Model | Status | PMID Hit@10 | MRR@10 | nDCG@10 | Chunk Hit@10 | Failures@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {status} | {pmid_hit_at_10} | {mrr_at_10} | {ndcg_at_10} | {chunk_hit_at_10} | {failure_count_at_10} |".format(
                model=row.get("model"),
                status=row.get("status"),
                pmid_hit_at_10=_fmt(row.get("pmid_hit_at_10")),
                mrr_at_10=_fmt(row.get("mrr_at_10")),
                ndcg_at_10=_fmt(row.get("ndcg_at_10")),
                chunk_hit_at_10=_fmt(row.get("chunk_hit_at_10")),
                failure_count_at_10=row.get("failure_count_at_10", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()

