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
        summary_path = model_output_dir(args.output_root, model_slug, args.precision) / "pubmedqa_pipeline" / "summary.json"
        if not summary_path.exists():
            rows.append({"model": model_slug, "status": "missing_summary", "summary_path": str(summary_path)})
            continue
        summary = load_json(summary_path)
        metrics = summary.get("metrics") or {}
        rows.append(
            {
                "model": model_slug,
                "status": "ok",
                "case_count": summary.get("case_count"),
                "label_accuracy": metrics.get("label_accuracy"),
                "label_macro_f1": metrics.get("label_macro_f1"),
                "case_pass_rate": metrics.get("case_pass_rate"),
                "final_pmid_hit_at_1": metrics.get("final_pmid_hit_at_1"),
                "final_pmid_hit_at_3": metrics.get("final_pmid_hit_at_3"),
                "final_pmid_hit_at_10": metrics.get("final_pmid_hit_at_10"),
                "final_pmid_mrr_at_10": metrics.get("final_pmid_mrr_at_10"),
                "rrf_pmid_hit_at_10": metrics.get("rrf_pmid_hit_at_10"),
                "metadata_boosted_pmid_hit_at_10": metrics.get("metadata_boosted_pmid_hit_at_10"),
                "summary_path": str(summary_path),
            }
        )
    rows = sorted(rows, key=_rank_key, reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "pubmedqa_pipeline_comparison.csv"
    json_path = args.out_dir / "pubmedqa_pipeline_comparison.json"
    markdown_path = args.out_dir / "pubmedqa_pipeline_comparison.md"
    _write_csv(csv_path, rows)
    write_json_atomic(json_path, {"created_at": now_iso(), "rows": rows})
    _write_markdown(markdown_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PubMedQA pipeline embedding evaluation summaries.")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "reports" / "pubmedqa_pipeline")
    return parser.parse_args()


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    if row.get("status") != "ok":
        return (-1.0, -1.0, -1.0, -1.0)
    return (
        _float(row.get("label_macro_f1")),
        _float(row.get("label_accuracy")),
        _float(row.get("final_pmid_hit_at_10")),
        _float(row.get("final_pmid_mrr_at_10")),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "status",
        "case_count",
        "label_accuracy",
        "label_macro_f1",
        "case_pass_rate",
        "final_pmid_hit_at_1",
        "final_pmid_hit_at_3",
        "final_pmid_hit_at_10",
        "final_pmid_mrr_at_10",
        "rrf_pmid_hit_at_10",
        "metadata_boosted_pmid_hit_at_10",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PubMedQA Pipeline Embedding Comparison",
        "",
        "| Model | Status | Label Acc | Macro F1 | Case Pass | Final PMID Hit@10 | Final PMID MRR@10 | RRF PMID Hit@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {status} | {label_accuracy} | {label_macro_f1} | {case_pass_rate} | {final_pmid_hit_at_10} | {final_pmid_mrr_at_10} | {rrf_pmid_hit_at_10} |".format(
                model=row.get("model"),
                status=row.get("status"),
                label_accuracy=_fmt(row.get("label_accuracy")),
                label_macro_f1=_fmt(row.get("label_macro_f1")),
                case_pass_rate=_fmt(row.get("case_pass_rate")),
                final_pmid_hit_at_10=_fmt(row.get("final_pmid_hit_at_10")),
                final_pmid_mrr_at_10=_fmt(row.get("final_pmid_mrr_at_10")),
                rrf_pmid_hit_at_10=_fmt(row.get("rrf_pmid_hit_at_10")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
