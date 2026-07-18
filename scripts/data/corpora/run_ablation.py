#!/usr/bin/env python3
"""Orchestrate the corpus ablation study.

For each configuration in ``ABLATION_CONFIGS`` build ``data/processed/chunks.parquet``
using only the listed corpora, then invoke ``scripts/rag/03_evaluate_retrieval.py``
on the standard PubMedQA / NICE / StatPearls samples.  The final JSON summary
is written to ``reports/corpus_ablation_summary.json`` and rendered to
``reports/corpus_ablation.md``.

This script is intentionally minimal: it shells out to the existing scripts
instead of importing them, so that each configuration runs in a clean process.
Building the merged parquet and reindexing Qdrant is skipped when ``--dry-run``
is passed - useful when the datasets are not yet materialised locally.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MERGER = PROJECT_ROOT / "scripts" / "data" / "corpora" / "build_processed_chunks.py"
REPORTS_DIR = PROJECT_ROOT / "reports"


ABLATION_CONFIGS = [
    {
        "name": "pubmed_only",
        "corpora": ["pubmed_reviews_v1"],
        "corpus_versions": ["pubmed-reviews-v1"],
    },
    {
        "name": "pubmed_plus_nice",
        "corpora": ["pubmed_reviews_v1", "nice_guidelines_v1"],
        "corpus_versions": ["pubmed-reviews-v1", "nice-guidelines-v1"],
    },
    {
        "name": "pubmed_plus_statpearls",
        "corpora": ["pubmed_reviews_v1", "statpearls_v1"],
        "corpus_versions": ["pubmed-reviews-v1", "statpearls-v1"],
    },
    {
        "name": "all",
        "corpora": ["pubmed_reviews_v1", "nice_guidelines_v1", "statpearls_v1"],
        "corpus_versions": ["pubmed-reviews-v1", "nice-guidelines-v1", "statpearls-v1"],
    },
]


BENCHMARKS = [
    {
        "id": "pubmedqa",
        "dataset": "data/benchmarks/retrieval/eval_retrieval_sample.json",
        "top_k": "5,10",
    },
    {
        "id": "nice",
        "dataset": "data/benchmarks/retrieval/eval_nice_guidelines_sample.json",
        "top_k": "5,10",
    },
    {
        "id": "statpearls",
        "dataset": "data/benchmarks/retrieval/eval_statpearls_sample.json",
        "top_k": "5,10",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Corpus ablation runner.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and skip execution.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Restrict to a subset of ablation config names.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=REPORTS_DIR / "corpus_ablation.md",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPORTS_DIR / "corpus_ablation_summary.json",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip index rebuild; assumes each ablation index is already built.",
    )
    args = parser.parse_args()

    configs = ABLATION_CONFIGS
    if args.configs:
        configs = [c for c in ABLATION_CONFIGS if c["name"] in args.configs]
        if not configs:
            print(f"No matching configs: {args.configs}", file=sys.stderr)
            return 2

    results: list[dict] = []
    for cfg in configs:
        print(f"\n=== ablation: {cfg['name']} ({', '.join(cfg['corpora'])}) ===", flush=True)
        merged_path = PROJECT_ROOT / "data" / "processed" / f"chunks_{cfg['name']}.parquet"
        manifest_path = PROJECT_ROOT / "data" / "processed" / f"manifest_{cfg['name']}.json"

        merge_cmd = [
            sys.executable,
            str(MERGER),
            "--corpora",
            *cfg["corpora"],
            "--out-chunks",
            str(merged_path),
            "--manifest-out",
            str(manifest_path),
        ]
        entry: dict = {
            "name": cfg["name"],
            "corpora": cfg["corpora"],
            "corpus_versions": cfg["corpus_versions"],
            "merged_chunks": str(merged_path),
            "benchmarks": {},
        }
        if args.dry_run:
            print(f"[dry-run] would run: {' '.join(merge_cmd)}")
        else:
            subprocess.run(merge_cmd, check=True)

        if args.skip_index:
            print("[skip-index] skipping Qdrant reindex; assumes index is already prepared")
        else:
            print(
                "[note] this script does not automate Qdrant reindex; run "
                "scripts/rag/01_build_index.py manually for each ablation before evaluating.",
                flush=True,
            )

        for bench in BENCHMARKS:
            bench_out = REPORTS_DIR / f"ablation_{cfg['name']}_{bench['id']}.json"
            bench_md = REPORTS_DIR / f"ablation_{cfg['name']}_{bench['id']}.md"
            bench_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "rag" / "03_evaluate_retrieval.py"),
                "--dataset",
                str(PROJECT_ROOT / bench["dataset"]),
                "--top-k",
                bench["top_k"],
                "--json-out",
                str(bench_out),
                "--md-out",
                str(bench_md),
            ]
            entry["benchmarks"][bench["id"]] = {
                "dataset": bench["dataset"],
                "json_out": str(bench_out),
                "md_out": str(bench_md),
            }
            if args.dry_run:
                print(f"[dry-run] would run: {' '.join(bench_cmd)}")
            else:
                subprocess.run(bench_cmd, check=False)

        results.append(entry)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "results": results,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _write_markdown(args.report_md, summary, dry_run=args.dry_run)
    print(f"\nWrote {args.summary_json}")
    print(f"Wrote {args.report_md}")
    return 0


def _write_markdown(path: Path, summary: dict, *, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corpus ablation report",
        "",
        f"Generated: {summary['created_at']}",
        "",
        ("(dry-run: benchmark results below are placeholders until the ablation is actually executed)" if dry_run else ""),
        "",
        "## Configurations",
        "",
        "| Name | Corpora | Merged chunks |",
        "| --- | --- | --- |",
    ]
    for entry in summary["results"]:
        lines.append(f"| {entry['name']} | {', '.join(entry['corpora'])} | `{entry['merged_chunks']}` |")

    lines += [
        "",
        "## Metrics",
        "",
        "Populate this section from `reports/ablation_<name>_<bench>.json` after running:",
        "",
        "```",
        "python scripts/data/corpora/run_ablation.py",
        "```",
        "",
        "Expected columns per benchmark: MRR, Recall@5, Recall@10, nDCG@10, source_hit_at_3.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
