#!/usr/bin/env python3
"""Sanity gate #4 (design doc, Faza 0 step 4): before writing any SLD prompts,
check that ``extract_stats_profile`` actually finds something on real PQA-L
abstracts. Statistical triggers built on regex extraction that mostly misses
are triggers that need redesigning, not prompts built on top of them.

Coverage counts a document as "hit" if extract_stats_profile finds at least
one marker anywhere in it. Reports the same breakdown per marker category,
plus the fraction of documents with at least one RESULTS-tagged sentence
(what findings_auditor needs to have anything to cite).

Usage:
  python3 scripts/agents/sld_stats_coverage.py
  python3 scripts/agents/sld_stats_coverage.py --corpus path/to/corpus.json --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.sld.segmentation import (  # noqa: E402
    extract_abstract_text,
    extract_stats_profile,
    split_sentences,
    tag_sections,
)

DEFAULT_CORPUS = (
    PROJECT_ROOT / "data" / "benchmarks" / "pubmedqa" / "official_pqal_test" / "corpus.json"
)
DEFAULT_THRESHOLD = 0.60

CATEGORIES = (
    "p_values",
    "confidence_intervals",
    "percentages",
    "sample_sizes",
    "effect_sizes",
    "diagnostic_metrics",
)


def _load_docs(corpus_path: Path) -> list[dict]:
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError(f"Unsupported corpus format: {corpus_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum fraction of docs that must carry >=1 stats marker (default 0.60)",
    )
    args = parser.parse_args()

    docs = _load_docs(args.corpus)
    n = len(docs)
    if n == 0:
        print(f"No documents found in {args.corpus}")
        raise SystemExit(1)

    any_hit = 0
    any_results_sentence = 0
    per_category_hits = {category: 0 for category in CATEGORIES}

    for doc in docs:
        abstract = extract_abstract_text(str(doc.get("content") or ""))
        sentences = split_sentences(abstract)
        profile = extract_stats_profile(sentences)
        tags = tag_sections(sentences)

        if not profile.is_empty():
            any_hit += 1
        if any(tag == "RESULTS" for tag in tags.values()):
            any_results_sentence += 1
        for category in CATEGORIES:
            if getattr(profile, category):
                per_category_hits[category] += 1

    coverage = any_hit / n
    results_coverage = any_results_sentence / n

    print(f"Corpus: {args.corpus}")
    print(f"Documents: {n}")
    print(f"Any stats marker:       {any_hit}/{n} = {coverage:.3f}")
    print(f"Any RESULTS sentence:   {any_results_sentence}/{n} = {results_coverage:.3f}")
    print("Per-category coverage:")
    for category in CATEGORIES:
        hits = per_category_hits[category]
        print(f"  {category:22s} {hits:4d}/{n} = {hits / n:.3f}")

    print()
    if coverage < args.threshold:
        print(
            f"FAIL: stats marker coverage {coverage:.3f} < threshold {args.threshold:.2f}. "
            "Statistical triggers need redesigning before R1 prompts are written."
        )
        raise SystemExit(1)
    print(f"PASS: stats marker coverage {coverage:.3f} >= threshold {args.threshold:.2f}.")


if __name__ == "__main__":
    main()
