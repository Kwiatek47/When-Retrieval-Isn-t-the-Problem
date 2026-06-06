from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ERROR_TYPES = [
    "underconfidence",
    "negation_miss",
    "scope_confusion",
    "aim_vs_result",
    "hedging_language",
    "other",
]


def main() -> None:
    args = _parse_args()
    report = _read_json(args.report)
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError(f"Expected `cases` list in {args.report}.")

    error_items = [_error_item(case) for case in cases if not bool(case.get("label_pass"))]
    pair_counts = Counter(f"{item['true_label']}->{item['predicted_label']}" for item in error_items)
    data = {
        "benchmark": "PubMedQA official PQA-L 500",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(args.report),
        "error_type_taxonomy": ERROR_TYPES,
        "manual_fields": ["error_type", "evidence_excerpt", "difficulty", "notes"],
        "selection": "label_errors_only",
        "summary": {
            "error_count": len(error_items),
            "predicted_vs_true": dict(sorted(pair_counts.items())),
        },
        "errors": error_items,
    }
    _write_json(args.out, data)
    print(f"Wrote error-analysis seed: {args.out} ({len(error_items)} errors)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed manual error analysis from an official PQA-L 500 eval report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}.")
    return data


def _error_item(case: dict[str, Any]) -> dict[str, Any]:
    top_source = {}
    final_documents = case.get("final_documents")
    if isinstance(final_documents, list) and final_documents:
        top_source = final_documents[0] if isinstance(final_documents[0], dict) else {}

    return {
        "id": str(case.get("id") or ""),
        "true_label": case.get("expected_label"),
        "predicted_label": case.get("predicted_label"),
        "retrieval_status": case.get("retrieval_status"),
        "error_bucket": case.get("error_bucket"),
        "source_hit_at_1": case.get("source_hit_at_1"),
        "source_hit_at_3": case.get("source_hit_at_3"),
        "citation_pass": case.get("citation_pass"),
        "hallucination_rate": case.get("hallucination_rate"),
        "evidence_decision": {
            "status": (case.get("evidence_decision") or {}).get("status"),
            "method": (case.get("evidence_decision") or {}).get("method"),
            "answer_label": (case.get("evidence_decision") or {}).get("answer_label"),
            "confidence": (case.get("evidence_decision") or {}).get("confidence"),
            "rationale": (case.get("evidence_decision") or {}).get("rationale"),
            "notes": (case.get("evidence_decision") or {}).get("notes") or [],
        },
        "top_source": {
            "pmid": top_source.get("pmid"),
            "documentId": top_source.get("documentId"),
            "chunkId": top_source.get("chunkId"),
            "title": top_source.get("title"),
            "score": top_source.get("score"),
            "evidenceScore": top_source.get("evidenceScore"),
            "queryTermCoverage": top_source.get("queryTermCoverage"),
            "benchmarkFullEvidence": top_source.get("benchmarkFullEvidence"),
        },
        "top_candidate_pmids": _candidate_pmids(case.get("metadata_boosted_candidates")),
        "evidence_excerpt": top_source.get("content_preview"),
        "answer": case.get("answer"),
        "error_type": None,
        "difficulty": None,
        "notes": None,
    }


def _candidate_pmids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    pmids = []
    for item in value[:10]:
        if not isinstance(item, dict):
            continue
        pmid = item.get("pmid")
        if pmid is not None:
            pmids.append(str(pmid))
    return pmids


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
