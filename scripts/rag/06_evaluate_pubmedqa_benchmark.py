from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from statistics import mean
import sys
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys_path = str(PROJECT_ROOT)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from app.rag.answer_contract import extract_yes_no_maybe_label

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "benchmarks" / "pubmedqa" / "eval_pubmedqa_benchmark.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class PubMedQACase:
    id: str
    question: str
    expected_label: str
    relevant_document_ids: set[str]
    relevant_pmids: set[str]


@dataclass(frozen=True)
class PubMedQAResult:
    id: str
    expected_label: str
    predicted_label: str | None
    retrieval_status: str
    source_hit_at_1: bool
    source_hit_at_3: bool
    citation_pass: bool
    groundedness: float | None
    hallucination_rate: float | None
    latency_ms: float
    answer: str
    final_documents: list[dict[str, Any]]
    rrf_candidates: list[dict[str, Any]]
    metadata_boosted_candidates: list[dict[str, Any]]
    evidence_decision: dict[str, Any]

    @property
    def label_pass(self) -> bool:
        return self.predicted_label == self.expected_label

    @property
    def case_pass(self) -> bool:
        return (
            self.retrieval_status == "grounded"
            and self.source_hit_at_3
            and self.citation_pass
            and self.label_pass
        )

    @property
    def answer_quality_pass(self) -> bool:
        return self.hallucination_rate is None or self.hallucination_rate <= 0.25

    @property
    def strict_case_pass(self) -> bool:
        return self.case_pass and self.answer_quality_pass


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args.dataset)
    results = [
        _evaluate_case(
            case,
            api_url=args.api_url,
            model=args.model,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            temperature=args.temperature,
            mode=args.mode,
        )
        for case in cases
    ]
    report = _build_report(cases=cases, results=results, args=args)
    _write_json(args.json_out, report)
    _write_markdown(args.md_out, report)
    print(f"Wrote PubMedQA eval JSON report: {args.json_out}")
    print(f"Wrote PubMedQA eval Markdown report: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    label = os.getenv("PUBMEDQA_EVAL_LABEL", "pubmedqa_eval")
    parser = argparse.ArgumentParser(description="Evaluate end-to-end RAG on a PubMedQA benchmark sample.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--api-url", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.getenv("PUBMEDQA_EVAL_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b")))
    parser.add_argument("--candidate-k", type=int, default=int(os.getenv("PUBMEDQA_EVAL_CANDIDATE_K", "20")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("PUBMEDQA_EVAL_TOP_K", "3")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("PUBMEDQA_EVAL_TEMPERATURE", "0.0")))
    parser.add_argument("--mode", default=os.getenv("PUBMEDQA_EVAL_MODE", "benchmark_pqal"))
    parser.add_argument("--json-out", type=Path, default=DEFAULT_REPORT_DIR / f"{label}.json")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_REPORT_DIR / f"{label}.md")
    return parser.parse_args()


def _load_cases(path: Path) -> list[PubMedQACase]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list of PubMedQA eval cases in {path}.")

    cases = []
    for item in data:
        cases.append(
            PubMedQACase(
                id=str(item["id"]),
                question=str(item["question"]),
                expected_label=str(item["expected_label"]).lower(),
                relevant_document_ids={str(value) for value in item.get("relevant_document_ids", [])},
                relevant_pmids={str(value) for value in item.get("relevant_pmids", [])},
            )
        )
    return cases


def _evaluate_case(
    case: PubMedQACase,
    *,
    api_url: str,
    model: str,
    candidate_k: int,
    top_k: int,
    temperature: float,
    mode: str,
) -> PubMedQAResult:
    started_at = perf_counter()
    trace = _post_json(
        f"{api_url.rstrip()}/api/rag/trace",
        {
            "messages": [{"role": "user", "content": case.question}],
            "candidate_k": candidate_k,
            "top_k": top_k,
            "mode": mode,
        },
    )
    chat = _post_json(
        f"{api_url.rstrip()}/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": case.question}],
            "temperature": temperature,
            "candidate_k": candidate_k,
            "top_k": top_k,
            "mode": mode,
        },
    )
    latency_ms = (perf_counter() - started_at) * 1000.0

    final_documents = list(trace.get("final_documents") or [])
    rrf_candidates = list(trace.get("rrf_candidates") or [])
    metadata_boosted_candidates = list(trace.get("metadata_boosted_candidates") or [])
    retrieval = chat.get("retrieval") or trace.get("retrieval") or {}
    answer = str((chat.get("message") or {}).get("content") or "")
    citation_validation = chat.get("citation_validation") or {}
    answer_quality = chat.get("answer_quality") or {}
    evidence_decision = chat.get("evidence_decision") or {}

    return PubMedQAResult(
        id=case.id,
        expected_label=case.expected_label,
        predicted_label=_extract_pubmedqa_label(answer),
        retrieval_status=str(retrieval.get("status") or ""),
        source_hit_at_1=_source_hit_at_k(case, final_documents, 1),
        source_hit_at_3=_source_hit_at_k(case, final_documents, 3),
        citation_pass=bool(citation_validation.get("passed", False)),
        groundedness=_optional_float(answer_quality.get("groundedness")),
        hallucination_rate=_optional_float(answer_quality.get("hallucination_rate")),
        latency_ms=latency_ms,
        answer=answer,
        final_documents=final_documents,
        rrf_candidates=rrf_candidates,
        metadata_boosted_candidates=metadata_boosted_candidates,
        evidence_decision=evidence_decision,
    )


def _extract_pubmedqa_label(answer: str) -> str | None:
    contracted_label = extract_yes_no_maybe_label(answer)
    if contracted_label:
        return contracted_label
    normalized = re.sub(r"\[[Ss][1-9][0-9]*\]", "", answer).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    for pattern, label in (
        (r"^(answer\s*:\s*)?yes\b", "yes"),
        (r"^(answer\s*:\s*)?no\b", "no"),
        (r"^(answer\s*:\s*)?maybe\b", "maybe"),
        (r"\bthe answer is yes\b", "yes"),
        (r"\bthe answer is no\b", "no"),
        (r"\bthe answer is maybe\b", "maybe"),
        (r"\bthere is insufficient evidence\b", "maybe"),
    ):
        if re.search(pattern, normalized):
            return label
    first_hits = [
        (normalized.find(label), label)
        for label in ("yes", "no", "maybe")
        if re.search(rf"\b{label}\b", normalized)
    ]
    if not first_hits:
        return None
    return sorted(first_hits)[0][1]


def _source_hit_at_k(case: PubMedQACase, documents: list[dict[str, Any]], k: int) -> bool:
    return any(_is_relevant(case, document) for document in documents[:k])


def _is_relevant(case: PubMedQACase, document: dict[str, Any]) -> bool:
    metadata = document.get("metadata") or {}
    document_ids = {
        str(document.get("id") or ""),
        str(metadata.get("chunkId") or ""),
        str(metadata.get("chunk_id") or ""),
        str(metadata.get("documentId") or ""),
        str(metadata.get("document_id") or ""),
    }
    pmid = str(metadata.get("pmid") or "")
    return bool(document_ids & case.relevant_document_ids) or pmid in case.relevant_pmids


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_report(
    *,
    cases: list[PubMedQACase],
    results: list[PubMedQAResult],
    args: argparse.Namespace,
) -> dict[str, Any]:
    groundedness_values = [result.groundedness for result in results if result.groundedness is not None]
    hallucination_values = [result.hallucination_rate for result in results if result.hallucination_rate is not None]
    labels = sorted({case.expected_label for case in cases})
    confusion_matrix = _confusion_matrix(results)
    summary = {
        "dataset": str(args.dataset),
        "api_url": args.api_url,
        "model": args.model,
        "case_count": len(results),
        "case_pass_rate": _rate(result.case_pass for result in results),
        "strict_case_pass_rate": _rate(result.strict_case_pass for result in results),
        "label_accuracy": _rate(result.label_pass for result in results),
        "answer_quality_pass_rate": _rate(result.answer_quality_pass for result in results),
        "grounded_status_rate": _rate(result.retrieval_status == "grounded" for result in results),
        "source_hit_at_1": _rate(result.source_hit_at_1 for result in results),
        "source_hit_at_3": _rate(result.source_hit_at_3 for result in results),
        "citation_pass_rate": _rate(result.citation_pass for result in results),
        "mean_groundedness": mean(groundedness_values) if groundedness_values else None,
        "mean_hallucination_rate": mean(hallucination_values) if hallucination_values else None,
        "mean_latency_ms": mean(result.latency_ms for result in results),
        "labels": {
            label: {
                "count": sum(1 for case in cases if case.expected_label == label),
                "accuracy": _rate(
                    result.label_pass
                    for result in results
                    if result.expected_label == label
                ),
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix,
        "prediction_pairs": _prediction_pairs(results),
        "error_buckets": _counts(_error_bucket(result) for result in results if not result.case_pass),
        "answer_quality_fail_count": sum(1 for result in results if not result.answer_quality_pass),
        "predicted_labels": {
            label: sum(1 for result in results if result.predicted_label == label)
            for label in ("yes", "no", "maybe")
        },
        "evidence_decision": {
            "statuses": {
                status: sum(1 for result in results if result.evidence_decision.get("status") == status)
                for status in ("supported", "refuted", "uncertain", "insufficient", "skipped")
            },
            "labels": {
                label: sum(1 for result in results if result.evidence_decision.get("answer_label") == label)
                for label in ("yes", "no", "maybe")
            },
            "methods": _counts(str(result.evidence_decision.get("method") or "") for result in results),
        },
        "config": {
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "mode": args.mode,
            "RAG_EVIDENCE_FILTER_ENABLED": os.getenv("RAG_EVIDENCE_FILTER_ENABLED", ""),
            "RAG_EVIDENCE_JUDGE_METHOD": os.getenv("RAG_EVIDENCE_JUDGE_METHOD", ""),
            "RAG_EVIDENCE_CLASSIFIER_ENABLED": os.getenv("RAG_EVIDENCE_CLASSIFIER_ENABLED", ""),
            "RAG_EVIDENCE_CLASSIFIER_MODEL_PATH": os.getenv("RAG_EVIDENCE_CLASSIFIER_MODEL_PATH", ""),
            "RAG_EVIDENCE_CLASSIFIER_FAST_THRESHOLD": os.getenv("RAG_EVIDENCE_CLASSIFIER_FAST_THRESHOLD", ""),
            "RAG_EVIDENCE_CLASSIFIER_HINT_THRESHOLD": os.getenv("RAG_EVIDENCE_CLASSIFIER_HINT_THRESHOLD", ""),
            "RAG_ANSWER_QUALITY_GATE_ENABLED": os.getenv("RAG_ANSWER_QUALITY_GATE_ENABLED", ""),
            "RAG_CORPUS_VERSION": os.getenv("RAG_CORPUS_VERSION", ""),
            "RAG_RETRIEVER": os.getenv("RAG_RETRIEVER", ""),
            "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", ""),
        },
    }
    return {
        "summary": summary,
        "cases": [_case_to_dict(result) for result in results],
    }


def _rate(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _case_to_dict(result: PubMedQAResult) -> dict[str, Any]:
    label_pass = result.label_pass
    error_bucket = None if result.case_pass else _error_bucket(result)
    return {
        "id": result.id,
        "expected_label": result.expected_label,
        "predicted_label": result.predicted_label,
        "label_pass": label_pass,
        "retrieval_status": result.retrieval_status,
        "source_hit_at_1": result.source_hit_at_1,
        "source_hit_at_3": result.source_hit_at_3,
        "citation_pass": result.citation_pass,
        "groundedness": result.groundedness,
        "hallucination_rate": result.hallucination_rate,
        "latency_ms": result.latency_ms,
        "case_pass": result.case_pass,
        "strict_case_pass": result.strict_case_pass,
        "answer_quality_pass": result.answer_quality_pass,
        "error_bucket": error_bucket,
        "answer": result.answer,
        "evidence_decision": result.evidence_decision,
        "final_documents": [_debug_document(document) for document in result.final_documents],
        "rrf_candidates": [_debug_document(document) for document in result.rrf_candidates[:10]],
        "metadata_boosted_candidates": [
            _debug_document(document) for document in result.metadata_boosted_candidates[:10]
        ],
    }


def _debug_document(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    return {
        "rank": document.get("rank"),
        "id": document.get("id"),
        "title": document.get("title"),
        "source": document.get("source"),
        "score": document.get("score"),
        "pmid": metadata.get("pmid"),
        "documentId": metadata.get("documentId"),
        "chunkId": metadata.get("chunkId") or metadata.get("chunk_id"),
        "parentChunkId": metadata.get("parentChunkId") or metadata.get("parent_chunk_id"),
        "evidenceScore": metadata.get("evidenceScore"),
        "queryTermCoverage": metadata.get("queryTermCoverage"),
        "rrfScore": metadata.get("rrfScore"),
        "preMetadataBoostScore": metadata.get("preMetadataBoostScore"),
        "metadataBoost": metadata.get("metadataBoost"),
        "matchedQueryCount": metadata.get("matchedQueryCount"),
        "benchmarkFullEvidence": metadata.get("benchmarkFullEvidence"),
        "fullEvidenceChars": metadata.get("fullEvidenceChars"),
        "selectedContentChars": metadata.get("selectedContentChars"),
        "content_preview": document.get("content_preview"),
        "metadata": metadata,
    }


def _confusion_matrix(results: list[PubMedQAResult]) -> dict[str, dict[str, int]]:
    labels = ("yes", "no", "maybe", "none")
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for result in results:
        gold = result.expected_label if result.expected_label in labels else "none"
        pred = result.predicted_label if result.predicted_label in labels else "none"
        matrix[gold][pred] += 1
    return matrix


def _prediction_pairs(results: list[PubMedQAResult]) -> dict[str, int]:
    return _counts(
        f"{result.expected_label}->{result.predicted_label or 'none'}"
        for result in results
    )


def _error_bucket(result: PubMedQAResult) -> str:
    if not result.source_hit_at_3:
        return "retrieval_miss"
    if result.retrieval_status != "grounded":
        return f"retrieval_status_{result.retrieval_status or 'unknown'}"
    if not result.citation_pass:
        return "citation_fail"
    if not result.label_pass:
        decision_label = result.evidence_decision.get("answer_label")
        if decision_label == result.predicted_label:
            return "evidence_judge_label_error"
        return "answer_label_extraction_or_writer_error"
    if not result.answer_quality_pass:
        return "answer_quality_fail"
    return "other_case_fail"


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "# PubMedQA Benchmark Eval",
        "",
        "## Summary",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Model: `{summary['model']}`",
        f"- Cases: {summary['case_count']}",
        f"- Case pass rate: {summary['case_pass_rate']:.3f}",
        f"- Strict case pass rate: {summary['strict_case_pass_rate']:.3f}",
        f"- Label accuracy: {summary['label_accuracy']:.3f}",
        f"- Answer quality pass rate: {summary['answer_quality_pass_rate']:.3f}",
        f"- Grounded status rate: {summary['grounded_status_rate']:.3f}",
        f"- Source hit@1: {summary['source_hit_at_1']:.3f}",
        f"- Source hit@3: {summary['source_hit_at_3']:.3f}",
        f"- Citation pass rate: {summary['citation_pass_rate']:.3f}",
        f"- Mean groundedness: {_format_optional(summary['mean_groundedness'])}",
        f"- Mean hallucination rate: {_format_optional(summary['mean_hallucination_rate'])}",
        f"- Mean latency: {summary['mean_latency_ms']:.1f} ms",
        "",
        "## Labels",
        "",
        "| Label | Count | Accuracy |",
        "|---|---:|---:|",
    ]
    for label, metrics in summary["labels"].items():
        lines.append(f"| {label} | {metrics['count']} | {metrics['accuracy']:.3f} |")

    lines.extend(["", "## Confusion Matrix", "", "| True \\ Pred | yes | no | maybe | none |", "|---|---:|---:|---:|---:|"])
    for true_label, row in summary["confusion_matrix"].items():
        lines.append(
            f"| {true_label} | {row.get('yes', 0)} | {row.get('no', 0)} | "
            f"{row.get('maybe', 0)} | {row.get('none', 0)} |"
        )

    lines.extend(["", "## Error Buckets", "", "| Bucket | Count |", "|---|---:|"])
    for bucket, count in summary["error_buckets"].items():
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Evidence Judge",
            "",
            "| Type | Value | Count |",
            "|---|---|---:|",
        ]
    )
    for label, count in summary["predicted_labels"].items():
        lines.append(f"| predicted_label | {label} | {count} |")
    for status, count in summary["evidence_decision"]["statuses"].items():
        lines.append(f"| evidence_status | {status} | {count} |")
    for label, count in summary["evidence_decision"]["labels"].items():
        lines.append(f"| evidence_label | {label} | {count} |")
    for method, count in summary["evidence_decision"]["methods"].items():
        lines.append(f"| evidence_method | {method} | {count} |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Expected | Predicted | Hit@1 | Hit@3 | Citation | Pass | Top source |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for case in report["cases"]:
        top = case["final_documents"][0] if case["final_documents"] else {}
        top_source = f"{top.get('pmid') or top.get('documentId') or '-'} / {top.get('title') or '-'}"
        lines.append(
            "| "
            f"`{case['id']}` | "
            f"{case['expected_label']} | "
            f"{case['predicted_label']} | "
            f"{case['source_hit_at_1']} | "
            f"{case['source_hit_at_3']} | "
            f"{case['citation_pass']} | "
            f"{case['case_pass']} | "
            f"{top_source} |"
        )

    lines.extend(["", "## Config", ""])
    for key, value in summary["config"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc


if __name__ == "__main__":
    main()
