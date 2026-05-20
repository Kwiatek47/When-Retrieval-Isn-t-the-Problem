from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "eval_rag_english_real_sources.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_status: str
    relevant_document_ids: set[str]
    relevant_pmids: set[str]
    expected_answer_terms: list[str]
    forbidden_answer_terms: list[str]

    @property
    def expects_sources(self) -> bool:
        return bool(self.relevant_document_ids or self.relevant_pmids)


@dataclass(frozen=True)
class CaseResult:
    id: str
    question: str
    expected_status: str
    retrieval_status: str
    provider: str
    latency_ms: float
    final_documents: list[dict[str, Any]]
    source_hit_at_1: bool | None
    source_hit_at_3: bool | None
    status_pass: bool
    citation_pass: bool
    groundedness: float | None
    hallucination_rate: float | None
    forbidden_terms_present: list[str]
    expected_terms_missing: list[str]
    answer: str

    @property
    def case_pass(self) -> bool:
        if self.expected_status == "grounded":
            return (
                self.status_pass
                and self.source_hit_at_3 is True
                and self.citation_pass
                and not self.expected_terms_missing
                and not self.forbidden_terms_present
                and (self.hallucination_rate is None or self.hallucination_rate <= 0.25)
            )
        return (
            self.status_pass
            and self.citation_pass
            and not self.forbidden_terms_present
        )


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args.dataset)
    if not cases:
        raise RuntimeError(f"No eval cases found in {args.dataset}.")

    results = [
        _evaluate_case(
            case,
            api_url=args.api_url,
            model=args.model,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        for case in cases
    ]
    report = _build_report(cases=cases, results=results, args=args)
    _write_json(args.json_out, report)
    _write_markdown(args.md_out, report)
    print(f"Wrote RAG eval JSON report: {args.json_out}")
    print(f"Wrote RAG eval Markdown report: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    label = os.getenv("RAG_EVAL_LABEL", "rag_eval")
    parser = argparse.ArgumentParser(description="Evaluate end-to-end RAG trace and chat behavior.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--api-url", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.getenv("RAG_EVAL_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b")))
    parser.add_argument("--candidate-k", type=int, default=int(os.getenv("RAG_EVAL_CANDIDATE_K", "20")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_EVAL_TOP_K", "3")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("RAG_EVAL_TEMPERATURE", "0.0")))
    parser.add_argument("--json-out", type=Path, default=DEFAULT_REPORT_DIR / f"{label}.json")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_REPORT_DIR / f"{label}.md")
    return parser.parse_args()


def _load_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list of eval cases in {path}.")

    cases = []
    for item in data:
        cases.append(
            EvalCase(
                id=str(item["id"]),
                question=str(item["question"]),
                expected_status=str(item.get("expected_status") or "grounded"),
                relevant_document_ids={str(value) for value in item.get("relevant_document_ids", [])},
                relevant_pmids={str(value) for value in item.get("relevant_pmids", [])},
                expected_answer_terms=[str(value) for value in item.get("expected_answer_terms", [])],
                forbidden_answer_terms=[str(value) for value in item.get("forbidden_answer_terms", [])],
            )
        )
    return cases


def _evaluate_case(
    case: EvalCase,
    *,
    api_url: str,
    model: str,
    candidate_k: int,
    top_k: int,
    temperature: float,
) -> CaseResult:
    started_at = perf_counter()
    trace = _post_json(
        f"{api_url.rstrip()}/api/rag/trace",
        {
            "messages": [{"role": "user", "content": case.question}],
            "candidate_k": candidate_k,
            "top_k": top_k,
        },
    )
    chat = _post_json(
        f"{api_url.rstrip()}/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": case.question}],
            "temperature": temperature,
        },
    )
    latency_ms = (perf_counter() - started_at) * 1000.0

    retrieval = chat.get("retrieval") or trace.get("retrieval") or {}
    retrieval_status = str(retrieval.get("status") or "")
    final_documents = list(trace.get("final_documents") or [])
    answer = str((chat.get("message") or {}).get("content") or "")
    citation_validation = chat.get("citation_validation") or {}
    answer_quality = chat.get("answer_quality") or {}

    return CaseResult(
        id=case.id,
        question=case.question,
        expected_status=case.expected_status,
        retrieval_status=retrieval_status,
        provider=str(retrieval.get("provider") or trace.get("provider") or ""),
        latency_ms=latency_ms,
        final_documents=final_documents,
        source_hit_at_1=_source_hit_at_k(case, final_documents, 1),
        source_hit_at_3=_source_hit_at_k(case, final_documents, 3),
        status_pass=_status_pass(case.expected_status, retrieval_status),
        citation_pass=bool(citation_validation.get("passed", False)),
        groundedness=_optional_float(answer_quality.get("groundedness")),
        hallucination_rate=_optional_float(answer_quality.get("hallucination_rate")),
        forbidden_terms_present=_terms_present(answer, case.forbidden_answer_terms),
        expected_terms_missing=_terms_missing(answer, case.expected_answer_terms)
        if retrieval_status == "grounded"
        else [],
        answer=answer,
    )


def _status_pass(expected_status: str, actual_status: str) -> bool:
    if expected_status == "low_evidence":
        return actual_status in {"low_evidence", "no_sources"}
    return actual_status == expected_status


def _source_hit_at_k(case: EvalCase, documents: list[dict[str, Any]], k: int) -> bool | None:
    if not case.expects_sources:
        return None
    return any(_is_relevant(case, document) for document in documents[:k])


def _is_relevant(case: EvalCase, document: dict[str, Any]) -> bool:
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


def _terms_present(answer: str, terms: list[str]) -> list[str]:
    lower_answer = answer.lower()
    return [term for term in terms if term.lower() in lower_answer]


def _terms_missing(answer: str, terms: list[str]) -> list[str]:
    lower_answer = answer.lower()
    return [term for term in terms if term.lower() not in lower_answer]


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_report(
    *,
    cases: list[EvalCase],
    results: list[CaseResult],
    args: argparse.Namespace,
) -> dict[str, Any]:
    answerable = [result for result in results if result.expected_status == "grounded"]
    negative = [result for result in results if result.expected_status == "low_evidence"]
    groundedness_values = [result.groundedness for result in results if result.groundedness is not None]
    hallucination_values = [result.hallucination_rate for result in results if result.hallucination_rate is not None]

    summary = {
        "dataset": str(args.dataset),
        "api_url": args.api_url,
        "model": args.model,
        "case_count": len(results),
        "answerable_count": len(answerable),
        "negative_count": len(negative),
        "case_pass_rate": _rate(result.case_pass for result in results),
        "status_accuracy": _rate(result.status_pass for result in results),
        "answerable_grounded_rate": _rate(result.retrieval_status == "grounded" for result in answerable),
        "negative_refusal_rate": _rate(result.retrieval_status in {"low_evidence", "no_sources"} for result in negative),
        "source_hit_at_1": _rate(result.source_hit_at_1 is True for result in answerable),
        "source_hit_at_3": _rate(result.source_hit_at_3 is True for result in answerable),
        "citation_pass_rate": _rate(result.citation_pass for result in results),
        "forbidden_term_violation_rate": _rate(bool(result.forbidden_terms_present) for result in results),
        "expected_term_missing_rate": _rate(bool(result.expected_terms_missing) for result in answerable),
        "mean_groundedness": mean(groundedness_values) if groundedness_values else None,
        "mean_hallucination_rate": mean(hallucination_values) if hallucination_values else None,
        "mean_latency_ms": mean(result.latency_ms for result in results),
        "config": {
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "RAG_EVIDENCE_FILTER_ENABLED": os.getenv("RAG_EVIDENCE_FILTER_ENABLED", ""),
            "RAG_ANSWER_QUALITY_GATE_ENABLED": os.getenv("RAG_ANSWER_QUALITY_GATE_ENABLED", ""),
            "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", ""),
            "RAG_CORPUS_VERSION": os.getenv("RAG_CORPUS_VERSION", ""),
        },
    }
    return {
        "summary": summary,
        "cases": [_case_result_to_dict(result) for result in results],
    }


def _rate(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _case_result_to_dict(result: CaseResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "question": result.question,
        "expected_status": result.expected_status,
        "retrieval_status": result.retrieval_status,
        "provider": result.provider,
        "latency_ms": result.latency_ms,
        "source_hit_at_1": result.source_hit_at_1,
        "source_hit_at_3": result.source_hit_at_3,
        "status_pass": result.status_pass,
        "citation_pass": result.citation_pass,
        "groundedness": result.groundedness,
        "hallucination_rate": result.hallucination_rate,
        "forbidden_terms_present": result.forbidden_terms_present,
        "expected_terms_missing": result.expected_terms_missing,
        "case_pass": result.case_pass,
        "answer": result.answer,
        "final_documents": [
            {
                "rank": document.get("rank"),
                "title": document.get("title"),
                "score": document.get("score"),
                "pmid": (document.get("metadata") or {}).get("pmid"),
                "documentId": (document.get("metadata") or {}).get("documentId"),
                "evidenceScore": (document.get("metadata") or {}).get("evidenceScore"),
                "queryTermCoverage": (document.get("metadata") or {}).get("queryTermCoverage"),
                "metadataBoost": (document.get("metadata") or {}).get("metadataBoost"),
            }
            for document in result.final_documents
        ],
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "# End-to-End RAG Evaluation",
        "",
        "## Summary",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Model: `{summary['model']}`",
        f"- Cases: {summary['case_count']} ({summary['answerable_count']} answerable, {summary['negative_count']} negative)",
        f"- Case pass rate: {summary['case_pass_rate']:.3f}",
        f"- Status accuracy: {summary['status_accuracy']:.3f}",
        f"- Answerable grounded rate: {summary['answerable_grounded_rate']:.3f}",
        f"- Negative refusal rate: {summary['negative_refusal_rate']:.3f}",
        f"- Source hit@1: {summary['source_hit_at_1']:.3f}",
        f"- Source hit@3: {summary['source_hit_at_3']:.3f}",
        f"- Citation pass rate: {summary['citation_pass_rate']:.3f}",
        f"- Forbidden term violation rate: {summary['forbidden_term_violation_rate']:.3f}",
        f"- Expected term missing rate: {summary['expected_term_missing_rate']:.3f}",
        f"- Mean groundedness: {_format_optional(summary['mean_groundedness'])}",
        f"- Mean hallucination rate: {_format_optional(summary['mean_hallucination_rate'])}",
        f"- Mean latency: {summary['mean_latency_ms']:.1f} ms",
        "",
        "## Cases",
        "",
        "| Case | Expected | Actual | Hit@3 | Citations | Pass | Top source |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for case in report["cases"]:
        top = case["final_documents"][0] if case["final_documents"] else {}
        top_source = f"{top.get('pmid') or top.get('documentId') or '-'} / {top.get('title') or '-'}"
        hit_at_3 = "-" if case["source_hit_at_3"] is None else str(case["source_hit_at_3"])
        lines.append(
            "| "
            f"`{case['id']}` | "
            f"{case['expected_status']} | "
            f"{case['retrieval_status']} | "
            f"{hit_at_3} | "
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
