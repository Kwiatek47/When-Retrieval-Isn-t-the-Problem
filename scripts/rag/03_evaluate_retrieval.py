from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval" / "eval_retrieval_sample.json"
DEFAULT_JSON_REPORT_PATH = PROJECT_ROOT / "reports" / "retrieval_quality_report.json"
DEFAULT_MD_REPORT_PATH = PROJECT_ROOT / "reports" / "retrieval_quality_report.md"


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    relevant_document_ids: set[str]
    relevant_pmids: set[str]
    intent: str = "general"
    acceptable_publication_types: set[str] | None = None
    must_not_answer_without_sources: bool = True

    @property
    def relevant_count(self) -> int:
        return max(len(self.relevant_document_ids), len(self.relevant_pmids))


@dataclass(frozen=True)
class RetrievedItem:
    rank: int
    chunk_id: str
    document_id: str
    pmid: str
    score: float
    title: str
    source: str
    url: str


@dataclass(frozen=True)
class CaseMetrics:
    id: str
    question: str
    latency_ms: float
    result_count: int
    retrieved: list[RetrievedItem]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float
    first_relevant_rank: int | None


def main() -> None:
    args = _parse_args()
    top_k_values = _parse_top_k(args.top_k)
    if not top_k_values:
        raise RuntimeError("--top-k must contain at least one positive integer.")

    cases = _load_cases(args.dataset)
    if not cases:
        raise RuntimeError(f"No evaluation cases found in {args.dataset}.")

    max_k = max(top_k_values)
    results = [
        _evaluate_case(
            case,
            base_url=args.api_url,
            search_path=args.search_path,
            limit=max_k,
            top_k_values=top_k_values,
        )
        for case in cases
    ]
    report = _build_report(
        cases=cases,
        results=results,
        top_k_values=top_k_values,
        args=args,
    )

    _write_json(args.json_out, report)
    _write_markdown(args.md_out, report)
    print(f"Wrote retrieval JSON report: {args.json_out}")
    print(f"Wrote retrieval Markdown report: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality through the public /search API.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--api-url", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--top-k", default=os.getenv("EVAL_TOP_K", "5,10,50"))
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_REPORT_PATH)
    parser.add_argument("--search-path", default=os.getenv("RAG_SEARCH_PATH", "/search"))
    return parser.parse_args()


def _parse_top_k(value: str) -> tuple[int, ...]:
    top_k = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed > 0:
            top_k.add(parsed)
    return tuple(sorted(top_k))


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
                relevant_document_ids={str(value) for value in item.get("relevant_document_ids", [])},
                relevant_pmids={str(value) for value in item.get("relevant_pmids", [])},
                intent=str(item.get("intent") or "general"),
                acceptable_publication_types={
                    str(value) for value in item.get("acceptable_publication_types", [])
                }
                or None,
                must_not_answer_without_sources=bool(item.get("must_not_answer_without_sources", True)),
            )
        )
    return cases


def _evaluate_case(
    case: EvalCase,
    *,
    base_url: str,
    search_path: str,
    limit: int,
    top_k_values: tuple[int, ...],
) -> CaseMetrics:
    started_at = perf_counter()
    response = _search(base_url=base_url, search_path=search_path, query=case.question, limit=limit)
    latency_ms = (perf_counter() - started_at) * 1000.0

    retrieved = [
        _retrieved_item(index=index, item=item)
        for index, item in enumerate(response.get("results", []), start=1)
    ]

    recall_at_k = {k: _recall_at_k(case, retrieved, k) for k in top_k_values}
    precision_at_k = {k: _precision_at_k(case, retrieved, k) for k in top_k_values}
    ndcg_at_k = {k: _ndcg_at_k(case, retrieved, k) for k in top_k_values}
    first_rank = _first_relevant_rank(case, retrieved)

    return CaseMetrics(
        id=case.id,
        question=case.question,
        latency_ms=latency_ms,
        result_count=len(retrieved),
        retrieved=retrieved,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        ndcg_at_k=ndcg_at_k,
        mrr=(1.0 / first_rank) if first_rank else 0.0,
        first_relevant_rank=first_rank,
    )


def _search(*, base_url: str, search_path: str, query: str, limit: int) -> dict[str, Any]:
    normalized_path = "/" + search_path.strip("/")
    url = f"{base_url.rstrip('/')}{normalized_path}?{urlencode({'q': query, 'top_k': limit})}"
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def _retrieved_item(*, index: int, item: dict[str, Any]) -> RetrievedItem:
    metadata = item.get("metadata") or {}
    return RetrievedItem(
        rank=index,
        chunk_id=str(item.get("chunk_id") or metadata.get("chunkId") or metadata.get("chunk_id") or ""),
        document_id=str(metadata.get("documentId") or metadata.get("document_id") or ""),
        pmid=str(item.get("pmid") or metadata.get("pmid") or ""),
        score=float(item.get("score") or 0.0),
        title=str(item.get("title") or ""),
        source=str(item.get("source") or ""),
        url=str(item.get("url") or metadata.get("url") or ""),
    )


def _recall_at_k(case: EvalCase, retrieved: list[RetrievedItem], k: int) -> float:
    if case.relevant_count == 0:
        return 0.0
    return len(_relevant_found(case, retrieved[:k])) / case.relevant_count


def _precision_at_k(case: EvalCase, retrieved: list[RetrievedItem], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(_relevant_found(case, retrieved[:k])) / k


def _ndcg_at_k(case: EvalCase, retrieved: list[RetrievedItem], k: int) -> float:
    dcg = 0.0
    seen_relevant: set[str] = set()
    for index, item in enumerate(retrieved[:k], start=1):
        relevance_key = _relevance_key(case, item)
        if relevance_key and relevance_key not in seen_relevant:
            seen_relevant.add(relevance_key)
            dcg += 1.0 / math.log2(index + 1)

    ideal_relevant = min(case.relevant_count, k)
    if ideal_relevant == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_relevant + 1))
    return dcg / ideal_dcg


def _first_relevant_rank(case: EvalCase, retrieved: list[RetrievedItem]) -> int | None:
    for item in retrieved:
        if _is_relevant(case, item):
            return item.rank
    return None


def _relevant_found(case: EvalCase, retrieved: list[RetrievedItem]) -> set[str]:
    found = set()
    for item in retrieved:
        relevance_key = _relevance_key(case, item)
        if relevance_key:
            found.add(relevance_key)
    return found


def _is_relevant(case: EvalCase, item: RetrievedItem) -> bool:
    return _relevance_key(case, item) is not None


def _relevance_key(case: EvalCase, item: RetrievedItem) -> str | None:
    if item.chunk_id in case.relevant_document_ids:
        return item.chunk_id
    if item.document_id in case.relevant_document_ids:
        return item.document_id
    if item.pmid in case.relevant_pmids:
        return item.pmid
    return None


def _build_report(
    *,
    cases: list[EvalCase],
    results: list[CaseMetrics],
    top_k_values: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count": len(cases),
        "api_url": args.api_url,
        "search_path": args.search_path,
        "dataset": str(args.dataset),
        "top_k": list(top_k_values),
        "mean_latency_ms": mean(result.latency_ms for result in results),
        "mrr": mean(result.mrr for result in results),
        "recall_at_k": {},
        "precision_at_k": {},
        "ndcg_at_k": {},
        "config": {
            "RAG_CANDIDATE_K": os.getenv("RAG_CANDIDATE_K", "50"),
            "RAG_TOP_K": os.getenv("RAG_TOP_K", "5"),
            "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder"),
            "RAG_RETRIEVER": os.getenv("RAG_RETRIEVER", "embedding_service"),
            "RAG_CORPUS_VERSION": os.getenv("RAG_CORPUS_VERSION", ""),
        },
        "intents": _intent_counts(cases),
    }
    for k in top_k_values:
        summary["recall_at_k"][str(k)] = mean(result.recall_at_k[k] for result in results)
        summary["precision_at_k"][str(k)] = mean(result.precision_at_k[k] for result in results)
        summary["ndcg_at_k"][str(k)] = mean(result.ndcg_at_k[k] for result in results)

    return {
        "summary": summary,
        "cases": [_case_to_dict(result) for result in results],
    }


def _case_to_dict(result: CaseMetrics) -> dict[str, Any]:
    return {
        "id": result.id,
        "question": result.question,
        "latency_ms": result.latency_ms,
        "result_count": result.result_count,
        "recall_at_k": {str(key): value for key, value in result.recall_at_k.items()},
        "precision_at_k": {str(key): value for key, value in result.precision_at_k.items()},
        "ndcg_at_k": {str(key): value for key, value in result.ndcg_at_k.items()},
        "mrr": result.mrr,
        "first_relevant_rank": result.first_relevant_rank,
        "retrieved": [
            {
                "rank": item.rank,
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "pmid": item.pmid,
                "score": item.score,
                "title": item.title,
                "source": item.source,
                "url": item.url,
            }
            for item in result.retrieved
        ],
    }


def _intent_counts(cases: list[EvalCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.intent] = counts.get(case.intent, 0) + 1
    return counts


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    top_k_values = [int(value) for value in summary["top_k"]]

    lines = [
        "# Retrieval Quality Report",
        "",
        "## Summary",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- API: `{summary['api_url']}{summary['search_path']}`",
        f"- Cases: {summary['case_count']}",
        f"- Mean latency: {summary['mean_latency_ms']:.1f} ms",
        f"- MRR: {summary['mrr']:.4f}",
        "",
        "| k | Recall@k | Precision@k | nDCG@k |",
        "|---:|---:|---:|---:|",
    ]
    for k in top_k_values:
        lines.append(
            "| "
            f"{k} | "
            f"{summary['recall_at_k'][str(k)]:.4f} | "
            f"{summary['precision_at_k'][str(k)]:.4f} | "
            f"{summary['ndcg_at_k'][str(k)]:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Configuration",
            "",
        ]
    )
    for key, value in summary["config"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Intents", ""])
    for intent, count in sorted(summary.get("intents", {}).items()):
        lines.append(f"- `{intent}`: {count}")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | First relevant rank | MRR | Top retrieved PMID/title |",
            "|---|---:|---:|---|",
        ]
    )
    for case in report["cases"]:
        first = case["first_relevant_rank"] if case["first_relevant_rank"] is not None else "-"
        top = case["retrieved"][0] if case["retrieved"] else {}
        top_label = f"{top.get('pmid') or '-'} / {top.get('title') or '-'}"
        lines.append(f"| `{case['id']}` | {first} | {case['mrr']:.4f} | {top_label} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
