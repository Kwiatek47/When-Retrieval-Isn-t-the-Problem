from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH", PROJECT_ROOT / "data" / "eval_retrieval_sample.json"))
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8081").rstrip("/")
TOP_K_VALUES = tuple(
    int(item.strip())
    for item in os.getenv("EVAL_TOP_K", "1,3,5").split(",")
    if item.strip()
)
GROUNDING_OVERLAP_THRESHOLD = float(os.getenv("GROUNDING_OVERLAP_THRESHOLD", "0.35"))

_CITATION_PATTERN = re.compile(r"\[S([1-9][0-9]*)\]")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"[\w]+", re.IGNORECASE)
_STOPWORDS = {
    "and",
    "are",
    "bez",
    "dla",
    "jest",
    "lub",
    "nie",
    "oraz",
    "pod",
    "przez",
    "przy",
    "sie",
    "the",
    "with",
}


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    relevant_document_ids: set[str]
    relevant_pmids: set[str]
    answer: str | None = None


@dataclass(frozen=True)
class CaseMetrics:
    id: str
    retrieved_document_ids: list[str]
    retrieved_pmids: list[str]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    groundedness: float | None
    hallucination_rate: float | None
    unsupported_statements: list[str]


def main() -> None:
    if not TOP_K_VALUES:
        raise RuntimeError("EVAL_TOP_K must define at least one k value.")

    cases = _load_cases(EVAL_DATASET_PATH)
    max_k = max(TOP_K_VALUES)
    results = [_evaluate_case(case, limit=max_k) for case in cases]
    summary = _summarize(results)

    output = {
        "dataset": str(EVAL_DATASET_PATH),
        "embedding_service_url": EMBEDDING_SERVICE_URL,
        "top_k": list(TOP_K_VALUES),
        "summary": summary,
        "cases": [
            {
                "id": result.id,
                "retrieved_document_ids": result.retrieved_document_ids,
                "retrieved_pmids": result.retrieved_pmids,
                "recall_at_k": {str(key): value for key, value in result.recall_at_k.items()},
                "precision_at_k": {str(key): value for key, value in result.precision_at_k.items()},
                "groundedness": result.groundedness,
                "hallucination_rate": result.hallucination_rate,
                "unsupported_statements": result.unsupported_statements,
            }
            for result in results
        ],
    }
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


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
                answer=str(item["answer"]) if item.get("answer") else None,
            )
        )
    return cases


def _evaluate_case(case: EvalCase, *, limit: int) -> CaseMetrics:
    documents = _hybrid_query(case.question, limit=limit)
    retrieved_document_ids = [_document_id(document) for document in documents]
    retrieved_pmids = [_pmid(document) for document in documents if _pmid(document)]

    recall_at_k = {
        k: _recall_at_k(case, retrieved_document_ids, retrieved_pmids, k)
        for k in TOP_K_VALUES
    }
    precision_at_k = {
        k: _precision_at_k(case, retrieved_document_ids, retrieved_pmids, k)
        for k in TOP_K_VALUES
    }

    groundedness, hallucination_rate, unsupported = _answer_grounding(case.answer, documents)
    return CaseMetrics(
        id=case.id,
        retrieved_document_ids=retrieved_document_ids,
        retrieved_pmids=retrieved_pmids,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        groundedness=groundedness,
        hallucination_rate=hallucination_rate,
        unsupported_statements=unsupported,
    )


def _hybrid_query(text: str, *, limit: int) -> list[dict[str, Any]]:
    response = _request_json(
        "POST",
        f"{EMBEDDING_SERVICE_URL}/embed/hybrid/query",
        {"text": text, "limit": limit},
    )
    return list(response.get("documents", []))


def _recall_at_k(case: EvalCase, document_ids: list[str], pmids: list[str], k: int) -> float:
    relevant = _relevant_found(case, document_ids[:k], pmids[:k])
    total_relevant = max(len(case.relevant_document_ids), len(case.relevant_pmids))
    if total_relevant == 0:
        return 0.0
    return len(relevant) / total_relevant


def _precision_at_k(case: EvalCase, document_ids: list[str], pmids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = _relevant_found(case, document_ids[:k], pmids[:k])
    return len(relevant) / k


def _relevant_found(case: EvalCase, document_ids: list[str], pmids: list[str]) -> set[str]:
    found = set()
    for index, document_id in enumerate(document_ids):
        if document_id in case.relevant_document_ids:
            found.add(document_id)
            continue
        pmid = pmids[index] if index < len(pmids) else ""
        if pmid in case.relevant_pmids:
            found.add(pmid)
    return found


def _answer_grounding(answer: str | None, documents: list[dict[str, Any]]) -> tuple[float | None, float | None, list[str]]:
    if not answer:
        return None, None, []

    source_text_by_id = {
        f"S{index + 1}": str(document.get("content") or "")
        for index, document in enumerate(documents)
    }
    all_context = " ".join(source_text_by_id.values())
    statements = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_PATTERN.split(answer.strip())
        if sentence.strip()
    ]
    if not statements:
        return None, None, []

    unsupported = []
    supported_count = 0
    for statement in statements:
        cited_ids = [f"S{match}" for match in _CITATION_PATTERN.findall(statement)]
        context = " ".join(source_text_by_id[source_id] for source_id in cited_ids if source_id in source_text_by_id)
        if not context:
            context = all_context
        if _is_supported(statement, context):
            supported_count += 1
        else:
            unsupported.append(statement)

    groundedness = supported_count / len(statements)
    hallucination_rate = len(unsupported) / len(statements)
    return groundedness, hallucination_rate, unsupported


def _is_supported(statement: str, context: str) -> bool:
    statement_tokens = _content_tokens(statement)
    if not statement_tokens:
        return True
    context_tokens = _content_tokens(context)
    if not context_tokens:
        return False
    overlap = len(statement_tokens & context_tokens) / len(statement_tokens)
    return overlap >= GROUNDING_OVERLAP_THRESHOLD


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in (value.lower() for value in _TOKEN_PATTERN.findall(text))
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _document_id(document: dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    return str(metadata.get("documentId") or document.get("id") or "")


def _pmid(document: dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    return str(metadata.get("pmid") or "")


def _summarize(results: list[CaseMetrics]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count": len(results),
        "recall_at_k": {},
        "precision_at_k": {},
    }
    for k in TOP_K_VALUES:
        summary["recall_at_k"][str(k)] = _mean(result.recall_at_k[k] for result in results)
        summary["precision_at_k"][str(k)] = _mean(result.precision_at_k[k] for result in results)

    groundedness_values = [result.groundedness for result in results if result.groundedness is not None]
    hallucination_values = [
        result.hallucination_rate
        for result in results
        if result.hallucination_rate is not None
    ]
    summary["groundedness"] = _mean(groundedness_values) if groundedness_values else None
    summary["hallucination_rate"] = _mean(hallucination_values) if hallucination_values else None
    return summary


def _mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


if __name__ == "__main__":
    main()
