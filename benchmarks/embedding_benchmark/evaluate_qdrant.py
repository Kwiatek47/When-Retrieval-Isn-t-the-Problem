from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from embedding_benchmark.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QUERY_SET_PATH,
    DEFAULT_REGISTRY_PATH,
    append_jsonl,
    collection_name_for,
    get_model_spec,
    human_seconds,
    load_registry,
    model_output_dir,
    monotonic_seconds,
    now_iso,
    ProgressTracker,
    read_jsonl,
    setup_logging,
    write_json_atomic,
)
from embedding_benchmark.model_adapters import create_adapter


def main() -> None:
    args = _parse_args()
    registry = load_registry(args.registry)
    spec = get_model_spec(registry, args.model)
    output_dir = model_output_dir(args.output_root, spec.slug, args.precision)
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "logs" / "evaluate_qdrant.log")
    collection = args.collection or collection_name_for(spec, prefix=args.collection_prefix, dtype="f16")
    queries = _load_queries(args.query_set, limit=args.limit)
    detailed_path = eval_dir / "qdrant_results.jsonl"
    summary_path = eval_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"

    if summary_path.exists() and detailed_path.exists() and not args.force:
        log.info("Evaluation already exists, skipping: %s", summary_path)
        return
    if args.force and detailed_path.exists():
        detailed_path.unlink()

    adapter = create_adapter(spec, device=args.device, precision=args.precision)
    client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)
    started_at = monotonic_seconds()
    per_query = []
    progress = ProgressTracker(total=len(queries), label=f"evaluate:{spec.slug}", logger=log, log_every_seconds=10.0)
    try:
        for index, query_case in enumerate(queries, start=1):
            query_started = monotonic_seconds()
            query_vector = adapter.encode_queries([str(query_case["query"])], batch_size=1)[0]
            points = _query_qdrant(
                client=client,
                collection=collection,
                vector_name=args.dense_vector_name,
                query_vector=query_vector,
                limit=args.top_k,
            )
            elapsed = monotonic_seconds() - query_started
            result = _score_query(query_case, points, elapsed_seconds=elapsed)
            per_query.append(result)
            append_jsonl(detailed_path, result)
            log.info(
                "Evaluated query %d/%d id=%s hit@10=%s mrr=%.4f elapsed=%s",
                index,
                len(queries),
                query_case["query_id"],
                result["metrics"].get("hit_at_10"),
                result["metrics"].get("mrr_at_10", 0.0),
                human_seconds(elapsed),
            )
            progress.update(index, extra=f"last_query={query_case['query_id']}")
    finally:
        client.close()
        adapter.close()

    elapsed_total = monotonic_seconds() - started_at
    progress.update(len(per_query), force=True, extra="complete")
    summary = _summarize(per_query, model=spec.slug, collection=collection, elapsed_seconds=elapsed_total)
    write_json_atomic(summary_path, summary)
    _write_markdown_report(eval_dir / "report.md", summary, per_query)
    append_jsonl(
        metrics_path,
        {
            "event": "evaluation_complete",
            "created_at": now_iso(),
            "model": spec.slug,
            "collection": collection,
            "query_count": len(per_query),
            "elapsed_seconds": elapsed_total,
            "summary_path": str(summary_path),
            "details_path": str(detailed_path),
            "metrics": summary["metrics_overall"],
        },
    )
    log.info("Evaluation complete model=%s queries=%d elapsed=%s", spec.slug, len(per_query), human_seconds(elapsed_total))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Qdrant benchmark collection against the static query set.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--query-set", type=Path, default=DEFAULT_QUERY_SET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--qdrant-timeout", type=float, default=120.0)
    parser.add_argument("--collection")
    parser.add_argument("--collection-prefix", default="pubmed_v1")
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_queries(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    required = {"query_id", "query", "expected_chunk_ids", "expected_pmids", "expected_doc_ids"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"Query {row.get('query_id')} missing fields: {', '.join(sorted(missing))}")
    return rows


def _query_qdrant(
    *,
    client: QdrantClient,
    collection: str,
    vector_name: str,
    query_vector: np.ndarray,
    limit: int,
) -> list[Any]:
    result = client.query_points(
        collection_name=collection,
        query=query_vector.astype(float).tolist(),
        using=vector_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(getattr(result, "points", result))


def _score_query(query_case: dict[str, Any], points: list[Any], *, elapsed_seconds: float) -> dict[str, Any]:
    expected_chunks = {str(item) for item in query_case.get("expected_chunk_ids", [])}
    expected_pmids = {str(item) for item in query_case.get("expected_pmids", [])}
    expected_doc_ids = {str(item) for item in query_case.get("expected_doc_ids", [])}

    results = []
    chunk_hit_rank = None
    pmid_hit_rank = None
    doc_hit_rank = None
    for rank, point in enumerate(points, start=1):
        payload = dict(getattr(point, "payload", None) or {})
        chunk_id = str(payload.get("chunkId") or payload.get("chunk_id") or "")
        pmid = str(payload.get("pmid") or "")
        doc_id = str(payload.get("documentId") or payload.get("document_id") or "")
        score = float(getattr(point, "score", 0.0) or 0.0)
        is_chunk_hit = chunk_id in expected_chunks
        is_pmid_hit = pmid in expected_pmids if pmid else False
        is_doc_hit = doc_id in expected_doc_ids if doc_id else False
        if is_chunk_hit and chunk_hit_rank is None:
            chunk_hit_rank = rank
        if is_pmid_hit and pmid_hit_rank is None:
            pmid_hit_rank = rank
        if is_doc_hit and doc_hit_rank is None:
            doc_hit_rank = rank
        results.append(
            {
                "rank": rank,
                "score": score,
                "hit_chunk": is_chunk_hit,
                "hit_pmid": is_pmid_hit,
                "hit_doc": is_doc_hit,
                "chunk_id": chunk_id,
                "pmid": pmid,
                "doc_id": doc_id,
                "title": str(payload.get("title") or ""),
                "year": payload.get("year"),
                "journal": str(payload.get("journal") or ""),
                "publication_types": payload.get("publicationTypes") or [],
                "text_preview": str(payload.get("text") or "")[:900],
                "payload": payload,
            }
        )

    metrics = {}
    for k in (1, 5, 10, 20):
        metrics[f"chunk_hit_at_{k}"] = int(chunk_hit_rank is not None and chunk_hit_rank <= k)
        metrics[f"pmid_hit_at_{k}"] = int(pmid_hit_rank is not None and pmid_hit_rank <= k)
        metrics[f"doc_hit_at_{k}"] = int(doc_hit_rank is not None and doc_hit_rank <= k)
    metrics["mrr_at_10"] = 0.0 if pmid_hit_rank is None or pmid_hit_rank > 10 else 1.0 / pmid_hit_rank
    metrics["ndcg_at_10"] = _dcg_at_10(results) / 1.0
    metrics["hit_at_10"] = metrics["pmid_hit_at_10"]

    return {
        "query_id": query_case["query_id"],
        "query": query_case["query"],
        "language": query_case.get("language"),
        "category": query_case.get("category"),
        "source": query_case.get("source"),
        "expected": {
            "chunk_ids": sorted(expected_chunks),
            "pmids": sorted(expected_pmids),
            "doc_ids": sorted(expected_doc_ids),
            "title": query_case.get("expected_title"),
            "reason": query_case.get("reason"),
        },
        "metrics": metrics,
        "hit_ranks": {
            "chunk": chunk_hit_rank,
            "pmid": pmid_hit_rank,
            "doc": doc_hit_rank,
        },
        "elapsed_seconds": elapsed_seconds,
        "qdrant_results": results,
    }


def _dcg_at_10(results: list[dict[str, Any]]) -> float:
    dcg = 0.0
    for item in results[:10]:
        relevance = 1.0 if item["hit_pmid"] or item["hit_doc"] else 0.0
        if relevance <= 0:
            continue
        dcg += relevance / np.log2(item["rank"] + 1)
    return float(dcg)


def _summarize(
    per_query: list[dict[str, Any]],
    *,
    model: str,
    collection: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    metric_keys = sorted({key for row in per_query for key in row["metrics"]})
    overall = {
        key: sum(float(row["metrics"].get(key, 0.0)) for row in per_query) / max(len(per_query), 1)
        for key in metric_keys
    }
    by_category = _group_summary(per_query, "category", metric_keys)
    by_language = _group_summary(per_query, "language", metric_keys)
    failures = [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "expected_pmids": row["expected"]["pmids"],
            "top_results": [
                {
                    "rank": item["rank"],
                    "score": item["score"],
                    "pmid": item["pmid"],
                    "title": item["title"],
                }
                for item in row["qdrant_results"][:5]
            ],
        }
        for row in per_query
        if not row["metrics"].get("pmid_hit_at_10")
    ]
    return {
        "created_at": now_iso(),
        "model": model,
        "collection": collection,
        "query_count": len(per_query),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": human_seconds(elapsed_seconds),
        "metrics_overall": overall,
        "metrics_by_category": by_category,
        "metrics_by_language": by_language,
        "failure_count_at_10": len(failures),
        "failures_at_10": failures,
        "justification": _justification(overall, by_category, by_language),
    }


def _group_summary(rows: list[dict[str, Any]], field: str, metric_keys: list[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(field) or "unknown"), []).append(row)
    return {
        group: {
            "count": len(items),
            "metrics": {
                key: sum(float(row["metrics"].get(key, 0.0)) for row in items) / max(len(items), 1)
                for key in metric_keys
            },
        }
        for group, items in sorted(groups.items())
    }


def _justification(overall: dict[str, float], by_category: dict[str, Any], by_language: dict[str, Any]) -> list[str]:
    lines = [
        f"Primary retrieval quality is judged by PMID-level recall@10={overall.get('pmid_hit_at_10', 0.0):.4f} and MRR@10={overall.get('mrr_at_10', 0.0):.4f}.",
        f"Chunk-level recall@10={overall.get('chunk_hit_at_10', 0.0):.4f}; PMID-level is more tolerant because most abstracts have one or two chunks.",
    ]
    for language, payload in by_language.items():
        metrics = payload["metrics"]
        lines.append(
            f"Language={language}: pmid_hit@10={metrics.get('pmid_hit_at_10', 0.0):.4f}, mrr@10={metrics.get('mrr_at_10', 0.0):.4f} over {payload['count']} queries."
        )
    weak_categories = [
        category
        for category, payload in by_category.items()
        if payload["metrics"].get("pmid_hit_at_10", 0.0) < overall.get("pmid_hit_at_10", 0.0)
    ]
    if weak_categories:
        lines.append("Below-average categories: " + ", ".join(weak_categories) + ".")
    return lines


def _write_markdown_report(path: Path, summary: dict[str, Any], per_query: list[dict[str, Any]]) -> None:
    lines = [
        f"# Embedding Benchmark: {summary['model']}",
        "",
        f"- Collection: `{summary['collection']}`",
        f"- Queries: `{summary['query_count']}`",
        f"- Elapsed: `{summary['elapsed_human']}`",
        "",
        "## Overall Metrics",
        "",
    ]
    for key, value in summary["metrics_overall"].items():
        lines.append(f"- `{key}`: `{value:.4f}`")
    lines.extend(["", "## Justification", ""])
    for item in summary["justification"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Sample Failures At 10", ""])
    for failure in summary["failures_at_10"][:20]:
        lines.append(f"- `{failure['query_id']}` {failure['query']} expected={failure['expected_pmids']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
