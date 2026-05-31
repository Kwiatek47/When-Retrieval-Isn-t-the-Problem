from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from qdrant_client import QdrantClient

from app.providers.ollama import OllamaProvider
from app.rag.answer_contract import extract_yes_no_maybe_label
from app.rag.answer_extraction import extract_answer_content
from app.rag.citation_validation import normalize_citation_format, validate_citations
from app.rag.models import PreRetrievalResult, RetrievedDocument, RetrievalResult
from app.rag.pipeline import RagPipeline
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import (
    _expanded_candidate_limit,
    _metadata_boost_documents,
    _query_texts,
    _weighted_rrf_merge,
)
from app.schemas import ChatMessage
from app.services.chat_service import benchmark_pqal_system_prompt
from embedding_benchmark.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_PATH,
    ProgressTracker,
    append_jsonl,
    collection_name_for,
    get_model_spec,
    human_seconds,
    load_json,
    load_registry,
    model_output_dir,
    monotonic_seconds,
    now_iso,
    setup_logging,
    write_json_atomic,
)
from embedding_benchmark.model_adapters import create_adapter


LABELS = ("yes", "no", "maybe")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
DEFAULT_DATASET = Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json")
DEFAULT_CORPUS = Path("data/benchmarks/pubmedqa/official_pqal_test/corpus.json")


@dataclass(frozen=True)
class PubMedQACase:
    id: str
    question: str
    expected_label: str
    relevant_document_ids: set[str]
    relevant_pmids: set[str]


class BenchmarkEmbeddingRetriever:
    provider_name = "embedding_benchmark_qdrant"

    def __init__(
        self,
        *,
        adapter: Any,
        qdrant_client: QdrantClient,
        collection_name: str,
        dense_vector_name: str,
        expansion_multiplier: int,
        expanded_limit_max: int,
    ) -> None:
        self.adapter = adapter
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.expansion_multiplier = expansion_multiplier
        self.expanded_limit_max = expanded_limit_max
        self.last_debug: dict[str, Any] = {}

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        query_texts = _query_texts(query)
        expanded_limit = _expanded_candidate_limit(
            limit,
            query,
            multiplier=self.expansion_multiplier,
            max_limit=self.expanded_limit_max,
        )
        results = []
        for query_text in query_texts:
            dense_vector = self.adapter.encode_queries([query_text], batch_size=1)[0]
            points = await self._query_qdrant(dense_vector=dense_vector, limit=expanded_limit)
            results.append([self._map_point(point) for point in points])
        merged_documents = _weighted_rrf_merge(results, query_texts=query_texts, limit=expanded_limit)
        boosted_documents = _metadata_boost_documents(merged_documents, query=query, limit=0)
        documents = boosted_documents if limit <= 0 else boosted_documents[:limit]
        self.last_debug = {
            "rrf_documents": merged_documents,
            "metadata_boosted_documents": boosted_documents,
            "expanded_limit": expanded_limit,
        }
        return RetrievalResult(
            query=query,
            documents=documents,
            provider=self.provider_name,
            debug=self.last_debug,
        )

    async def _query_qdrant(self, *, dense_vector: np.ndarray, limit: int) -> list[Any]:
        kwargs = {
            "collection_name": self.collection_name,
            "query": dense_vector.astype(float).tolist(),
            "using": self.dense_vector_name,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
        }
        call = self.qdrant_client.query_points
        if inspect.iscoroutinefunction(call):
            result = await call(**kwargs)
        else:
            result = await asyncio.to_thread(call, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        return list(getattr(result, "points", result))

    def _map_point(self, point: Any) -> RetrievedDocument:
        payload = dict(getattr(point, "payload", None) or {})
        title = str(payload.get("title") or "Untitled medical chunk")
        content = str(payload.get("text") or payload.get("content") or "")
        pmid = str(payload.get("pmid") or "")
        source = str(payload.get("source") or "qdrant")
        if source and pmid:
            source = f"{source}: PMID {pmid}"
        elif pmid:
            source = f"PMID {pmid}"
        return RetrievedDocument(
            id=str(getattr(point, "id", "") or ""),
            title=title,
            content=content,
            source=source,
            score=float(getattr(point, "score", 0.0) or 0.0),
            metadata=_metadata_from_payload(payload),
        )


async def async_main() -> None:
    args = _parse_args()
    registry = load_registry(args.registry)
    spec = get_model_spec(registry, args.model)
    output_dir = model_output_dir(args.output_root, spec.slug, args.precision)
    eval_dir = output_dir / "pubmedqa_pipeline"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "logs" / "evaluate_pubmedqa_pipeline.log")

    summary_path = eval_dir / "summary.json"
    detail_path = eval_dir / "results.jsonl"
    report_path = eval_dir / "report.md"
    metrics_path = output_dir / "metrics.jsonl"
    if summary_path.exists() and detail_path.exists() and not args.force:
        log.info("PubMedQA pipeline eval already exists, skipping: %s", summary_path)
        return
    if args.force and detail_path.exists():
        detail_path.unlink()

    cases = _load_cases(args.dataset, limit=args.limit)
    collection = args.collection or collection_name_for(spec, prefix=args.collection_prefix, dtype="f16")
    adapter = create_adapter(spec, device=args.device, precision=args.precision)
    client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)
    retriever = BenchmarkEmbeddingRetriever(
        adapter=adapter,
        qdrant_client=client,
        collection_name=collection,
        dense_vector_name=args.dense_vector_name,
        expansion_multiplier=args.retrieval_expansion_multiplier,
        expanded_limit_max=args.retrieval_expanded_limit_max,
    )
    pipeline = _build_pipeline(args, retriever=retriever)
    llm_provider = OllamaProvider(
        base_url=args.ollama_url,
        timeout=args.ollama_timeout,
        keep_alive=args.ollama_keep_alive,
        num_predict=args.ollama_num_predict,
        num_ctx=args.ollama_num_ctx,
    )

    started_at = monotonic_seconds()
    results: list[dict[str, Any]] = []
    progress = ProgressTracker(total=len(cases), label=f"pubmedqa_pipeline:{spec.slug}", logger=log, log_every_seconds=10.0)
    try:
        for index, case in enumerate(cases, start=1):
            case_started = monotonic_seconds()
            messages = [ChatMessage(role="user", content=f"Answer yes, no, or maybe based on retrieved evidence: {case.question}")]
            rag_result = await pipeline.run(
                messages=messages,
                system_prompt=benchmark_pqal_system_prompt(),
                mode="benchmark_pqal",
                candidate_limit=args.candidate_k,
                final_documents_limit=args.top_k,
            )
            llm_response = await llm_provider.chat(
                model=args.llm_model,
                messages=rag_result.messages,
                temperature=args.temperature,
            )
            answer = normalize_citation_format(extract_answer_content(llm_response.message.content))
            predicted_label = extract_yes_no_maybe_label(answer)
            citation_validation = validate_citations(answer, rag_result.citations)
            elapsed = monotonic_seconds() - case_started
            result = _score_case(
                case=case,
                rag_result=rag_result,
                answer=answer,
                raw_answer=llm_response.message.content,
                predicted_label=predicted_label,
                citation_validation=citation_validation,
                retriever_debug=retriever.last_debug,
                elapsed_seconds=elapsed,
            )
            results.append(result)
            append_jsonl(detail_path, result)
            log.info(
                "PubMedQA pipeline case %d/%d id=%s expected=%s predicted=%s status=%s pmid_hit@3=%s elapsed=%s",
                index,
                len(cases),
                case.id,
                case.expected_label,
                result["predicted_label"],
                result["retrieval_status"],
                result["retrieval_metrics"]["final_pmid_hit_at_3"],
                human_seconds(elapsed),
            )
            progress.update(index, extra=f"last_case={case.id}")
    finally:
        client.close()
        adapter.close()

    elapsed_total = monotonic_seconds() - started_at
    progress.update(len(results), force=True, extra="complete")
    summary = _summarize(
        results,
        model=spec.slug,
        collection=collection,
        dataset=args.dataset,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        elapsed_seconds=elapsed_total,
        args=args,
    )
    write_json_atomic(summary_path, summary)
    _write_markdown(report_path, summary)
    append_jsonl(
        metrics_path,
        {
            "event": "pubmedqa_pipeline_eval_complete",
            "created_at": now_iso(),
            "model": spec.slug,
            "collection": collection,
            "dataset": str(args.dataset),
            "llm_model": args.llm_model,
            "case_count": len(results),
            "elapsed_seconds": elapsed_total,
            "summary_path": str(summary_path),
            "details_path": str(detail_path),
            "metrics": summary["metrics"],
        },
    )
    log.info("PubMedQA pipeline eval complete model=%s cases=%d elapsed=%s", spec.slug, len(results), human_seconds(elapsed_total))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one embedding model inside the current PubMedQA RAG pipeline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pubmedqa-corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--llm-model", default=os.getenv("PUBMEDQA_EVAL_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("PUBMEDQA_EVAL_TEMPERATURE", "0.0")))
    parser.add_argument("--ollama-timeout", type=float, default=float(os.getenv("OLLAMA_TIMEOUT", "300")))
    parser.add_argument("--ollama-keep-alive", default=os.getenv("OLLAMA_KEEP_ALIVE", "30m"))
    parser.add_argument("--ollama-num-predict", type=int, default=int(os.getenv("OLLAMA_NUM_PREDICT", "400")))
    parser.add_argument("--ollama-num-ctx", type=int, default=int(os.getenv("OLLAMA_NUM_CTX", "2048")))
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--qdrant-timeout", type=float, default=120.0)
    parser.add_argument("--collection")
    parser.add_argument("--collection-prefix", default="pubmed_v1")
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--max-excerpt-chars", type=int, default=1600)
    parser.add_argument("--cross-encoder-model", default="ncbi/MedCPT-Cross-Encoder")
    parser.add_argument("--cross-encoder-max-length", type=int, default=512)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=8)
    parser.add_argument("--cross-encoder-device")
    parser.add_argument("--evidence-filter-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retrieval-expansion-multiplier", type=int, default=2)
    parser.add_argument("--retrieval-expanded-limit-max", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _build_pipeline(args: argparse.Namespace, *, retriever: BenchmarkEmbeddingRetriever) -> RagPipeline:
    pre_retriever = PreRetriever(
        ollama_base_url="http://127.0.0.1:11434",
        rewrite_model="",
        rewrite_timeout=1.0,
        active_corpus_version="",
    )
    post_retriever = PostRetriever(
        max_context_chars=args.max_context_chars,
        final_documents_limit=args.top_k,
        max_excerpt_chars=args.max_excerpt_chars,
        cross_encoder_model_name=args.cross_encoder_model,
        cross_encoder_max_length=args.cross_encoder_max_length,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
        cross_encoder_device=args.cross_encoder_device,
        evidence_filter_enabled=args.evidence_filter_enabled,
    )
    return RagPipeline(
        pre_retriever=pre_retriever,
        retriever=retriever,
        post_retriever=post_retriever,
        retrieval_candidate_limit=max(args.candidate_k, args.top_k),
        adaptive_retrieval_enabled=False,
        adaptive_max_rounds=0,
        pubmedqa_official_corpus_path=args.pubmedqa_corpus,
    )


def _load_cases(path: Path, *, limit: int | None) -> list[PubMedQACase]:
    data = load_json(path)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list in PubMedQA dataset: {path}")
    cases = []
    for item in data:
        label = str(item.get("expected_label") or "").strip().lower()
        if label not in LABEL_TO_ID:
            raise RuntimeError(f"Invalid expected_label={label!r} in {path}")
        question = str(item.get("benchmark_question") or item.get("question") or "").strip()
        if not question:
            raise RuntimeError(f"Missing question for case {item.get('id')}")
        cases.append(
            PubMedQACase(
                id=str(item.get("id") or item.get("pmid") or len(cases)),
                question=question,
                expected_label=label,
                relevant_document_ids={str(value) for value in item.get("relevant_document_ids", [])},
                relevant_pmids={str(value) for value in item.get("relevant_pmids", [])},
            )
        )
    return cases[:limit] if limit is not None else cases


def _score_case(
    *,
    case: PubMedQACase,
    rag_result: Any,
    answer: str,
    raw_answer: str,
    predicted_label: str | None,
    citation_validation: Any,
    retriever_debug: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    final_documents = list(rag_result.source_documents or [])
    rrf_documents = list(retriever_debug.get("rrf_documents") or [])
    metadata_boosted = list(retriever_debug.get("metadata_boosted_documents") or [])
    retrieval_metrics = {
        **_retrieval_metrics(case, final_documents, prefix="final"),
        **_retrieval_metrics(case, metadata_boosted, prefix="metadata_boosted"),
        **_retrieval_metrics(case, rrf_documents, prefix="rrf"),
    }
    return {
        "id": case.id,
        "question": case.question,
        "expected_label": case.expected_label,
        "predicted_label": predicted_label,
        "label_pass": predicted_label == case.expected_label,
        "case_pass": predicted_label == case.expected_label and bool(retrieval_metrics["final_pmid_hit_at_3"]),
        "retrieval_status": rag_result.retrieval.status if rag_result.retrieval else "skipped",
        "retrieval_metrics": retrieval_metrics,
        "answer": answer,
        "raw_answer": raw_answer,
        "citation_validation": citation_validation.model_dump(),
        "elapsed_seconds": elapsed_seconds,
        "final_documents": [_document_payload(document, rank=index) for index, document in enumerate(final_documents, start=1)],
        "metadata_boosted_candidates": [
            _document_payload(document, rank=index)
            for index, document in enumerate(metadata_boosted[:20], start=1)
        ],
        "rrf_candidates": [
            _document_payload(document, rank=index)
            for index, document in enumerate(rrf_documents[:20], start=1)
        ],
    }


def _retrieval_metrics(case: PubMedQACase, documents: list[RetrievedDocument], *, prefix: str) -> dict[str, int | float]:
    pmid_rank = None
    doc_rank = None
    chunk_rank = None
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        ids = {
            str(document.id or ""),
            str(metadata.get("chunkId") or ""),
            str(metadata.get("chunk_id") or ""),
            str(metadata.get("documentId") or ""),
            str(metadata.get("document_id") or ""),
            str(metadata.get("selectedChunkId") or ""),
        }
        pmid = str(metadata.get("pmid") or "")
        if chunk_rank is None and ids & case.relevant_document_ids:
            chunk_rank = rank
        if doc_rank is None and ids & case.relevant_document_ids:
            doc_rank = rank
        if pmid_rank is None and pmid in case.relevant_pmids:
            pmid_rank = rank
    metrics: dict[str, int | float] = {}
    for k in (1, 3, 5, 10, 20):
        metrics[f"{prefix}_chunk_hit_at_{k}"] = int(chunk_rank is not None and chunk_rank <= k)
        metrics[f"{prefix}_doc_hit_at_{k}"] = int(doc_rank is not None and doc_rank <= k)
        metrics[f"{prefix}_pmid_hit_at_{k}"] = int(pmid_rank is not None and pmid_rank <= k)
    metrics[f"{prefix}_pmid_mrr_at_10"] = 0.0 if pmid_rank is None or pmid_rank > 10 else 1.0 / pmid_rank
    return metrics


def _summarize(
    results: list[dict[str, Any]],
    *,
    model: str,
    collection: str,
    dataset: Path,
    candidate_k: int,
    top_k: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = [str(row["expected_label"]) for row in results]
    predictions = [row.get("predicted_label") for row in results]
    classification = _classification_metrics(labels=labels, predictions=predictions)
    retrieval_keys = sorted({key for row in results for key in row["retrieval_metrics"]})
    retrieval = {
        key: mean(float(row["retrieval_metrics"].get(key, 0.0)) for row in results) if results else 0.0
        for key in retrieval_keys
    }
    return {
        "created_at": now_iso(),
        "model": model,
        "collection": collection,
        "dataset": str(dataset),
        "llm_model": args.llm_model,
        "llm_provider": "ollama",
        "ollama_url": args.ollama_url,
        "temperature": args.temperature,
        "case_count": len(results),
        "candidate_k": candidate_k,
        "top_k": top_k,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": human_seconds(elapsed_seconds),
        "pipeline_components": {
            "pre_retriever": "app.rag.pre_retrieval.PreRetriever",
            "retriever": "benchmarks BenchmarkEmbeddingRetriever using benchmark model Qdrant collection",
            "post_retriever": "app.rag.post_retrieval.PostRetriever",
            "cross_encoder_model": args.cross_encoder_model,
            "evidence_filter_enabled": args.evidence_filter_enabled,
            "benchmark_evidence_expansion": str(args.pubmedqa_corpus),
            "answer_source": "benchmark_pqal LLM response parsed with app.rag.answer_contract.extract_yes_no_maybe_label",
        },
        "metrics": {
            "label_accuracy": classification["accuracy"],
            "label_macro_f1": classification["macro_f1"],
            "case_pass_rate": mean(float(row["case_pass"]) for row in results) if results else 0.0,
            "final_pmid_hit_at_1": retrieval.get("final_pmid_hit_at_1", 0.0),
            "final_pmid_hit_at_3": retrieval.get("final_pmid_hit_at_3", 0.0),
            "final_pmid_hit_at_10": retrieval.get("final_pmid_hit_at_10", 0.0),
            "final_pmid_mrr_at_10": retrieval.get("final_pmid_mrr_at_10", 0.0),
            "rrf_pmid_hit_at_10": retrieval.get("rrf_pmid_hit_at_10", 0.0),
            "metadata_boosted_pmid_hit_at_10": retrieval.get("metadata_boosted_pmid_hit_at_10", 0.0),
        },
        "classification": classification,
        "retrieval": retrieval,
        "predicted_labels": dict(Counter(str(item) for item in predictions)),
        "retrieval_statuses": dict(Counter(str(row["retrieval_status"]) for row in results)),
        "citation_validation_pass_rate": mean(
            float((row.get("citation_validation") or {}).get("passed", False)) for row in results
        ) if results else 0.0,
        "failures": [
            {
                "id": row["id"],
                "question": row["question"],
                "expected_label": row["expected_label"],
                "predicted_label": row["predicted_label"],
                "answer": row["answer"],
                "retrieval_status": row["retrieval_status"],
                "final_pmid_hit_at_3": row["retrieval_metrics"].get("final_pmid_hit_at_3", 0),
                "top_final_pmids": [item["pmid"] for item in row["final_documents"][:5]],
            }
            for row in results
            if not row["case_pass"]
        ][:50],
    }


def _classification_metrics(*, labels: list[str], predictions: list[Any]) -> dict[str, Any]:
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    unknown_predictions = 0
    for gold, pred in zip(labels, predictions):
        if pred not in LABEL_TO_ID:
            unknown_predictions += 1
            continue
        confusion[LABEL_TO_ID[gold]][LABEL_TO_ID[str(pred)]] += 1
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, label in enumerate(LABELS):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(LABELS)) if row != index)
        support = sum(1 for gold in labels if gold == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(1 for gold, pred in zip(labels, predictions) if gold == pred)
    return {
        "accuracy": correct / max(len(labels), 1),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_label": per_label,
        "confusion_matrix": confusion,
        "unknown_predictions": unknown_predictions,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    classification = summary["classification"]
    lines = [
        f"# PubMedQA Pipeline Embedding Eval: {summary['model']}",
        "",
        f"- Collection: `{summary['collection']}`",
        f"- Dataset: `{summary['dataset']}`",
        f"- LLM model: `{summary['llm_model']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Candidate K: `{summary['candidate_k']}`",
        f"- Top K after rerank/filter: `{summary['top_k']}`",
        f"- Cross encoder: `{summary['pipeline_components']['cross_encoder_model']}`",
        f"- Answer source: `{summary['pipeline_components']['answer_source']}`",
        f"- Elapsed: `{summary['elapsed_human']}`",
        "",
        "## Overall",
        "",
        f"- Label accuracy: `{metrics['label_accuracy']:.4f}`",
        f"- Label macro F1: `{metrics['label_macro_f1']:.4f}`",
        f"- Case pass rate: `{metrics['case_pass_rate']:.4f}`",
        f"- Final PMID hit@1: `{metrics['final_pmid_hit_at_1']:.4f}`",
        f"- Final PMID hit@3: `{metrics['final_pmid_hit_at_3']:.4f}`",
        f"- Final PMID hit@10: `{metrics['final_pmid_hit_at_10']:.4f}`",
        f"- Final PMID MRR@10: `{metrics['final_pmid_mrr_at_10']:.4f}`",
        "",
        "## Per Label",
        "",
        "| Label | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = classification["per_label"][label]
        lines.append(
            f"| {label} | {row['support']} | {float(row['precision']):.4f} | "
            f"{float(row['recall']):.4f} | {float(row['f1']):.4f} |"
        )
    lines.extend(["", "## Sample Failures", ""])
    for failure in summary["failures"][:20]:
        lines.append(
            f"- `{failure['id']}` expected={failure['expected_label']} predicted={failure['predicted_label']} "
            f"status={failure['retrieval_status']} final_pmid_hit@3={failure['final_pmid_hit_at_3']} "
            f"top_final_pmids={failure['top_final_pmids']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _document_payload(document: RetrievedDocument, *, rank: int) -> dict[str, Any]:
    metadata = document.metadata or {}
    return {
        "rank": rank,
        "id": document.id,
        "score": document.score,
        "title": document.title,
        "source": document.source,
        "pmid": str(metadata.get("pmid") or ""),
        "chunk_id": str(metadata.get("chunkId") or metadata.get("chunk_id") or ""),
        "doc_id": str(metadata.get("documentId") or metadata.get("document_id") or ""),
        "metadata": metadata,
        "text_preview": document.content[:900],
    }


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata_keys = (
        "chunkId",
        "pmid",
        "doi",
        "journal",
        "year",
        "authors",
        "meshTerms",
        "section",
        "chunkIndex",
        "parentChunkId",
        "parentWordCount",
        "documentId",
        "url",
        "publicationDate",
        "publicationTypes",
        "isReview",
        "isSystematicReview",
        "wordCount",
        "textHash",
        "embeddingModel",
        "corpusVersion",
        "corpusType",
        "sourceAuthority",
    )
    return {key: payload[key] for key in metadata_keys if key in payload}


if __name__ == "__main__":
    asyncio.run(async_main())
