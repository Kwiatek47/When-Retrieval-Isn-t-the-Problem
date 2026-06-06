from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any

from app.rag.models import PostRetrievalResult, RetrievedDocument
from app.schemas import Citation, RetrievalInfo


logger = logging.getLogger(__name__)


def expand_pubmedqa_benchmark_evidence(
    result: PostRetrievalResult,
    *,
    corpus_path: Path,
) -> PostRetrievalResult:
    """Replace selected PQA-L chunks with full official abstracts by PMID.

    This is intentionally benchmark-only. It does not choose a new PMID, does not
    use the gold label, and does not affect medical chat mode.
    """

    if not result.source_documents:
        return result
    corpus = _load_pubmedqa_corpus(corpus_path)
    if not corpus:
        return result

    expanded_documents = []
    expanded_count = 0
    for document in result.source_documents:
        pmid = str(document.metadata.get("pmid") or "").strip()
        full_document = corpus.get(pmid)
        if not pmid or full_document is None:
            expanded_documents.append(document)
            continue
        expanded_documents.append(_merge_full_evidence(document, full_document))
        expanded_count += 1

    if expanded_count == 0:
        return result

    citations = [
        Citation(
            id=f"S{index}",
            title=document.title,
            source=document.source,
            url=_citation_url(document),
            score=document.score,
            metadata={str(key): str(value) for key, value in document.metadata.items()},
        )
        for index, document in enumerate(expanded_documents, start=1)
    ]
    retrieval = result.retrieval
    if retrieval and retrieval.status == "low_evidence":
        retrieval = RetrievalInfo(
            enabled=retrieval.enabled,
            status="grounded",
            provider=f"{retrieval.provider}+pubmedqa_full_evidence",
            query=retrieval.query,
            documents_count=len(expanded_documents),
        )

    logger.info(
        "pubmedqa benchmark evidence expanded documents=%d corpus_path=%s",
        expanded_count,
        corpus_path,
    )
    return replace(
        result,
        source_documents=expanded_documents,
        citations=citations,
        retrieval=retrieval,
    )


def _merge_full_evidence(selected: RetrievedDocument, full_document: RetrievedDocument) -> RetrievedDocument:
    metadata = {
        **full_document.metadata,
        **selected.metadata,
        "benchmarkFullEvidence": True,
        "benchmarkFullEvidenceSource": "official_pqal_corpus",
        "selectedChunkId": selected.metadata.get("chunkId") or selected.metadata.get("chunk_id") or selected.id,
        "selectedContentChars": len(selected.content),
        "fullEvidenceChars": len(full_document.content),
    }
    return replace(
        selected,
        id=full_document.id or selected.id,
        title=full_document.title or selected.title,
        content=full_document.content,
        source=full_document.source or selected.source,
        metadata=metadata,
    )


@lru_cache(maxsize=4)
def _load_pubmedqa_corpus(corpus_path: Path) -> dict[str, RetrievedDocument]:
    if not corpus_path.exists():
        logger.warning("PubMedQA benchmark corpus not found: %s", corpus_path)
        return {}
    with corpus_path.open(encoding="utf-8") as handle:
        raw_items = json.load(handle)
    if not isinstance(raw_items, list):
        logger.warning("PubMedQA benchmark corpus has unexpected format: %s", corpus_path)
        return {}

    documents = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        metadata = _metadata(item.get("metadata"))
        pmid = str(metadata.get("pmid") or "").strip()
        if not pmid:
            continue
        documents[pmid] = RetrievedDocument(
            id=str(item.get("id") or f"pubmedqa-official-{pmid}"),
            title=str(item.get("title") or "Untitled PubMedQA source"),
            content=str(item.get("content") or ""),
            source=str(item.get("source") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
            score=float(item.get("score") or 0.0),
            metadata=metadata,
        )
    return documents


def _metadata(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _citation_url(document: RetrievedDocument) -> str | None:
    for key in ("url", "source_url", "sourceUrl"):
        value = str(document.metadata.get(key) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    if document.source.startswith(("http://", "https://")):
        return document.source
    pmid = str(document.metadata.get("pmid") or "").strip()
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return None
