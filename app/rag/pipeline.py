import logging
from dataclasses import replace
from time import perf_counter

from app.rag.models import PostRetrievalResult, PreRetrievalResult, RetrievedDocument, RetrievalResult
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import MedicalKnowledgeRetriever
from app.schemas import ChatMessage


logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(
        self,
        *,
        pre_retriever: PreRetriever,
        retriever: MedicalKnowledgeRetriever,
        post_retriever: PostRetriever,
        retrieval_candidate_limit: int,
        adaptive_retrieval_enabled: bool = False,
        adaptive_max_rounds: int = 0,
    ) -> None:
        self.pre_retriever = pre_retriever
        self.retriever = retriever
        self.post_retriever = post_retriever
        self.retrieval_candidate_limit = retrieval_candidate_limit
        self.adaptive_retrieval_enabled = adaptive_retrieval_enabled
        self.adaptive_max_rounds = max(adaptive_max_rounds, 0)

    async def run(self, *, messages: list[ChatMessage], system_prompt: str) -> PostRetrievalResult:
        started_at = perf_counter()
        pre_retrieval = await self.pre_retriever.prepare(messages)
        pre_retrieval_done_at = perf_counter()
        if pre_retrieval.requires_retrieval:
            retrieval = await self.retriever.retrieve(pre_retrieval, limit=self.retrieval_candidate_limit)
        else:
            retrieval = RetrievalResult(query=pre_retrieval, documents=[], provider="skipped")
        retrieval_done_at = perf_counter()

        result = self.post_retriever.assemble(
            messages=messages,
            system_prompt=system_prompt,
            pre_retrieval=pre_retrieval,
            retrieval=retrieval,
        )

        if self._should_run_adaptive_retrieval(pre_retrieval, result):
            retrieval, result = await self._run_adaptive_retrieval(
                messages=messages,
                system_prompt=system_prompt,
                pre_retrieval=pre_retrieval,
                first_retrieval=retrieval,
                first_result=result,
            )
        post_retrieval_done_at = perf_counter()

        logger.info(
            "rag_pipeline timing pre_retrieval=%.3fs retrieval=%.3fs post_retrieval=%.3fs total=%.3fs "
            "requires_retrieval=%s provider=%s candidate_documents=%d final_documents=%d "
            "intent=%s queries=%s source_pmids=%s source_scores=%s publication_types=%s status=%s",
            pre_retrieval_done_at - started_at,
            retrieval_done_at - pre_retrieval_done_at,
            post_retrieval_done_at - retrieval_done_at,
            post_retrieval_done_at - started_at,
            pre_retrieval.requires_retrieval,
            retrieval.provider,
            len(retrieval.documents),
            result.retrieval.documents_count,
            pre_retrieval.intent,
            pre_retrieval.search_queries,
            [document.metadata.get("pmid") for document in result.source_documents],
            [round(document.score, 6) for document in result.source_documents],
            [document.metadata.get("publicationTypes") for document in result.source_documents],
            result.retrieval.status,
        )
        return result

    def _should_run_adaptive_retrieval(
        self,
        pre_retrieval: PreRetrievalResult,
        result: PostRetrievalResult,
    ) -> bool:
        return (
            self.adaptive_retrieval_enabled
            and self.adaptive_max_rounds > 0
            and pre_retrieval.requires_retrieval
            and result.retrieval.status == "low_evidence"
        )

    async def _run_adaptive_retrieval(
        self,
        *,
        messages: list[ChatMessage],
        system_prompt: str,
        pre_retrieval: PreRetrievalResult,
        first_retrieval: RetrievalResult,
        first_result: PostRetrievalResult,
    ) -> tuple[RetrievalResult, PostRetrievalResult]:
        retrieval_rounds = [first_retrieval]
        result = first_result
        for round_index in range(self.adaptive_max_rounds):
            adaptive_query = self._adaptive_pre_retrieval_query(pre_retrieval, round_index=round_index)
            if adaptive_query.search_queries == pre_retrieval.search_queries:
                break
            next_retrieval = await self.retriever.retrieve(
                adaptive_query,
                limit=self.retrieval_candidate_limit,
            )
            retrieval_rounds.append(next_retrieval)
            combined_retrieval = self._merge_retrieval_rounds(pre_retrieval, retrieval_rounds)
            result = self.post_retriever.assemble(
                messages=messages,
                system_prompt=system_prompt,
                pre_retrieval=pre_retrieval,
                retrieval=combined_retrieval,
            )
            if result.retrieval.status != "low_evidence":
                return combined_retrieval, result

        return self._merge_retrieval_rounds(pre_retrieval, retrieval_rounds), result

    def _adaptive_pre_retrieval_query(
        self,
        pre_retrieval: PreRetrievalResult,
        *,
        round_index: int,
    ) -> PreRetrievalResult:
        preferred_terms = " ".join(pre_retrieval.preferred_publication_types[:4])
        intent_terms = _adaptive_intent_terms(pre_retrieval.intent)
        fallback_terms = "systematic review guideline meta-analysis randomized controlled trial"
        adaptive_suffix = " ".join(term for term in (preferred_terms, intent_terms or fallback_terms) if term)
        adaptive_query = " ".join([pre_retrieval.normalized_query, adaptive_suffix]).strip()

        search_queries = _unique_queries([*pre_retrieval.search_queries, adaptive_query])
        notes = [
            *pre_retrieval.notes,
            f"Adaptive retrieval round {round_index + 1} added publication-type focused query.",
        ]
        return replace(pre_retrieval, search_queries=search_queries, notes=notes)

    def _merge_retrieval_rounds(
        self,
        pre_retrieval: PreRetrievalResult,
        retrieval_rounds: list[RetrievalResult],
    ) -> RetrievalResult:
        merged: dict[str, tuple[RetrievedDocument, float, list[float], list[int]]] = {}
        rrf_k = 60
        for round_index, retrieval in enumerate(retrieval_rounds):
            round_weight = 1.0 if round_index == 0 else 0.75
            for rank, document in enumerate(retrieval.documents, start=1):
                if not document.content.strip():
                    continue
                key = _document_merge_key(document)
                rrf_score = round_weight / float(rrf_k + rank)
                if key not in merged:
                    merged[key] = (document, rrf_score, [document.score], [round_index + 1])
                    continue
                current_document, current_score, raw_scores, rounds = merged[key]
                best_document = document if document.score > max(raw_scores) else current_document
                merged[key] = (
                    best_document,
                    current_score + rrf_score,
                    [*raw_scores, document.score],
                    [*rounds, round_index + 1],
                )

        documents = []
        for document, score, raw_scores, rounds in merged.values():
            metadata = {
                **document.metadata,
                "adaptiveRrfScore": round(score, 6),
                "adaptiveRounds": sorted(set(rounds)),
                "adaptiveRoundScores": raw_scores,
            }
            documents.append(replace(document, score=score, metadata=metadata))

        provider = retrieval_rounds[0].provider if retrieval_rounds else "none"
        if len(retrieval_rounds) > 1:
            provider = f"{provider}+adaptive"
        return RetrievalResult(
            query=pre_retrieval,
            documents=sorted(documents, key=lambda item: item.score, reverse=True),
            provider=provider,
        )


def _unique_queries(queries: list[str]) -> list[str]:
    unique = []
    seen = set()
    for query in queries:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if normalized and key not in seen:
            unique.append(normalized)
            seen.add(key)
    return unique


def _document_merge_key(document: RetrievedDocument) -> str:
    metadata = document.metadata
    for key in ("parentChunkId", "parent_chunk_id", "chunkId", "chunk_id", "documentId", "document_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return document.id or f"{document.source}:{document.title}"


def _adaptive_intent_terms(intent: str) -> str:
    terms_by_intent = {
        "treatment": "efficacy safety outcomes adverse effects contraindications dosing recommendation",
        "diagnosis": "diagnostic accuracy sensitivity specificity screening criteria recommendation",
        "adverse_effects": "safety adverse effects contraindications risk warnings monitoring",
        "prognosis": "mortality survival outcomes risk prediction cohort",
        "mechanism": "pathophysiology mechanism biomarker molecular pathway",
    }
    return terms_by_intent.get(intent, "")
