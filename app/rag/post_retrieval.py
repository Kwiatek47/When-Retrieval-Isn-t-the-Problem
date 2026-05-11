from dataclasses import replace
import logging
from time import perf_counter
from typing import Any

from app.rag.conflict_detection import detect_evidence_conflicts
from app.rag.models import PostRetrievalResult, PreRetrievalResult, RetrievedDocument, RetrievalResult
from app.schemas import ChatMessage, Citation, EvidenceConflictInfo, RetrievalInfo


logger = logging.getLogger(__name__)


class PostRetriever:
    """Rank retrieved documents and assemble the grounded LLM prompt."""

    def __init__(
        self,
        max_context_chars: int,
        final_documents_limit: int,
        cross_encoder_model_name: str | None = None,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.final_documents_limit = final_documents_limit
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder = self._load_cross_encoder(cross_encoder_model_name)

    def assemble(
        self,
        *,
        messages: list[ChatMessage],
        system_prompt: str,
        pre_retrieval: PreRetrievalResult,
        retrieval: RetrievalResult,
    ) -> PostRetrievalResult:
        documents = self._rank_documents(retrieval.documents, pre_retrieval)
        context_block, citations = self._build_context(documents)
        evidence_conflicts = detect_evidence_conflicts(
            documents=documents,
            citations=citations,
            query=pre_retrieval.normalized_query,
        )

        system_message = ChatMessage(
            role="system",
            content=self._build_system_prompt(
                system_prompt=system_prompt,
                context_block=context_block,
                has_documents=bool(documents),
                conflict_block=self._build_conflict_block(evidence_conflicts),
            ),
        )
        user_visible_messages = [message for message in messages if message.role != "system"]
        retrieval_info = RetrievalInfo(
            enabled=pre_retrieval.requires_retrieval,
            status=self._status(pre_retrieval, documents),
            provider=retrieval.provider,
            query=pre_retrieval.normalized_query,
            documents_count=len(documents),
        )

        return PostRetrievalResult(
            messages=[system_message, *user_visible_messages],
            citations=citations,
            retrieval=retrieval_info,
            evidence_conflicts=evidence_conflicts,
            source_documents=documents[: len(citations)],
        )

    def _rank_documents(
        self,
        documents: list[RetrievedDocument],
        pre_retrieval: PreRetrievalResult,
    ) -> list[RetrievedDocument]:
        documents = self._score_documents_with_cross_encoder(documents, pre_retrieval)
        deduplicated = {}
        for document in sorted(documents, key=lambda item: item.score, reverse=True):
            if not document.content.strip():
                continue
            key = document.id or f"{document.source}:{document.title}"
            deduplicated.setdefault(key, document)
        ranked_documents = list(deduplicated.values())
        if self.final_documents_limit <= 0:
            return ranked_documents
        return ranked_documents[: self.final_documents_limit]

    def _score_documents_with_cross_encoder(
        self,
        documents: list[RetrievedDocument],
        pre_retrieval: PreRetrievalResult,
    ) -> list[RetrievedDocument]:
        if self.cross_encoder is None or not documents:
            return documents

        query = pre_retrieval.normalized_query
        pairs = [(query, document.content) for document in documents]
        started_at = perf_counter()
        scores = self.cross_encoder.predict(pairs)
        finished_at = perf_counter()

        logger.info(
            "post_retrieval rerank timing cross_encoder=%.3fs documents=%d model=%s",
            finished_at - started_at,
            len(documents),
            self.cross_encoder_model_name,
        )

        return [
            replace(document, score=float(score))
            for document, score in zip(documents, scores, strict=True)
        ]

    def _load_cross_encoder(self, model_name: str | None) -> Any | None:
        if not model_name:
            return None
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        started_at = perf_counter()
        model = CrossEncoder(model_name)
        logger.info(
            "post_retrieval cross_encoder loaded model=%s load=%.3fs",
            model_name,
            perf_counter() - started_at,
        )
        return model

    def _build_context(self, documents: list[RetrievedDocument]) -> tuple[str, list[Citation]]:
        if not documents:
            return "No verified medical knowledge-base documents were retrieved.", []

        remaining_chars = self.max_context_chars
        context_parts = []
        citations = []

        for document in documents:
            if remaining_chars <= 0:
                break

            citation_id = f"S{len(citations) + 1}"
            content = document.content.strip()
            if len(content) > remaining_chars:
                truncated = content[:remaining_chars].rsplit(" ", 1)[0].strip()
                content = truncated or content[:remaining_chars].strip()
            if not content:
                continue

            context_parts.append(
                "\n".join(
                    [
                        f"[{citation_id}] {document.title}",
                        f"Source: {document.source}",
                        f"Score: {document.score:.4f}",
                        content,
                    ]
                )
            )
            citations.append(
                Citation(
                    id=citation_id,
                    title=document.title,
                    source=document.source,
                    score=document.score,
                    metadata=self._stringify_metadata(document.metadata),
                )
            )
            remaining_chars -= len(content)

        if not context_parts:
            return "No verified medical knowledge-base documents were retrieved.", []

        return "\n\n".join(context_parts), citations

    def _build_system_prompt(
        self,
        *,
        system_prompt: str,
        context_block: str,
        has_documents: bool,
        conflict_block: str,
    ) -> str:
        source_policy = (
            "Answer only from the MEDICAL_KNOWLEDGE_BASE context. "
            "Every medical claim must include an inline citation with labels like [S1]. "
            "Do not use prior knowledge, training data, or assumptions to add medical facts. "
            "Do not invent citations."
        )
        reasoning_policy = (
            "Before writing the final response, think step-by-step privately. "
            "Analyze the user's query, identify relevant symptoms, conditions, interventions, or outcomes, "
            "and cross-reference them only with cited MEDICAL_KNOWLEDGE_BASE entries. "
            "Do not reveal hidden chain-of-thought or uncited reasoning. "
            "If you need to show your evidence check, keep it brief inside <thinking> tags and include only "
            "the relevant citation labels and whether the evidence is sufficient. "
            "Put the final concise user-facing response inside <answer> tags."
        )
        if not has_documents:
            source_policy = (
                "No verified medical knowledge-base context is available for this answer. "
                "Do not answer the user's medical question from prior knowledge. "
                "Say that the knowledge base did not return sources, so you cannot provide a grounded answer. "
                "You may only advise consulting a qualified clinician for personal medical decisions."
            )
            reasoning_policy = (
                "Do not perform or output step-by-step reasoning because there are no sources to reason from. "
                "Return only a concise refusal inside <answer> tags."
            )

        return "\n\n".join(
            [
                system_prompt,
                "RAG instructions:",
                source_policy,
                "Reasoning and output format:",
                reasoning_policy,
                "If retrieved evidence is insufficient or conflicting, say so explicitly.",
                "If CONFLICTING_EVIDENCE_FLAG is present, do not blend competing recommendations. "
                "Present both positions with citations and abstain from a specific directive unless the "
                "retrieved sources establish a clear priority.",
                "Always advise consulting a qualified clinician for personal medical decisions.",
                "CONFLICTING_EVIDENCE_FLAG:",
                conflict_block,
                "MEDICAL_KNOWLEDGE_BASE:",
                context_block,
            ]
        )

    def _build_conflict_block(self, evidence_conflicts: EvidenceConflictInfo) -> str:
        if not evidence_conflicts.detected:
            return "No source-level recommendation conflicts detected."

        lines = [evidence_conflicts.instruction]
        for pair in evidence_conflicts.pairs:
            newer = f"; newer source: {pair.newer_source_id}" if pair.newer_source_id else ""
            terms = ", ".join(pair.shared_terms) or "overlapping clinical terms"
            lines.append(
                f"- {', '.join(pair.source_ids)} conflict on {terms}{newer}. Reason: {pair.reason}"
            )
        return "\n".join(lines)

    def _status(self, pre_retrieval: PreRetrievalResult, documents: list[RetrievedDocument]) -> str:
        if not pre_retrieval.requires_retrieval:
            return "skipped"
        if documents:
            return "grounded"
        return "no_sources"

    def _stringify_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        return {str(key): str(value) for key, value in metadata.items()}
