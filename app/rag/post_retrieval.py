from typing import Any

from app.rag.models import PostRetrievalResult, PreRetrievalResult, RetrievedDocument, RetrievalResult
from app.schemas import ChatMessage, Citation, RetrievalInfo


class PostRetriever:
    """Rank retrieved documents and assemble the grounded LLM prompt."""

    def __init__(self, max_context_chars: int) -> None:
        self.max_context_chars = max_context_chars

    def assemble(
        self,
        *,
        messages: list[ChatMessage],
        system_prompt: str,
        pre_retrieval: PreRetrievalResult,
        retrieval: RetrievalResult,
    ) -> PostRetrievalResult:
        documents = self._rank_documents(retrieval.documents)
        context_block, citations = self._build_context(documents)

        system_message = ChatMessage(
            role="system",
            content=self._build_system_prompt(
                system_prompt=system_prompt,
                context_block=context_block,
                has_documents=bool(documents),
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
        )

    def _rank_documents(self, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        deduplicated = {}
        for document in sorted(documents, key=lambda item: item.score, reverse=True):
            if not document.content.strip():
                continue
            key = document.id or f"{document.source}:{document.title}"
            deduplicated.setdefault(key, document)
        return list(deduplicated.values())

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

    def _build_system_prompt(self, *, system_prompt: str, context_block: str, has_documents: bool) -> str:
        source_policy = (
            "Answer only from the MEDICAL_KNOWLEDGE_BASE context. "
            "Every medical claim must include an inline citation with labels like [S1]. "
            "Do not use prior knowledge, training data, or assumptions to add medical facts. "
            "Do not invent citations."
        )
        if not has_documents:
            source_policy = (
                "No verified medical knowledge-base context is available for this answer. "
                "Do not answer the user's medical question from prior knowledge. "
                "Say that the knowledge base did not return sources, so you cannot provide a grounded answer. "
                "You may only advise consulting a qualified clinician for personal medical decisions."
            )

        return "\n\n".join(
            [
                system_prompt,
                "RAG instructions:",
                source_policy,
                "If retrieved evidence is insufficient or conflicting, say so explicitly.",
                "Always advise consulting a qualified clinician for personal medical decisions.",
                "MEDICAL_KNOWLEDGE_BASE:",
                context_block,
            ]
        )

    def _status(self, pre_retrieval: PreRetrievalResult, documents: list[RetrievedDocument]) -> str:
        if not pre_retrieval.requires_retrieval:
            return "skipped"
        if documents:
            return "grounded"
        return "no_sources"

    def _stringify_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        return {str(key): str(value) for key, value in metadata.items()}
