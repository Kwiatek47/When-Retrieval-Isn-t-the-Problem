from dataclasses import replace
from datetime import datetime
import logging
import re
from time import perf_counter
from typing import Any

from app.rag.answer_contract import is_yes_no_maybe_task_text
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
        max_excerpt_chars: int = 1600,
        cross_encoder_model_name: str | None = None,
        cross_encoder_max_length: int = 512,
        cross_encoder_batch_size: int = 8,
        cross_encoder_device: str | None = None,
        evidence_filter_enabled: bool = True,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.final_documents_limit = final_documents_limit
        self.max_excerpt_chars = max_excerpt_chars
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder_max_length = cross_encoder_max_length
        self.cross_encoder_batch_size = cross_encoder_batch_size
        self.cross_encoder_device = cross_encoder_device
        self.evidence_filter_enabled = evidence_filter_enabled
        self.cross_encoder = self._load_cross_encoder(cross_encoder_model_name)

    def assemble(
        self,
        *,
        messages: list[ChatMessage],
        system_prompt: str,
        pre_retrieval: PreRetrievalResult,
        retrieval: RetrievalResult,
        final_documents_limit: int | None = None,
    ) -> PostRetrievalResult:
        documents, low_evidence = self._rank_documents(
            retrieval.documents,
            pre_retrieval,
            final_documents_limit=final_documents_limit,
        )
        preliminary_citations = self._citations_for_documents(documents)
        preliminary_conflicts = detect_evidence_conflicts(
            documents=documents,
            citations=preliminary_citations,
            query=pre_retrieval.normalized_query,
        )
        documents = self._order_conflicting_documents(documents, preliminary_conflicts)
        context_block, citations = self._build_context(documents, pre_retrieval)
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
                answer_contract_block=self._build_answer_contract_block(pre_retrieval),
            ),
        )
        user_visible_messages = [message for message in messages if message.role != "system"]
        retrieval_info = RetrievalInfo(
            enabled=pre_retrieval.requires_retrieval,
            status=self._status(pre_retrieval, documents, low_evidence),
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
            pre_retrieval=pre_retrieval,
        )

    def _rank_documents(
        self,
        documents: list[RetrievedDocument],
        pre_retrieval: PreRetrievalResult,
        final_documents_limit: int | None = None,
    ) -> tuple[list[RetrievedDocument], bool]:
        deduplicated = {}
        for document in sorted(documents, key=lambda item: item.score, reverse=True):
            if not document.content.strip():
                continue
            key = self._document_key(document)
            deduplicated.setdefault(key, document)
        deduplicated_documents = list(deduplicated.values())
        if not deduplicated_documents:
            return [], False

        reranked_documents = self._score_documents_with_cross_encoder(deduplicated_documents, pre_retrieval)
        ranked_documents = self._score_documents_for_evidence(reranked_documents, pre_retrieval)
        threshold = self._evidence_threshold(pre_retrieval)
        filtered_documents = [
            document
            for document in ranked_documents
            if float(document.metadata.get("evidenceScore") or 0.0) >= threshold
            and self._passes_query_alignment(document, pre_retrieval)
        ]
        minimum_sources = self._minimum_source_count(pre_retrieval)
        enough_sources = len(filtered_documents) >= minimum_sources or self._has_high_authority_source(filtered_documents)
        low_evidence = pre_retrieval.requires_retrieval and not enough_sources
        selected_documents = filtered_documents if filtered_documents else ranked_documents[:minimum_sources]

        selected_limit = self.final_documents_limit if final_documents_limit is None else final_documents_limit
        if selected_limit <= 0:
            return selected_documents, low_evidence
        return selected_documents[: selected_limit], low_evidence

    def _score_documents_with_cross_encoder(
        self,
        documents: list[RetrievedDocument],
        pre_retrieval: PreRetrievalResult,
    ) -> list[RetrievedDocument]:
        if self.cross_encoder is None or not documents:
            return documents

        query = self._rerank_query(pre_retrieval)
        pairs = [(query, document.content) for document in documents]
        started_at = perf_counter()
        scores = self.cross_encoder.predict(
            pairs,
            batch_size=self.cross_encoder_batch_size,
            show_progress_bar=False,
        )
        finished_at = perf_counter()

        logger.info(
            "post_retrieval rerank timing cross_encoder=%.3fs documents=%d model=%s batch_size=%d max_length=%d",
            finished_at - started_at,
            len(documents),
            self.cross_encoder_model_name,
            self.cross_encoder_batch_size,
            self.cross_encoder_max_length,
        )

        if len(scores) != len(documents):
            raise RuntimeError(f"Expected {len(documents)} reranker scores, got {len(scores)}.")

        return [
            replace(document, score=float(score))
            for document, score in zip(documents, scores)
        ]

    def _score_documents_for_evidence(
        self,
        documents: list[RetrievedDocument],
        pre_retrieval: PreRetrievalResult,
    ) -> list[RetrievedDocument]:
        sorted_documents = sorted(documents, key=lambda item: item.score, reverse=True)
        query_term_sets = self._query_term_sets(pre_retrieval)
        query_count = self._query_count(pre_retrieval)
        denominator = max(len(sorted_documents) - 1, 1)
        scored_documents = []

        for index, document in enumerate(sorted_documents):
            rank_score = 1.0 if len(sorted_documents) == 1 else 1.0 - (index / denominator)
            coverage_score = self._query_term_coverage(document, query_term_sets)
            publication_score = self._publication_type_score(document, pre_retrieval)
            recency_score = self._recency_score(document, pre_retrieval)
            multi_query_score = self._multi_query_match_score(document, query_count)
            evidence_score = (
                0.25 * rank_score
                + 0.45 * coverage_score
                + 0.15 * publication_score
                + 0.05 * recency_score
                + 0.10 * multi_query_score
            )
            if query_term_sets and coverage_score == 0.0:
                evidence_score *= 0.4
            if coverage_score >= 0.45:
                evidence_score += 0.20

            metadata = {
                **document.metadata,
                "rerankerScore": document.score,
                "evidenceScore": round(evidence_score, 6),
                "queryTermCoverage": round(coverage_score, 6),
                "publicationTypeScore": round(publication_score, 6),
                "recencyScore": round(recency_score, 6),
                "multiQueryMatchScore": round(multi_query_score, 6),
                "evidenceFilterPassed": self._passes_query_alignment_score(coverage_score, pre_retrieval),
            }
            scored_documents.append(replace(document, score=evidence_score, metadata=metadata))

        return sorted(scored_documents, key=lambda item: item.score, reverse=True)

    def _query_term_coverage(self, document: RetrievedDocument, query_term_sets: list[set[str]]) -> float:
        if not query_term_sets:
            return 0.0
        document_terms = self._token_set(f"{document.title} {document.content}")
        scores = []
        for query_terms in query_term_sets:
            if not query_terms:
                continue
            matched = len(query_terms & document_terms)
            scores.append(matched / len(query_terms))
        return max(scores, default=0.0)

    def _multi_query_match_score(self, document: RetrievedDocument, query_count: int) -> float:
        if query_count <= 1:
            return 1.0
        matched_query_count = self._optional_int(document.metadata.get("matchedQueryCount")) or 1
        return min(max(matched_query_count / query_count, 0.0), 1.0)

    def _publication_type_score(
        self,
        document: RetrievedDocument,
        pre_retrieval: PreRetrievalResult,
    ) -> float:
        metadata = document.metadata
        publication_types = {item.lower() for item in self._publication_types(metadata)}
        preferred = {item.lower() for item in pre_retrieval.preferred_publication_types}
        scores = [0.35]

        if metadata.get("isSystematicReview") is True or str(metadata.get("isSystematicReview")).lower() == "true":
            scores.append(0.95)
        if "practice guideline" in publication_types:
            scores.append(1.0)
        if "guideline" in publication_types:
            scores.append(0.95)
        if "systematic review" in publication_types:
            scores.append(0.90)
        if "meta-analysis" in publication_types:
            scores.append(0.85)
        if "review" in publication_types:
            scores.append(0.70)
        if "randomized controlled trial" in publication_types:
            scores.append(0.65)
        if "clinical trial" in publication_types:
            scores.append(0.55)
        if publication_types & preferred:
            scores.append(0.80)
        if publication_types & {"case reports", "letter", "editorial", "comment"}:
            scores.append(0.15)

        return max(scores)

    def _recency_score(self, document: RetrievedDocument, pre_retrieval: PreRetrievalResult) -> float:
        year = self._optional_int(document.metadata.get("year"))
        if year is None:
            return 0.0 if pre_retrieval.requires_recent_evidence else 0.35

        current_year = datetime.now().year
        age = max(current_year - year, 0)
        if pre_retrieval.requires_recent_evidence:
            if pre_retrieval.min_year is not None and year < pre_retrieval.min_year:
                return 0.0
            return max(0.2, 1.0 - (age / 10.0))
        return max(0.35, 1.0 - (age / 25.0))

    def _evidence_threshold(self, pre_retrieval: PreRetrievalResult) -> float:
        if pre_retrieval.intent in {"treatment", "diagnosis", "adverse_effects"}:
            return 0.32
        return 0.24

    def _passes_query_alignment(
        self,
        document: RetrievedDocument,
        pre_retrieval: PreRetrievalResult,
    ) -> bool:
        if not self.evidence_filter_enabled:
            return True
        if not pre_retrieval.requires_retrieval:
            return True
        coverage = float(document.metadata.get("queryTermCoverage") or 0.0)
        return self._passes_query_alignment_score(coverage, pre_retrieval)

    def _passes_query_alignment_score(
        self,
        coverage: float,
        pre_retrieval: PreRetrievalResult,
    ) -> bool:
        if not pre_retrieval.requires_retrieval:
            return True
        return coverage >= self._minimum_query_term_coverage(pre_retrieval)

    def _minimum_query_term_coverage(self, pre_retrieval: PreRetrievalResult) -> float:
        if pre_retrieval.intent in {"treatment", "diagnosis", "adverse_effects"}:
            return 0.30
        return 0.20

    def _minimum_source_count(self, pre_retrieval: PreRetrievalResult) -> int:
        if is_yes_no_maybe_task_text(pre_retrieval.original_query):
            return 1
        if pre_retrieval.intent in {"treatment", "diagnosis", "adverse_effects"}:
            return 2
        return 1

    def _has_high_authority_source(self, documents: list[RetrievedDocument]) -> bool:
        if not documents:
            return False
        high_authority_types = {"practice guideline", "guideline", "systematic review", "meta-analysis"}
        for document in documents:
            publication_types = {item.lower() for item in self._publication_types(document.metadata)}
            if publication_types & high_authority_types:
                return True
            if str(document.metadata.get("isSystematicReview")).lower() == "true":
                return True
        return False

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
        model = CrossEncoder(
            model_name,
            max_length=self.cross_encoder_max_length,
            device=self.cross_encoder_device,
        )
        logger.info(
            "post_retrieval cross_encoder loaded model=%s load=%.3fs max_length=%d batch_size=%d device=%s",
            model_name,
            perf_counter() - started_at,
            self.cross_encoder_max_length,
            self.cross_encoder_batch_size,
            self.cross_encoder_device or "auto",
        )
        return model

    def _build_context(
        self,
        documents: list[RetrievedDocument],
        pre_retrieval: PreRetrievalResult,
    ) -> tuple[str, list[Citation]]:
        if not documents:
            return "No verified medical knowledge-base documents were retrieved.", []

        remaining_chars = self.max_context_chars
        context_parts = []
        citations = []
        query_terms = set().union(*self._query_term_sets(pre_retrieval))

        for document in documents:
            if remaining_chars <= 0:
                break

            citation_id = f"S{len(citations) + 1}"
            excerpt_sentence_limit = 5 if pre_retrieval.intent == "treatment" else 3
            content = self._extract_excerpt(
                document.content,
                query_terms,
                sentence_limit=excerpt_sentence_limit,
            ).strip()
            content_limit = min(remaining_chars, self.max_excerpt_chars)
            if len(content) > content_limit:
                truncated = content[:content_limit].rsplit(" ", 1)[0].strip()
                content = truncated or content[:content_limit].strip()
            if not content:
                continue

            context_parts.append(
                "\n".join(
                    [
                        f"[{citation_id}] {document.title}",
                        f"Source: {document.source}",
                        f"PMID: {document.metadata.get('pmid') or '-'}",
                        f"DOI: {document.metadata.get('doi') or '-'}",
                        f"Year: {document.metadata.get('year') or '-'}",
                        f"Publication types: {', '.join(self._publication_types(document.metadata)) or '-'}",
                        f"Evidence score: {document.score:.4f}",
                        "Selected excerpt:",
                        content,
                    ]
                )
            )
            citations.append(
                Citation(
                    id=citation_id,
                    title=document.title,
                    source=document.source,
                    url=self._citation_url(document),
                    score=document.score,
                    metadata=self._stringify_metadata(document.metadata),
                )
            )
            remaining_chars -= len(content)

        if not context_parts:
            return "No verified medical knowledge-base documents were retrieved.", []

        priority_block = self._build_priority_block(citations)
        return "\n\n".join([priority_block, *context_parts]), citations

    def _build_priority_block(self, citations: list[Citation]) -> str:
        if not citations:
            return ""
        primary_ids = " ".join(f"[{citation.id}]" for citation in citations[:2])
        return (
            "SOURCE_PRIORITY:\n"
            f"Use {primary_ids} first when they directly answer the question. "
            "Later sources are supporting or conflicting evidence."
        )

    def _build_system_prompt(
        self,
        *,
        system_prompt: str,
        context_block: str,
        has_documents: bool,
        conflict_block: str,
        answer_contract_block: str,
    ) -> str:
        source_policy = (
            "Answer only from the MEDICAL_KNOWLEDGE_BASE context. "
            "ALMANAC-style citation contract: every substantive sentence in the final answer must end with "
            "inline citations in the exact bracket format [S1]. "
            "When citing multiple sources, use separate labels like [S1] [S3], not grouped labels like [S1, S3]. "
            "Do not use parenthetical citations like (S1). "
            "Do not place a bare citation after a paragraph; each citation must support the sentence immediately before it. "
            "Use the smallest set of citations that directly supports each sentence; do not attach every source to a "
            "broad sentence just to satisfy citation requirements. "
            "Do not cite more than three sources in one sentence; split broad summaries into separate sentences by outcome. "
            "Do not use prior knowledge, training data, or assumptions to add medical facts. "
            "Do not invent citations."
        )
        evidence_policy = (
            "Do not invent exact percentages, effect sizes, guideline recommendations, standard-of-care statements, "
            "mechanisms, populations, contraindications, or safety claims unless they are explicitly stated in the cited "
            "source excerpt. "
            "If the retrieved context only supports a broad conclusion, write a broad conclusion. "
            "Do not say 'recommended', 'guidelines recommend', 'standard of care', or 'should be used' unless the "
            "cited excerpt explicitly states a recommendation or guideline. Prefer neutral wording such as "
            "'retrieved evidence describes', 'reported', or 'is discussed' for review abstracts. "
            "Use exact numbers only when you attribute them to the cited source, for example "
            "'the retrieved meta-analysis reported ... [S4]'. "
            "For broad evidence summaries, separate clinical outcomes, kidney/cardiorenal effects, mechanisms, and "
            "limitations only when each category is directly supported by retrieved evidence. "
            "For questions about benefits and risks, answer in natural prose unless the user explicitly asks for "
            "bullets or a structured table. Cover benefits, risks, and population caveats only when they are directly "
            "supported by retrieved excerpts. Do not add treatment targets, blood-pressure thresholds, mechanisms, "
            "guidelines, or broad meta-analysis support unless the user explicitly asks for those details. "
            "If retrieved excerpts do not directly support benefits or risks, say that the retrieved evidence is "
            "insufficient for that part instead of inferring it. "
            "Do not mix population-specific findings into a general answer without naming the population. "
            "Do not cite kidney-outcome-only evidence for general blood-pressure targets or adverse-event claims unless "
            "the selected excerpt explicitly supports that exact claim. "
            "When the user asks about medication options or therapies, include every medication class or named drug "
            "that is explicitly listed in the selected excerpt; do not stop after the first option. "
            "When the user asks about benefits or outcomes, include every outcome category explicitly listed in the "
            "selected excerpt if it directly answers the question. "
            "For questions asking to summarize evidence, prefer this compact structure when supported: "
            "'Heart failure evidence: ... [Sx] [Sy]. CKD/cardiorenal evidence: ... [Sx]. Mechanistic evidence: ... [Sx].' "
            "Omit unsupported categories. "
            "Prefer stronger evidence types such as systematic reviews, meta-analyses, reviews, or guidelines when they "
            "are retrieved, but do not overstate review abstracts as patient-specific treatment advice."
        )
        reasoning_policy = (
            "Return only the final user-facing response inside <answer> tags. "
            "Do not output hidden reasoning, <thinking> tags, source audits, or markdown headings. "
            "Keep the answer concise: usually 3-5 sentences or short bullets. "
            "Do not add a generic 'more research is needed' conclusion unless the retrieved evidence specifically "
            "supports uncertainty. "
            "Use the user's language. "
            "If evidence is insufficient, say exactly what is missing instead of filling gaps from prior knowledge."
        )
        if not has_documents:
            source_policy = (
                "No verified medical knowledge-base context is available for this answer. "
                "Do not answer the user's medical question from prior knowledge. "
                "Say that the knowledge base did not return sources, so you cannot provide a grounded answer. "
                "You may only advise consulting a qualified clinician for personal medical decisions."
            )
            evidence_policy = (
                "Do not infer medical facts, mechanisms, risks, benefits, guidelines, or treatment options without "
                "retrieved sources."
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
                "Evidence and wording policy:",
                evidence_policy,
                "Reasoning and output format:",
                reasoning_policy,
                answer_contract_block,
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

    def _build_answer_contract_block(self, pre_retrieval: PreRetrievalResult) -> str:
        if not is_yes_no_maybe_task_text(pre_retrieval.original_query):
            return (
                "Task-specific answer contract: Use the user's requested format when it is explicit; otherwise "
                "answer naturally and cite every substantive medical claim."
            )
        return (
            "Task-specific answer contract for PubMedQA-style questions: "
            "Inside <answer>, the first line must be exactly one of `Answer: yes`, `Answer: no`, or "
            "`Answer: maybe`. Privately classify the retrieved evidence before writing the final answer. "
            "Use `yes` when the retrieved excerpts directionally support the proposition in the question. "
            "Use `no` when the retrieved excerpts directly refute it or report no meaningful association, "
            "benefit, diagnostic/prognostic value, reliability, usefulness, advantage, or effect. Use `maybe` only when "
            "evidence is explicitly mixed, inconclusive, indirect, conflicting, or does not directly answer the "
            "question. Do not choose `maybe` merely because there is one retrieved abstract, a small study, or cautious "
            "scientific wording. The second line must start with `Evidence:` "
            "and contain one concise cited explanation. Do not add text before the answer label."
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

    def _status(
        self,
        pre_retrieval: PreRetrievalResult,
        documents: list[RetrievedDocument],
        low_evidence: bool,
    ) -> str:
        if not pre_retrieval.requires_retrieval:
            return "skipped"
        if low_evidence:
            return "low_evidence"
        if documents:
            return "grounded"
        return "no_sources"

    def _stringify_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        return {str(key): str(value) for key, value in metadata.items()}

    def _document_key(self, document: RetrievedDocument) -> str:
        for key in ("parentChunkId", "parent_chunk_id", "chunkId", "chunk_id", "documentId", "document_id"):
            value = document.metadata.get(key)
            if value:
                return str(value)
        return document.id or f"{document.source}:{document.title}"

    def _publication_types(self, metadata: dict[str, Any]) -> list[str]:
        value = metadata.get("publicationTypes") or metadata.get("publication_types") or []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                stripped = stripped.strip("[]")
            return [item.strip().strip("'\"") for item in re.split(r"[;,]", stripped) if item.strip().strip("'\"")]
        return [str(item).strip() for item in value if str(item).strip()]

    def _optional_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _rerank_query(self, pre_retrieval: PreRetrievalResult) -> str:
        all_query_texts = self._query_texts_for_scoring(pre_retrieval)
        if not all_query_texts:
            return pre_retrieval.normalized_query

        query_texts = self._unique_texts(
            [
                pre_retrieval.original_query,
                pre_retrieval.normalized_query,
                all_query_texts[-1],
            ]
        )
        if len(query_texts) == 1:
            return query_texts[0][:512].strip()
        return " ".join(query_texts)[:512].strip()

    def _query_term_sets(self, pre_retrieval: PreRetrievalResult) -> list[set[str]]:
        return [
            terms
            for terms in (
                self._meaningful_terms(query_text)
                for query_text in self._query_texts_for_scoring(pre_retrieval)
            )
            if terms
        ]

    def _query_count(self, pre_retrieval: PreRetrievalResult) -> int:
        return max(len(self._query_texts_for_scoring(pre_retrieval)), 1)

    def _query_texts_for_scoring(self, pre_retrieval: PreRetrievalResult) -> list[str]:
        return self._unique_texts(
            [*pre_retrieval.search_queries, pre_retrieval.original_query, pre_retrieval.normalized_query]
        )

    def _unique_texts(self, values: list[str]) -> list[str]:
        unique = []
        seen = set()
        for value in values:
            text = value.strip()
            key = text.lower()
            if text and key not in seen:
                unique.append(text)
                seen.add(key)
        return unique

    def _citations_for_documents(self, documents: list[RetrievedDocument]) -> list[Citation]:
        return [
            Citation(
                id=f"S{index + 1}",
                title=document.title,
                source=document.source,
                url=self._citation_url(document),
                score=document.score,
                metadata=self._stringify_metadata(document.metadata),
            )
            for index, document in enumerate(documents)
        ]

    def _citation_url(self, document: RetrievedDocument) -> str | None:
        metadata = document.metadata or {}
        for key in ("url", "source_url", "sourceUrl"):
            value = str(metadata.get(key) or "").strip()
            if value.startswith(("http://", "https://")):
                return value

        source = str(document.source or "").strip()
        if source.startswith(("http://", "https://")):
            return source.split(": PMID", 1)[0].strip()

        pmid = str(metadata.get("pmid") or "").strip()
        if pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        doi = str(metadata.get("doi") or "").strip()
        if doi:
            return f"https://doi.org/{doi}"

        return None

    def _order_conflicting_documents(
        self,
        documents: list[RetrievedDocument],
        evidence_conflicts: EvidenceConflictInfo,
    ) -> list[RetrievedDocument]:
        if not evidence_conflicts.detected:
            return documents

        document_by_label = {f"S{index + 1}": document for index, document in enumerate(documents)}
        grouped_labels = []
        seen_labels = set()
        for pair in evidence_conflicts.pairs:
            for label in pair.source_ids:
                if label in document_by_label and label not in seen_labels:
                    grouped_labels.append(label)
                    seen_labels.add(label)
        if not grouped_labels:
            return documents

        grouped_documents = [document_by_label[label] for label in grouped_labels]
        remaining_documents = [
            document
            for index, document in enumerate(documents, start=1)
            if f"S{index}" not in seen_labels
        ]
        return [*grouped_documents, *remaining_documents]

    def _extract_excerpt(self, content: str, query_terms: set[str], *, sentence_limit: int = 3) -> str:
        sentences = self._sentences(content)
        if not sentences:
            return content.strip()
        if not query_terms:
            return " ".join(sentences[:2])

        ranked_sentences = sorted(
            enumerate(sentences),
            key=lambda item: (
                self._sentence_coverage(item[1], query_terms),
                -item[0],
            ),
            reverse=True,
        )
        selected_indexes = sorted(index for index, sentence in ranked_sentences[:sentence_limit] if sentence.strip())
        if not selected_indexes:
            return " ".join(sentences[:2])
        return " ".join(sentences[index] for index in selected_indexes)

    def _sentences(self, content: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", content).strip()
        if not normalized:
            return []
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]

    def _sentence_coverage(self, sentence: str, query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        sentence_terms = self._token_set(sentence)
        return len(query_terms & sentence_terms) / len(query_terms)

    def _meaningful_terms(self, text: str) -> set[str]:
        stopwords = {
            "about",
            "jak",
            "jakie",
            "jaka",
            "jest",
            "known",
            "lek",
            "leki",
            "medication",
            "medications",
            "czy",
            "oraz",
            "dla",
            "pod",
            "nad",
            "stosuje",
            "used",
            "uses",
            "the",
            "and",
            "with",
            "without",
            "what",
            "which",
            "are",
            "is",
            "was",
            "were",
            "patient",
            "patients",
            "pacjent",
            "pacjentow",
            "pacjentów",
            "choroba",
            "chorobie",
            "benefit",
            "benefits",
            "did",
            "disease",
            "during",
            "first",
            "for",
            "from",
            "help",
            "into",
            "line",
            "option",
            "options",
            "of",
            "onto",
            "over",
            "prevent",
            "recommended",
            "regimen",
            "should",
            "therapy",
            "therapies",
            "treatment",
            "used",
            "using",
            "what",
        }
        terms = re.findall(r"[\w]+", text.lower())
        return {term for term in terms if len(term) > 2 and term not in stopwords}

    def _token_set(self, text: str) -> set[str]:
        return set(re.findall(r"[\w]+", text.lower()))
