import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest

try:
    import httpx  # noqa: F401
except ModuleNotFoundError:
    sys.modules["httpx"] = types.SimpleNamespace(AsyncClient=object, HTTPError=Exception)

from app.rag.answer_contract import enforce_yes_no_maybe_contract, is_yes_no_maybe_task
from app.rag.answer_guardrails import apply_answer_guardrails, repair_missing_citations
from app.rag.evidence_judge import EvidenceJudge, answer_from_evidence_decision
from app.rag.models import PostRetrievalResult, PreRetrievalResult, RetrievedDocument, RetrievalResult
from app.rag.pipeline import RagPipeline
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import _expanded_candidate_limit, _metadata_boost_documents, _weighted_rrf_merge
from app.schemas import ChatMessage, ChatResponse, EvidenceConflictInfo, EvidenceDecisionInfo, RetrievalInfo
from app.services.chat_service import has_urgent_red_flags, low_evidence_refusal, urgent_red_flag_response


class RagRankingTests(unittest.TestCase):
    def test_expands_candidate_pool_for_clinical_metadata_ranking(self) -> None:
        query = PreRetrievalResult(
            original_query="Jak leczyc PAD?",
            normalized_query="Jak leczyc PAD?",
            search_queries=["Jak leczyc PAD?", "treatment peripheral artery disease"],
            requires_retrieval=True,
            intent="treatment",
            preferred_publication_types=["Systematic Review", "Guideline"],
        )

        self.assertEqual(_expanded_candidate_limit(50, query), 100)
        self.assertEqual(_expanded_candidate_limit(80, query), 100)

    def test_metadata_boost_promotes_stronger_evidence_type(self) -> None:
        query = PreRetrievalResult(
            original_query="leczenie PAD",
            normalized_query="leczenie PAD",
            search_queries=["leczenie PAD"],
            requires_retrieval=True,
            intent="treatment",
            preferred_publication_types=["Systematic Review", "Guideline"],
        )
        weak = RetrievedDocument(
            id="case",
            title="Case report",
            content="Peripheral artery disease case report.",
            source="pubmed",
            score=0.0110,
            metadata={"publicationTypes": ["Case Reports"]},
        )
        strong = RetrievedDocument(
            id="review",
            title="Systematic review",
            content="Peripheral artery disease treatment review.",
            source="pubmed",
            score=0.0100,
            metadata={"publicationTypes": ["Systematic Review"], "isSystematicReview": True},
        )

        ranked = _metadata_boost_documents([weak, strong], query=query, limit=2)

        self.assertEqual(ranked[0].id, "review")
        self.assertIn("systematic_review", ranked[0].metadata["metadataBoostReasons"])
        self.assertIn("weak_publication_type", ranked[1].metadata["metadataBoostReasons"])

    def test_rerank_query_keeps_original_query_and_rewrite(self) -> None:
        post_retriever = PostRetriever(max_context_chars=1000, final_documents_limit=3)
        query = PreRetrievalResult(
            original_query="leki na PAD",
            normalized_query="leki na PAD",
            search_queries=[
                "leki na PAD",
                "leki na PAD peripheral artery disease choroba tetnic obwodowych",
                "leczenie symptomatic peripheral artery disease guideline therapy",
            ],
            requires_retrieval=True,
        )

        rerank_query = post_retriever._rerank_query(query)

        self.assertIn("PAD", rerank_query)
        self.assertIn("guideline therapy", rerank_query)
        self.assertLessEqual(len(rerank_query), 512)

    def test_evidence_scoring_uses_best_query_variant_for_term_coverage(self) -> None:
        post_retriever = PostRetriever(max_context_chars=1000, final_documents_limit=3)
        query = PreRetrievalResult(
            original_query="leki na PAD",
            normalized_query="leki na PAD",
            search_queries=["leki na PAD", "leczenie symptomatic peripheral artery disease"],
            requires_retrieval=True,
            intent="treatment",
            preferred_publication_types=["Systematic Review"],
        )
        document = RetrievedDocument(
            id="review",
            title="Peripheral artery disease review",
            content="Treatment of symptomatic peripheral artery disease is discussed in this review.",
            source="pubmed",
            score=1.0,
            metadata={
                "matchedQueryCount": 2,
                "publicationTypes": ["Systematic Review"],
                "year": 2025,
            },
        )

        scored = post_retriever._score_documents_for_evidence([document], query)

        self.assertGreater(scored[0].metadata["queryTermCoverage"], 0.5)
        self.assertEqual(scored[0].metadata["multiQueryMatchScore"], 1.0)

    def test_low_query_alignment_marks_treatment_result_as_low_evidence(self) -> None:
        post_retriever = PostRetriever(max_context_chars=1000, final_documents_limit=3)
        query = PreRetrievalResult(
            original_query="Jakie leki stosuje się w PAD?",
            normalized_query="Jakie leki stosuje się w PAD?",
            search_queries=[
                "Jakie leki stosuje się w PAD?",
                "Jakie leki stosuje się w PAD? peripheral artery disease choroba tetnic obwodowych",
            ],
            requires_retrieval=True,
            intent="treatment",
            preferred_publication_types=["Systematic Review", "Guideline"],
        )
        unrelated = RetrievedDocument(
            id="bp-trial",
            title="A Randomized Trial of Intensive versus Standard Blood-Pressure Control",
            content=(
                "This trial studied systolic blood pressure control and cardiovascular risk. "
                "The record mentions chronic kidney disease and heart failure outcomes."
            ),
            source="pubmed",
            score=1.0,
            metadata={"publicationTypes": ["Randomized Controlled Trial"], "year": 2021, "matchedQueryCount": 2},
        )

        ranked, low_evidence = post_retriever._rank_documents([unrelated], query)

        self.assertTrue(low_evidence)
        self.assertEqual(ranked[0].id, "bp-trial")
        self.assertLess(ranked[0].metadata["queryTermCoverage"], 0.18)

    def test_query_term_coverage_uses_whole_tokens_not_substrings(self) -> None:
        post_retriever = PostRetriever(max_context_chars=1000, final_documents_limit=3)
        query = PreRetrievalResult(
            original_query="PAD treatment",
            normalized_query="PAD treatment",
            search_queries=["PAD treatment"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="unrelated",
            title="Padding protocol",
            content="The document discusses padded dressings and treatment schedules.",
            source="pubmed",
            score=1.0,
        )

        scored = post_retriever._score_documents_for_evidence([document], query)

        self.assertEqual(scored[0].metadata["queryTermCoverage"], 0.0)

    def test_trace_documents_keep_structured_retrieval_metadata(self) -> None:
        _install_fastapi_stub()
        from app.api.routes import _debug_documents, _trace_documents

        rrf_document = RetrievedDocument(
            id="review",
            title="Review",
            content="Evidence.",
            source="pubmed",
            score=0.016,
            metadata={
                "rrfScore": 0.016,
                "matchedQueryCount": 2,
                "rawRetrievalScores": [0.91, 0.87],
            },
        )
        retrieval = RetrievalResult(
            query=PreRetrievalResult(
                original_query="PAD",
                normalized_query="PAD",
                search_queries=["PAD"],
                requires_retrieval=True,
            ),
            documents=[],
            provider="fake",
            debug={"rrf_documents": [rrf_document]},
        )

        trace_documents = _trace_documents(_debug_documents(retrieval, "rrf_documents"))

        self.assertEqual(trace_documents[0].metadata["matchedQueryCount"], 2)
        self.assertEqual(trace_documents[0].metadata["rawRetrievalScores"], [0.91, 0.87])

    def test_parent_child_chunks_are_deduplicated_by_parent(self) -> None:
        query = PreRetrievalResult(
            original_query="leczenie PAD",
            normalized_query="leczenie PAD",
            search_queries=["leczenie PAD"],
            requires_retrieval=True,
            intent="treatment",
            preferred_publication_types=["Systematic Review"],
        )
        child_one = RetrievedDocument(
            id="child-1",
            title="PAD review",
            content="Peripheral artery disease treatment is reviewed.",
            source="pubmed",
            score=0.9,
            metadata={"chunkId": "p1:part:0", "parentChunkId": "p1", "publicationTypes": ["Systematic Review"]},
        )
        child_two = RetrievedDocument(
            id="child-2",
            title="PAD review",
            content="Peripheral artery disease safety is reviewed.",
            source="pubmed",
            score=0.8,
            metadata={"chunkId": "p1:part:1", "parentChunkId": "p1", "publicationTypes": ["Systematic Review"]},
        )

        merged = _weighted_rrf_merge([[child_one, child_two]], query_texts=["leczenie PAD"], limit=10)
        ranked, _low_evidence = PostRetriever(max_context_chars=1000, final_documents_limit=3)._rank_documents(
            [child_one, child_two],
            query,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(ranked), 1)

    def test_context_includes_priority_header_for_small_models(self) -> None:
        post_retriever = PostRetriever(max_context_chars=2000, final_documents_limit=2)
        query = PreRetrievalResult(
            original_query="leczenie PAD",
            normalized_query="leczenie PAD",
            search_queries=["leczenie PAD", "peripheral artery disease treatment"],
            requires_retrieval=True,
            intent="treatment",
            preferred_publication_types=["Systematic Review", "Guideline"],
        )
        documents = [
            RetrievedDocument(
                id="review",
                title="PAD review",
                content="Peripheral artery disease treatment is discussed in this systematic review.",
                source="pubmed",
                score=1.0,
                metadata={"publicationTypes": ["Systematic Review"], "matchedQueryCount": 2},
            ),
            RetrievedDocument(
                id="guideline",
                title="PAD guideline",
                content="Peripheral artery disease treatment is discussed in this guideline.",
                source="pubmed",
                score=0.9,
                metadata={"publicationTypes": ["Guideline"], "matchedQueryCount": 2},
            ),
        ]

        context, citations = post_retriever._build_context(documents, query)

        self.assertEqual([citation.id for citation in citations], ["S1", "S2"])
        self.assertIn("SOURCE_PRIORITY", context)
        self.assertIn("Use [S1] [S2] first", context)

    def test_answer_guardrail_keeps_guideline_recommendation(self) -> None:
        answer = "SGLT2 inhibitors are recommended for CKD [S1]."
        source = RetrievedDocument(
            id="guideline",
            title="Guideline",
            content="SGLT2 inhibitors are recommended for chronic kidney disease.",
            source="pubmed",
            metadata={"publicationTypes": ["Guideline"]},
        )

        self.assertEqual(apply_answer_guardrails(answer, [source]), answer)

    def test_missing_citation_repair_cites_supported_sentence_only(self) -> None:
        source = RetrievedDocument(
            id="doac-review",
            title="Bleeding Risk in Nonvalvular Atrial Fibrillation",
            content=(
                "Direct oral anticoagulants were associated with fewer bleeding events than warfarin. "
                "Apixaban showed lower bleeding risk while rivaroxaban did not show the same reduction."
            ),
            source="pubmed",
        )
        answer = (
            "Direct oral anticoagulants had fewer bleeding events than warfarin [S1]. "
            "Apixaban showed lower bleeding risk while rivaroxaban did not show the same reduction. "
            "Aspirin and statins are standard PAD medications."
        )

        repaired = repair_missing_citations(answer, [source])

        self.assertIn("same reduction [S1].", repaired)
        self.assertIn("Aspirin and statins are standard PAD medications.", repaired)

    def test_pubmedqa_prompt_contract_requires_first_line_label(self) -> None:
        post_retriever = PostRetriever(max_context_chars=2000, final_documents_limit=1)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Does aspirin help?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Does aspirin help?",
            search_queries=["Does aspirin help?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc",
            title="Aspirin trial",
            content="The trial suggests aspirin reduced vascular events.",
            source="pubmed",
            score=1.0,
            metadata={"matchedQueryCount": 1},
        )

        result = post_retriever.assemble(
            messages=[ChatMessage(role="user", content=query.original_query)],
            system_prompt="System prompt.",
            pre_retrieval=query,
            retrieval=RetrievalResult(query=query, documents=[document], provider="fake"),
        )

        self.assertIn("first line must be exactly one of `Answer: yes`", result.messages[0].content)
        self.assertTrue(is_yes_no_maybe_task(result.messages[1:]))

    def test_pubmedqa_task_allows_single_source_for_top_one_eval(self) -> None:
        post_retriever = PostRetriever(max_context_chars=2000, final_documents_limit=1)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Does aspirin help?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Does aspirin help?",
            search_queries=["Does aspirin help?"],
            requires_retrieval=True,
            intent="treatment",
        )

        self.assertEqual(post_retriever._minimum_source_count(query), 1)

    def test_yes_no_maybe_contract_repairs_missing_label_with_citation(self) -> None:
        source = RetrievedDocument(
            id="source",
            title="Managed care study",
            content="The retrieved evidence is insufficient and inconclusive.",
            source="pubmed",
        )
        answer = "The retrieved evidence is insufficient to definitively answer this question."

        contracted = enforce_yes_no_maybe_contract(answer, [source])

        self.assertTrue(contracted.startswith("Answer: maybe\nEvidence:"))
        self.assertIn("[S1]", contracted)

    def test_yes_no_maybe_contract_can_infer_directional_positive_evidence(self) -> None:
        source = RetrievedDocument(
            id="source",
            title="GP hospital study",
            content="Conclusion: GP hospitals seem to reduce the utilisation of general hospitals.",
            source="pubmed",
        )
        answer = "Conclusion: GP hospitals seem to reduce the utilisation of general hospitals [S1]."

        contracted = enforce_yes_no_maybe_contract(answer, [source])

        self.assertTrue(contracted.startswith("Answer: yes\nEvidence:"))


class PreRetrievalPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_acronym_expansion_adds_query_without_duplicates(self) -> None:
        pre_retriever = PreRetriever(
            ollama_base_url="http://localhost:11434",
            rewrite_model="",
            rewrite_timeout=0.01,
        )

        result = await pre_retriever.prepare([ChatMessage(role="user", content="leki na PAD i T2DM")])

        self.assertLessEqual(len(result.search_queries), 4)
        self.assertTrue(any("peripheral artery disease" in query for query in result.search_queries))
        self.assertTrue(any("type 2 diabetes mellitus" in query for query in result.search_queries))
        self.assertTrue(any("Query expanded deterministically" in note for note in result.notes))


class EvidenceJudgeArchitectureTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_evidence_judge_outputs_control_decision_before_writer(self) -> None:
        provider = _FakeJudgeProvider(
            '{"status":"refuted","answer":"no","confidence":0.82,'
            '"rationale":"The abstract reports no meaningful diagnostic value [S1]."}'
        )
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Is the test useful?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Is the test useful?",
            search_queries=["Is the test useful?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="Diagnostic test study",
            content="The test had no meaningful diagnostic value in this cohort.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=provider,
            model="fake-judge",
        )

        self.assertEqual(decision.method, "llm")
        self.assertEqual(decision.status, "refuted")
        self.assertEqual(decision.answer_label, "no")
        self.assertEqual(decision.source_ids, ["doc-1"])
        self.assertIn("evidence judge", provider.messages[0][0].content.lower())

    async def test_trace_uses_rule_judge_without_llm_provider(self) -> None:
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?",
            search_queries=["Does treatment help?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="Treatment study",
            content="Treatment significantly improved outcomes in the study population.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=None,
            model="",
        )

        self.assertEqual(decision.method, "rules")
        self.assertEqual(decision.answer_label, "yes")

    async def test_llm_evidence_judge_adds_primary_citation_when_missing(self) -> None:
        provider = _FakeJudgeProvider(
            '{"status":"supported","answer":"yes","rationale":"The abstract reports improved outcomes."}'
        )
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?",
            search_queries=["Does treatment help?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="Treatment study",
            content="Treatment improved outcomes.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=provider,
            model="fake-judge",
        )

        self.assertIn("[S1]", decision.rationale)
        self.assertEqual(decision.citations, ["S1"])
        self.assertEqual(decision.source_ids, ["doc-1"])

    async def test_llm_evidence_judge_calibrates_direct_negative_yes_to_no(self) -> None:
        provider = _FakeJudgeProvider(
            '{"status":"supported","answer":"yes","confidence":0.91,'
            '"rationale":"The study suggests the intervention may help despite no significant difference [S1]."}'
        )
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Does the intervention improve outcomes?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Does the intervention improve outcomes?",
            search_queries=["Does the intervention improve outcomes?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="Outcome trial",
            content="Abstract context: The outcomes did not differ between groups and there was no significant benefit.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=provider,
            model="fake-judge",
        )

        self.assertEqual(decision.answer_label, "no")
        self.assertEqual(decision.status, "refuted")
        self.assertTrue(any("Calibrated yes to no" in note for note in decision.notes))

    async def test_llm_evidence_judge_calibrates_suggestive_yes_to_maybe(self) -> None:
        provider = _FakeJudgeProvider(
            '{"status":"supported","answer":"yes","confidence":0.88,'
            '"rationale":"The study suggests a potential association, but confounding weakened the evidence [S1]."}'
        )
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Is exposure a risk factor?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Is exposure a risk factor?",
            search_queries=["Is exposure a risk factor?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="Risk study",
            content="Abstract context: Estimates suggested a potential association, but bias and confounding weakened validity.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=provider,
            model="fake-judge",
        )

        self.assertEqual(decision.answer_label, "maybe")
        self.assertEqual(decision.status, "uncertain")
        self.assertTrue(any("Calibrated yes to maybe" in note for note in decision.notes))

    async def test_llm_evidence_judge_keeps_direct_positive_yes(self) -> None:
        provider = _FakeJudgeProvider(
            '{"status":"supported","answer":"yes","confidence":0.86,'
            '"rationale":"The trial reports significantly improved outcomes [S1]."}'
        )
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?",
            search_queries=["Does treatment help?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="Treatment trial",
            content="Abstract context: Treatment significantly improved outcomes in the study population.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=provider,
            model="fake-judge",
        )

        self.assertEqual(decision.answer_label, "yes")
        self.assertEqual(decision.status, "supported")
        self.assertEqual(decision.notes, [])

    async def test_llm_evidence_judge_v3_fixes_contradictory_no_rationale(self) -> None:
        provider = _FakeJudgeProvider(
            '{"status":"refuted","answer":"no","confidence":0.73,'
            '"rationale":"Interns ordered significantly more arterial blood gases, supporting an influence [S1]."}'
        )
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1)
        query = PreRetrievalResult(
            original_query=(
                "Answer yes, no, or maybe based on retrieved evidence: "
                "Does pediatric housestaff experience influence tests ordered?"
            ),
            normalized_query=(
                "Answer yes, no, or maybe based on retrieved evidence: "
                "Does pediatric housestaff experience influence tests ordered?"
            ),
            search_queries=["Does pediatric housestaff experience influence tests ordered?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="NICU ordering study",
            content="Abstract context: Interns ordered significantly more arterial blood gases than residents.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=provider,
            model="fake-judge",
        )

        self.assertEqual(decision.answer_label, "yes")
        self.assertEqual(decision.status, "supported")
        self.assertTrue(any("Calibrated no to yes" in note for note in decision.notes))

    async def test_llm_evidence_judge_voting_uses_majority_label(self) -> None:
        provider = _FakeJudgeProvider(
            [
                '{"status":"supported","answer":"yes","confidence":0.74,"rationale":"The marker improved [S1]."}',
                '{"status":"uncertain","answer":"maybe","confidence":0.69,"rationale":"The evidence is partial [S1]."}',
                '{"status":"uncertain","answer":"maybe","confidence":0.77,"rationale":"The source only partially answers it [S1]."}',
            ]
        )
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1, voting_enabled=True, voting_rounds=3)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Is the marker reliable?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Is the marker reliable?",
            search_queries=["Is the marker reliable?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="Marker study",
            content="Abstract context: The marker helped in some cases but only partially answered reliability.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=provider,
            model="fake-judge",
        )

        self.assertEqual(decision.method, "llm_voting")
        self.assertEqual(decision.answer_label, "maybe")
        self.assertEqual(len(provider.messages), 3)
        self.assertTrue(any("Vote labels" in note for note in decision.notes))

    async def test_llm_evidence_judge_voting_tie_breaks_to_maybe(self) -> None:
        provider = _FakeJudgeProvider(
            [
                '{"status":"supported","answer":"yes","confidence":0.74,"rationale":"The source supports it [S1]."}',
                '{"status":"refuted","answer":"no","confidence":0.74,"rationale":"The source does not support it [S1]."}',
            ]
        )
        judge = EvidenceJudge(enabled=True, method="llm", max_sources=1, voting_enabled=True, voting_rounds=2)
        query = PreRetrievalResult(
            original_query="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?",
            normalized_query="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?",
            search_queries=["Does treatment help?"],
            requires_retrieval=True,
        )
        document = RetrievedDocument(
            id="doc-1",
            title="Treatment study",
            content="Abstract context: Evidence was mixed.",
            source="pubmed",
        )

        decision = await judge.judge(
            messages=[ChatMessage(role="user", content=query.original_query)],
            pre_retrieval=query,
            source_documents=[document],
            retrieval_status="grounded",
            llm_provider=provider,
            model="fake-judge",
        )

        self.assertEqual(decision.answer_label, "maybe")
        self.assertEqual(decision.status, "uncertain")

    def test_evidence_decision_is_attached_to_writer_prompt(self) -> None:
        _install_fastapi_stub()
        from app.api.routes import _with_evidence_decision

        decision = EvidenceDecisionInfo(
            status="supported",
            method="llm",
            answer_label="yes",
            confidence=0.8,
            rationale="Treatment improved outcomes [S1].",
            citations=["S1"],
        )
        result = PostRetrievalResult(
            messages=[ChatMessage(role="system", content="System prompt."), ChatMessage(role="user", content="Q")],
            citations=[],
            retrieval=RetrievalInfo(enabled=True, status="grounded", provider="fake", query="Q", documents_count=1),
            evidence_conflicts=EvidenceConflictInfo(detected=False),
            source_documents=[],
        )

        updated = _with_evidence_decision(result, decision)

        self.assertEqual(updated.evidence_decision, decision)
        self.assertIn("EVIDENCE_JUDGE_DECISION", updated.messages[0].content)
        self.assertIn("answer_label: yes", updated.messages[0].content)
        self.assertIn("Use this evidence judge decision", updated.messages[0].content)

    def test_yes_no_answer_can_be_built_directly_from_evidence_decision(self) -> None:
        decision = EvidenceDecisionInfo(
            status="refuted",
            method="rules",
            answer_label="no",
            rationale="The abstract reports no association [S1].",
            citations=["S1"],
        )

        answer = answer_from_evidence_decision(
            decision,
            [RetrievedDocument(id="doc", title="Study", content="No association.", source="pubmed")],
        )

        self.assertTrue(answer.startswith("Answer: no\nEvidence:"))
        self.assertIn("[S1]", answer)


class ChildChunkingTests(unittest.TestCase):
    def test_child_chunking_preserves_short_record_and_splits_long_record(self) -> None:
        module = _load_child_chunk_module()
        short = {"chunk_id": "pmid-1", "text": "Short abstract.", "word_count": 2, "chunk_index": 0}
        long_text = " ".join(f"Sentence {index} has several medical words." for index in range(180))
        long = {"chunk_id": "pmid-2", "text": long_text, "word_count": 900, "chunk_index": 0}

        rows = module.build_child_rows([short, long], max_words=120, split_threshold_words=150)

        self.assertEqual(rows[0]["chunk_id"], "pmid-1")
        self.assertEqual(rows[0]["parent_chunk_id"], "")
        split_rows = [row for row in rows if str(row["chunk_id"]).startswith("pmid-2:part:")]
        self.assertGreater(len(split_rows), 1)
        self.assertTrue(all(row["parent_chunk_id"] == "pmid-2" for row in split_rows))
        self.assertTrue(all(row["parent_word_count"] == 900 for row in split_rows))


class PubMedQADatasetBuilderTests(unittest.TestCase):
    def test_strict_corpus_omits_conclusion_and_final_label_metadata(self) -> None:
        module = _load_pubmedqa_builder_module()
        item = {
            "pmid": "123",
            "QUESTION": "Does treatment improve outcomes?",
            "CONTEXTS": ["Methods text.", "Results text."],
            "LONG_ANSWER": "Treatment improved outcomes.",
            "MESHES": ["Treatment Outcome"],
            "LABELS": ["BACKGROUND", "RESULTS"],
            "YEAR": "2024",
            "final_decision": "yes",
        }

        corpus_item = module._to_corpus_item(item, strict=True)

        self.assertIn("Results text.", corpus_item["content"])
        self.assertNotIn("Conclusion:", corpus_item["content"])
        self.assertNotIn("Treatment improved outcomes.", corpus_item["content"])
        self.assertEqual(corpus_item["metadata"]["corpusType"], "pubmedqa_strict_benchmark")
        self.assertNotIn("pubmedqaFinalDecision", corpus_item["metadata"])
        self.assertNotIn("pubmedqaLabels", corpus_item["metadata"])


class AdaptiveRetrievalTests(unittest.TestCase):
    def test_low_evidence_runs_one_adaptive_round(self) -> None:
        pre_retriever = _FakePreRetriever()
        retriever = _FakeRetriever()
        pipeline = RagPipeline(
            pre_retriever=pre_retriever,
            retriever=retriever,
            post_retriever=PostRetriever(max_context_chars=2000, final_documents_limit=5),
            retrieval_candidate_limit=5,
            adaptive_retrieval_enabled=True,
            adaptive_max_rounds=1,
        )

        result = asyncio.run(
            pipeline.run(
                messages=[ChatMessage(role="user", content="leczenie PAD")],
                system_prompt="System prompt.",
            )
        )

        self.assertEqual(retriever.calls, 2)
        self.assertEqual(result.retrieval.status, "grounded")
        self.assertEqual(pre_retriever.allow_rewrite_values, [True])

    def test_benchmark_mode_skips_query_rewrite_and_adaptive_retrieval(self) -> None:
        pre_retriever = _FakePreRetriever()
        retriever = _FakeRetriever()
        pipeline = RagPipeline(
            pre_retriever=pre_retriever,
            retriever=retriever,
            post_retriever=PostRetriever(max_context_chars=2000, final_documents_limit=5),
            retrieval_candidate_limit=5,
            adaptive_retrieval_enabled=True,
            adaptive_max_rounds=1,
        )

        result = asyncio.run(
            pipeline.run(
                messages=[ChatMessage(role="user", content="Answer yes, no, or maybe based on retrieved evidence: Does treatment help?")],
                system_prompt="System prompt.",
                mode="benchmark_pqal",
            )
        )

        self.assertEqual(pre_retriever.allow_rewrite_values, [False])
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(result.retrieval.status, "low_evidence")

    def test_adaptive_query_adds_intent_specific_follow_up_terms(self) -> None:
        pipeline = RagPipeline(
            pre_retriever=_FakePreRetriever(),
            retriever=_FakeRetriever(),
            post_retriever=PostRetriever(max_context_chars=2000, final_documents_limit=5),
            retrieval_candidate_limit=5,
            adaptive_retrieval_enabled=True,
            adaptive_max_rounds=1,
        )
        pre_retrieval = PreRetrievalResult(
            original_query="leczenie PAD",
            normalized_query="leczenie PAD",
            search_queries=["leczenie PAD"],
            requires_retrieval=True,
            intent="treatment",
            preferred_publication_types=["Systematic Review", "Guideline"],
        )

        adaptive = pipeline._adaptive_pre_retrieval_query(pre_retrieval, round_index=0)

        self.assertIn("efficacy", adaptive.search_queries[-1])
        self.assertIn("safety", adaptive.search_queries[-1])


class LowEvidenceSafetyRefusalTests(unittest.TestCase):
    def test_low_evidence_refusal_adds_emergency_instruction_for_red_flags(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content="I have sudden weakness on one side of the body and trouble speaking. What should I do?",
            )
        ]

        refusal = low_evidence_refusal(messages, [])

        self.assertTrue(has_urgent_red_flags(messages))
        self.assertIn("seek emergency care immediately", refusal)
        self.assertIn("cannot provide a reliable cited medical answer", refusal)

    def test_urgent_red_flag_response_is_immediate_and_explicit(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content="I have crushing chest pain, shortness of breath, and sweating. Can I wait until tomorrow?",
            )
        ]

        response = urgent_red_flag_response(messages)

        self.assertTrue(has_urgent_red_flags(messages))
        self.assertIn("Seek emergency care immediately", response)
        self.assertIn("911/112", response)

    def test_low_evidence_refusal_does_not_add_emergency_instruction_for_routine_question(self) -> None:
        messages = [ChatMessage(role="user", content="What is known about mild seasonal allergies?")]

        refusal = low_evidence_refusal(messages, [])

        self.assertFalse(has_urgent_red_flags(messages))
        self.assertNotIn("seek emergency care immediately", refusal)


class _FakePreRetriever:
    def __init__(self) -> None:
        self.allow_rewrite_values: list[bool] = []

    async def prepare(self, _messages: list[ChatMessage], *, allow_rewrite: bool = True) -> PreRetrievalResult:
        self.allow_rewrite_values.append(allow_rewrite)
        return PreRetrievalResult(
            original_query="leczenie PAD",
            normalized_query="leczenie PAD",
            search_queries=["leczenie PAD", "peripheral artery disease treatment"],
            requires_retrieval=True,
            intent="treatment",
            preferred_publication_types=["Systematic Review", "Guideline"],
        )


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        self.calls += 1
        if self.calls == 1:
            documents = [
                RetrievedDocument(
                    id="weak",
                    title="Unrelated case",
                    content="This abstract is unrelated to the user question.",
                    source="pubmed",
                    score=1.0,
                    metadata={"publicationTypes": ["Case Reports"], "year": 2018},
                )
            ]
        else:
            documents = [
                RetrievedDocument(
                    id="review-1",
                    title="Peripheral artery disease treatment review",
                    content="Treatment of peripheral artery disease is discussed in this systematic review.",
                    source="pubmed",
                    score=1.0,
                    metadata={"publicationTypes": ["Systematic Review"], "year": 2025, "matchedQueryCount": 2},
                ),
                RetrievedDocument(
                    id="guideline-1",
                    title="Peripheral artery disease guideline",
                    content="Guideline evidence discusses treatment of peripheral artery disease.",
                    source="pubmed",
                    score=0.9,
                    metadata={"publicationTypes": ["Guideline"], "year": 2025, "matchedQueryCount": 2},
                ),
            ]
        return RetrievalResult(query=query, documents=documents[:limit], provider="fake")


class _FakeJudgeProvider:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = [content] if isinstance(content, str) else list(content)
        self.messages: list[list[ChatMessage]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
    ) -> ChatResponse:
        self.messages.append(messages)
        index = min(len(self.messages) - 1, len(self.contents) - 1)
        return ChatResponse(
            model=model,
            message=ChatMessage(role="assistant", content=self.contents[index]),
            done=True,
        )


def _load_child_chunk_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rag" / "00_build_child_chunks.py"
    spec = importlib.util.spec_from_file_location("build_child_chunks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_pubmedqa_builder_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rag" / "05_build_pubmedqa_benchmark_dataset.py"
    spec = importlib.util.spec_from_file_location("build_pubmedqa_benchmark_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _install_fastapi_stub() -> None:
    if "fastapi" in sys.modules:
        return

    class _FakeAPIRouter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def post(self, *_args, **_kwargs):
            return lambda func: func

        def get(self, *_args, **_kwargs):
            return lambda func: func

    class _FakeHTTPException(Exception):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args)
            self.kwargs = kwargs

    def _depends(value=None, **_kwargs):
        return value

    def _query(default=None, **_kwargs):
        return default

    sys.modules["fastapi"] = types.SimpleNamespace(
        APIRouter=_FakeAPIRouter,
        Depends=_depends,
        HTTPException=_FakeHTTPException,
        Query=_query,
        status=types.SimpleNamespace(
            HTTP_400_BAD_REQUEST=400,
            HTTP_502_BAD_GATEWAY=502,
            HTTP_503_SERVICE_UNAVAILABLE=503,
        ),
    )


if __name__ == "__main__":
    unittest.main()
