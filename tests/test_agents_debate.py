"""Tests for supervisor-free multi-agent clinical debate skeleton."""

from __future__ import annotations

import asyncio
import unittest

from pydantic import ValidationError

from app.agents import (
    ClinicalAgent,
    ClinicalOpinion,
    DebateOrchestrator,
    MockInferenceBackend,
    build_default_agents,
)
from app.agents.aggregation import aggregate_pubmedqa_decision, extract_label, majority_vote, opinion_label
from app.agents.backends import EvidenceHint, NullEvidenceHint, parse_clinical_opinion_json
from app.agents.prompts import build_messages
from app.schemas import ChatMessage

SAMPLE_CASE = "45-year-old with fever and cough."


class ClinicalOpinionSchemaTests(unittest.TestCase):
    def test_requires_exact_fields(self) -> None:
        opinion = ClinicalOpinion(
            top_1_diagnosis="Pneumonia",
            top_3_differential_diagnoses=["Pneumonia", "Bronchitis", "PE"],
            pros=["Fever"],
            cons=["No imaging"],
            required_further_tests=["CXR"],
            confidence_level=0.6,
            sources_used=["guideline"],
            red_flags=["Hypoxia"],
            missing_information="SpO2 unknown",
        )
        payload = opinion.model_dump()
        self.assertEqual(
            set(payload.keys()),
            {
                "top_1_diagnosis",
                "evidence_conclusiveness",
                "top_3_differential_diagnoses",
                "pros",
                "cons",
                "required_further_tests",
                "confidence_level",
                "sources_used",
                "red_flags",
                "missing_information",
                "information_requests",
            },
        )

    def test_rejects_invalid_confidence(self) -> None:
        with self.assertRaises(ValidationError):
            ClinicalOpinion(
                top_1_diagnosis="x",
                top_3_differential_diagnoses=["a"],
                confidence_level=1.5,
            )


class DebateOrchestratorTests(unittest.TestCase):
    def test_runs_three_rounds_without_supervisor(self) -> None:
        backend = MockInferenceBackend()
        agents = build_default_agents(backend)
        self.assertEqual(len(agents), 4)

        result = asyncio.run(DebateOrchestrator(agents, rounds=3).run(SAMPLE_CASE))

        self.assertEqual(len(result.rounds), 3)
        self.assertEqual(len(result.final_opinions), 4)
        self.assertEqual(result.final_opinions, result.rounds[-1])
        for round_opinions in result.rounds:
            self.assertEqual(len(round_opinions), 4)
            self.assertEqual(
                {entry.agent_id for entry in round_opinions},
                {
                    "generalist",
                    "evidence_skeptic",
                    "differential_expander",
                    "safety_officer",
                },
            )
        self.assertTrue(all("(revised" in e.opinion.top_1_diagnosis for e in result.rounds[1]))
        self.assertTrue(all("(revised" in e.opinion.top_1_diagnosis for e in result.rounds[2]))

    def test_round_one_has_no_peer_context(self) -> None:
        peer_flags: list[bool] = []

        class SpyBackend(MockInferenceBackend):
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                user = next(m.content for m in messages if m.role == "user")
                peer_flags.append("PEER OPINIONS" in user)
                return await super().complete(messages, temperature=temperature)

        agents = build_default_agents(SpyBackend())
        asyncio.run(DebateOrchestrator(agents, rounds=2).run(SAMPLE_CASE))

        self.assertEqual(peer_flags.count(False), 4)
        self.assertEqual(peer_flags.count(True), 4)

    def test_rejects_invalid_round_count(self) -> None:
        agents = build_default_agents(MockInferenceBackend())
        with self.assertRaises(ValueError):
            DebateOrchestrator(agents, rounds=1)
        with self.assertRaises(ValueError):
            DebateOrchestrator(agents, rounds=4)

    def test_early_exit_skips_later_rounds(self) -> None:
        from app.agents.orchestrator import labels_unanimous

        agents = build_default_agents(MockInferenceBackend(), task_mode="pubmedqa")

        def stop_after_round1(round_number: int, opinions: list) -> bool:
            return round_number == 1 and labels_unanimous(opinions)

        # Force unanimous labels via a custom backend.
        class UnanimousBackend(MockInferenceBackend):
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                from app.agents.models import ClinicalOpinion

                return ClinicalOpinion(
                    top_1_diagnosis="yes",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.9,
                ).model_dump_json()

        agents = build_default_agents(UnanimousBackend(), task_mode="pubmedqa")
        orch = DebateOrchestrator(agents, rounds=3, early_exit=stop_after_round1)
        result = asyncio.run(orch.run("RESEARCH QUESTION:\nIs X useful?\nEVIDENCE:\nvaluable"))
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual(orch.early_exits, 1)

    def test_round_robin_context_grows_within_round(self) -> None:
        """Round 2+ is a true round-robin: later speakers see earlier speakers' turns."""
        captured: list[str] = []

        class SpyBackend(MockInferenceBackend):
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                user = next(m.content for m in messages if m.role == "user")
                captured.append(user)
                return await super().complete(messages, temperature=temperature)

        agents = build_default_agents(SpyBackend())
        asyncio.run(DebateOrchestrator(agents, rounds=2).run(SAMPLE_CASE))

        self.assertEqual(len(captured), 8)
        round2_calls = captured[4:]
        peer_counts = [call.count('"agent_id":') for call in round2_calls]
        # Each round-2 speaker sees the 3 other peers from round 1, plus everyone
        # who has already taken their turn this round (0, then 1, then 2, then 3).
        self.assertEqual(peer_counts, [3, 4, 5, 6])

    def test_orchestrator_round_survives_one_agent_backend_failure(self) -> None:
        """A single flaky backend must not crash the whole debate round."""

        class ExplodingBackend:
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                raise RuntimeError("simulated backend failure")

        good_backend = MockInferenceBackend()
        agents = [
            ClinicalAgent(agent_id="generalist", persona="generalist", backend=good_backend),
            ClinicalAgent(agent_id="evidence_skeptic", persona="evidence_skeptic", backend=ExplodingBackend()),
            ClinicalAgent(agent_id="differential_expander", persona="differential_expander", backend=good_backend),
            ClinicalAgent(agent_id="safety_officer", persona="safety_officer", backend=good_backend),
        ]
        result = asyncio.run(DebateOrchestrator(agents, rounds=2).run(SAMPLE_CASE))

        self.assertEqual(len(result.final_opinions), 4)
        failing_entry = next(e for e in result.final_opinions if e.agent_id == "evidence_skeptic")
        self.assertEqual(failing_entry.opinion.sources_used, ["fallback"])


class ClinicalAgentTests(unittest.TestCase):
    def test_retries_invalid_json_once(self) -> None:
        class FlakyBackend:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "not-json"
                return ClinicalOpinion(
                    top_1_diagnosis="Bronchitis",
                    top_3_differential_diagnoses=["Bronchitis", "Pneumonia", "Asthma"],
                    pros=["Cough"],
                    cons=["No labs"],
                    required_further_tests=["CXR"],
                    confidence_level=0.4,
                    sources_used=[],
                    red_flags=[],
                    missing_information="Need exam",
                ).model_dump_json()

        backend = FlakyBackend()
        agent = ClinicalAgent(
            agent_id="generalist",
            persona="generalist",
            backend=backend,
            hint_provider=NullEvidenceHint(),
        )
        opinion = asyncio.run(agent.generate_opinion(SAMPLE_CASE))
        self.assertEqual(opinion.top_1_diagnosis, "Bronchitis")
        self.assertEqual(backend.calls, 2)

    def test_backend_exception_falls_back_instead_of_raising(self) -> None:
        class ExplodingBackend:
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                raise ConnectionError("simulated timeout")

        agent = ClinicalAgent(
            agent_id="generalist",
            persona="generalist",
            backend=ExplodingBackend(),
            hint_provider=NullEvidenceHint(),
        )
        opinion = asyncio.run(agent.generate_opinion(SAMPLE_CASE))
        self.assertEqual(opinion.top_1_diagnosis, "maybe")
        self.assertEqual(opinion.sources_used, ["fallback"])


class PromptAndParseTests(unittest.TestCase):
    def test_prompt_includes_evidence_hint_slot(self) -> None:
        messages = build_messages(
            agent_id="generalist",
            persona="generalist",
            patient_case=SAMPLE_CASE,
            evidence_hint=EvidenceHint(label="yes", confidence=0.72, model_path="seed47"),
        )
        user = messages[1].content
        self.assertIn("EVIDENCE CLASSIFIER HINT", user)
        self.assertIn("yes", user)
        self.assertIn("0.720", user)

    def test_pubmedqa_prompt_requires_yes_no_maybe(self) -> None:
        messages = build_messages(
            agent_id="generalist",
            persona="generalist",
            patient_case="Question: Is X useful?\nEvidence: ...",
            task_mode="pubmedqa",
        )
        system = messages[0].content
        self.assertIn("task_mode=pubmedqa", system)
        self.assertIn('"yes", "no", "maybe"', system)

    def test_parse_fenced_json(self) -> None:
        opinion = ClinicalOpinion(
            top_1_diagnosis="PE",
            top_3_differential_diagnoses=["PE", "Pneumonia", "ACS"],
            confidence_level=0.5,
        )
        fenced = f"```json\n{opinion.model_dump_json()}\n```"
        parsed = parse_clinical_opinion_json(fenced)
        self.assertEqual(parsed.top_1_diagnosis, "PE")

    def test_parse_normalizes_empty_differentials(self) -> None:
        raw = (
            '{"top_1_diagnosis":"yes","top_3_differential_diagnoses":[],'
            '"pros":[],"cons":[],"required_further_tests":[],'
            '"confidence_level":0.7,"sources_used":[],"red_flags":[],'
            '"missing_information":""}'
        )
        parsed = parse_clinical_opinion_json(raw)
        self.assertEqual(parsed.top_1_diagnosis, "yes")
        self.assertGreaterEqual(len(parsed.top_3_differential_diagnoses), 1)
        self.assertIn("yes", parsed.top_3_differential_diagnoses)


class AggregationTests(unittest.TestCase):
    def test_extract_label(self) -> None:
        self.assertEqual(extract_label("yes"), "yes")
        self.assertEqual(extract_label("Answer: maybe"), "maybe")
        self.assertIsNone(extract_label("pneumonia"))

    def test_majority_vote_prefers_count_then_confidence(self) -> None:
        opinions = [
            ClinicalOpinion(top_1_diagnosis="yes", top_3_differential_diagnoses=["yes"], confidence_level=0.4),
            ClinicalOpinion(top_1_diagnosis="yes", top_3_differential_diagnoses=["yes"], confidence_level=0.5),
            ClinicalOpinion(top_1_diagnosis="no", top_3_differential_diagnoses=["no"], confidence_level=0.9),
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.9),
        ]
        label, share = majority_vote(opinions)
        self.assertEqual(label, "yes")
        self.assertAlmostEqual(share["yes"], 0.5)

    def test_opinion_label(self) -> None:
        opinion = ClinicalOpinion(
            top_1_diagnosis="maybe",
            top_3_differential_diagnoses=["maybe", "yes", "no"],
            confidence_level=0.3,
        )
        self.assertEqual(opinion_label(opinion), "maybe")

    def test_bert_gate_trusts_high_confidence_yes_no(self) -> None:
        agents = [
            ClinicalOpinion(top_1_diagnosis="no", top_3_differential_diagnoses=["no"], confidence_level=0.4),
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.4),
            ClinicalOpinion(top_1_diagnosis="yes", top_3_differential_diagnoses=["yes"], confidence_level=0.4),
            ClinicalOpinion(top_1_diagnosis="no", top_3_differential_diagnoses=["no"], confidence_level=0.4),
        ]
        bert = ClinicalOpinion(
            top_1_diagnosis="yes",
            top_3_differential_diagnoses=["yes", "no", "maybe"],
            confidence_level=0.96,
        )
        label, _, rule = aggregate_pubmedqa_decision(
            agents, bert_opinion=bert, mode="bert_gate", bert_gate_confidence=0.9
        )
        self.assertEqual(label, "yes")
        self.assertEqual(rule, "bert_gate")

    def test_bert_gate_allows_unanimous_panel_override(self) -> None:
        agents = [
            ClinicalOpinion(top_1_diagnosis="no", top_3_differential_diagnoses=["no"], confidence_level=0.8),
            ClinicalOpinion(top_1_diagnosis="no", top_3_differential_diagnoses=["no"], confidence_level=0.8),
            ClinicalOpinion(top_1_diagnosis="no", top_3_differential_diagnoses=["no"], confidence_level=0.8),
            ClinicalOpinion(top_1_diagnosis="no", top_3_differential_diagnoses=["no"], confidence_level=0.8),
        ]
        bert = ClinicalOpinion(
            top_1_diagnosis="yes",
            top_3_differential_diagnoses=["yes"],
            confidence_level=0.96,
        )
        label, _, rule = aggregate_pubmedqa_decision(
            agents, bert_opinion=bert, mode="bert_gate", bert_gate_confidence=0.9
        )
        self.assertEqual(label, "no")
        self.assertEqual(rule, "panel_unanimous_override")

    def test_bert_gate_ignores_unanimous_maybe_override(self) -> None:
        agents = [
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.7),
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.7),
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.7),
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.7),
        ]
        bert = ClinicalOpinion(
            top_1_diagnosis="yes",
            top_3_differential_diagnoses=["yes"],
            confidence_level=0.94,
        )
        label, _, rule = aggregate_pubmedqa_decision(
            agents, bert_opinion=bert, mode="bert_gate", bert_gate_confidence=0.9
        )
        self.assertEqual(label, "yes")
        self.assertEqual(rule, "bert_gate")

class PubmedqaDebateTests(unittest.TestCase):
    def test_mock_pubmedqa_debate_produces_labels(self) -> None:
        agents = build_default_agents(MockInferenceBackend(), task_mode="pubmedqa")
        result = asyncio.run(
            DebateOrchestrator(agents, rounds=2).run(
                "RESEARCH QUESTION:\nIs therapy valuable?\nEVIDENCE:\nSignificantly improved outcomes."
            )
        )
        labels = [opinion_label(entry.opinion) for entry in result.final_opinions]
        self.assertTrue(all(label in {"yes", "no", "maybe"} for label in labels))
        vote, _ = majority_vote([entry.opinion for entry in result.final_opinions])
        self.assertIn(vote, {"yes", "no", "maybe"})


if __name__ == "__main__":
    unittest.main()