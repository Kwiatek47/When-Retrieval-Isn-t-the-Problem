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
from app.agents.aggregation import (
    aggregate_pubmedqa_decision,
    build_consensus_decision,
    extract_label,
    majority_vote,
    opinion_label,
)
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
                "safety_opinion",
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
                if "PATIENT CASE:" in user:
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
            DebateOrchestrator(agents, rounds=6)

    def test_allows_up_to_five_rounds(self) -> None:
        agents = build_default_agents(MockInferenceBackend())
        orch = DebateOrchestrator(agents, rounds=5)
        self.assertEqual(orch.max_rounds, 5)
        result = asyncio.run(orch.run(SAMPLE_CASE))
        self.assertEqual(len(result.rounds), 5)

    def test_adaptive_rounds_stops_after_min_when_unanimous(self) -> None:
        class UnanimousBackend(MockInferenceBackend):
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                return ClinicalOpinion(
                    top_1_diagnosis="yes",
                    evidence_conclusiveness="conclusive",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.9,
                    sources_used=["abstract"],
                ).model_dump_json()

        agents = build_default_agents(UnanimousBackend(), task_mode="pubmedqa")
        orch = DebateOrchestrator(
            agents,
            adaptive_rounds=True,
            min_rounds=2,
            max_rounds=5,
            debate_mode="peer",
        )
        result = asyncio.run(
            orch.run("RESEARCH QUESTION:\nIs X useful?\nEVIDENCE:\nvaluable")
        )
        self.assertEqual(len(result.rounds), 2)
        self.assertEqual(orch.adaptive_stops, 1)

    def test_adaptive_rounds_continues_on_conflict(self) -> None:
        class AlternatingBackend(MockInferenceBackend):
            def __init__(self) -> None:
                self.n = 0

            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                self.n += 1
                label = "yes" if self.n % 2 else "no"
                return ClinicalOpinion(
                    top_1_diagnosis=label,
                    evidence_conclusiveness="inconclusive",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.5,
                    sources_used=["abstract"],
                ).model_dump_json()

        agents = build_default_agents(AlternatingBackend(), task_mode="pubmedqa")
        orch = DebateOrchestrator(
            agents,
            adaptive_rounds=True,
            min_rounds=2,
            max_rounds=4,
            debate_mode="peer",
        )
        result = asyncio.run(
            orch.run("RESEARCH QUESTION:\nIs X useful?\nEVIDENCE:\nmixed")
        )
        self.assertEqual(len(result.rounds), 4)
        self.assertEqual(orch.adaptive_stops, 0)

    def test_early_exit_skips_later_rounds(self) -> None:
        from app.agents.orchestrator import check_early_exit_asymmetric_veto

        def stop_after_round1(
            round_number: int, opinions: list, patient_case: str
        ) -> bool:
            return round_number == 1 and check_early_exit_asymmetric_veto(
                opinions, patient_case=patient_case
            )

        # Force unanimous high-confidence yes via a custom backend.
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

    def test_asymmetric_veto_blocks_maybe_and_low_confidence(self) -> None:
        from app.agents.models import AgentRoundOpinion
        from app.agents.orchestrator import check_early_exit_asymmetric_veto

        def _entry(agent_id: str, label: str, confidence: float) -> AgentRoundOpinion:
            return AgentRoundOpinion(
                agent_id=agent_id,
                persona=agent_id,
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis=label,
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=confidence,
                ),
            )

        panel_yes = [
            _entry("generalist", "yes", 0.9),
            _entry("evidence_skeptic", "yes", 0.85),
            _entry("differential_expander", "yes", 0.8),
            _entry("uncertainty_advocate", "yes", 0.9),
        ]
        self.assertTrue(check_early_exit_asymmetric_veto(panel_yes))

        advocate_maybe = list(panel_yes)
        advocate_maybe[-1] = _entry("uncertainty_advocate", "maybe", 0.9)
        self.assertFalse(check_early_exit_asymmetric_veto(advocate_maybe))

        advocate_low_conf = list(panel_yes)
        advocate_low_conf[-1] = _entry("uncertainty_advocate", "yes", 0.5)
        self.assertFalse(check_early_exit_asymmetric_veto(advocate_low_conf))

        unanimous_maybe = [
            _entry("generalist", "maybe", 0.9),
            _entry("evidence_skeptic", "maybe", 0.9),
            _entry("differential_expander", "maybe", 0.9),
            _entry("uncertainty_advocate", "maybe", 0.9),
        ]
        self.assertFalse(check_early_exit_asymmetric_veto(unanimous_maybe))

        # Without advocate, binary unanimity alone is enough.
        no_advocate = panel_yes[:3]
        self.assertTrue(check_early_exit_asymmetric_veto(no_advocate))

        # Keyword heuristic blocks early-exit even on unanimous yes.
        self.assertFalse(
            check_early_exit_asymmetric_veto(
                panel_yes,
                patient_case="EVIDENCE: Further research is needed before conclusions.",
            )
        )

    def test_blind_critic_hides_hint_from_advocate_in_round1(self) -> None:
        from app.agents.agent import ClinicalAgent
        from app.agents.backends import EvidenceHint

        flags: dict[str, list[bool]] = {}

        class SpyAgent(ClinicalAgent):
            async def generate_opinion(
                self,
                patient_case: str,
                context=None,
                *,
                include_evidence_hint: bool = True,
            ):
                flags.setdefault(self.agent_id, []).append(include_evidence_hint)
                return await super().generate_opinion(
                    patient_case,
                    context=context,
                    include_evidence_hint=include_evidence_hint,
                )

        class StaticHint:
            def get_hint(self, patient_case: str) -> EvidenceHint | None:
                return EvidenceHint(label="yes", confidence=0.95, model_path="test")

        backend = MockInferenceBackend()
        hint = StaticHint()
        agents = [
            SpyAgent(
                agent_id=aid,
                persona=persona,
                backend=backend,
                hint_provider=hint,  # type: ignore[arg-type]
                task_mode="pubmedqa",
            )
            for aid, persona in (
                ("generalist", "generalist"),
                ("evidence_skeptic", "evidence_skeptic"),
                ("differential_expander", "differential_expander"),
                ("uncertainty_advocate", "uncertainty_advocate"),
            )
        ]
        orch = DebateOrchestrator(agents, rounds=2, debate_mode="peer")
        asyncio.run(orch.run("RESEARCH QUESTION:\nQ?\nEVIDENCE:\nstrong result"))

        self.assertEqual(flags["generalist"][0], True)
        self.assertEqual(flags["differential_expander"][0], True)
        self.assertEqual(flags["uncertainty_advocate"][0], False)
        # Round 2: advocate sees hint again.
        self.assertEqual(flags["uncertainty_advocate"][1], True)

    def test_abstract_suggests_inconclusive(self) -> None:
        from app.agents.orchestrator import abstract_suggests_inconclusive

        self.assertTrue(
            abstract_suggests_inconclusive("Due to small sample size, results are tentative.")
        )
        self.assertTrue(
            abstract_suggests_inconclusive("LIMITATION: selection bias may affect results.")
        )
        self.assertFalse(
            abstract_suggests_inconclusive("A large RCT showed a clear benefit.")
        )

    def test_supervisor_uses_optional_separate_backend(self) -> None:
        agent_backend = MockInferenceBackend()
        supervisor_calls = {"n": 0}

        class SupervisorOnlyBackend(MockInferenceBackend):
            async def complete(
                self, messages: list[ChatMessage], *, temperature: float = 0.3
            ) -> str:
                supervisor_calls["n"] += 1
                return await super().complete(messages, temperature=temperature)

        agents = build_default_agents(agent_backend)
        orch = DebateOrchestrator(
            agents,
            rounds=2,
            supervisor_backend=SupervisorOnlyBackend(),
        )
        asyncio.run(orch.run(SAMPLE_CASE))
        self.assertGreaterEqual(supervisor_calls["n"], 1)
        self.assertIs(orch.supervisor.backend.__class__, SupervisorOnlyBackend)

    def test_supervisor_instructions_injected_on_round_2(self) -> None:
        """Round 2+ prompts include moderation instructions from the Supervisor."""
        captured: list[str] = []

        class SpyBackend(MockInferenceBackend):
            async def complete(
                self, messages: list[ChatMessage], *, temperature: float = 0.3
            ) -> str:
                user = next(m.content for m in messages if m.role == "user")
                if "PATIENT CASE:" in user:
                    captured.append(user)
                return await super().complete(messages, temperature=temperature)

        agents = build_default_agents(SpyBackend())
        asyncio.run(DebateOrchestrator(agents, rounds=2).run(SAMPLE_CASE))

        self.assertEqual(len(captured), 8)
        round1_calls = captured[:4]
        round2_calls = captured[4:]

        self.assertTrue(
            all(
                "Oto wnioski i instrukcje od Supervisora z poprzedniej rundy" not in call
                for call in round1_calls
            )
        )
        self.assertTrue(
            all(
                "Oto wnioski i instrukcje od Supervisora z poprzedniej rundy"
                in call
                for call in round2_calls
            )
        )

    def test_peer_mode_skips_supervisor_moderation(self) -> None:
        agents = build_default_agents(MockInferenceBackend())
        result = asyncio.run(
            DebateOrchestrator(agents, rounds=2, debate_mode="peer").run(SAMPLE_CASE)
        )
        self.assertEqual(result.supervisor_moderation, [])
        self.assertEqual(DebateOrchestrator(agents, rounds=2, debate_mode="peer").ARCHITECTURE, "peer_round_robin")

    def test_hybrid_mode_includes_peer_and_supervisor_context(self) -> None:
        captured: list[str] = []

        class SpyBackend(MockInferenceBackend):
            async def complete(
                self, messages: list[ChatMessage], *, temperature: float = 0.3
            ) -> str:
                user = next(m.content for m in messages if m.role == "user")
                if "PATIENT CASE:" in user and "PEER OPINIONS" in user:
                    captured.append(user)
                return await super().complete(messages, temperature=temperature)

        agents = build_default_agents(SpyBackend())
        asyncio.run(DebateOrchestrator(agents, rounds=2, debate_mode="hybrid").run(SAMPLE_CASE))

        self.assertGreaterEqual(len(captured), 1)
        self.assertTrue(
            any(
                "Oto wnioski i instrukcje od Supervisora z poprzedniej rundy" in call
                and "generalist" in call
                for call in captured
            )
        )

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

    def test_advocate_label_rule_omits_maybe_warning(self) -> None:
        advocate = build_messages(
            agent_id="uncertainty_advocate",
            persona="uncertainty_advocate",
            patient_case="Question: Is X useful?\nEvidence: ...",
            task_mode="pubmedqa",
        )[0].content
        generalist = build_messages(
            agent_id="generalist",
            persona="generalist",
            patient_case="Question: Is X useful?\nEvidence: ...",
            task_mode="pubmedqa",
        )[0].content
        self.assertNotIn("WARNING", advocate)
        self.assertIn("express genuine uncertainty", advocate)
        self.assertIn("WARNING", generalist)

    def test_director_prompt_includes_case_and_transcript(self) -> None:
        from app.agents.prompts import SUPERVISOR_DIRECTOR_PROMPT

        filled = SUPERVISOR_DIRECTOR_PROMPT.format(
            patient_case="CASE_TEXT_XYZ",
            full_debate_transcript="DEBATE_TEXT_XYZ",
            biolinkbert_hint="HINT_TEXT_XYZ",
        )
        self.assertIn("CASE_TEXT_XYZ", filled)
        self.assertIn("DEBATE_TEXT_XYZ", filled)
        self.assertIn("HINT_TEXT_XYZ", filled)
        self.assertIn("fallback", filled.lower())
        self.assertNotIn("ClinicalOpinion JSON schema", filled)

    def test_parse_recovers_json_embedded_in_prose(self) -> None:
        raw = (
            'Here is my answer:\n'
            '{"top_1_diagnosis":"yes","top_3_differential_diagnoses":["yes","no","maybe"],'
            '"confidence_level":0.8}\nThanks'
        )
        parsed = parse_clinical_opinion_json(raw)
        self.assertEqual(parsed.top_1_diagnosis, "yes")

    def test_parse_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            parse_clinical_opinion_json("   ")


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

    def test_build_consensus_decision_marks_unanimous_as_consensus(self) -> None:
        opinions = [
            ClinicalOpinion(
                top_1_diagnosis="yes",
                evidence_conclusiveness="conclusive",
                top_3_differential_diagnoses=["yes"],
                confidence_level=0.9,
                sources_used=["abstract"],
            )
            for _ in range(4)
        ]
        decision = build_consensus_decision(
            opinions,
            predicted_label="yes",
            vote_share={"yes": 1.0, "no": 0.0, "maybe": 0.0},
        )
        self.assertEqual(decision.mode, "consensus")
        self.assertEqual(decision.final_label, "yes")


class MultiOllamaHelpersTests(unittest.TestCase):
    def test_parse_ollama_base_urls(self) -> None:
        from app.agents.backends import parse_ollama_base_urls, sticky_ollama_url

        self.assertEqual(
            parse_ollama_base_urls(None, default="http://localhost:11434"),
            ["http://localhost:11434"],
        )
        self.assertEqual(
            parse_ollama_base_urls(
                "http://127.0.0.1:11434, http://127.0.0.1:11435/",
                default="http://localhost:11434",
            ),
            ["http://127.0.0.1:11434", "http://127.0.0.1:11435"],
        )
        urls = ["http://a:1", "http://b:2", "http://c:3"]
        self.assertEqual(sticky_ollama_url(urls, 1), "http://a:1")
        self.assertEqual(sticky_ollama_url(urls, 2), "http://b:2")
        self.assertEqual(sticky_ollama_url(urls, 4), "http://a:1")


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