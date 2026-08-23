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


class EvaluateDebateCliTests(unittest.TestCase):
    def test_parse_supervisor_base_url(self) -> None:
        from scripts.agents.evaluate_debate_pubmedqa import _parse_args
        from unittest.mock import patch

        argv = [
            "evaluate_debate_pubmedqa.py",
            "--backend",
            "ollama",
            "--supervisor-model",
            "qwen2.5:14b",
            "--supervisor-base-url",
            "http://127.0.0.1:11437",
        ]
        with patch("sys.argv", argv):
            args = _parse_args()
        self.assertEqual(args.supervisor_model, "qwen2.5:14b")
        self.assertEqual(args.supervisor_base_url, "http://127.0.0.1:11437")


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
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
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
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
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
        self.assertFalse(result.exhausted_without_consensus)

    def test_adaptive_rounds_continues_on_conflict(self) -> None:
        class ConflictBackend(MockInferenceBackend):
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
                system = messages[0].content if messages else ""
                if "agent_id=generalist" in system:
                    label = "yes"
                elif "agent_id=differential_expander" in system:
                    label = "yes"
                elif "agent_id=evidence_skeptic" in system:
                    label = "no"
                elif "agent_id=uncertainty_advocate" in system:
                    label = "no"
                else:
                    label = "maybe"
                return ClinicalOpinion(
                    top_1_diagnosis=label,
                    evidence_conclusiveness="inconclusive",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.5,
                    sources_used=["abstract"],
                ).model_dump_json()

        agents = build_default_agents(ConflictBackend(), task_mode="pubmedqa")
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
        self.assertTrue(result.exhausted_without_consensus)

    def test_fundamental_conflict_2_2_and_guardian_alliance(self) -> None:
        from app.agents.models import AgentRoundOpinion
        from app.agents.orchestrator import fundamental_panel_conflict

        def _panel(labels: dict[str, str]) -> list[AgentRoundOpinion]:
            return [
                AgentRoundOpinion(
                    agent_id=agent_id,
                    persona=agent_id,
                    round=4,
                    opinion=ClinicalOpinion(
                        top_1_diagnosis=label,
                        top_3_differential_diagnoses=["yes", "no", "maybe"],
                        confidence_level=0.8,
                        sources_used=["abstract"],
                    ),
                )
                for agent_id, label in labels.items()
            ]

        self.assertTrue(
            fundamental_panel_conflict(
                _panel(
                    {
                        "generalist": "yes",
                        "differential_expander": "yes",
                        "evidence_skeptic": "no",
                        "uncertainty_advocate": "no",
                    }
                )
            )
        )
        self.assertTrue(
            fundamental_panel_conflict(
                _panel(
                    {
                        "generalist": "yes",
                        "differential_expander": "yes",
                        "evidence_skeptic": "no",
                        "uncertainty_advocate": "maybe",
                    }
                )
            )
        )
        self.assertTrue(
            fundamental_panel_conflict(
                _panel(
                    {
                        "generalist": "yes",
                        "differential_expander": "no",
                        "evidence_skeptic": "maybe",
                        "uncertainty_advocate": "maybe",
                    }
                )
            )
        )
        # Only uncertainty_advocate maybe (evidence_skeptic agrees) → not fundamental.
        self.assertFalse(
            fundamental_panel_conflict(
                _panel(
                    {
                        "generalist": "yes",
                        "differential_expander": "yes",
                        "evidence_skeptic": "yes",
                        "uncertainty_advocate": "maybe",
                    }
                )
            )
        )

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
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
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
        advocate_maybe[3] = _entry("uncertainty_advocate", "maybe", 0.9)
        self.assertFalse(check_early_exit_asymmetric_veto(advocate_maybe))

        advocate_low_conf = list(panel_yes)
        advocate_low_conf[3] = _entry("uncertainty_advocate", "yes", 0.5)
        self.assertFalse(check_early_exit_asymmetric_veto(advocate_low_conf))

        unanimous_maybe = [
            _entry("generalist", "maybe", 0.9),
            _entry("evidence_skeptic", "maybe", 0.9),
            _entry("differential_expander", "maybe", 0.9),
            _entry("uncertainty_advocate", "maybe", 0.9),
        ]
        self.assertFalse(check_early_exit_asymmetric_veto(unanimous_maybe))

        # Without uncertainty experts, binary unanimity alone is enough.
        no_experts = panel_yes[:3]
        self.assertTrue(check_early_exit_asymmetric_veto(no_experts))

        # Keyword heuristic blocks early-exit even on unanimous yes.
        self.assertFalse(
            check_early_exit_asymmetric_veto(
                panel_yes,
                patient_case="EVIDENCE: Further research is needed before conclusions.",
            )
        )

    def test_blind_critic_hides_hint_from_advocate_all_rounds_by_default(self) -> None:
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
                frozen_label: str | None = None,
                num_predict: int | None = None,
                moderator_instruction: str | None = None,
                round_number: int | None = None,
            ):
                flags.setdefault(self.agent_id, []).append(include_evidence_hint)
                return await super().generate_opinion(
                    patient_case,
                    context=context,
                    include_evidence_hint=include_evidence_hint,
                    frozen_label=frozen_label,
                    num_predict=num_predict,
                    moderator_instruction=moderator_instruction,
                    round_number=round_number,
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
        # Default all-rounds: uncertainty expert stays blind in round 2.
        self.assertEqual(flags["uncertainty_advocate"][1], False)

    def test_blind_critic_r1_only_restores_hint_in_round2(self) -> None:
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
                frozen_label: str | None = None,
                num_predict: int | None = None,
                moderator_instruction: str | None = None,
                round_number: int | None = None,
            ):
                flags.setdefault(self.agent_id, []).append(include_evidence_hint)
                return await super().generate_opinion(
                    patient_case,
                    context=context,
                    include_evidence_hint=include_evidence_hint,
                    frozen_label=frozen_label,
                    num_predict=num_predict,
                    moderator_instruction=moderator_instruction,
                    round_number=round_number,
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
        orch = DebateOrchestrator(
            agents, rounds=2, debate_mode="peer", blind_critic="r1-only"
        )
        asyncio.run(orch.run("RESEARCH QUESTION:\nQ?\nEVIDENCE:\nstrong result"))

        self.assertEqual(flags["uncertainty_advocate"][0], False)
        self.assertEqual(flags["uncertainty_advocate"][1], True)

    def test_frozen_stance_flag_controls_round2_frozen_label(self) -> None:
        from app.agents.agent import ClinicalAgent

        frozen_seen: dict[str, list[str | None]] = {}

        class SpyAgent(ClinicalAgent):
            async def generate_opinion(
                self,
                patient_case: str,
                context=None,
                *,
                include_evidence_hint: bool = True,
                frozen_label: str | None = None,
                num_predict: int | None = None,
                moderator_instruction: str | None = None,
                round_number: int | None = None,
            ):
                frozen_seen.setdefault(self.agent_id, []).append(frozen_label)
                return await super().generate_opinion(
                    patient_case,
                    context=context,
                    include_evidence_hint=include_evidence_hint,
                    frozen_label=frozen_label,
                    num_predict=num_predict,
                    moderator_instruction=moderator_instruction,
                    round_number=round_number,
                )

        backend = MockInferenceBackend()
        agents = [
            SpyAgent(
                agent_id=aid,
                persona=persona,
                backend=backend,
                task_mode="pubmedqa",
            )
            for aid, persona in (
                ("generalist", "generalist"),
                ("evidence_skeptic", "evidence_skeptic"),
                ("differential_expander", "differential_expander"),
                ("uncertainty_advocate", "uncertainty_advocate"),
            )
        ]
        case = "RESEARCH QUESTION:\nQ?\nEVIDENCE:\nstrong result"

        asyncio.run(
            DebateOrchestrator(
                agents, rounds=2, debate_mode="peer", frozen_stance=False
            ).run(case)
        )
        for agent_id, values in frozen_seen.items():
            self.assertEqual(values, [None, None], msg=agent_id)

        frozen_seen.clear()
        asyncio.run(
            DebateOrchestrator(
                agents, rounds=2, debate_mode="peer", frozen_stance=True
            ).run(case)
        )
        for agent_id, values in frozen_seen.items():
            self.assertIsNone(values[0], msg=agent_id)
            self.assertIsNotNone(values[1], msg=agent_id)

    def test_round3_uses_higher_num_predict(self) -> None:
        from app.agents.agent import ClinicalAgent

        predict_seen: list[int | None] = []

        class SpyAgent(ClinicalAgent):
            async def generate_opinion(
                self,
                patient_case: str,
                context=None,
                *,
                include_evidence_hint: bool = True,
                frozen_label: str | None = None,
                num_predict: int | None = None,
                moderator_instruction: str | None = None,
                round_number: int | None = None,
            ):
                predict_seen.append(num_predict)
                return await super().generate_opinion(
                    patient_case,
                    context=context,
                    include_evidence_hint=include_evidence_hint,
                    frozen_label=frozen_label,
                    num_predict=num_predict,
                    moderator_instruction=moderator_instruction,
                    round_number=round_number,
                )

        backend = MockInferenceBackend()
        agents = [
            SpyAgent(
                agent_id=aid,
                persona=persona,
                backend=backend,
                task_mode="pubmedqa",
            )
            for aid, persona in (
                ("generalist", "generalist"),
                ("evidence_skeptic", "evidence_skeptic"),
                ("differential_expander", "differential_expander"),
                ("uncertainty_advocate", "uncertainty_advocate"),
            )
        ]
        asyncio.run(
            DebateOrchestrator(
                agents,
                rounds=3,
                debate_mode="peer",
                agent_num_predict_round3=1500,
            ).run("RESEARCH QUESTION:\nQ?\nEVIDENCE:\nstrong result")
        )
        # 4 agents x 3 rounds = 12 calls; round 3 should use 1500
        self.assertEqual(predict_seen[:4], [None, None, None, None])
        self.assertEqual(predict_seen[4:8], [None, None, None, None])
        self.assertEqual(predict_seen[8:], [1500, 1500, 1500, 1500])

    def test_safety_red_flag_halts_debate_by_default(self) -> None:
        from app.agents.agent import ClinicalAgent
        from app.agents.models import SafetyOpinion

        class SafetySpy(ClinicalAgent):
            async def generate_opinion(self, patient_case: str, context=None, **kwargs):
                if self.agent_id == "safety_officer":
                    return ClinicalOpinion(
                        top_1_diagnosis="Acute coronary syndrome",
                        top_3_differential_diagnoses=["ACS", "PE", "Aortic dissection"],
                        confidence_level=0.9,
                        sources_used=["case"],
                        safety_opinion=SafetyOpinion(
                            safety_passed=False,
                            red_flags_detected=["chest pain + hypoxia"],
                            immediate_intervention_required=True,
                            reasoning="Can't-miss emergency",
                        ),
                    )
                return await super().generate_opinion(
                    patient_case, context=context, **kwargs
                )

        backend = MockInferenceBackend()
        agents = [
            SafetySpy(agent_id=aid, persona=persona, backend=backend)
            for aid, persona in (
                ("generalist", "generalist"),
                ("evidence_skeptic", "evidence_skeptic"),
                ("differential_expander", "differential_expander"),
                ("safety_officer", "safety_officer"),
            )
        ]
        orch = DebateOrchestrator(agents, rounds=3, debate_mode="peer")
        result = asyncio.run(orch.run(SAMPLE_CASE))
        self.assertTrue(result.safety_halted)
        self.assertEqual(len(result.rounds), 1)
        self.assertIn("RED FLAG", result.safety_red_flag_reason or "")
        self.assertEqual(orch.safety_halts, 1)

    def test_safety_red_flag_defer_continues_debate(self) -> None:
        from app.agents.agent import ClinicalAgent
        from app.agents.models import SafetyOpinion

        class SafetySpy(ClinicalAgent):
            async def generate_opinion(self, patient_case: str, context=None, **kwargs):
                if self.agent_id == "safety_officer":
                    return ClinicalOpinion(
                        top_1_diagnosis="Acute coronary syndrome",
                        top_3_differential_diagnoses=["ACS", "PE", "Aortic dissection"],
                        confidence_level=0.9,
                        sources_used=["case"],
                        safety_opinion=SafetyOpinion(
                            safety_passed=False,
                            red_flags_detected=["chest pain"],
                            immediate_intervention_required=True,
                            reasoning="urgent",
                        ),
                    )
                return await super().generate_opinion(
                    patient_case, context=context, **kwargs
                )

        backend = MockInferenceBackend()
        agents = [
            SafetySpy(agent_id=aid, persona=persona, backend=backend)
            for aid, persona in (
                ("generalist", "generalist"),
                ("evidence_skeptic", "evidence_skeptic"),
                ("differential_expander", "differential_expander"),
                ("safety_officer", "safety_officer"),
            )
        ]
        orch = DebateOrchestrator(
            agents, rounds=3, debate_mode="peer", safety_red_flag="defer"
        )
        result = asyncio.run(orch.run(SAMPLE_CASE))
        self.assertFalse(result.safety_halted)
        self.assertEqual(len(result.rounds), 3)

    def test_abstract_suggests_inconclusive(self) -> None:
        from app.agents.heuristics import abstract_suggests_inconclusive

        self.assertTrue(
            abstract_suggests_inconclusive("Due to small sample size, results are tentative.")
        )
        self.assertTrue(
            abstract_suggests_inconclusive("Results remain unclear; further research is needed.")
        )
        # Broad "limitation(s)" alone must NOT trip the heuristic.
        self.assertFalse(
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
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.3,
                num_predict: int | None = None,
            ) -> str:
                supervisor_calls["n"] += 1
                return await super().complete(
                    messages, temperature=temperature, num_predict=num_predict
                )

        agents = build_default_agents(agent_backend)
        orch = DebateOrchestrator(
            agents,
            rounds=2,
            supervisor_backend=SupervisorOnlyBackend(),
        )
        asyncio.run(orch.run(SAMPLE_CASE))
        self.assertGreaterEqual(supervisor_calls["n"], 1)
        self.assertIs(orch.supervisor.backend.__class__, SupervisorOnlyBackend)

    def test_supervisor_fail_peer_round_skips_empty_moderation(self) -> None:
        agent_backend = MockInferenceBackend()
        captured: list[str] = []

        class BrokenSupervisorBackend:
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
                return "NOT_VALID_JSON{{"

        class SpyAgentBackend(MockInferenceBackend):
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
                user = next(m.content for m in messages if m.role == "user")
                if "PATIENT CASE:" in user:
                    captured.append(user)
                return await super().complete(messages, temperature=temperature)

        agents = build_default_agents(SpyAgentBackend())
        orch = DebateOrchestrator(
            agents,
            rounds=2,
            debate_mode="moderated",
            supervisor_backend=BrokenSupervisorBackend(),
            supervisor_fail="peer-round",
        )
        result = asyncio.run(orch.run(SAMPLE_CASE))
        self.assertEqual(orch.supervisor_failovers, 1)
        self.assertEqual(len(result.supervisor_moderation), 0)
        self.assertTrue(orch.supervisor.last_moderation_failed)
        # Round 2 should be peer critique, not empty supervisor JSON noise.
        round2 = captured[4:]
        self.assertTrue(any("PEER OPINIONS FROM THE PREVIOUS ROUND" in text for text in round2))
        self.assertFalse(any("re-run moderation" in text for text in round2))

    def test_supervisor_fail_empty_defer_injects_fallback(self) -> None:
        class BrokenSupervisorBackend:
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
                return "NOT_VALID_JSON{{"

        agents = build_default_agents(MockInferenceBackend())
        orch = DebateOrchestrator(
            agents,
            rounds=2,
            debate_mode="moderated",
            supervisor_backend=BrokenSupervisorBackend(),
            supervisor_fail="empty-defer",
        )
        result = asyncio.run(orch.run(SAMPLE_CASE))
        self.assertEqual(orch.supervisor_failovers, 0)
        self.assertEqual(len(result.supervisor_moderation), 1)
        self.assertIn("re-run moderation", result.supervisor_moderation[0].round_instructions[0])

    def test_supervisor_instructions_injected_on_round_2(self) -> None:
        """Round 2+ prompts include moderation instructions from the Supervisor."""
        captured: list[str] = []

        class SpyBackend(MockInferenceBackend):
            async def complete(
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.3,
                num_predict: int | None = None,
            ) -> str:
                user = next(m.content for m in messages if m.role == "user")
                if "PATIENT CASE:" in user:
                    captured.append(user)
                return await super().complete(
                    messages, temperature=temperature, num_predict=num_predict
                )

        agents = build_default_agents(SpyBackend())
        asyncio.run(DebateOrchestrator(agents, rounds=2).run(SAMPLE_CASE))

        self.assertEqual(len(captured), 8)
        round1_calls = captured[:4]
        round2_calls = captured[4:]

        self.assertTrue(
            all(
                "[SYSTEM INSTRUCTION FROM MODERATOR]" not in call
                for call in round1_calls
            )
        )
        self.assertTrue(
            all(
                "[SYSTEM INSTRUCTION FROM MODERATOR]" in call
                and "Supervisor moderation from the previous round" in call
                for call in round2_calls
            )
        )

    def test_peer_mode_skips_supervisor_moderation(self) -> None:
        agents = build_default_agents(MockInferenceBackend())
        result = asyncio.run(
            DebateOrchestrator(agents, rounds=2, debate_mode="peer").run(SAMPLE_CASE)
        )
        self.assertEqual(result.supervisor_moderation, [])
        self.assertEqual(DebateOrchestrator(agents, rounds=2, debate_mode="peer").ARCHITECTURE, "peer_parallel")

    def test_parallel_peer_round_sees_only_previous_round(self) -> None:
        """R2 agents must not observe same-round peer drafts (no round-robin)."""
        captured: list[str] = []

        class SpyBackend(MockInferenceBackend):
            async def complete(
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.3,
                num_predict: int | None = None,
            ) -> str:
                user = next(m.content for m in messages if m.role == "user")
                if "PATIENT CASE:" in user and "PEER OPINIONS" in user:
                    captured.append(user)
                return await super().complete(
                    messages, temperature=temperature, num_predict=num_predict
                )

        agents = build_default_agents(SpyBackend())
        asyncio.run(
            DebateOrchestrator(agents, rounds=2, debate_mode="peer").run(SAMPLE_CASE)
        )
        self.assertEqual(len(captured), 4)
        for user in captured:
            self.assertIn("PEER OPINIONS FROM THE PREVIOUS ROUND", user)
            self.assertIn("write in isolation", user)
            # Round-1 opinions are marked [R1]; no same-round [R2] peer drafts.
            self.assertNotIn("[R2]", user)
            self.assertNotIn("[R2*]", user)

    def test_hybrid_mode_includes_peer_and_supervisor_context(self) -> None:
        captured: list[str] = []

        class SpyBackend(MockInferenceBackend):
            async def complete(
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.3,
                num_predict: int | None = None,
            ) -> str:
                user = next(m.content for m in messages if m.role == "user")
                if "PATIENT CASE:" in user and "PEER OPINIONS" in user:
                    captured.append(user)
                return await super().complete(
                    messages, temperature=temperature, num_predict=num_predict
                )

        agents = build_default_agents(SpyBackend())
        asyncio.run(DebateOrchestrator(agents, rounds=2, debate_mode="hybrid").run(SAMPLE_CASE))

        self.assertGreaterEqual(len(captured), 1)
        self.assertTrue(
            any(
                "[SYSTEM INSTRUCTION FROM MODERATOR]" in call
                and "Supervisor moderation from the previous round:" in call
                and ("generalist" in call or "clinical debate agent `generalist`" in call)
                for call in captured
            )
        )

    def test_orchestrator_round_survives_one_agent_backend_failure(self) -> None:
        """A single flaky backend must not crash the whole debate round."""

        class ExplodingBackend:
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
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

            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
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
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None) -> str:
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

    def test_peer_context_nl_renders_natural_language(self) -> None:
        from app.agents.models import AgentRoundOpinion
        from app.agents.prompts import format_opinion_nl

        entry = AgentRoundOpinion(
            agent_id="generalist",
            persona="generalist",
            round=1,
            opinion=ClinicalOpinion(
                top_1_diagnosis="yes",
                evidence_conclusiveness="conclusive",
                top_3_differential_diagnoses=["yes", "no", "maybe"],
                pros=["authors report significant primary benefit"],
                cons=["small sample size"],
                confidence_level=0.8,
                sources_used=["abstract"],
            ),
        )
        nl = format_opinion_nl(entry, current_round=2)
        self.assertIn("[R1] generalist (conf=0.80, conclusive): yes", nl)
        self.assertIn("Pro: authors report significant primary benefit", nl)
        self.assertIn("Con: small sample size", nl)

        messages = build_messages(
            agent_id="evidence_skeptic",
            persona="evidence_skeptic",
            patient_case=SAMPLE_CASE,
            context=[entry],
            peer_context="nl",
            task_mode="pubmedqa",
        )
        user = messages[1].content
        self.assertIn("PEER OPINIONS FROM THE PREVIOUS ROUND", user)
        self.assertIn("generalist (conf=0.80, conclusive): yes", user)
        self.assertNotIn('"agent_id"', user)
        self.assertIn("[uncertainty_advocate]", user)
        self.assertIn("MUST explicitly name an agent you disagree with", user)
        self.assertIn("Synthesize your own counter-arguments", user)
        self.assertIn("valid JSON", user)

    def test_round1_prompt_omits_structured_criticism_rule(self) -> None:
        messages = build_messages(
            agent_id="generalist",
            persona="generalist",
            patient_case=SAMPLE_CASE,
            task_mode="pubmedqa",
        )
        user = messages[1].content
        self.assertIn("independent first-round opinion", user)
        self.assertNotIn("MUST explicitly name an agent you disagree with", user)

    def test_peer_context_compact_json_still_dumps_json(self) -> None:
        from app.agents.models import AgentRoundOpinion

        entry = AgentRoundOpinion(
            agent_id="generalist",
            persona="generalist",
            round=1,
            opinion=ClinicalOpinion(
                top_1_diagnosis="no",
                evidence_conclusiveness="inconclusive",
                top_3_differential_diagnoses=["no", "yes", "maybe"],
                pros=["null primary"],
                cons=[],
                confidence_level=0.6,
                sources_used=["abstract"],
            ),
        )
        messages = build_messages(
            agent_id="evidence_skeptic",
            persona="evidence_skeptic",
            patient_case=SAMPLE_CASE,
            context=[entry],
            peer_context="compact-json",
            task_mode="pubmedqa",
        )
        user = messages[1].content
        self.assertIn('"agent_id"', user)
        self.assertIn('"label"', user)

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
        self.assertIn("Coverage Auditor", advocate)
        # Measured at 87% "maybe" fire rate, the persona was a constant rather than a
        # detector; the prompt must push it back toward discriminating.
        self.assertIn("DEFAULT: most abstracts DO answer their own question", advocate)
        self.assertIn("You are a DETECTOR, not an advocate", advocate)
        self.assertIn("These do NOT justify 'maybe'", advocate)

    def test_frozen_stance_injects_adversarial_directive_in_round2(self) -> None:
        from app.agents.models import AgentRoundOpinion

        entry = AgentRoundOpinion(
            agent_id="generalist",
            persona="generalist",
            round=2,
            opinion=ClinicalOpinion(
                top_1_diagnosis="yes",
                evidence_conclusiveness="conclusive",
                top_3_differential_diagnoses=["yes", "no", "maybe"],
                pros=["primary endpoint met"],
                cons=[],
                confidence_level=0.8,
                sources_used=["abstract"],
            ),
        )
        messages = build_messages(
            agent_id="evidence_skeptic",
            persona="evidence_skeptic",
            patient_case=SAMPLE_CASE,
            context=[entry],
            task_mode="pubmedqa",
            frozen_label="no",
        )
        system = messages[0].content
        self.assertIn("[SYSTEM ARCHITECTURE OVERRIDE]", system)
        self.assertIn("defense attorney for the 'no' label", system)
        self.assertIn("internal_monologue", system)
        self.assertIn("best_evidence_supporting_my_label", system)
        self.assertIn("explicit_attack_on_opposing_peers", system)
        # internal_monologue must precede top_1_diagnosis in the defense schema block.
        self.assertLess(
            system.index("- internal_monologue"),
            system.index("- top_1_diagnosis (string, MUST BE EXACTLY"),
        )
        self.assertIn("MUST BE EXACTLY 'no'", system)
        self.assertNotIn("ALIGNMENT RULE", system)
        # Defense schema must not ask for generic pros/cons keys.
        self.assertNotIn("- pros (array of strings)", system)
        self.assertNotIn("- cons (array of strings)", system)

        round1_messages = build_messages(
            agent_id="evidence_skeptic",
            persona="evidence_skeptic",
            patient_case=SAMPLE_CASE,
            task_mode="pubmedqa",
            frozen_label=None,
        )
        self.assertNotIn("[SYSTEM ARCHITECTURE OVERRIDE]", round1_messages[0].content)
        self.assertNotIn("best_evidence_supporting_my_label", round1_messages[0].content)

    def test_parse_maps_defense_semantic_keys_to_pros_cons(self) -> None:
        raw = (
            '{"internal_monologue":"My label is no; [generalist] overclaims significance.",'
            '"top_1_diagnosis":"no","evidence_conclusiveness":"conclusive",'
            '"top_3_differential_diagnoses":["no","yes","maybe"],'
            '"best_evidence_supporting_my_label":["Null primary endpoint p=0.45"],'
            '"explicit_attack_on_opposing_peers":["[generalist] Ignores non-significance"],'
            '"confidence_level":0.8,"sources_used":["abstract"],'
            '"red_flags":[],"missing_information":""}'
        )
        parsed = parse_clinical_opinion_json(raw)
        self.assertEqual(parsed.top_1_diagnosis, "no")
        self.assertEqual(parsed.pros, ["Null primary endpoint p=0.45"])
        self.assertEqual(parsed.cons, ["[generalist] Ignores non-significance"])
        dumped = parsed.model_dump()
        self.assertNotIn("internal_monologue", dumped)

    def test_clinical_opinion_alias_choices_accept_semantic_keys(self) -> None:
        opinion = ClinicalOpinion.model_validate(
            {
                "top_1_diagnosis": "maybe",
                "top_3_differential_diagnoses": ["maybe", "yes", "no"],
                "best_evidence_supporting_my_label": ["Mixed endpoints"],
                "explicit_attack_on_opposing_peers": ["[evidence_skeptic] Overclaims"],
                "confidence_level": 0.7,
            }
        )
        self.assertEqual(opinion.pros, ["Mixed endpoints"])
        self.assertEqual(opinion.cons, ["[evidence_skeptic] Overclaims"])
        # Field-name construction still works for history / aggregation.
        via_name = ClinicalOpinion(
            top_1_diagnosis="yes",
            top_3_differential_diagnoses=["yes", "no", "maybe"],
            pros=["supports yes"],
            cons=["caveat"],
            confidence_level=0.6,
        )
        self.assertEqual(via_name.pros, ["supports yes"])
        dumped = via_name.model_dump()
        self.assertIn("pros", dumped)
        self.assertNotIn("best_evidence_supporting_my_label", dumped)

    def test_moderator_instruction_prepended_inside_peer_opinions_block(self) -> None:
        from app.agents.models import AgentRoundOpinion, SupervisorModerationOutput
        from app.agents.prompts import format_moderator_instruction_block

        moderation = SupervisorModerationOutput(
            agreements=["Panel agrees primary endpoint was significant"],
            contradictions=["Split on maybe vs yes"],
            round_instructions=["Reconcile using explicit p-values from the abstract"],
            primary_endpoint_result="p=0.02 favoring intervention",
            author_conclusion="yes",
            residual_uncertainty=[],
        )
        moderator_block = format_moderator_instruction_block(moderation, peer_context="nl")
        entry = AgentRoundOpinion(
            agent_id="generalist",
            persona="generalist",
            round=1,
            opinion=ClinicalOpinion(
                top_1_diagnosis="yes",
                evidence_conclusiveness="conclusive",
                top_3_differential_diagnoses=["yes", "no", "maybe"],
                pros=["significant endpoint"],
                cons=[],
                confidence_level=0.8,
                sources_used=["abstract"],
            ),
        )
        messages = build_messages(
            agent_id="evidence_skeptic",
            persona="evidence_skeptic",
            patient_case=SAMPLE_CASE,
            context=[entry],
            moderator_instruction=moderator_block,
            task_mode="pubmedqa",
        )
        user = messages[1].content
        self.assertIn("PEER OPINIONS FROM THE PREVIOUS ROUND", user)
        mod_idx = user.index("[SYSTEM INSTRUCTION FROM MODERATOR]")
        peer_idx = user.index("PEER OPINIONS FROM THE PREVIOUS ROUND")
        self.assertGreater(mod_idx, peer_idx)
        self.assertIn("Supervisor moderation from the previous round:", user)
        self.assertIn("generalist (conf=0.80, conclusive): yes", user)
        self.assertNotIn("[R1] supervisor", user)

    def test_director_prompt_includes_case_and_transcript(self) -> None:
        from app.agents.prompts import SUPERVISOR_DIRECTOR_PROMPT

        filled = SUPERVISOR_DIRECTOR_PROMPT.format(
            patient_case="CASE_TEXT_XYZ",
            full_debate_transcript="TRANSCRIPT_TEXT_XYZ",
            panel_vote_summary="VOTE_SUMMARY_XYZ",
        )
        self.assertIn("CASE_TEXT_XYZ", filled)
        self.assertIn("TRANSCRIPT_TEXT_XYZ", filled)
        self.assertIn("VOTE_SUMMARY_XYZ", filled)
        self.assertIn("CRITICAL RULES FOR CHOOSING THE LABEL", filled)
        self.assertIn("forced stubbornness", filled)
        self.assertIn("Devil's Advocate", filled)
        self.assertIn("BOILERPLATE", filled)
        # Votes are calibrated evidence now; the old absolute ban made the Director
        # follow whichever agent argued hardest, usually the lone maybe-hunter.
        self.assertNotIn("DO NOT TALLY VOTES", filled)
        self.assertIn("LONE DISSENT DOES NOT WIN BY DEFAULT", filled)
        self.assertIn("ROUND 1 IS THE HONEST SIGNAL", filled)
        self.assertNotIn("MUST output \"yes\" or \"no\"", filled)
        self.assertIn("final_label", filled)
        self.assertNotIn("ClinicalOpinion JSON schema", filled)
        self.assertNotIn("top_1_diagnosis", filled)

    def test_parse_recovers_json_embedded_in_prose(self) -> None:
        raw = (
            'Here is my answer:\n'
            '{"top_1_diagnosis":"yes","top_3_differential_diagnoses":["yes","no","maybe"],'
            '"confidence_level":0.8}\nThanks'
        )
        parsed = parse_clinical_opinion_json(raw)
        self.assertEqual(parsed.top_1_diagnosis, "yes")

    def test_parse_recovers_truncated_json_with_label(self) -> None:
        raw = (
            '{\n    "top_1_diagnosis": "no",\n    "evidence_conclusiveness": "conclusive",\n'
            '    "top_3_differential_diagnoses": ["no", "maybe", "no"],\n'
            '    "pros": ["Study shows high vitamin D '
        )
        parsed = parse_clinical_opinion_json(raw)
        self.assertEqual(parsed.top_1_diagnosis, "no")
        self.assertEqual(parsed.evidence_conclusiveness, "conclusive")

    def test_parse_coerces_structured_cons_objects(self) -> None:
        raw = (
            '{"top_1_diagnosis":"no","evidence_conclusiveness":"conclusive",'
            '"top_3_differential_diagnoses":["no","maybe","yes"],'
            '"pros":[],"cons":[{"uncertainty_advocate":"Overstates lack of evidence"}],'
            '"confidence_level":0.7}'
        )
        parsed = parse_clinical_opinion_json(raw)
        self.assertEqual(parsed.top_1_diagnosis, "no")
        self.assertEqual(len(parsed.cons), 1)
        self.assertIn("[uncertainty_advocate]", parsed.cons[0])

    def test_parse_sanitizes_possessive_in_pros(self) -> None:
        # Unescaped apostrophe inside a JSON string (common LLM mistake).
        raw = (
            '{"top_1_diagnosis":"yes","evidence_conclusiveness":"conclusive",'
            '"top_3_differential_diagnoses":["yes","no","maybe"],'
            '"pros":["Significant correlation in UC, supports calprotectin\'s role"],'
            '"cons":[],"confidence_level":0.8,"sources_used":["abstract"],'
            '"red_flags":[],"missing_information":""}'
        )
        parsed = parse_clinical_opinion_json(raw)
        self.assertEqual(parsed.top_1_diagnosis, "yes")
        self.assertTrue(parsed.pros)

    def test_parse_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            parse_clinical_opinion_json("   ")

    def test_maybe_director_gate_downgrades_yes(self) -> None:
        from app.agents.aggregation import apply_maybe_director_gate
        from app.agents.models import AgentRoundOpinion, SharedDebateReport, SupervisorDirectorOutput

        opinions = [
            AgentRoundOpinion(
                agent_id="generalist",
                persona="generalist",
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="yes",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.55,
                    sources_used=["abstract"],
                ),
            ),
            AgentRoundOpinion(
                agent_id="evidence_skeptic",
                persona="evidence_skeptic",
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="maybe",
                    top_3_differential_diagnoses=["maybe", "yes", "no"],
                    confidence_level=0.6,
                    sources_used=["abstract"],
                ),
            ),
            AgentRoundOpinion(
                agent_id="differential_expander",
                persona="differential_expander",
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="no",
                    top_3_differential_diagnoses=["no", "yes", "maybe"],
                    confidence_level=0.7,
                    sources_used=["abstract"],
                ),
            ),
            AgentRoundOpinion(
                agent_id="uncertainty_advocate",
                persona="uncertainty_advocate",
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="maybe",
                    top_3_differential_diagnoses=["maybe", "yes", "no"],
                    confidence_level=0.85,
                    sources_used=["abstract"],
                ),
            ),
        ]
        out = SupervisorDirectorOutput(
            final_label="yes",
            consensus_type="differential",
            rationale="directional but partial",
            primary_endpoint_answers_question=True,
            findings_decisive_for_question=False,
            authors_state_uncertainty=False,
            question_coverage="partial",
        )
        gated = apply_maybe_director_gate(
            out,
            patient_case="EVIDENCE: results remain unclear; further research is needed.",
            final_opinions=opinions,
            shared_report=SharedDebateReport(
                author_conclusion="maybe",
                residual_uncertainty=["mixed primary endpoints"],
            ),
        )
        self.assertEqual(gated.final_label, "maybe")

    def test_maybe_director_gate_preserves_unanimous_binary(self) -> None:
        from app.agents.aggregation import apply_maybe_director_gate
        from app.agents.models import AgentRoundOpinion, SharedDebateReport, SupervisorDirectorOutput

        opinions = [
            AgentRoundOpinion(
                agent_id=aid,
                persona=aid,
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="no",
                    top_3_differential_diagnoses=["no", "yes", "maybe"],
                    confidence_level=0.9,
                    sources_used=["abstract"],
                ),
            )
            for aid in (
                "generalist",
                "evidence_skeptic",
                "differential_expander",
                "uncertainty_advocate",
            )
        ]
        out = SupervisorDirectorOutput(
            final_label="no",
            consensus_type="consensus",
            rationale="clear null",
            primary_endpoint_answers_question=False,
            findings_decisive_for_question=True,
            authors_state_uncertainty=True,
            question_coverage="partial",
        )
        gated = apply_maybe_director_gate(
            out,
            patient_case="EVIDENCE: further research is needed.",
            final_opinions=opinions,
            shared_report=SharedDebateReport(
                author_conclusion="maybe",
                residual_uncertainty=["boilerplate only"],
            ),
        )
        self.assertEqual(gated.final_label, "no")

    def test_maybe_gate_blocks_partial_when_panel_and_bert_agree(self) -> None:
        from app.agents.aggregation import apply_maybe_director_gate
        from app.agents.models import AgentRoundOpinion, SupervisorDirectorOutput

        opinions = [
            AgentRoundOpinion(
                agent_id=aid,
                persona=aid,
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="yes",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.82,
                    sources_used=["abstract"],
                ),
            )
            for aid in (
                "generalist",
                "evidence_skeptic",
                "differential_expander",
                "uncertainty_advocate",
            )
        ]
        # Force one soft maybe voter with low weight so conf vote still leans yes.
        opinions[3] = AgentRoundOpinion(
            agent_id="uncertainty_advocate",
            persona="uncertainty_advocate",
            round=1,
            opinion=ClinicalOpinion(
                top_1_diagnosis="maybe",
                top_3_differential_diagnoses=["maybe", "yes", "no"],
                confidence_level=0.4,
                sources_used=["abstract"],
            ),
        )
        out = SupervisorDirectorOutput(
            final_label="yes",
            consensus_type="differential",
            rationale="partial coverage claimed",
            primary_endpoint_answers_question=True,
            findings_decisive_for_question=False,
            authors_state_uncertainty=False,
            question_coverage="partial",
        )
        gated = apply_maybe_director_gate(
            out,
            patient_case="EVIDENCE: clear positive primary endpoint.",
            final_opinions=opinions,
            bert_label="yes",
        )
        self.assertEqual(gated.final_label, "yes")

    def test_maybe_gate_promotes_on_coverage_none(self) -> None:
        from app.agents.aggregation import apply_maybe_director_gate
        from app.agents.models import AgentRoundOpinion, SupervisorDirectorOutput

        opinions = [
            AgentRoundOpinion(
                agent_id="generalist",
                persona="generalist",
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="yes",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.7,
                    sources_used=["abstract"],
                ),
            ),
            AgentRoundOpinion(
                agent_id="uncertainty_advocate",
                persona="uncertainty_advocate",
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="maybe",
                    top_3_differential_diagnoses=["maybe", "yes", "no"],
                    confidence_level=0.7,
                    sources_used=["abstract"],
                ),
            ),
        ]
        out = SupervisorDirectorOutput(
            final_label="yes",
            consensus_type="consensus",
            rationale="bert-like",
            question_coverage="none",
        )
        gated = apply_maybe_director_gate(
            out,
            patient_case="EVIDENCE: unrelated surrogate only.",
            final_opinions=opinions,
            bert_label="yes",
        )
        self.assertEqual(gated.final_label, "maybe")

    def test_specialist_persistent_maybe_forces_maybe_after_3_rounds(self) -> None:
        from app.agents.aggregation import apply_maybe_director_gate
        from app.agents.models import AgentRoundOpinion, SupervisorDirectorOutput

        opinions = [
            AgentRoundOpinion(
                agent_id=aid,
                persona=aid,
                round=3,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="yes",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.85,
                    sources_used=["abstract"],
                ),
            )
            for aid in ("generalist", "evidence_skeptic", "differential_expander")
        ]
        opinions.append(
            AgentRoundOpinion(
                agent_id="uncertainty_advocate",
                persona="uncertainty_advocate",
                round=3,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="maybe",
                    top_3_differential_diagnoses=["maybe", "yes", "no"],
                    confidence_level=0.80,
                    sources_used=["abstract"],
                ),
            )
        )
        out = SupervisorDirectorOutput(
            final_label="yes",
            consensus_type="consensus",
            rationale="panel agrees",
            question_coverage="full",
        )
        gated = apply_maybe_director_gate(
            out,
            patient_case="EVIDENCE: clear positive result.",
            final_opinions=opinions,
            rounds_completed=3,
        )
        self.assertEqual(gated.final_label, "maybe")
        self.assertIn("persistent maybe", gated.rationale)

        # Same scenario but only 2 rounds — should NOT trigger Path E.
        gated_r2 = apply_maybe_director_gate(
            out,
            patient_case="EVIDENCE: clear positive result.",
            final_opinions=opinions,
            rounds_completed=2,
        )
        self.assertEqual(gated_r2.final_label, "yes")

    def test_aggregate_llm_director_default_skips_posthoc_gate(self) -> None:
        import asyncio
        from app.agents.aggregation import aggregate_with_llm_director
        from app.agents.models import AgentRoundOpinion, SupervisorDirectorOutput

        class _FakeSupervisor:
            def __init__(self) -> None:
                self.last_director_output = None
                self.last_kwargs = None

            async def synthesize_decision(self, *args, **kwargs):
                self.last_kwargs = dict(kwargs)
                if args:
                    # Positional fallback if callers ever change signature.
                    keys = ("patient_case", "debate_transcript", "biolinkbert_hint")
                    for idx, key in enumerate(keys):
                        if idx < len(args) and key not in self.last_kwargs:
                            self.last_kwargs[key] = args[idx]
                return SupervisorDirectorOutput(
                    final_label="yes",
                    consensus_type="consensus",
                    rationale="authors affirm",
                    question_coverage="none",
                    primary_endpoint_answers_question=False,
                    findings_decisive_for_question=False,
                    authors_state_uncertainty=True,
                )

        r1 = [
            AgentRoundOpinion(
                agent_id="generalist",
                persona="generalist",
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="yes",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.7,
                    sources_used=["abstract"],
                    pros=["Authors affirm benefit."],
                    cons=["Sample is small."],
                ),
            ),
            AgentRoundOpinion(
                agent_id="uncertainty_advocate",
                persona="uncertainty_advocate",
                round=1,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="maybe",
                    top_3_differential_diagnoses=["maybe", "yes", "no"],
                    confidence_level=0.8,
                    sources_used=["abstract"],
                    pros=["Question coverage incomplete."],
                    cons=[],
                ),
            ),
        ]
        r2 = [
            AgentRoundOpinion(
                agent_id="generalist",
                persona="generalist",
                round=2,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="yes",
                    top_3_differential_diagnoses=["yes", "no", "maybe"],
                    confidence_level=0.75,
                    sources_used=["abstract"],
                    pros=["Primary endpoint positive."],
                ),
            ),
            AgentRoundOpinion(
                agent_id="uncertainty_advocate",
                persona="uncertainty_advocate",
                round=2,
                opinion=ClinicalOpinion(
                    top_1_diagnosis="maybe",
                    top_3_differential_diagnoses=["maybe", "yes", "no"],
                    confidence_level=0.85,
                    sources_used=["abstract"],
                    pros=["Coverage gap unresolved."],
                ),
            ),
        ]
        supervisor = _FakeSupervisor()
        label_off = asyncio.run(
            aggregate_with_llm_director(
                "CASE",
                [r1, r2],
                '{"label":"yes"}',
                supervisor=supervisor,  # type: ignore[arg-type]
                director_maybe_gate="off",
            )
        )
        self.assertEqual(label_off, "yes")
        self.assertEqual(supervisor.last_director_output.final_label, "yes")
        transcript = supervisor.last_kwargs["debate_transcript"]
        self.assertIn("=== ROUND 1 ===", transcript)
        self.assertIn("=== ROUND 2 ===", transcript)
        self.assertIn("CONFLICT", transcript)
        self.assertIn("Coverage gap unresolved.", transcript)
        self.assertIsNone(supervisor.last_kwargs.get("debate_brief"))

        label_legacy = asyncio.run(
            aggregate_with_llm_director(
                "CASE",
                [r1, r2],
                '{"label":"yes"}',
                supervisor=supervisor,  # type: ignore[arg-type]
                director_maybe_gate="legacy",
            )
        )
        self.assertEqual(label_legacy, "maybe")
        self.assertEqual(supervisor.last_director_output.final_label, "maybe")

    def test_abstract_suggests_inconclusive_ignores_broad_limitation(self) -> None:
        from app.agents.heuristics import abstract_suggests_inconclusive

        self.assertFalse(
            abstract_suggests_inconclusive("A clear RCT benefit; study limitations are discussed.")
        )
        self.assertTrue(
            abstract_suggests_inconclusive("Results remain unclear and further research is needed.")
        )

    def test_confidence_aware_vote_zeros_fallback(self) -> None:
        from app.agents.aggregation import confidence_aware_vote
        from app.agents.backends import fallback_clinical_opinion

        strong = ClinicalOpinion(
            top_1_diagnosis="no",
            top_3_differential_diagnoses=["no", "yes", "maybe"],
            confidence_level=0.9,
            sources_used=["abstract"],
        )
        fallback = fallback_clinical_opinion(label="yes", reason="broken")
        label, share, _ = confidence_aware_vote([fallback, strong])
        self.assertEqual(label, "no")
        self.assertGreater(share["no"], share["yes"])


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


class PanelVoteSummaryTests(unittest.TestCase):
    """Mechanical panel telemetry fed to the Director (and to the eval report)."""

    @staticmethod
    def _entry(agent_id: str, label: str, *, round_number: int = 1, confidence: float = 0.8):
        from app.agents.models import AgentRoundOpinion

        return AgentRoundOpinion(
            agent_id=agent_id,
            persona=agent_id,
            round=round_number,
            opinion=ClinicalOpinion(
                top_1_diagnosis=label,
                top_3_differential_diagnoses=["yes", "no", "maybe"],
                confidence_level=confidence,
            ),
        )

    def _panel(self, advocate_label: str, *, round_number: int = 1) -> list:
        return [
            self._entry("generalist", "yes", round_number=round_number),
            self._entry("evidence_skeptic", "yes", round_number=round_number),
            self._entry("differential_expander", "yes", round_number=round_number),
            self._entry("uncertainty_advocate", advocate_label, round_number=round_number),
        ]

    def test_lone_dissenter_identifies_role_and_label(self) -> None:
        from app.agents.aggregation import lone_dissenter

        self.assertEqual(
            lone_dissenter(self._panel("maybe")), ("uncertainty_advocate", "maybe")
        )
        self.assertIsNone(lone_dissenter(self._panel("yes")))

    def test_panel_conflict_kind_separates_lone_dissent_from_even_split(self) -> None:
        from app.agents.aggregation import panel_conflict_kind

        self.assertEqual(panel_conflict_kind(self._panel("yes")), "unanimous")
        self.assertEqual(panel_conflict_kind(self._panel("maybe")), "lone_dissent")

        even_split = [
            self._entry("generalist", "yes"),
            self._entry("evidence_skeptic", "yes"),
            self._entry("differential_expander", "no"),
            self._entry("uncertainty_advocate", "no"),
        ]
        self.assertEqual(panel_conflict_kind(even_split), "fundamental")

    def test_vote_summary_flags_structurally_biased_lone_dissenter(self) -> None:
        from app.agents.aggregation import build_panel_vote_summary

        summary = build_panel_vote_summary(
            [self._panel("maybe"), self._panel("maybe", round_number=2)],
            biolinkbert_label="yes",
            panel_saw_bert_hint=True,
        )
        self.assertIn("ROUND 1", summary)
        self.assertIn("yes=3", summary)
        self.assertIn("all agents agree except one", summary)
        self.assertIn("The single dissenter is uncertainty_advocate", summary)
        self.assertIn("system-instructed to hunt for 'maybe'", summary)

    def test_vote_summary_never_leaks_consensus_type_vocabulary(self) -> None:
        """Enum-like tokens in the prompt get copied into the Director's
        ``consensus_type``, which fails validation and forces a fallback maybe."""
        from app.agents.aggregation import build_panel_vote_summary

        panels = [self._panel("maybe"), self._panel("yes")]
        panels.append(
            [
                self._entry("generalist", "yes"),
                self._entry("evidence_skeptic", "yes"),
                self._entry("differential_expander", "no"),
                self._entry("uncertainty_advocate", "no"),
            ]
        )
        for panel in panels:
            summary = build_panel_vote_summary([panel], panel_saw_bert_hint=True)
            for token in ("lone_dissent", "fundamental", "unanimous", "split"):
                self.assertNotIn(token, summary)

    def test_vote_summary_warns_bert_is_not_independent_of_panel(self) -> None:
        """With --hint biolinkbert the agents saw the classifier label, so panel
        agreement with it is the same source counted twice."""
        from app.agents.aggregation import build_panel_vote_summary

        seen = build_panel_vote_summary(
            [self._panel("maybe")], biolinkbert_label="yes", panel_saw_bert_hint=True
        )
        self.assertIn("INDEPENDENCE WARNING", seen)
        self.assertIn("same source counted twice", seen)

        # --hint none: the panel really is independent, so no warning belongs here.
        blind = build_panel_vote_summary(
            [self._panel("maybe")], biolinkbert_label="yes", panel_saw_bert_hint=False
        )
        self.assertNotIn("INDEPENDENCE WARNING", blind)

    def test_vote_summary_omits_bias_note_for_ordinary_dissenter(self) -> None:
        from app.agents.aggregation import build_panel_vote_summary

        panel = [
            self._entry("generalist", "yes"),
            self._entry("evidence_skeptic", "no"),
            self._entry("differential_expander", "yes"),
            self._entry("uncertainty_advocate", "yes"),
        ]
        summary = build_panel_vote_summary([panel])
        self.assertIn("The single dissenter is evidence_skeptic", summary)
        self.assertNotIn("system-instructed", summary)

    def test_vote_summary_handles_empty_history(self) -> None:
        from app.agents.aggregation import build_panel_vote_summary

        self.assertIn("no debate rounds", build_panel_vote_summary([]))


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

    def test_bert_gate_unanimous_maybe_vetoes_high_conf_bert(self) -> None:
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
        self.assertEqual(label, "maybe")
        self.assertEqual(rule, "panel_maybe_veto")

    def test_bert_gate_majority_maybe_veto(self) -> None:
        agents = [
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.7),
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.7),
            ClinicalOpinion(top_1_diagnosis="maybe", top_3_differential_diagnoses=["maybe"], confidence_level=0.7),
            ClinicalOpinion(top_1_diagnosis="yes", top_3_differential_diagnoses=["yes"], confidence_level=0.7),
        ]
        bert = ClinicalOpinion(
            top_1_diagnosis="no",
            top_3_differential_diagnoses=["no"],
            confidence_level=0.95,
        )
        # Default unanimous: 3/4 maybe is not enough → trust BERT
        label, _, rule = aggregate_pubmedqa_decision(
            agents,
            bert_opinion=bert,
            mode="bert_gate",
            bert_gate_confidence=0.9,
            panel_maybe_veto="unanimous",
        )
        self.assertEqual(label, "no")
        self.assertEqual(rule, "bert_gate")

        label, _, rule = aggregate_pubmedqa_decision(
            agents,
            bert_opinion=bert,
            mode="bert_gate",
            bert_gate_confidence=0.9,
            panel_maybe_veto="majority",
        )
        self.assertEqual(label, "maybe")
        self.assertEqual(rule, "panel_maybe_veto")

    def test_bert_gate_legacy_off_keeps_bert_vs_panel_maybe(self) -> None:
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
            agents,
            bert_opinion=bert,
            mode="bert_gate",
            bert_gate_confidence=0.9,
            panel_maybe_veto="off",
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
    def test_pubmedqa_panel_has_four_agents(self) -> None:
        agents = build_default_agents(MockInferenceBackend(), task_mode="pubmedqa")
        self.assertEqual(len(agents), 4)
        self.assertEqual(
            {agent.agent_id for agent in agents},
            {
                "generalist",
                "evidence_skeptic",
                "differential_expander",
                "uncertainty_advocate",
            },
        )

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

class DirectorOutputRobustnessTests(unittest.TestCase):
    """An off-vocabulary descriptive field must not discard a usable final_label."""

    def _director(self, payload: dict):
        import asyncio, json as _json
        from app.agents.supervisor_agent import SupervisorAgent

        class _Backend:
            async def complete(self, messages, **kwargs):
                return _json.dumps(payload)

        supervisor = SupervisorAgent(backend=_Backend())
        return asyncio.run(
            supervisor.synthesize_decision(
                patient_case="CASE", debate_transcript="T", biolinkbert_hint="{}"
            )
        )

    def test_unknown_consensus_type_is_coerced_not_fatal(self) -> None:
        out = self._director(
            {"final_label": "no", "consensus_type": "lone_dissent", "rationale": "r"}
        )
        self.assertEqual(out.final_label, "no")
        self.assertEqual(out.consensus_type, "differential")

    def test_valid_consensus_type_is_preserved(self) -> None:
        out = self._director(
            {"final_label": "yes", "consensus_type": "consensus", "rationale": "r"}
        )
        self.assertEqual(out.final_label, "yes")
        self.assertEqual(out.consensus_type, "consensus")


class MaybeDetectorTests(unittest.TestCase):
    """Standalone answer-split detector (docs/agents/maybe-detector-spec.md)."""

    def _detect(self, payload):
        import asyncio, json as _json
        from app.agents.maybe_detector import detect_answer_split

        class _Backend:
            async def complete(self, messages, **kwargs):
                return payload if isinstance(payload, str) else _json.dumps(payload)

        return asyncio.run(
            detect_answer_split(_Backend(), question="Q?", abstract="A.")
        )

    def test_split_with_quoted_findings_is_detected(self) -> None:
        v = self._detect(
            {
                "split_detected": True,
                "split_kind": "subgroup",
                "conflicting_findings": ["significant in UC", "not significant in CD"],
                "confidence": 0.8,
            }
        )
        self.assertTrue(v.is_usable)
        self.assertEqual(v.split_kind, "subgroup")

    def test_split_claim_without_quotes_is_downgraded(self) -> None:
        """An unsupported assertion of uncertainty is the failure mode to avoid."""
        v = self._detect(
            {"split_detected": True, "split_kind": "subgroup", "conflicting_findings": []}
        )
        self.assertFalse(v.split_detected)
        self.assertFalse(v.is_usable)
        self.assertEqual(v.split_kind, "none")

    def test_single_quote_is_not_a_split(self) -> None:
        v = self._detect(
            {
                "split_detected": True,
                "split_kind": "outcome_conflict",
                "conflicting_findings": ["only one finding"],
            }
        )
        self.assertFalse(v.is_usable)

    def test_unparseable_output_reports_no_split(self) -> None:
        """A detector failure must never fabricate uncertainty."""
        v = self._detect("the model rambled without JSON")
        self.assertFalse(v.split_detected)
        self.assertEqual(v.split_kind, "none")

    def test_unknown_split_kind_is_normalized(self) -> None:
        v = self._detect(
            {
                "split_detected": True,
                "split_kind": "totally_made_up",
                "conflicting_findings": ["a", "b"],
            }
        )
        self.assertEqual(v.split_kind, "none")

    def test_apply_only_ever_produces_maybe_never_flips_direction(self) -> None:
        from app.agents.maybe_detector import AnswerSplitVerdict, apply_split_detector

        usable = AnswerSplitVerdict(
            split_detected=True,
            split_kind="subgroup",
            conflicting_findings=["a", "b"],
            confidence=0.9,
        )
        self.assertEqual(apply_split_detector("yes", usable), ("maybe", True))
        self.assertEqual(apply_split_detector("no", usable), ("maybe", True))
        # Never rewrites an existing maybe, and never turns yes into no.
        self.assertEqual(apply_split_detector("maybe", usable), ("maybe", False))
        self.assertEqual(apply_split_detector(None, usable), (None, False))

    def test_apply_respects_confidence_floor(self) -> None:
        from app.agents.maybe_detector import AnswerSplitVerdict, apply_split_detector

        weak = AnswerSplitVerdict(
            split_detected=True,
            split_kind="subgroup",
            conflicting_findings=["a", "b"],
            confidence=0.2,
        )
        self.assertEqual(
            apply_split_detector("yes", weak, min_confidence=0.6), ("yes", False)
        )

    def test_outcome_conflict_is_detected_but_not_acted_on(self) -> None:
        """Measured at precision 0.150 on balanced90 — below the 0.333 base rate.

        Acting on it drops composed accuracy from 0.689 to 0.556, so it is
        recorded for analysis but never overrides a binary answer.
        """
        from app.agents.maybe_detector import AnswerSplitVerdict, apply_split_detector

        verdict = AnswerSplitVerdict(
            split_detected=True,
            split_kind="outcome_conflict",
            conflicting_findings=["marker a up", "marker b flat"],
            confidence=1.0,
        )
        self.assertTrue(verdict.is_usable)
        self.assertEqual(apply_split_detector("yes", verdict), ("yes", False))

    def test_compound_question_split_is_actionable(self) -> None:
        from app.agents.maybe_detector import AnswerSplitVerdict, apply_split_detector

        verdict = AnswerSplitVerdict(
            split_detected=True,
            split_kind="compound_question",
            conflicting_findings=["recalled the weight", "could not explain it"],
            confidence=1.0,
        )
        self.assertEqual(apply_split_detector("no", verdict), ("maybe", True))
