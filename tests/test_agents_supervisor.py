"""Tests for the supervisor architectures: partitioning, anonymization, routing, closure."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import build_default_agents
from app.agents.backends import MockInferenceBackend
from app.agents.metrics import adoption_rate, flip_counts, rescue_rate, subversion_rate
from app.agents.models import AgentRoundOpinion, ClinicalOpinion
from app.agents.partitioning import partition_patient_case, split_sentences
from app.agents.prompts import build_messages
from app.agents.supervisor import (
    SupervisorAgent,
    SupervisorVerdict,
    apply_decision_rule,
    eligibility_gate,
)
from app.agents.supervisor_orchestrator import SupervisorOrchestrator
from app.agents.token_meter import CountingBackend
from app.schemas import ChatMessage

CASE = """Task: Answer yes/no/maybe from the evidence only.

RESEARCH QUESTION:
Is anorectal endosonography valuable in dyschesia?

EVIDENCE:
SOURCE pubmedqa-official-12377809
Title: Is anorectal endosonography valuable in dyschesia?
Dyschesia can be provoked by inappropriate defecation movements. The aim of this study was to demonstrate dysfunction of the anal sphincter. Twenty consecutive patients underwent linear anorectal endosonography. The anal sphincter became paradoxically shorter during straining in 85% of patients. Changes in sphincter length were significantly different in patients."""


def _panel(backend, count: int = 4):
    agents = build_default_agents(backend, task_mode="pubmedqa", neutral=True)
    return agents[:count]


def _opinion(label: str, **kwargs) -> ClinicalOpinion:
    return ClinicalOpinion(
        top_1_diagnosis=label,
        top_3_differential_diagnoses=["yes", "no", "maybe"],
        confidence_level=0.5,
        **kwargs,
    )


def _entry(agent_id: str, label: str, round_number: int) -> AgentRoundOpinion:
    return AgentRoundOpinion(
        agent_id=agent_id,
        persona="neutral_analyst",
        round=round_number,
        opinion=_opinion(label),
    )


class PartitioningTests(unittest.TestCase):
    def test_split_sentences_finds_boundaries(self) -> None:
        parts = split_sentences("First one. Second one! Third one? Fourth.")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "First one.")

    def test_partitions_are_deterministic(self) -> None:
        first = partition_patient_case(CASE, k=3, overlap=1)
        second = partition_patient_case(CASE, k=3, overlap=1)
        self.assertEqual(
            [p.patient_case for p in first],
            [p.patient_case for p in second],
        )

    def test_core_spans_cover_every_sentence_exactly_once(self) -> None:
        # With overlap 0 the spans must tile the evidence: nothing dropped, nothing
        # duplicated, or some agent silently never sees part of the case.
        partitions = partition_patient_case(CASE, k=3, overlap=0)
        covered: list[int] = []
        for part in partitions:
            covered.extend(range(*part.sentence_span))
        self.assertEqual(covered, sorted(covered))
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(covered[0], 0)

    def test_overlap_widens_spans_without_losing_order(self) -> None:
        tight = partition_patient_case(CASE, k=3, overlap=0)
        loose = partition_patient_case(CASE, k=3, overlap=1)
        self.assertEqual(len(tight), len(loose))
        self.assertGreater(
            sum(end - start for start, end in (p.sentence_span for p in loose)),
            sum(end - start for start, end in (p.sentence_span for p in tight)),
        )

    def test_question_survives_in_every_partition(self) -> None:
        for part in partition_patient_case(CASE, k=4, overlap=0):
            self.assertIn("RESEARCH QUESTION:", part.patient_case)
            self.assertIn("anorectal endosonography valuable in dyschesia", part.patient_case)

    def test_partitions_hold_different_evidence(self) -> None:
        partitions = partition_patient_case(CASE, k=3, overlap=0)
        texts = [p.evidence_text for p in partitions]
        self.assertEqual(len(set(texts)), len(texts))
        # The decisive result sentence is held by exactly one agent.
        holders = [text for text in texts if "85%" in text]
        self.assertEqual(len(holders), 1)

    def test_partitions_round_trip_through_the_case_parser(self) -> None:
        from app.agents.backends import parse_pubmedqa_patient_case

        for part in partition_patient_case(CASE, k=3, overlap=0):
            question, documents = parse_pubmedqa_patient_case(part.patient_case)
            self.assertIn("dyschesia", question)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].id, "pubmedqa-official-12377809")

    def test_more_partitions_than_sentences_does_not_produce_empty_segments(self) -> None:
        partitions = partition_patient_case(CASE, k=50, overlap=0)
        self.assertLessEqual(len(partitions), 5)
        for part in partitions:
            self.assertTrue(part.evidence_text.strip())

    def test_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            partition_patient_case(CASE, k=0)
        with self.assertRaises(ValueError):
            partition_patient_case(CASE, k=2, overlap=-1)


class EligibilityGateTests(unittest.TestCase):
    SEGMENTS = {
        "a1": "The anal sphincter became paradoxically shorter during straining.",
        "a2": "Twenty consecutive patients underwent linear endosonography.",
        "a3": "Dyschesia can be provoked by inappropriate defecation movements.",
    }

    def test_routes_only_to_segments_holding_the_term(self) -> None:
        eligible = eligibility_gate("What happened to the sphincter?", self.SEGMENTS)
        self.assertEqual(eligible, ["a1"])

    def test_returns_nothing_when_no_segment_matches(self) -> None:
        self.assertEqual(eligibility_gate("What was the cholesterol?", self.SEGMENTS), [])

    def test_stopwords_alone_do_not_qualify_a_segment(self) -> None:
        # "what/was/the" appear everywhere; matching on them would make the gate
        # nominate every segment and stop filtering anything.
        self.assertEqual(eligibility_gate("What was the?", self.SEGMENTS), [])

    def test_orders_by_overlap_strength(self) -> None:
        eligible = eligibility_gate("sphincter straining patients", self.SEGMENTS)
        self.assertEqual(eligible[0], "a1")


class DecisionRuleTests(unittest.TestCase):
    def test_failed_check_forces_maybe_even_when_model_says_yes(self) -> None:
        label, rule = apply_decision_rule(
            {"question_addressed": True, "direction_established": False}, "yes"
        )
        self.assertEqual(label, "maybe")
        self.assertIn("direction_established", rule)

    def test_passing_checks_keep_the_proposed_label(self) -> None:
        checks = dict.fromkeys(
            ("question_addressed", "direction_established", "opposite_reading_excluded", "hedging_absent"),
            True,
        )
        self.assertEqual(apply_decision_rule(checks, "no"), ("no", "rigor_checks_passed"))

    def test_missing_label_degrades_to_maybe(self) -> None:
        self.assertEqual(apply_decision_rule({}, None), ("maybe", "no_label_proposed"))


class SupervisorAgentTests(unittest.TestCase):
    def test_parses_a_verdict_from_fenced_json(self) -> None:
        class FencedBackend:
            async def complete(self, messages, *, temperature=0.3) -> str:
                return (
                    '```json\n{"rigor_checks": {"question_addressed": true, '
                    '"direction_established": true, "opposite_reading_excluded": true, '
                    '"hedging_absent": true}, "label": "yes", "confidence": 0.8, '
                    '"rationale": "ok"}\n```'
                )

        verdict = asyncio.run(
            SupervisorAgent(FencedBackend()).moderate(question="q", arguments=[])
        )
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.label, "yes")

    def test_backend_failure_returns_error_verdict_instead_of_raising(self) -> None:
        class ExplodingBackend:
            async def complete(self, messages, *, temperature=0.3) -> str:
                raise RuntimeError("simulated supervisor failure")

        verdict = asyncio.run(
            SupervisorAgent(ExplodingBackend()).moderate(question="q", arguments=[])
        )
        self.assertFalse(verdict.ok)
        self.assertTrue(verdict.error)

    def test_routing_drops_targets_the_gate_did_not_allow(self) -> None:
        class OverreachingBackend:
            async def complete(self, messages, *, temperature=0.3) -> str:
                return json.dumps({"r1": ["a1", "a9"]})

        routed = asyncio.run(
            SupervisorAgent(OverreachingBackend()).route(
                question="q", requests={"r1": "about sphincter"}, eligible={"r1": ["a1"]}
            )
        )
        self.assertEqual(routed["r1"], ["a1"])

    def test_routing_falls_back_to_gate_order_when_model_fails(self) -> None:
        class ExplodingBackend:
            async def complete(self, messages, *, temperature=0.3) -> str:
                raise RuntimeError("boom")

        routed = asyncio.run(
            SupervisorAgent(ExplodingBackend()).route(
                question="q", requests={"r1": "q"}, eligible={"r1": ["a2", "a3"]}
            )
        )
        self.assertEqual(routed["r1"], ["a2"])


class AnonymizationTests(unittest.TestCase):
    CONTEXT = [_entry("analyst_1", "yes", 1), _entry("analyst_2", "no", 1)]

    def test_identified_transcript_carries_identity(self) -> None:
        messages = build_messages(
            agent_id="analyst_3",
            persona="neutral_analyst",
            patient_case=CASE,
            context=self.CONTEXT,
            task_mode="pubmedqa",
        )
        user = messages[1].content
        self.assertIn("PEER OPINIONS SO FAR", user)
        self.assertIn('"agent_id"', user)

    def test_anonymized_transcript_withholds_every_identity_field(self) -> None:
        messages = build_messages(
            agent_id="analyst_3",
            persona="neutral_analyst",
            patient_case=CASE,
            context=self.CONTEXT,
            task_mode="pubmedqa",
            anonymize=True,
            shuffle_seed=1,
        )
        user = messages[1].content
        self.assertIn("ARGUMENTS SUBMITTED FOR REVIEW", user)
        for leaked in ('"agent_id"', '"persona"', '"round"', "already_spoken_this_round"):
            self.assertNotIn(leaked, user)
        self.assertIn('"argument_id"', user)

    def test_anonymized_transcript_keeps_the_arguments_themselves(self) -> None:
        messages = build_messages(
            agent_id="analyst_3",
            persona="neutral_analyst",
            patient_case=CASE,
            context=self.CONTEXT,
            task_mode="pubmedqa",
            anonymize=True,
            shuffle_seed=1,
        )
        user = messages[1].content
        self.assertIn('"yes"', user)
        self.assertIn('"no"', user)

    def test_neutral_persona_header_names_no_role(self) -> None:
        system = build_messages(
            agent_id="analyst_1",
            persona="neutral_analyst",
            patient_case=CASE,
            task_mode="pubmedqa",
        )[0].content
        self.assertNotIn("with persona", system)
        # The machine-readable markers the mock backend parses must survive.
        self.assertIn("agent_id=analyst_1", system)
        self.assertIn("task_mode=pubmedqa", system)

    def test_partial_evidence_asks_for_information_requests(self) -> None:
        user = build_messages(
            agent_id="analyst_1",
            persona="neutral_analyst",
            patient_case=CASE,
            task_mode="pubmedqa",
            partial_evidence=True,
        )[1].content
        self.assertIn("SCOPE:", user)
        self.assertIn("information_requests", user)

    def test_baseline_prompt_does_not_mention_information_requests(self) -> None:
        # The shared-context arms must keep the baseline's exact prompt to stay a
        # fair comparison.
        system = build_messages(
            agent_id="generalist",
            persona="generalist",
            patient_case=CASE,
            task_mode="pubmedqa",
        )[0].content
        self.assertNotIn("information_requests", system)

    def test_routed_questions_reach_the_answering_agent(self) -> None:
        user = build_messages(
            agent_id="analyst_2",
            persona="neutral_analyst",
            patient_case=CASE,
            task_mode="pubmedqa",
            info_requests=["What was the control group size?"],
        )[1].content
        self.assertIn("QUESTIONS ROUTED TO YOU", user)
        self.assertIn("control group size", user)


class SupervisorOrchestratorTests(unittest.TestCase):
    def _build(self, **kwargs):
        backend = MockInferenceBackend()
        agents = _panel(backend)
        return SupervisorOrchestrator(agents, SupervisorAgent(backend), **kwargs)

    def test_architecture_name_reflects_the_enabled_mechanisms(self) -> None:
        self.assertEqual(self._build().ARCHITECTURE, "supervisor_shared_context")
        self.assertEqual(self._build(anonymize=True).ARCHITECTURE, "supervisor_anonymized")
        self.assertEqual(
            self._build(anonymize=True, partitions=4).ARCHITECTURE,
            "supervisor_asymmetric_infonav",
        )

    def test_shared_context_run_keeps_the_debate_result_contract(self) -> None:
        orchestrator = self._build(rounds=2)
        result = asyncio.run(orchestrator.run(CASE))
        self.assertEqual(len(result.rounds), 2)
        self.assertEqual(result.final_opinions, result.rounds[-1])
        for round_entries in result.rounds:
            self.assertEqual(len(round_entries), 4)

    def test_agent_ids_are_stable_across_rounds(self) -> None:
        # compute_uncertainty matches agents by id between the first and last
        # round; unstable ids silently degrade flip_rate to zero.
        result = asyncio.run(self._build(rounds=3).run(CASE))
        id_sets = [{entry.agent_id for entry in rnd} for entry in [None] for rnd in result.rounds]
        self.assertTrue(all(ids == id_sets[0] for ids in id_sets))
        self.assertEqual(len(id_sets[0]), 4)

    def test_asymmetric_run_gives_each_agent_a_different_segment(self) -> None:
        orchestrator = self._build(anonymize=True, partitions=4, partition_overlap=0)
        result = asyncio.run(orchestrator.run(CASE))
        segments = {entry.agent_id: entry.segment_id for entry in result.rounds[0]}
        self.assertEqual(len(set(segments.values())), 4)
        views = {part.evidence_text for part in orchestrator.last_partitions}
        self.assertEqual(len(views), 4)

    def test_asymmetric_run_records_the_information_exchange(self) -> None:
        orchestrator = self._build(anonymize=True, partitions=3, partition_overlap=1)
        asyncio.run(orchestrator.run(CASE))
        self.assertTrue(orchestrator.last_exchange)
        for record in orchestrator.last_exchange:
            self.assertIn("asked_by", record)
            self.assertNotIn(record["asked_by"], record["routed_to"])
            for target in record["routed_to"]:
                self.assertIn(target, record["eligible"])

    def test_asymmetric_round_count_follows_max_info_rounds(self) -> None:
        orchestrator = self._build(anonymize=True, partitions=3, max_info_rounds=2)
        self.assertEqual(orchestrator.rounds, 3)
        result = asyncio.run(orchestrator.run(CASE))
        self.assertEqual(len(result.rounds), 3)

    def test_supervisor_failure_falls_back_to_majority_and_says_so(self) -> None:
        class ExplodingBackend:
            async def complete(self, messages, *, temperature=0.3) -> str:
                raise RuntimeError("simulated supervisor failure")

        agents = _panel(MockInferenceBackend())
        orchestrator = SupervisorOrchestrator(agents, SupervisorAgent(ExplodingBackend()), rounds=2)
        asyncio.run(orchestrator.run(CASE))
        verdict = orchestrator.last_verdict
        self.assertEqual(verdict.rule, "supervisor_fallback_majority")
        self.assertIn(verdict.label, ("yes", "no", "maybe"))
        self.assertTrue(verdict.error)

    def test_one_failing_agent_does_not_break_the_round(self) -> None:
        class ExplodingBackend:
            async def complete(self, messages, *, temperature=0.3) -> str:
                raise RuntimeError("simulated agent failure")

        backend = MockInferenceBackend()
        agents = _panel(backend)
        agents[1].backend = ExplodingBackend()
        orchestrator = SupervisorOrchestrator(agents, SupervisorAgent(backend), rounds=2)
        result = asyncio.run(orchestrator.run(CASE))
        self.assertEqual(len(result.final_opinions), 4)
        broken = next(e for e in result.final_opinions if e.agent_id == agents[1].agent_id)
        self.assertEqual(broken.opinion.sources_used, ["fallback"])

    def test_rejects_invalid_configuration(self) -> None:
        backend = MockInferenceBackend()
        supervisor = SupervisorAgent(backend)
        with self.assertRaises(ValueError):
            SupervisorOrchestrator(_panel(backend, 1), supervisor)
        with self.assertRaises(ValueError):
            SupervisorOrchestrator(_panel(backend), supervisor, partitions=1)
        with self.assertRaises(ValueError):
            SupervisorOrchestrator(_panel(backend), supervisor, rounds=9)

    def test_verdict_share_is_a_one_hot_over_the_labels(self) -> None:
        orchestrator = self._build()
        share = orchestrator.verdict_share(SupervisorVerdict(label="no"))
        self.assertEqual(share, {"yes": 0.0, "no": 1.0, "maybe": 0.0})


class TokenMeterTests(unittest.TestCase):
    def test_counts_every_call_and_estimates_when_usage_is_absent(self) -> None:
        meter = CountingBackend(MockInferenceBackend())
        agents = _panel(meter)
        orchestrator = SupervisorOrchestrator(agents, SupervisorAgent(meter), rounds=2)
        asyncio.run(orchestrator.run(CASE))
        snapshot = meter.snapshot()
        # 4 agents x 2 rounds + 1 supervisor closure.
        self.assertEqual(snapshot.calls, 9)
        self.assertGreater(snapshot.total_tokens, 0)
        self.assertTrue(snapshot.is_estimated)

    def test_reports_exact_counts_when_the_backend_supplies_them(self) -> None:
        from app.agents.backends import LAST_USAGE

        class UsageBackend:
            async def complete(self, messages, *, temperature=0.3) -> str:
                LAST_USAGE.set((11, 7))
                return "{}"

        meter = CountingBackend(UsageBackend())
        asyncio.run(meter.complete([ChatMessage(role="user", content="hi")]))
        snapshot = meter.snapshot()
        self.assertEqual((snapshot.prompt_tokens, snapshot.completion_tokens), (11, 7))
        self.assertFalse(snapshot.is_estimated)

    def test_concurrent_calls_do_not_lose_counts(self) -> None:
        # Usage travels through a ContextVar precisely so parallel agents cannot
        # overwrite each other's numbers.
        from app.agents.backends import LAST_USAGE

        class SlowUsageBackend:
            async def complete(self, messages, *, temperature=0.3) -> str:
                await asyncio.sleep(0)
                LAST_USAGE.set((5, 5))
                return "{}"

        meter = CountingBackend(SlowUsageBackend())

        async def run() -> None:
            await asyncio.gather(
                *[meter.complete([ChatMessage(role="user", content="x")]) for _ in range(4)]
            )

        asyncio.run(run())
        self.assertEqual(meter.snapshot().calls, 4)
        self.assertEqual(meter.snapshot().total_tokens, 40)

    def test_reset_clears_the_counters(self) -> None:
        meter = CountingBackend(MockInferenceBackend())
        asyncio.run(meter.complete([ChatMessage(role="user", content="x")]))
        meter.reset()
        self.assertEqual(meter.snapshot().calls, 0)


class MetricsTests(unittest.TestCase):
    def test_flip_counts_separate_subversion_from_rescue(self) -> None:
        rounds = [
            [_entry("a", "yes", 1), _entry("b", "no", 1), _entry("c", "yes", 1)],
            [_entry("a", "no", 2), _entry("b", "yes", 2), _entry("c", "yes", 2)],
        ]
        counts = flip_counts(rounds, "yes")
        self.assertEqual(counts.subverted, 1)
        self.assertEqual(counts.rescued, 1)
        self.assertEqual(counts.stable_correct, 1)
        self.assertEqual(counts.evaluated, 3)

    def test_rates_share_a_denominator_so_they_are_comparable(self) -> None:
        rounds = [
            [_entry("a", "yes", 1), _entry("b", "no", 1)],
            [_entry("a", "no", 2), _entry("b", "no", 2)],
        ]
        counts = [flip_counts(rounds, "yes")]
        self.assertAlmostEqual(subversion_rate(counts), 0.5)
        self.assertAlmostEqual(rescue_rate(counts), 0.0)

    def test_single_round_has_no_flips(self) -> None:
        counts = flip_counts([[_entry("a", "yes", 1)]], "yes")
        self.assertEqual(counts.evaluated, 0)
        self.assertEqual(subversion_rate([counts]), 0.0)

    def test_adoption_counts_only_agents_that_moved_to_the_peer_majority(self) -> None:
        rounds = [
            [_entry("a", "yes", 1), _entry("b", "no", 1), _entry("c", "no", 1)],
            [_entry("a", "no", 2), _entry("b", "no", 2), _entry("c", "no", 2)],
        ]
        # Only `a` disagreed with its peers' majority and then joined it; b and c
        # already held it, so nothing moved for them.
        self.assertAlmostEqual(adoption_rate(rounds), 1.0)

    def test_tied_peers_are_skipped_rather_than_broken_arbitrarily(self) -> None:
        # b and c each see one `yes` and one `no` among their peers. Breaking that
        # tie by list order would make the metric depend on agent ordering.
        rounds = [
            [_entry("a", "yes", 1), _entry("b", "no", 1), _entry("c", "no", 1)],
            [_entry("a", "no", 2), _entry("b", "yes", 2), _entry("c", "yes", 2)],
        ]
        reversed_rounds = [list(reversed(rnd)) for rnd in rounds]
        self.assertAlmostEqual(adoption_rate(rounds), adoption_rate(reversed_rounds))

    def test_holding_out_against_the_majority_is_not_adoption(self) -> None:
        rounds = [
            [_entry("a", "yes", 1), _entry("b", "no", 1), _entry("c", "no", 1)],
            [_entry("a", "yes", 2), _entry("b", "no", 2), _entry("c", "no", 2)],
        ]
        self.assertAlmostEqual(adoption_rate(rounds), 0.0)


if __name__ == "__main__":
    unittest.main()
