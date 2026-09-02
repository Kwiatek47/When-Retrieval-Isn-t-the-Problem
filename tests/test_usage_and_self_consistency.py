"""Tests for LLM cost accounting and the self-consistency control arm."""

from __future__ import annotations

import asyncio
import unittest

from app.agents.agent import ClinicalAgent
from app.agents.backends import MockInferenceBackend
from app.core.usage import UsageTotals, current_usage, record_llm_usage, start_usage_scope, usage_scope
from app.core.config import resolve_ollama_base_url
from scripts.agents.evaluate_debate_pubmedqa import (
    DebateCaseResult,
    _summarize,
    _summarize_cost,
    report_has_measured_cost,
)
from scripts.agents.evaluate_self_consistency_pubmedqa import _evaluate_self_consistency

CASE = {
    "id": "case-1",
    "question": "Does X help Y?",
    "expected_label": "yes",
    "relevant_document_ids": [],
}


class UsageTotalsTests(unittest.TestCase):
    def test_records_calls_tokens_and_models(self) -> None:
        totals = UsageTotals()
        totals.record(model="qwen2.5:7b", prompt_tokens=100, completion_tokens=20)
        totals.record(model="qwen2.5:7b", prompt_tokens=50, completion_tokens=10)
        self.assertEqual(totals.llm_calls, 2)
        self.assertEqual(totals.total_tokens, 180)
        self.assertEqual(totals.calls_by_model, {"qwen2.5:7b": 2})

    def test_failed_calls_are_counted_apart_from_measured_ones(self) -> None:
        totals = UsageTotals()
        totals.record(model="m", prompt_tokens=10, completion_tokens=5)
        totals.record(model="m", failed=True)
        self.assertEqual(totals.llm_calls, 1)
        self.assertEqual(totals.failed_llm_calls, 1)
        self.assertEqual(totals.total_tokens, 15)

    def test_recording_outside_a_scope_is_a_noop(self) -> None:
        record_llm_usage(model="m", prompt_tokens=10)
        self.assertIsNone(current_usage())

    def test_scope_restores_the_previous_accumulator(self) -> None:
        with usage_scope() as outer:
            record_llm_usage(model="m", prompt_tokens=1)
            with usage_scope() as inner:
                record_llm_usage(model="m", prompt_tokens=2)
            self.assertEqual(inner.prompt_tokens, 2)
            record_llm_usage(model="m", prompt_tokens=4)
        self.assertEqual(outer.prompt_tokens, 5)
        self.assertIsNone(current_usage())

    def test_concurrent_tasks_do_not_share_counters(self) -> None:
        async def _one(n: int) -> int:
            totals = start_usage_scope()
            record_llm_usage(model="m", prompt_tokens=n)
            return totals.prompt_tokens

        async def _run() -> list[int]:
            return list(await asyncio.gather(_one(3), _one(11)))

        self.assertEqual(sorted(asyncio.run(_run())), [3, 11])


class SummarizeCostTests(unittest.TestCase):
    def test_ignores_unmeasured_rows(self) -> None:
        measured = DebateCaseResult(
            id="a",
            expected_label="yes",
            predicted_label="yes",
            vote_share={},
            agent_labels={},
            round1_vote_label="yes",
            cost_measured=True,
            llm_calls=8,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        old = DebateCaseResult(
            id="b",
            expected_label="yes",
            predicted_label="yes",
            vote_share={},
            agent_labels={},
            round1_vote_label="yes",
        )
        cost = _summarize_cost([measured, old])
        self.assertEqual(cost["measured_cases"], 1)
        self.assertEqual(cost["mean_llm_calls_per_case"], 8.0)
        self.assertEqual(cost["mean_total_tokens_per_case"], 120.0)
        self.assertTrue(
            report_has_measured_cost({"summary": {"cost": cost}})
        )
        self.assertFalse(
            report_has_measured_cost(
                {"summary": {"cost": {**cost, "measured_cases": 0, "mean_llm_calls_per_case": 0.0}}}
            )
        )
        self.assertFalse(report_has_measured_cost({"summary": {}}))

    def test_summarize_always_writes_mean_llm_calls_per_case(self) -> None:
        measured = DebateCaseResult(
            id="a",
            expected_label="yes",
            predicted_label="yes",
            vote_share={},
            agent_labels={},
            round1_vote_label="yes",
            cost_measured=True,
            llm_calls=8,
            label_pass=True,
            round1_pass=True,
        )
        summary = _summarize(
            [measured],
            backend="mock",
            rounds=2,
            dataset="x",
            hint="none",
            aggregation="majority",
            architecture="round_robin_no_supervisor",
            fast=False,
            early_exit_rate=0.0,
        )
        self.assertIn("mean_llm_calls_per_case", summary["cost"])
        self.assertEqual(summary["cost"]["mean_llm_calls_per_case"], 8.0)


class OllamaUrlResolutionTests(unittest.TestCase):
    def test_prefers_base_url_over_host(self) -> None:
        self.assertEqual(
            resolve_ollama_base_url(base_url="http://gpu:11434", host="localhost:9"),
            "http://gpu:11434",
        )

    def test_host_without_scheme_becomes_http(self) -> None:
        self.assertEqual(
            resolve_ollama_base_url(base_url="", host="192.168.1.4:11434"),
            "http://192.168.1.4:11434",
        )

    def test_strips_trailing_slash(self) -> None:
        self.assertEqual(
            resolve_ollama_base_url(base_url="http://localhost:11434/"),
            "http://localhost:11434",
        )


class SelfConsistencyIndependenceTests(unittest.TestCase):
    def test_samples_are_independent_and_counted(self) -> None:
        backend = MockInferenceBackend()

        def factory(_hint, _idx):
            return ClinicalAgent(
                agent_id="generalist",
                persona="generalist",
                backend=backend,
                task_mode="pubmedqa",
            )

        results = asyncio.run(
            _evaluate_self_consistency(
                [CASE],
                corpus={},
                agent_factory=factory,
                samples=4,
                hint_provider=None,
                inject_hint=False,
                case_concurrency=1,
                sample_concurrency=1,
                prior_results={},
                checkpoint_path=None,
            )
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].aggregation_rule, "self_consistency_majority_n4")
        self.assertEqual(results[0].llm_calls, 4)
        self.assertEqual(len(results[0].agent_labels), 4)


if __name__ == "__main__":
    unittest.main()
