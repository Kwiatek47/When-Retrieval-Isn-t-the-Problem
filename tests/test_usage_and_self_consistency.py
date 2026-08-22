"""Tests for LLM cost accounting and the self-consistency control arm.

Both exist to make the compute-matched comparison in the paper possible, so the
things asserted here are the ones that would silently corrupt it: per-case
isolation of the counters, tokens read from the Ollama response, and samples
drawn independently.
"""

from __future__ import annotations

import asyncio
import unittest

from app.agents import MockInferenceBackend, build_default_agents
from app.agents.agent import ClinicalAgent
from app.core.usage import UsageTotals, current_usage, record_llm_usage, usage_scope
from app.providers.base import ProviderError, ProviderUnavailableError
from app.providers.ollama import OllamaProvider
from app.schemas import ChatMessage
from scripts.agents.evaluate_debate_pubmedqa import DebateCaseResult, _summarize_cost
from scripts.agents.evaluate_self_consistency_pubmedqa import _evaluate_self_consistency

CASE = {
    "id": "case-1",
    "question": "Does X help Y?",
    "expected_label": "yes",
    "relevant_document_ids": [],
}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _chat_payload(*, prompt_tokens: int = 120, completion_tokens: int = 30, content: str = "ok") -> dict:
    return {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


class UsageTotalsTests(unittest.TestCase):
    def test_records_calls_tokens_and_models(self) -> None:
        totals = UsageTotals()
        totals.record(model="qwen2.5:7b", prompt_tokens=100, completion_tokens=20)
        totals.record(model="qwen2.5:7b", prompt_tokens=50, completion_tokens=10)
        self.assertEqual(totals.llm_calls, 2)
        self.assertEqual(totals.prompt_tokens, 150)
        self.assertEqual(totals.completion_tokens, 30)
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


class OllamaUsageTests(unittest.TestCase):
    def _provider(self, response: object | Exception) -> OllamaProvider:
        provider = OllamaProvider(base_url="http://localhost:11434", timeout=5.0)

        async def _post_chat(payload: dict) -> object:
            if isinstance(response, Exception):
                raise response
            return response

        provider._post_chat = _post_chat  # type: ignore[assignment]
        return provider

    def _chat(self, provider: OllamaProvider) -> None:
        asyncio.run(
            provider.chat(
                model="qwen2.5:7b",
                messages=[ChatMessage(role="user", content="hi")],
                temperature=0.7,
            )
        )

    def test_reads_prompt_eval_count_and_eval_count(self) -> None:
        provider = self._provider(_FakeResponse(_chat_payload()))
        with usage_scope() as totals:
            self._chat(provider)
        self.assertEqual(totals.llm_calls, 1)
        self.assertEqual(totals.prompt_tokens, 120)
        self.assertEqual(totals.completion_tokens, 30)
        self.assertEqual(totals.calls_by_model, {"qwen2.5:7b": 1})

    def test_transport_failure_counts_as_a_failed_call(self) -> None:
        provider = self._provider(ProviderUnavailableError("connection refused"))
        with usage_scope() as totals:
            with self.assertRaises(ProviderUnavailableError):
                self._chat(provider)
        self.assertEqual(totals.llm_calls, 0)
        self.assertEqual(totals.failed_llm_calls, 1)

    def test_empty_completion_still_charges_the_prompt(self) -> None:
        provider = self._provider(_FakeResponse(_chat_payload(completion_tokens=0, content="")))
        with usage_scope() as totals:
            with self.assertRaises(ProviderError):
                self._chat(provider)
        self.assertEqual(totals.llm_calls, 1)
        self.assertEqual(totals.prompt_tokens, 120)


class DebateCostAccountingTests(unittest.TestCase):
    def test_one_scope_per_case_does_not_leak_between_concurrent_cases(self) -> None:
        backend = MockInferenceBackend()

        async def _case(calls: int) -> UsageTotals:
            with usage_scope() as totals:
                agents = build_default_agents(backend, task_mode="pubmedqa")
                await asyncio.gather(
                    *[agents[0].generate_opinion("case", context=None) for _ in range(calls)]
                )
                return totals

        async def _both() -> tuple[UsageTotals, UsageTotals]:
            return await asyncio.gather(
                asyncio.create_task(_case(3)),
                asyncio.create_task(_case(7)),
            )

        first, second = asyncio.run(_both())
        self.assertEqual(first.llm_calls, 3)
        self.assertEqual(second.llm_calls, 7)

    def test_summary_averages_only_measured_cases(self) -> None:
        measured = DebateCaseResult(
            id="a",
            expected_label="yes",
            predicted_label="yes",
            vote_share={},
            agent_labels={},
            round1_vote_label="yes",
            cost_measured=True,
            llm_calls=9,
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
        )
        # Replayed from a checkpoint written before the counter existed.
        resumed = DebateCaseResult(
            id="b",
            expected_label="no",
            predicted_label="no",
            vote_share={},
            agent_labels={},
            round1_vote_label="no",
        )
        cost = _summarize_cost([measured, resumed])
        self.assertEqual(cost["measured_cases"], 1)
        self.assertEqual(cost["unmeasured_cases"], 1)
        self.assertEqual(cost["mean_llm_calls_per_case"], 9.0)
        self.assertEqual(cost["mean_total_tokens_per_case"], 1200.0)

    def test_summary_reports_nothing_when_every_case_was_resumed(self) -> None:
        resumed = DebateCaseResult(
            id="b",
            expected_label="no",
            predicted_label="no",
            vote_share={},
            agent_labels={},
            round1_vote_label="no",
        )
        cost = _summarize_cost([resumed])
        self.assertEqual(cost["measured_cases"], 0)
        self.assertNotIn("mean_llm_calls_per_case", cost)


class SelfConsistencyArmTests(unittest.TestCase):
    def _run(self, *, samples: int, backend: object | None = None) -> list[DebateCaseResult]:
        shared = backend or MockInferenceBackend()

        def _agent_factory(hint_provider: object, case_index: int) -> ClinicalAgent:
            return ClinicalAgent(
                agent_id="generalist",
                persona="generalist",
                backend=shared,
                hint_provider=hint_provider,  # type: ignore[arg-type]
                temperature=0.7,
                task_mode="pubmedqa",
            )

        return asyncio.run(
            _evaluate_self_consistency(
                [CASE],
                {},
                agent_factory=_agent_factory,
                samples=samples,
                hint_provider=None,
                inject_hint=False,
                case_concurrency=1,
                sample_concurrency=2,
                prior_results={},
                checkpoint_path=None,
            )
        )

    def test_draws_n_samples_and_charges_n_calls(self) -> None:
        (result,) = self._run(samples=6)
        self.assertEqual(len(result.agent_labels), 6)
        self.assertEqual(result.llm_calls, 6)
        self.assertTrue(result.cost_measured)
        self.assertEqual(result.rounds_run, 1)
        self.assertEqual(result.aggregation_rule, "self_consistency_majority_n6")

    def test_samples_never_see_each_other(self) -> None:
        """The whole point of the control arm: no peer context in any prompt."""
        seen: list[str] = []

        class SpyBackend(MockInferenceBackend):
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                seen.append("\n".join(message.content for message in messages))
                return await super().complete(messages, temperature=temperature)

        self._run(samples=4, backend=SpyBackend())
        self.assertEqual(len(seen), 4)
        for prompt in seen:
            self.assertNotIn("PEER OPINIONS", prompt)
        # Independent draws means every sample gets the identical prompt.
        self.assertEqual(len(set(seen)), 1)

    def test_majority_label_and_first_sample_baseline_are_both_reported(self) -> None:
        (result,) = self._run(samples=3)
        self.assertIn(result.predicted_label, {"yes", "no", "maybe"})
        self.assertEqual(result.round1_vote_label, result.agent_labels["sample_01"])
        self.assertTrue(result.unanimous_final)
        self.assertEqual(result.uncertainty_score, 0.0)
        self.assertEqual(result.uncertainty_signals["sample_agreement"], 1.0)

    def test_disagreeing_samples_raise_the_entropy_signal(self) -> None:
        labels = iter(["yes", "no", "maybe", "yes"])

        class RotatingBackend(MockInferenceBackend):
            async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
                label = next(labels)
                return (
                    '{"top_1_diagnosis": "%s", "top_3_differential_diagnoses": ["yes", "no", "maybe"], '
                    '"pros": [], "cons": [], "required_further_tests": [], "confidence_level": 0.5, '
                    '"sources_used": [], "red_flags": [], "missing_information": ""}' % label
                )

        (result,) = self._run(samples=4, backend=RotatingBackend())
        self.assertEqual(result.predicted_label, "yes")
        self.assertFalse(result.unanimous_final)
        self.assertGreater(result.uncertainty_score, 0.0)
        self.assertAlmostEqual(result.uncertainty_signals["sample_maybe_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
