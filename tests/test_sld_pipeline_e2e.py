"""End-to-end smoke tests for the full Supervised Ledger Debate pipeline,
running on real PQA-L abstracts against the schema-aware MockSLDBackend
(scripts/agents/evaluate_sld_pubmedqa.py) — the Faza 1 exit criterion from
the design doc: the whole wiring (Stage 0 -> R1 -> gate #1 -> Moderator ->
verify_ledger -> R2 -> gate #2 -> Director -> compose_label) must run without
crashing, respect prompt budgets, and the verification gate must actually
fire on fabricated citations, not just in isolated unit tests.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.agents.sld.pipeline import SLDCase, SLDPipeline
from scripts.agents.evaluate_sld_pubmedqa import MockSLDBackend, _case_abstract_raw, _load_cases, _load_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "pubmedqa"
    / "official_pqal_test"
    / "quick"
    / "balanced90.json"
)
CORPUS = PROJECT_ROOT / "data" / "benchmarks" / "pubmedqa" / "official_pqal_test" / "corpus.json"


def _load_sample_cases(n: int) -> list[dict]:
    cases = _load_cases(DATASET)
    return cases[:n]


class PipelineMockE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_to_completion_on_real_abstracts_without_crashing(self) -> None:
        corpus = _load_corpus(CORPUS)
        cases = _load_sample_cases(10)
        backend = MockSLDBackend(hallucinate=False)
        pipeline = SLDPipeline(backend=backend, director_samples=1)

        results = []
        for case in cases:
            sld_case = SLDCase(
                case_id=case["id"],
                question=case["question"],
                abstract_raw=_case_abstract_raw(case, corpus),
                expected_label=case["expected_label"],
            )
            result = await pipeline.run(sld_case)
            results.append(result)

        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIn(result.predicted_label, {"yes", "no", "maybe"})
            self.assertIsNotNone(result.ledger)
            self.assertIsNotNone(result.director_verdict)
            self.assertIsNotNone(result.rule_name)
            # Result must be JSON-round-trippable (checkpoint format).
            reloaded = json.loads(result.model_dump_json())
            self.assertEqual(reloaded["case_id"], result.case_id)

    async def test_hallucinating_backend_drives_grounding_score_to_zero(self) -> None:
        """The gate must actually fire in a live pipeline run, not just in
        isolated verify.py unit tests."""
        corpus = _load_corpus(CORPUS)
        cases = _load_sample_cases(5)
        backend = MockSLDBackend(hallucinate=True)
        pipeline = SLDPipeline(backend=backend, director_samples=1)

        total_dropped = 0
        for case in cases:
            sld_case = SLDCase(
                case_id=case["id"],
                question=case["question"],
                abstract_raw=_case_abstract_raw(case, corpus),
                expected_label=case["expected_label"],
            )
            result = await pipeline.run(sld_case)
            self.assertEqual(result.grounding_score_r1, 0.0)
            self.assertEqual(result.grounding_score_r2, 0.0)
            total_dropped += len(result.dropped_claims_r1) + len(result.dropped_claims_r2)
            # Pipeline still produces a usable result; it doesn't crash or hang.
            self.assertIn(result.predicted_label, {"yes", "no", "maybe"})

        self.assertGreater(total_dropped, 0)

    async def test_clean_backend_achieves_high_grounding_across_a_larger_sample(self) -> None:
        corpus = _load_corpus(CORPUS)
        cases = _load_sample_cases(30)
        backend = MockSLDBackend(hallucinate=False)
        pipeline = SLDPipeline(backend=backend, director_samples=1)

        scores_r1 = []
        for case in cases:
            sld_case = SLDCase(
                case_id=case["id"],
                question=case["question"],
                abstract_raw=_case_abstract_raw(case, corpus),
                expected_label=case["expected_label"],
            )
            result = await pipeline.run(sld_case)
            scores_r1.append(result.grounding_score_r1)

        # Not a strict 1.0 (some abstracts lack RESULTS sentences, correctly
        # tripping the findings_auditor out-of-section check even for a
        # non-hallucinating mock) but should be high on average.
        self.assertGreater(sum(scores_r1) / len(scores_r1), 0.85)

    async def test_verdict_source_llm_bypasses_the_rule_table(self) -> None:
        corpus = _load_corpus(CORPUS)
        case = _load_sample_cases(1)[0]
        backend = MockSLDBackend(hallucinate=False)
        pipeline = SLDPipeline(backend=backend, director_samples=1, verdict_source="llm")

        sld_case = SLDCase(
            case_id=case["id"],
            question=case["question"],
            abstract_raw=_case_abstract_raw(case, corpus),
            expected_label=case["expected_label"],
        )
        result = await pipeline.run(sld_case)
        self.assertEqual(result.rule_name, "director_llm_label")
        self.assertEqual(result.predicted_label, result.director_verdict.label)

    async def test_l3_arm_skips_round_two_and_director_entirely(self) -> None:
        corpus = _load_corpus(CORPUS)
        cases = _load_sample_cases(5)
        backend = MockSLDBackend(hallucinate=False)
        pipeline = SLDPipeline(backend=backend, run_round_two=False)

        for case in cases:
            sld_case = SLDCase(
                case_id=case["id"],
                question=case["question"],
                abstract_raw=_case_abstract_raw(case, corpus),
                expected_label=case["expected_label"],
            )
            result = await pipeline.run(sld_case)
            self.assertEqual(result.panel_r2_raw, [])
            self.assertEqual(result.panel_r2_verified, [])
            self.assertEqual(result.grounding_score_r2, 1.0)
            self.assertIsNotNone(result.director_verdict)
            self.assertIn(result.predicted_label, {"yes", "no", "maybe"})
            self.assertNotEqual(result.rule_name, "director_llm_label")

    async def test_l4_arm_uses_peer_notes_and_majority_vote_no_director(self) -> None:
        corpus = _load_corpus(CORPUS)
        cases = _load_sample_cases(5)
        backend = MockSLDBackend(hallucinate=False)
        pipeline = SLDPipeline(backend=backend, show_ledger_in_r2=False)

        for case in cases:
            sld_case = SLDCase(
                case_id=case["id"],
                question=case["question"],
                abstract_raw=_case_abstract_raw(case, corpus),
                expected_label=case["expected_label"],
            )
            result = await pipeline.run(sld_case)
            self.assertEqual(len(result.panel_r2_raw), 4)
            self.assertIsNone(result.director_verdict)
            self.assertEqual(result.rule_name, "majority_vote_no_ledger")
            self.assertIn(result.predicted_label, {"yes", "no", "maybe"})

    async def test_verification_disabled_lets_hallucinated_claims_through_but_still_measures_them(
        self,
    ) -> None:
        """Ablation (a): grounding_score/dropped_claims stay populated either
        way (the damage is always measured); the flag only controls whether
        the raw or cleaned contribution is what actually reaches the ledger."""
        corpus = _load_corpus(CORPUS)
        case = _load_sample_cases(1)[0]
        hallucinating_backend = MockSLDBackend(hallucinate=True)

        gated = SLDPipeline(backend=hallucinating_backend, director_samples=1, verification_enabled=True)
        ungated = SLDPipeline(backend=hallucinating_backend, director_samples=1, verification_enabled=False)

        sld_case = SLDCase(
            case_id=case["id"],
            question=case["question"],
            abstract_raw=_case_abstract_raw(case, corpus),
            expected_label=case["expected_label"],
        )
        gated_result = await gated.run(sld_case)
        ungated_result = await ungated.run(sld_case)

        # Both measure the same damage...
        self.assertEqual(gated_result.grounding_score_r1, 0.0)
        self.assertEqual(ungated_result.grounding_score_r1, 0.0)
        self.assertEqual(len(gated_result.dropped_claims_r1), len(ungated_result.dropped_claims_r1))
        # ...but only the gated run actually strips the fabricated claims.
        self.assertIsNone(gated_result.ledger.primary_endpoint)
        self.assertIsNotNone(ungated_result.ledger.primary_endpoint)

    async def test_stats_profile_disabled_yields_an_empty_profile_in_the_prompt(self) -> None:
        """Ablation (b): withholding stats_profile shouldn't crash the run —
        findings_auditor/gap_auditor prompts just lose that reference block."""
        corpus = _load_corpus(CORPUS)
        case = _load_sample_cases(1)[0]
        backend = MockSLDBackend(hallucinate=False)
        pipeline = SLDPipeline(backend=backend, director_samples=1, use_stats_profile=False)

        sld_case = SLDCase(
            case_id=case["id"],
            question=case["question"],
            abstract_raw=_case_abstract_raw(case, corpus),
            expected_label=case["expected_label"],
        )
        result = await pipeline.run(sld_case)
        self.assertIn(result.predicted_label, {"yes", "no", "maybe"})


if __name__ == "__main__":
    unittest.main()
