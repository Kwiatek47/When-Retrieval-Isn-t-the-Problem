"""Stage 0 segmentation tests, run against real PQA-L corpus abstracts.

corpus.json wraps every abstract in benchmark boilerplate ("PubMedQA ... source
PMID ... Research question: ... Abstract context: ..."); these tests exist
because getting that stripped wrong silently poisons every downstream S-ID.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.agents.sld.segmentation import (
    classify_question_type,
    extract_abstract_text,
    extract_stats_profile,
    split_sentences,
    tag_sections,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = (
    PROJECT_ROOT / "data" / "benchmarks" / "pubmedqa" / "official_pqal_test" / "corpus.json"
)


def _load_corpus() -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


class ExtractAbstractTextTests(unittest.TestCase):
    def test_strips_benchmark_boilerplate_prefix(self) -> None:
        raw = (
            "PubMedQA official PQA-L test source PMID 12377809. "
            "Research question: Is X valuable in Y? "
            "Abstract context: Real abstract sentence one. Real abstract sentence two."
        )
        abstract = extract_abstract_text(raw)
        self.assertEqual(
            abstract, "Real abstract sentence one. Real abstract sentence two."
        )
        self.assertNotIn("PMID", abstract)
        self.assertNotIn("Research question", abstract)

    def test_missing_marker_falls_back_to_full_text(self) -> None:
        raw = "No boilerplate marker here, just an abstract."
        self.assertEqual(extract_abstract_text(raw), raw)

    def test_every_official_pqal_doc_has_the_marker(self) -> None:
        corpus = _load_corpus()
        missing = [d["id"] for d in corpus if "Abstract context: " not in d["content"]]
        self.assertEqual(missing, [], f"{len(missing)} docs lack the Abstract context marker")


class SplitSentencesTests(unittest.TestCase):
    def test_stable_sequential_ids(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_sentences(text)
        self.assertEqual([sid for sid, _ in sentences], ["S1", "S2", "S3"])
        self.assertEqual(
            [s for _, s in sentences],
            ["First sentence.", "Second sentence.", "Third sentence."],
        )

    def test_empty_text_yields_no_sentences(self) -> None:
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   "), [])

    def test_does_not_split_on_p_value_decimal(self) -> None:
        text = "Changes were significant (p<0.01, chi(2) test) in patients."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 1)

    def test_does_not_split_on_percent_or_ci(self) -> None:
        text = "Risk increased to 85.5% (95% CI 1.2 to 3.4) among patients. This is real."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("85.5%", sentences[0][1])

    def test_does_not_split_on_common_abbreviations(self) -> None:
        text = "We compared groups (e.g. cases vs. controls) using a t-test. Results followed."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)

    def test_real_abstract_round_trips_without_dropping_content(self) -> None:
        corpus = _load_corpus()
        doc = next(d for d in corpus if d["id"] == "pubmedqa-official-12377809")
        abstract = extract_abstract_text(doc["content"])
        sentences = split_sentences(abstract)
        self.assertGreaterEqual(len(sentences), 5)
        rejoined = " ".join(s for _, s in sentences)
        # No sentence-splitting run should lose non-whitespace content.
        self.assertEqual(
            "".join(rejoined.split()),
            "".join(abstract.split()),
        )

    def test_ids_are_unique_and_ordered_across_a_sample_of_the_corpus(self) -> None:
        corpus = _load_corpus()
        for doc in corpus[:25]:
            abstract = extract_abstract_text(doc["content"])
            sentences = split_sentences(abstract)
            ids = [sid for sid, _ in sentences]
            self.assertEqual(ids, sorted(ids, key=lambda x: int(x[1:])))
            self.assertEqual(len(ids), len(set(ids)))


class TagSectionsTests(unittest.TestCase):
    def test_stats_bearing_sentence_is_results(self) -> None:
        sentences = [("S1", "The response rate was significantly higher at 85% (p<0.01).")]
        tags = tag_sections(sentences)
        self.assertEqual(tags["S1"], "RESULTS")

    def test_recruitment_sentence_is_methods(self) -> None:
        sentences = [("S1", "Patients were randomly assigned to two treatment arms.")]
        tags = tag_sections(sentences)
        self.assertEqual(tags["S1"], "METHODS")

    def test_purpose_sentence_is_background_even_when_naming_design(self) -> None:
        sentences = [
            ("S1", "The aim of this prospective study was to assess treatment outcomes.")
        ]
        tags = tag_sections(sentences)
        self.assertEqual(tags["S1"], "BACKGROUND")

    def test_unrelated_sentence_falls_back_to_other(self) -> None:
        sentences = [("S1", "Dyschesia can be provoked by inappropriate defecation movements.")]
        tags = tag_sections(sentences)
        self.assertEqual(tags["S1"], "OTHER")

    def test_real_abstract_tags_every_sentence_and_favors_precision_on_results(self) -> None:
        corpus = _load_corpus()
        doc = next(d for d in corpus if d["id"] == "pubmedqa-official-12377809")
        abstract = extract_abstract_text(doc["content"])
        sentences = split_sentences(abstract)
        tags = tag_sections(sentences)
        self.assertEqual(set(tags.keys()), {sid for sid, _ in sentences})
        # The two sentences carrying p<0.01 and both percentage comparisons
        # must be tagged RESULTS; this abstract has no other RESULTS content.
        results_ids = {sid for sid, label in tags.items() if label == "RESULTS"}
        self.assertIn("S6", results_ids)
        self.assertIn("S7", results_ids)


class ExtractStatsProfileTests(unittest.TestCase):
    def test_extracts_p_value_ci_percent_n_effect_size_and_diagnostic_metric(self) -> None:
        sentences = [
            ("S1", "Sensitivity was 82% and specificity was 75% (95% CI 1.1 to 2.0)."),
            ("S2", "The odds ratio was OR=2.3 with p<0.05 among n=120 patients."),
            ("S3", "The AUC was 0.81 and PPV was high."),
        ]
        profile = extract_stats_profile(sentences)
        self.assertTrue(any(hit.sentence_id == "S2" for hit in profile.p_values))
        self.assertTrue(any(hit.sentence_id == "S1" for hit in profile.confidence_intervals))
        self.assertTrue(any(hit.sentence_id == "S1" for hit in profile.percentages))
        self.assertTrue(any(hit.sentence_id == "S2" for hit in profile.sample_sizes))
        self.assertTrue(any(hit.sentence_id == "S2" for hit in profile.effect_sizes))
        self.assertTrue(any(hit.sentence_id == "S3" for hit in profile.diagnostic_metrics))
        self.assertFalse(profile.is_empty())

    def test_no_markers_yields_empty_profile(self) -> None:
        sentences = [("S1", "Dyschesia can be provoked by inappropriate defecation movements.")]
        profile = extract_stats_profile(sentences)
        self.assertTrue(profile.is_empty())
        self.assertEqual(profile.sentence_ids(), set())

    def test_sentence_ids_unions_across_categories(self) -> None:
        sentences = [("S1", "The rate was 50% with p<0.01, n=40, and AUC 0.7 (95% CI 0.6 to 0.8).")]
        profile = extract_stats_profile(sentences)
        self.assertEqual(profile.sentence_ids(), {"S1"})


class ClassifyQuestionTypeTests(unittest.TestCase):
    def test_diagnostic_accuracy_wins_over_utility(self) -> None:
        self.assertEqual(
            classify_question_type("Is X valuable in diagnosing Y?"), "diagnostic_accuracy"
        )

    def test_plain_utility(self) -> None:
        self.assertEqual(
            classify_question_type("Is anorectal endosonography valuable in dyschesia?"),
            "utility",
        )

    def test_causal(self) -> None:
        self.assertEqual(classify_question_type("Does X cause Y?"), "causal")

    def test_comparison(self) -> None:
        self.assertEqual(classify_question_type("Is X better than Y?"), "comparison")

    def test_prevalence(self) -> None:
        self.assertEqual(classify_question_type("What is the prevalence of X?"), "prevalence")

    def test_default_falls_back_to_association(self) -> None:
        self.assertEqual(classify_question_type("Is X related to Y outcomes?"), "association")


if __name__ == "__main__":
    unittest.main()
