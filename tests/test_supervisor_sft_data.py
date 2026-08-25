from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def _pubmedqa_item(label: str, *, question: str = "Is X effective?") -> dict:
    return {
        "QUESTION": question,
        "CONTEXTS": ["Primary endpoint result."],
        "LONG_ANSWER": f"Authors conclude {label}.",
        "final_decision": label,
    }


class SupervisorSftDataTests(unittest.TestCase):
    def test_prepare_script_runs_directly_from_repo_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "scripts/sft/prepare_pubmedqa_supervisor_sft.py",
                "--help",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_loader_excludes_every_heldout_pmid_and_unlabeled_source(self) -> None:
        from scripts.sft.prepare_pubmedqa_supervisor_sft import load_safe_examples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pqal = root / "ori_pqal.json"
            pqaa = root / "ori_pqaa.json"
            pqau = root / "ori_pqau.json"
            heldout = root / "eval.json"
            pqal.write_text(
                json.dumps(
                    {
                        "1": _pubmedqa_item("maybe"),
                        "2": _pubmedqa_item("yes"),
                    }
                ),
                encoding="utf-8",
            )
            pqaa.write_text(json.dumps({"3": _pubmedqa_item("no")}), encoding="utf-8")
            pqau.write_text(json.dumps({"4": {"QUESTION": "Unlabeled"}}), encoding="utf-8")
            heldout.write_text(json.dumps([{"relevant_pmids": ["2"]}]), encoding="utf-8")

            examples, blocked = load_safe_examples(
                pqal_path=pqal,
                pqaa_path=pqaa,
                heldout_eval=heldout,
            )

        self.assertEqual(blocked, {"2"})
        self.assertEqual({example.pmid for example in examples}, {"1", "3"})
        self.assertEqual({example.source_dataset for example in examples}, {"pqa_l", "pqa_a"})

    def test_three_way_split_is_deterministic_disjoint_and_keeps_all_pqal(self) -> None:
        from scripts.classifier.prepare_pubmedqa_deberta_dataset import ClassifierExample
        from scripts.sft.prepare_pubmedqa_supervisor_sft import split_examples_three_way

        examples = []
        for source_name, count in (("pqa_l", 30), ("pqa_a", 60)):
            for index in range(count):
                label = ("yes", "no", "maybe")[index % 3]
                examples.append(
                    ClassifierExample(
                        id=f"{source_name}-{index}",
                        pmid=f"{source_name}-{index}",
                        question="Q",
                        evidence="E",
                        label=label,
                        source_file=f"ori_{source_name.replace('_', '')}.json",
                        source_dataset=source_name,
                        long_answer="L",
                    )
                )

        first = split_examples_three_way(
            examples,
            seed=47,
            train_fraction=0.70,
            dev_fraction=0.15,
            max_pqaa_per_label=6,
        )
        second = split_examples_three_way(
            examples,
            seed=47,
            train_fraction=0.70,
            dev_fraction=0.15,
            max_pqaa_per_label=6,
        )

        self.assertEqual(
            [[item.pmid for item in first[name]] for name in ("train", "dev", "internal_test")],
            [[item.pmid for item in second[name]] for name in ("train", "dev", "internal_test")],
        )
        pmid_sets = {name: {item.pmid for item in rows} for name, rows in first.items()}
        self.assertFalse(pmid_sets["train"] & pmid_sets["dev"])
        self.assertFalse(pmid_sets["train"] & pmid_sets["internal_test"])
        self.assertFalse(pmid_sets["dev"] & pmid_sets["internal_test"])
        self.assertEqual(
            sum(item.source_dataset == "pqa_l" for rows in first.values() for item in rows),
            30,
        )
        self.assertLessEqual(
            sum(item.source_dataset == "pqa_a" for rows in first.values() for item in rows),
            18,
        )

    def test_leakage_audit_rejects_blocked_overlap(self) -> None:
        from scripts.classifier.prepare_pubmedqa_deberta_dataset import ClassifierExample
        from scripts.sft.prepare_pubmedqa_supervisor_sft import assert_no_heldout_overlap

        example = ClassifierExample(
            id="bad",
            pmid="123",
            question="Q",
            evidence="E",
            label="yes",
            source_file="ori_pqal.json",
            source_dataset="pqa_l",
        )
        with self.assertRaisesRegex(ValueError, "123"):
            assert_no_heldout_overlap({"train": [example]}, {"123"})

    def test_debate_record_contains_two_rounds_and_compact_brief(self) -> None:
        from app.agents.backends import EvidenceHint, MockInferenceBackend
        from scripts.sft.generate_pubmedqa_supervisor_debates import generate_debate_record

        source = {
            "id": "pubmedqa-1",
            "pmid": "1",
            "question": "Is X effective?",
            "evidence": "X improved the primary endpoint.",
            "long_answer": "The authors conclude that X is effective.",
            "label": "yes",
            "source_dataset": "pqa_l",
        }
        record = asyncio.run(
            generate_debate_record(
                source,
                backend=MockInferenceBackend(),
                hint=EvidenceHint(label="yes", confidence=0.9),
                agent_concurrency=2,
            )
        )

        self.assertEqual(record["pmid"], "1")
        self.assertEqual(record["gold_label"], "yes")
        self.assertEqual(len(record["history"]), 2)
        self.assertTrue(all(len(round_entries) == 4 for round_entries in record["history"]))
        self.assertEqual(record["debate_brief"]["rounds_completed"], 2)
        dual_read = record["debate_brief"]["dual_read"]
        self.assertIn("author_conclusion_reader", dual_read)
        self.assertIn("uncertainty_advocate", dual_read)
        self.assertNotIn("relevance_checker", dual_read)
        self.assertNotIn("data_skeptic", dual_read)
        self.assertEqual(record["biolinkbert_hint"]["label"], "yes")
        self.assertNotIn(record["long_answer"], record["patient_case"])

    def test_completed_ids_tolerates_partial_last_line(self) -> None:
        from scripts.sft.generate_pubmedqa_supervisor_debates import load_completed_ids

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            path.write_text('{"id":"a"}\n{"id":"b"}\n{"id":', encoding="utf-8")
            self.assertEqual(load_completed_ids(path), {"a", "b"})

    def test_deterministic_targets_match_supervisor_schemas(self) -> None:
        from app.agents.backends import EvidenceHint, MockInferenceBackend
        from app.agents.models import SupervisorDirectorOutput, SupervisorModerationOutput
        from scripts.sft.generate_pubmedqa_supervisor_debates import generate_debate_record
        from scripts.sft.prepare_pubmedqa_supervisor_sft import (
            build_director_sft_record,
            build_moderator_sft_record,
        )

        source = {
            "id": "pubmedqa-1",
            "pmid": "1",
            "question": "Is X effective?",
            "evidence": "X improved the primary endpoint.",
            "long_answer": "The authors conclude that X may be effective, but evidence is inconclusive.",
            "label": "maybe",
            "source_dataset": "pqa_l",
        }
        debate = asyncio.run(
            generate_debate_record(
                source,
                backend=MockInferenceBackend(),
                hint=EvidenceHint(label="yes", confidence=0.7),
            )
        )

        director = build_director_sft_record(debate)
        moderator = build_moderator_sft_record(debate)
        director_target = SupervisorDirectorOutput.model_validate_json(
            director["messages"][-1]["content"]
        )
        moderator_target = SupervisorModerationOutput.model_validate_json(
            moderator["messages"][-1]["content"]
        )

        self.assertEqual(director["task_type"], "director")
        self.assertEqual(moderator["task_type"], "moderator")
        self.assertEqual(director_target.final_label, "maybe")
        self.assertEqual(moderator_target.author_conclusion, "maybe")
        self.assertIn("=== ROUND 1 ===", director["messages"][1]["content"])
        self.assertNotIn(source["long_answer"], director["messages"][1]["content"])
        self.assertNotIn(source["long_answer"], moderator["messages"][1]["content"])
        self.assertNotIn("4-agent", moderator["messages"][1]["content"])
        self.assertIn('"author_conclusion"', moderator["messages"][1]["content"])
        self.assertIn('"primary_endpoint_result"', moderator["messages"][1]["content"])
        self.assertIn('"residual_uncertainty"', moderator["messages"][1]["content"])
        self.assertTrue(moderator_target.round_instructions)

    def test_sft_validator_rejects_heldout_pmid(self) -> None:
        from scripts.sft.prepare_pubmedqa_supervisor_sft import validate_sft_record

        record = {
            "id": "x",
            "pmid": "123",
            "gold_label": "yes",
            "task_type": "director",
            "messages": [
                {"role": "system", "content": "json"},
                {"role": "user", "content": "case"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "final_label": "yes",
                            "consensus_type": "consensus",
                        }
                    ),
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_sft_record(record, blocked_pmids={"123"})

    def test_class_aware_replay_raises_maybe_share_for_training_only(self) -> None:
        from scripts.sft.prepare_pubmedqa_supervisor_sft import class_aware_director_replay

        records = [
            {"id": f"yes-{index}", "gold_label": "yes"}
            for index in range(8)
        ] + [
            {"id": "maybe-0", "gold_label": "maybe"}
        ]
        replayed = class_aware_director_replay(records, min_maybe_share=0.20)
        maybe_count = sum(row["gold_label"] == "maybe" for row in replayed)
        self.assertGreaterEqual(maybe_count / len(replayed), 0.20)
        self.assertEqual(len({row["id"] for row in replayed}), len(replayed))


if __name__ == "__main__":
    unittest.main()
