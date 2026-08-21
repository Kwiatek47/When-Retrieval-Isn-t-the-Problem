from __future__ import annotations

import json
from pathlib import Path
import unittest


def _director(label: str) -> str:
    return json.dumps(
        {
            "final_label": label,
            "consensus_type": "consensus",
            "rationale": "grounded",
        }
    )


class SupervisorSftMetricsTests(unittest.TestCase):
    def test_export_modelfile_and_smoke_selection_cover_both_tasks(self) -> None:
        from scripts.sft.export_qwen14b_supervisor_ollama import (
            render_modelfile,
            select_smoke_records,
        )

        records = [
            {"id": f"{task}-{label}", "task_type": task, "gold_label": label}
            for task in ("director", "moderator")
            for label in ("yes", "no", "maybe")
        ]
        selected = select_smoke_records(records)
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            {(row["task_type"], row["gold_label"]) for row in selected},
            {
                (task, label)
                for task in ("director", "moderator")
                for label in ("yes", "no", "maybe")
            },
        )
        modelfile = render_modelfile(Path("/tmp/supervisor-q4_k_m.gguf"))
        self.assertIn("FROM /tmp/supervisor-q4_k_m.gguf", modelfile)
        self.assertIn("PARAMETER num_ctx 8192", modelfile)

    def test_director_metrics_include_macro_f1_maybe_recall_and_validity(self) -> None:
        from scripts.sft.evaluate_qwen14b_supervisor import compute_director_metrics

        rows = [
            {"gold_label": "yes"},
            {"gold_label": "no"},
            {"gold_label": "maybe"},
            {"gold_label": "maybe"},
        ]
        predictions = [_director("yes"), _director("no"), _director("maybe"), "not-json"]
        metrics, details = compute_director_metrics(rows, predictions)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["json_validity"], 0.75)
        self.assertAlmostEqual(metrics["per_label"]["maybe"]["recall"], 0.5)
        self.assertGreater(metrics["macro_f1"], 0.0)
        self.assertEqual(len(details), 4)

    def test_moderator_metrics_check_schema_and_author_conclusion(self) -> None:
        from scripts.sft.evaluate_qwen14b_supervisor import compute_moderator_metrics

        rows = [{"gold_label": "maybe"}, {"gold_label": "yes"}]
        predictions = [
            json.dumps(
                {
                    "agreements": [],
                    "contradictions": ["split"],
                    "round_instructions": ["resolve"],
                    "author_conclusion": "maybe",
                }
            ),
            "{}",
        ]
        metrics, _ = compute_moderator_metrics(rows, predictions)
        self.assertEqual(metrics["json_validity"], 1.0)
        self.assertEqual(metrics["pydantic_validity"], 1.0)
        self.assertEqual(metrics["author_conclusion_accuracy"], 0.5)
        self.assertEqual(metrics["instruction_completeness"], 0.5)

    def test_checkpoint_gate_uses_macro_f1_with_accuracy_floor(self) -> None:
        from scripts.sft.evaluate_qwen14b_supervisor import checkpoint_qualifies

        baseline = {"accuracy": 0.70, "macro_f1": 0.60}
        self.assertTrue(
            checkpoint_qualifies(
                {
                    "accuracy": 0.69,
                    "macro_f1": 0.65,
                    "json_validity": 0.995,
                    "moderator_pydantic_validity": 1.0,
                },
                baseline=baseline,
                baseline_moderator_pydantic_validity=1.0,
            )
        )
        self.assertFalse(
            checkpoint_qualifies(
                {
                    "accuracy": 0.68,
                    "macro_f1": 0.70,
                    "json_validity": 1.0,
                    "moderator_pydantic_validity": 1.0,
                },
                baseline=baseline,
                baseline_moderator_pydantic_validity=1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
