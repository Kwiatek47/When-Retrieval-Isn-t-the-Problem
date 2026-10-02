from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _arr(text: str) -> np.ndarray:
    """Labels from a compact string: y = yes, n = no, m = maybe."""
    return np.array([{"y": "yes", "n": "no", "m": "maybe"}[c] for c in text])


class LabelProbeAnalysisTests(unittest.TestCase):
    GOLD = _arr("mmmmyynn")
    HUMAN = _arr("mmmyyynn")   # 3 hits, 3 predicted -> F1 = 6/7
    MODEL = _arr("myyymynn")   # 1 hit, 2 predicted  -> F1 = 2/6

    def test_maybe_f1(self) -> None:
        from scripts.agents.analyze_label_probe import maybe_f1

        self.assertAlmostEqual(maybe_f1(self.HUMAN, self.GOLD), 6 / 7)
        self.assertAlmostEqual(maybe_f1(self.MODEL, self.GOLD), 2 / 6)
        self.assertEqual(maybe_f1(_arr("yyyyyynn"), self.GOLD), 0.0)

    def test_gap_and_conclusion_effect(self) -> None:
        from scripts.agents.analyze_label_probe import conclusion_effect, gap_to_human

        self.assertAlmostEqual(gap_to_human(self.MODEL, self.HUMAN, self.GOLD), 6 / 7 - 2 / 6)
        with_conclusion = _arr("mmyymynn")  # 2 hits, 3 predicted -> 4/7
        self.assertAlmostEqual(conclusion_effect(with_conclusion, self.MODEL, self.GOLD), 4 / 7 - 2 / 6)

    def test_verdicts_follow_the_registered_rules(self) -> None:
        from scripts.agents.analyze_label_probe import effect_verdict, gap_verdict

        self.assertEqual(gap_verdict({"ci_low": 0.1, "ci_high": 0.4}), "model below human")
        self.assertEqual(gap_verdict({"ci_low": -0.3, "ci_high": -0.1}), "model above human")
        self.assertEqual(gap_verdict({"ci_low": -0.1, "ci_high": 0.2}), "not distinguishable")
        self.assertEqual(effect_verdict({"ci_low": 0.01, "ci_high": 0.2}), "supported")
        self.assertEqual(effect_verdict({"ci_low": -0.2, "ci_high": 0.0}), "refuted")
        self.assertEqual(effect_verdict({"ci_low": -0.1, "ci_high": 0.1}), "inconclusive")

    def test_unanswered_questions_count_as_wrong_and_not_maybe(self) -> None:
        from scripts.agents.analyze_label_probe import NO_ANSWER, describe, load_labels

        rows = [
            {"pmid": "1", "repeat": 0, "label": "maybe"},
            {"pmid": "2", "repeat": 0, "label": None},
            {"pmid": "3", "repeat": 1, "label": "yes"},  # only the first repeat is scored
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"partial', encoding="utf-8")
            pred = load_labels(path, ["1", "2", "3"])
        self.assertEqual(list(pred), ["maybe", NO_ANSWER, NO_ANSWER])
        frame = {"gold": _arr("mmy"), "rr": _arr("mmy"), "rf": _arr("mmy")}
        d = describe(pred, frame)
        self.assertEqual(d["unanswered"], 2)
        self.assertAlmostEqual(d["accuracy_vs_gold"], round(1 / 3, 4))
        self.assertEqual(d["maybe_vs_gold"]["predicted"], 1)

    def test_analyze_mode_runs_end_to_end_on_synthetic_data(self) -> None:
        from scripts.agents.analyze_label_probe import analyze_mode

        rng = np.random.default_rng(0)
        labels = np.array(["yes", "no", "maybe"])
        frame = {k: labels[rng.integers(0, 3, size=80)] for k in ("gold", "rr", "rf")}
        without, with_conclusion = labels[rng.integers(0, 3, size=80)], labels[rng.integers(0, 3, size=80)]
        r = analyze_mode(without, with_conclusion, frame, np.random.default_rng(1), 40)
        self.assertIn(r["T1_gap_to_human"]["verdict"], ("model below human", "model above human", "not distinguishable"))
        changed = int(((without != "maybe") & (with_conclusion == "maybe")).sum())
        self.assertEqual(r["answers_changed_to_maybe"], changed)


class ThinkModeTests(unittest.TestCase):
    def test_think_mode_gets_its_own_file_and_blocks_mixed_resumes(self) -> None:
        from scripts.agents.probe_prompts import PROMPTS
        from scripts.agents.run_label_probe import default_out, done_keys

        spec = PROMPTS["label-defined@1"]
        on = default_out(spec, "context", "qwen3:30b", "test", "on")
        off = default_out(spec, "context", "qwen3:30b", "test", "off")
        self.assertNotEqual(on, off)
        self.assertIn("think-on", on.name)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            row = {"pmid": "1", "repeat": 0, "label": "yes", "input": "context", "model": "m", "think": "on", "prompt_sha": spec.sha}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(done_keys(path, "m", spec, "context", "on"), {("1", 0)})
            with self.assertRaises(SystemExit):
                done_keys(path, "m", spec, "context", "off")


if __name__ == "__main__":
    unittest.main()
