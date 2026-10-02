from __future__ import annotations

import json
from pathlib import Path
import string
import tempfile
import unittest


class PromptRegistryTests(unittest.TestCase):
    def test_registered_prompt_text_is_frozen(self) -> None:
        # H1b was pre-registered and fully run with this exact prompt (commit 73cd9bf).
        # If this fails, the registered prompt was edited: restore it and add a new version.
        from scripts.agents.probe_prompts import PROMPTS

        self.assertEqual(PROMPTS["conditionality@1"].sha, "77e624205b2a")
        self.assertEqual(PROMPTS["conditionality@1"].status, "registered")

    def test_every_prompt_renders_with_exactly_its_fields(self) -> None:
        from scripts.agents.probe_prompts import PROMPTS

        shas = set()
        for prompt_id, spec in PROMPTS.items():
            placeholders = {name for _, name, _, _ in string.Formatter().parse(spec.user) if name}
            self.assertEqual(placeholders, set(spec.fields), prompt_id)
            messages = spec.messages(**{f: f"<{f}>" for f in spec.fields})
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            json.dumps(spec.schema)
            self.assertIn(spec.status, ("registered", "candidate"))
            shas.add(spec.sha)
        self.assertEqual(len(shas), len(PROMPTS), "two prompt versions share one text")

    def test_missing_field_is_an_error(self) -> None:
        from scripts.agents.probe_prompts import PROMPTS

        with self.assertRaises(KeyError):
            PROMPTS["label-defined@1"].messages(question="Q")

    def test_prompts_never_mention_gold_or_annotator_fields(self) -> None:
        from scripts.agents.probe_prompts import PROMPTS

        for prompt_id, spec in PROMPTS.items():
            text = (spec.system + spec.user).lower()
            for leaked in ("final_decision", "reasoning_free", "reasoning_required", "gold label", "annotator"):
                self.assertNotIn(leaked, text, prompt_id)


class SnapshotTests(unittest.TestCase):
    def test_each_run_writes_a_snapshot_and_appends_the_registry(self) -> None:
        from scripts.agents.probe_prompts import PROMPTS, snapshot_run

        spec = PROMPTS["label-defined@1"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "probe.run.jsonl"
            record = snapshot_run(out, [spec], script="tests", model="m", meta={"input": "context"})
            snapshot = json.loads((Path(tmp) / "probe.run.prompts.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["prompt_version"], record["prompt_version"])
            self.assertEqual(snapshot["prompts"]["label-defined@1"]["sha"], spec.sha)
            self.assertEqual(snapshot["prompts"]["label-defined@1"]["user"], spec.user)
            self.assertEqual(snapshot["meta"]["model"], "m")
            self.assertIn("commit", snapshot["meta"]["git"])

            snapshot_run(out, [spec], script="tests", model="m")
            rows = (Path(tmp) / "prompt_versions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[0])["meta"]["prompt_shas"], {"label-defined@1": spec.sha})


class LabelProbeTests(unittest.TestCase):
    ITEM = {
        "QUESTION": "Does X help?",
        "CONTEXTS": ["We tested X.", "X helped women only."],
        "LABELS": ["OBJECTIVE", "RESULTS"],
        "LONG_ANSWER": "X may help women.",
    }

    def test_conclusion_is_shown_only_in_the_conclusion_condition(self) -> None:
        from scripts.agents.run_label_probe import evidence_text

        without = evidence_text(self.ITEM, "context")
        with_conclusion = evidence_text(self.ITEM, "context+conclusion")
        self.assertIn("RESULTS: X helped women only.", without)
        self.assertNotIn("X may help women.", without)
        self.assertIn("Authors' conclusion:\nX may help women.", with_conclusion)
        with self.assertRaises(ValueError):
            evidence_text(self.ITEM, "conclusion")

    def test_parse_label(self) -> None:
        from scripts.agents.run_label_probe import parse_label

        self.assertEqual(parse_label('{"rationale": "r", "label": "maybe", "confidence": 60}'), ("maybe", 60, "r"))
        self.assertEqual(parse_label('{"rationale": "r", "label": "maybe", "confidence": 900}'), ("maybe", None, "r"))
        self.assertEqual(parse_label('{"label": "unsure"}'), (None, None, ""))


class ConditionalityRaterTests(unittest.TestCase):
    def test_calibration_mode_rates_only_sheet_passages(self) -> None:
        from scripts.agents.rate_conditionality import build_tasks

        data = {
            "1": {"QUESTION": "Q1", "CONTEXTS": ["r1"], "LABELS": ["RESULTS"], "LONG_ANSWER": "c1"},
            "2": {"QUESTION": "Q2", "CONTEXTS": ["r2"], "LABELS": ["RESULTS"], "LONG_ANSWER": "c2"},
        }
        tasks = build_tasks(data, 2, set(), None, only={("2", "results")})
        self.assertEqual({(t["pmid"], t["part"], t["repeat"]) for t in tasks}, {("2", "results", 0), ("2", "results", 1)})

    def test_candidate_prompts_write_to_their_own_files(self) -> None:
        from scripts.agents.probe_prompts import PROMPTS
        from scripts.agents.rate_conditionality import OUT, default_out

        self.assertEqual(default_out(PROMPTS["conditionality@1"], False), OUT)
        v2 = default_out(PROMPTS["conditionality@2"], False)
        self.assertNotEqual(v2, OUT)
        self.assertTrue(default_out(PROMPTS["conditionality@2"], True).name.endswith(".calibration.jsonl"))


if __name__ == "__main__":
    unittest.main()
