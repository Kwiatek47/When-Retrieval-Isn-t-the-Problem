"""Tests for prompt snapshot / versioning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class PromptVersioningTests(unittest.TestCase):
    def test_hash_stable_and_snapshot_written(self) -> None:
        from app.agents.prompt_versioning import (
            collect_prompt_bundle,
            hash_prompt_bundle,
            snapshot_prompts_for_run,
        )

        bundle_a = collect_prompt_bundle()
        bundle_b = collect_prompt_bundle()
        self.assertEqual(hash_prompt_bundle(bundle_a), hash_prompt_bundle(bundle_b))
        self.assertIn("SUPERVISOR_DIRECTOR_PROMPT", bundle_a)
        self.assertIn("UNCERTAINTY_ADVOCATE_PROMPT", bundle_a)

        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            record = snapshot_prompts_for_run(
                report_dir,
                run_label="unit_test_run",
                script="tests",
                extra_meta={"k": "v"},
            )
            snap = report_dir / "unit_test_run.prompts.json"
            py_copy = report_dir / "unit_test_run.prompts.py"
            registry = report_dir / "prompt_versions.jsonl"
            self.assertTrue(snap.is_file())
            self.assertTrue(py_copy.is_file())
            self.assertTrue(registry.is_file())
            data = json.loads(snap.read_text(encoding="utf-8"))
            self.assertEqual(data["prompt_version"], record["prompt_version"])
            self.assertEqual(len(data["prompt_sha256"]), 64)
            self.assertEqual(data["meta"]["k"], "v")
            lines = registry.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            # Second run appends registry.
            snapshot_prompts_for_run(report_dir, run_label="unit_test_run2", script="tests")
            lines = registry.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
