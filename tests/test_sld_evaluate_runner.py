"""Tests for evaluate_sld_pubmedqa.py's own helpers (not the pipeline itself).

Covers the frozen-TriggerConfig loader: forgetting to wire a tuned config
into a later run would silently fall back to the all-enabled default and
waste the whole point of tune_sld_trigger_config.py.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.agents.sld.decision import TriggerConfig
from scripts.agents.evaluate_sld_pubmedqa import _load_trigger_config


class LoadTriggerConfigTests(unittest.TestCase):
    def test_no_path_returns_all_enabled_default(self) -> None:
        config = _load_trigger_config(None)
        self.assertEqual(config, TriggerConfig())

    def test_loads_a_frozen_config_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_trigger_config.json"
            path.write_text(
                json.dumps(
                    {
                        "trigger_config": {
                            "coverage_gap_enabled": True,
                            "mixed_findings_enabled": False,
                            "null_result_enabled": False,
                            "hedged_conclusion_enabled": False,
                        },
                        "config_hash": "deadbeef",
                    }
                ),
                encoding="utf-8",
            )
            config = _load_trigger_config(path)
            self.assertTrue(config.coverage_gap_enabled)
            self.assertFalse(config.mixed_findings_enabled)
            self.assertFalse(config.null_result_enabled)
            self.assertFalse(config.hedged_conclusion_enabled)


if __name__ == "__main__":
    unittest.main()
