"""Targeted tests for prompt-content ablations that aren't covered by the
budget regression suite: label-blind vs not (ablation d)."""

from __future__ import annotations

import unittest

from app.agents.sld import prompts
from app.agents.sld.segmentation import extract_stats_profile

SENTENCES = {
    "S1": "Patients received drug X and the primary outcome improved significantly (p<0.01).",
}
SECTION_TAGS = {"S1": "RESULTS"}
STATS_PROFILE = extract_stats_profile([("S1", SENTENCES["S1"])])


class LabelBlindAblationTests(unittest.TestCase):
    def test_default_prompts_never_mention_yes_no_maybe_as_the_task(self) -> None:
        builders = [
            prompts.build_question_framer_prompt("Is X valuable?", SENTENCES),
            prompts.build_findings_auditor_prompt(
                "Is X valuable?", SENTENCES, SECTION_TAGS, STATS_PROFILE
            ),
            prompts.build_gap_auditor_prompt("Is X valuable?", SENTENCES, STATS_PROFILE),
            prompts.build_conclusion_reconstructor_prompt("Is X valuable?", SENTENCES),
        ]
        for prompt in builders:
            self.assertIn("do NOT decide or hint at a yes/no/maybe answer", prompt)

    def test_not_label_blind_reveals_the_eventual_task(self) -> None:
        builders = [
            prompts.build_question_framer_prompt("Is X valuable?", SENTENCES, label_blind=False),
            prompts.build_findings_auditor_prompt(
                "Is X valuable?", SENTENCES, SECTION_TAGS, STATS_PROFILE, label_blind=False
            ),
            prompts.build_gap_auditor_prompt(
                "Is X valuable?", SENTENCES, STATS_PROFILE, label_blind=False
            ),
            prompts.build_conclusion_reconstructor_prompt(
                "Is X valuable?", SENTENCES, label_blind=False
            ),
        ]
        for prompt in builders:
            self.assertNotIn("do NOT decide or hint at a yes/no/maybe answer", prompt)
            self.assertIn("answer the research question yes, no, or maybe", prompt)


if __name__ == "__main__":
    unittest.main()
