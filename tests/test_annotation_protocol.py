from __future__ import annotations

import asyncio
import importlib.util
import json
import unittest

import numpy as np

HAS_APP_DEPS = importlib.util.find_spec("pydantic") is not None


class ScriptedBackend:
    """Answers by annotator: each call pops the next label from that annotator's script."""

    def __init__(self, scripts: dict[str, list[str]]) -> None:
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls: list[str] = []

    async def complete(self, messages, *, temperature=None, num_predict=None, response_format=None) -> str:
        text = messages[-1].content
        who = "with_conclusion" if "which you can see" in text or "Authors' conclusion" in text else "context_only"
        self.calls.append(text)
        label = self.scripts[who].pop(0)
        if label == "broken":
            return "not json"
        return json.dumps({"rationale": f"{who} says {label}", "label": label, "confidence": 80})


def _evidence() -> dict[str, str]:
    return {"with_conclusion": "Abstract.\n\nAuthors' conclusion:\nIt may help.", "context_only": "Abstract."}


@unittest.skipUnless(HAS_APP_DEPS, "needs the app dependencies (run with llm_env)")
class ProtocolTests(unittest.TestCase):
    def _run(self, scripts: dict[str, list[str]], max_rounds: int = 3):
        from app.agents.annotation_protocol import AnnotationProtocol
        from scripts.agents.probe_prompts import PROMPTS

        backend = ScriptedBackend(scripts)
        protocol = AnnotationProtocol(
            backend, label_spec=PROMPTS["label-defined@1"], negotiate_spec=PROMPTS["negotiate@1"], max_rounds=max_rounds
        )
        return asyncio.run(protocol.run("1", "Does X help?", _evidence())), backend

    def test_agreement_ends_at_round_zero(self) -> None:
        result, backend = self._run({"with_conclusion": ["maybe"], "context_only": ["maybe"]})
        self.assertEqual((result.status, result.final, result.rounds_used), ("agreed", "maybe", 0))
        self.assertEqual(result.final_follows, "both")
        self.assertEqual(len(backend.calls), 2)

    def test_dispute_settled_on_the_conclusion_side(self) -> None:
        result, _ = self._run({"with_conclusion": ["maybe", "maybe"], "context_only": ["yes", "maybe"]})
        self.assertEqual((result.status, result.final, result.rounds_used), ("negotiated", "maybe", 1))
        self.assertEqual(result.final_follows, "with_conclusion")
        self.assertEqual(result.initial, {"with_conclusion": "maybe", "context_only": "yes"})

    def test_unsettled_dispute_is_removed(self) -> None:
        result, _ = self._run({"with_conclusion": ["yes", "yes", "yes"], "context_only": ["no", "no", "no"]}, max_rounds=2)
        self.assertEqual((result.status, result.final, result.rounds_used), ("removed", None, 2))
        self.assertEqual(result.final_follows, "none")

    def test_negotiation_prompt_states_what_the_colleague_could_see(self) -> None:
        _, backend = self._run({"with_conclusion": ["yes", "yes"], "context_only": ["no", "yes"]})
        negotiation = [c for c in backend.calls if "Discussion round 1" in c]
        self.assertEqual(len(negotiation), 2)
        self.assertTrue(any("which you have not seen" in c for c in negotiation))
        self.assertTrue(all("Your colleague's current label" in c for c in negotiation))

    def test_unparseable_reply_marks_the_question_failed(self) -> None:
        result, _ = self._run({"with_conclusion": ["broken", "broken"], "context_only": ["yes"]})
        self.assertEqual(result.status, "failed")


class AnalysisTests(unittest.TestCase):
    def _data(self) -> dict:
        # (rf, rr, final): 2 unanimous maybe, 1 negotiated maybe, a yes/no dispute won by rf
        rows = [("maybe", "maybe", "maybe"), ("maybe", "maybe", "maybe"), ("maybe", "yes", "maybe"),
                ("yes", "no", "yes"), ("no", "yes", "yes"), ("yes", "yes", "yes")]
        return {str(i): {"reasoning_free_pred": a, "reasoning_required_pred": b, "final_decision": c}
                for i, (a, b, c) in enumerate(rows)}

    def test_human_reference_from_the_raw_labels(self) -> None:
        from scripts.agents.analyze_annotation_protocol import human_reference

        h = human_reference(self._data())
        self.assertAlmostEqual(h["maybe_through_negotiation"], 1 / 3)
        # disputes: (maybe,yes)->rf, (yes,no)->rf, (no,yes)->rr
        self.assertAlmostEqual(h["conclusion_wins"], 2 / 3)
        self.assertEqual(h["yes_no_dispute_to_maybe"], [0, 2])

    def test_replay_statistics(self) -> None:
        from scripts.agents.analyze_annotation_protocol import conclusion_wins, maybe_through_negotiation, protocol_gain

        status = np.array(["agreed", "negotiated", "negotiated", "removed"])
        final = np.array(["maybe", "maybe", "yes", "removed"])
        self.assertAlmostEqual(maybe_through_negotiation(status, final), 0.5)
        follows = np.array(["both", "with_conclusion", "context_only", "with_conclusion", "none"])
        self.assertAlmostEqual(conclusion_wins(follows), 2 / 3)
        gold = np.array(["maybe", "maybe", "yes", "no"])
        self.assertAlmostEqual(protocol_gain(final, np.array(["maybe", "yes", "yes", "no"]), gold), 1.0 - 2 / 3)

    def test_verdicts(self) -> None:
        from scripts.agents.analyze_annotation_protocol import gain_verdict, reproduction_verdict

        self.assertEqual(reproduction_verdict({"ci_low": 0.70, "ci_high": 0.85}, 0.79), "reproduced")
        self.assertEqual(reproduction_verdict({"ci_low": 0.20, "ci_high": 0.40}, 0.79), "not reproduced")
        self.assertEqual(reproduction_verdict({"ci_low": 0.55, "ci_high": 0.85}, 0.79), "inconclusive")
        self.assertEqual(gain_verdict({"ci_low": 0.01, "ci_high": 0.2}), "supported")
        self.assertEqual(gain_verdict({"ci_low": -0.2, "ci_high": 0.0}), "refuted")

    def test_load_run_skips_pilot_and_keeps_a_later_success(self) -> None:
        import tempfile
        from pathlib import Path

        from scripts.agents.analyze_annotation_protocol import load_run

        rows = [{"pmid": "1", "status": "failed"}, {"pmid": "1", "status": "agreed"}, {"pmid": "9", "status": "agreed"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            loaded = load_run(path, pilot=("9",))
        self.assertEqual([(r["pmid"], r["status"]) for r in loaded], [("1", "agreed")])


if __name__ == "__main__":
    unittest.main()
