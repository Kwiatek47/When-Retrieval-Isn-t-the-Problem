from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

CODEBOOK = Path(__file__).resolve().parents[1] / "docs/research/2026-10-04-rq5-codebook-taksonomia-maybe.md"


class CodingToolTests(unittest.TestCase):
    def test_categories_and_rules_match_the_registered_codebook(self) -> None:
        from scripts.agents.rq5_code import CATEGORIES, RULES

        text = CODEBOOK.read_text(encoding="utf-8")
        for code, name, when in CATEGORIES:
            self.assertIn(f"| {code} | {name} | {when} |", text)
        for rule in RULES:
            self.assertIn(rule, text)

    def test_parse_answer(self) -> None:
        from scripts.agents.rq5_code import parse_answer

        self.assertEqual(parse_answer("c"), ("code", "C"))
        self.assertEqual(parse_answer("B  mała próba "), ("code", "B|mała próba"))
        self.assertEqual(parse_answer(""), ("keep", ""))
        self.assertEqual(parse_answer("g 12"), ("goto", "12"))
        self.assertEqual(parse_answer("q"), ("quit", ""))
        self.assertEqual(parse_answer("G"), ("invalid", "G"))
        self.assertEqual(parse_answer("x"), ("invalid", "x"))

    def test_codes_survive_a_save_and_resume(self) -> None:
        from scripts.agents.rq5_code import first_uncoded, load_codes, save_codes

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "codes.csv"
            self.assertEqual(load_codes(path), {})
            save_codes(path, {3: {"item": "3", "code": "D", "note": "obserwacyjne, przyczynowe", "coded_at": "t"}})
            codes = load_codes(path)
            self.assertEqual(codes[3]["code"], "D")
            self.assertEqual(codes[3]["note"], "obserwacyjne, przyczynowe")
            self.assertEqual(first_uncoded([1, 2, 3], codes), 0)
            self.assertEqual(first_uncoded([1, 2, 3], codes, start=2), 0)  # wraps past the coded one
            self.assertIsNone(first_uncoded([3], codes))

    def test_score_reads_code_files(self) -> None:
        from scripts.agents.rq5_maybe_taxonomy import read_code_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            path.write_text('item,code,note\n1,c,x\n2,F,"y, z"\n', encoding="utf-8")
            self.assertEqual(read_code_file(path), {1: "C", 2: "F"})


if __name__ == "__main__":
    unittest.main()
