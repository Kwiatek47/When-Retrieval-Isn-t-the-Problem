"""Tests for the Figure 1 renderer (synthetic audit input only)."""

from __future__ import annotations

import unittest
import xml.dom.minidom

from scripts.agents.plot_label_matrix import (
    BAR_H,
    CELL,
    GAP,
    MARGIN_LEFT,
    render_latex,
    render_svg,
)

LABELS = ("yes", "no", "maybe")

# A 1000-item audit stub: the diagonal agrees, and two off-diagonal cells carry a split.
_COUNTS = {
    ("yes", "yes"): {"yes": 454, "no": 0, "maybe": 0},
    ("yes", "no"): {"yes": 12, "no": 85, "maybe": 1},
    ("yes", "maybe"): {"yes": 21, "no": 2, "maybe": 39},
    ("no", "yes"): {"yes": 42, "no": 10, "maybe": 1},
    ("no", "no"): {"yes": 0, "no": 224, "maybe": 0},
    ("no", "maybe"): {"yes": 0, "no": 8, "maybe": 17},
    ("maybe", "yes"): {"yes": 23, "no": 0, "maybe": 20},
    ("maybe", "no"): {"yes": 0, "no": 9, "maybe": 9},
    ("maybe", "maybe"): {"yes": 0, "no": 0, "maybe": 23},
}


def _audit() -> dict:
    cells = {}
    for (ctx, concl), final in _COUNTS.items():
        cells[f"context_only={ctx}|sees_conclusion={concl}"] = {
            "n": sum(final.values()),
            "annotators_agree": ctx == concl,
            "final_decision": final,
        }
    return {
        "label_matrix": {
            "n_items": sum(sum(f.values()) for f in _COUNTS.values()),
            "cells": cells,
            "marginals": {
                "context_only": {"yes": 614, "no": 302, "maybe": 84},
                "sees_conclusion": {"yes": 550, "no": 340, "maybe": 110},
                "final_decision": {"yes": 552, "no": 338, "maybe": 110},
            },
            "final_label_neither_annotator_proposed": {
                "n": 4,
                "by_final_label": {"maybe": 2, "no": 2},
            },
        }
    }


class RenderSvgTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svg = render_svg(_audit())

    def test_output_is_well_formed_xml(self) -> None:
        doc = xml.dom.minidom.parseString(self.svg)
        self.assertEqual(doc.documentElement.tagName, "svg")

    def test_every_cell_count_is_drawn(self) -> None:
        for final in _COUNTS.values():
            self.assertIn(f">{sum(final.values())}</text>", self.svg)

    def test_nothing_is_drawn_outside_the_canvas(self) -> None:
        doc = xml.dom.minidom.parseString(self.svg)
        root = doc.documentElement
        width = float(root.getAttribute("width"))
        height = float(root.getAttribute("height"))
        for rect in doc.getElementsByTagName("rect"):
            right = float(rect.getAttribute("x") or 0) + float(rect.getAttribute("width"))
            bottom = float(rect.getAttribute("y") or 0) + float(rect.getAttribute("height"))
            self.assertLessEqual(right, width + 0.5)
            self.assertLessEqual(bottom, height + 0.5)

    def test_bar_segments_fill_the_track_exactly(self) -> None:
        """Each cell's stacked bar must sum to the full bar width, or a split is being lost."""
        doc = xml.dom.minidom.parseString(self.svg)
        per_cell: dict[tuple[str, int], float] = {}
        for rect in doc.getElementsByTagName("rect"):
            if rect.getAttribute("height") != f"{BAR_H:.1f}":
                continue
            column = int((float(rect.getAttribute("x")) - MARGIN_LEFT) // (CELL + GAP))
            key = (rect.getAttribute("y"), column)
            per_cell[key] = per_cell.get(key, 0.0) + float(rect.getAttribute("width"))
        self.assertEqual(len(per_cell), 9, "expected one stacked bar per cell")
        for total in per_cell.values():
            self.assertAlmostEqual(total, CELL - 24, places=1)

    def test_a_zero_count_cell_does_not_emit_a_segment(self) -> None:
        audit = _audit()
        key = "context_only=maybe|sees_conclusion=no"
        audit["label_matrix"]["cells"][key] = {
            "n": 0,
            "annotators_agree": False,
            "final_decision": {"yes": 0, "no": 0, "maybe": 0},
        }
        svg = render_svg(audit)
        xml.dom.minidom.parseString(svg)
        self.assertIn("no items", svg)


class RenderLatexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tex = render_latex(_audit())

    def test_has_one_body_row_per_label_pair(self) -> None:
        body = [ln for ln in self.tex.splitlines() if ln.endswith("\\\\") and "&" in ln]
        # 9 cells + 2 header lines + 1 total line
        self.assertEqual(len(body), 12)

    def test_diagonal_rows_are_emphasised(self) -> None:
        self.assertIn("\\textit{yes} & \\textit{yes} & 454", self.tex)
        self.assertIn("{yes} & {no} & 98", self.tex)

    def test_totals_row_uses_the_final_decision_marginal(self) -> None:
        self.assertIn("& 1000 & 552 & 338 & 110 \\\\", self.tex)

    def test_underscore_is_escaped_for_latex(self) -> None:
        self.assertIn("final\\_decision", self.tex)
        self.assertNotIn("texttt{final_decision}", self.tex)


if __name__ == "__main__":
    unittest.main()
