#!/usr/bin/env python3
"""Insert the compute-matched SC table row from a finished SC JSON.

Reads numbers only from the report. Refuses if the file is missing or has no
accuracy. Does not invent values. Does not claim debate > SC.

Usage:
  .venv/bin/python scripts/agents/insert_sc_row.py
  .venv/bin/python scripts/agents/insert_sc_row.py --report reports/debate/sc_balanced90_ml4h_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "debate" / "sc_balanced90_ml4h_v1.json"
TEX_TARGETS = (
    PROJECT_ROOT / "paper" / "sections" / "05_results.tex",
    PROJECT_ROOT / "paper" / "overleaf" / "sections" / "05_results.tex",
)
PENDING_ROW = r"SC \(matched compute\) & pending & --- & --- \\\\"


def main() -> None:
    args = _parse_args()
    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(
            f"No SC report at {report_path}. Run "
            "`scripts/agents/run_ml4h_v1_arms.sh sc` first. Refusing to invent a number."
        )
    summary = (json.loads(report_path.read_text(encoding="utf-8")).get("summary") or {})
    acc = summary.get("label_accuracy")
    n = summary.get("cases")
    samples = summary.get("samples")
    if acc is None or n is None:
        raise SystemExit(
            f"{report_path} has no summary.label_accuracy / cases. Refusing to invent a number."
        )
    acc_f = float(acc)
    n_i = int(n)
    samples_s = f"N={int(samples)}" if samples is not None else "matched"
    tex_name = report_path.name.replace("_", r"\_")
    # Four backslashes so the written row ends with LaTeX \\ (re.sub eats one pair).
    row = f"SC ({samples_s}) & ${acc_f:.3f}$ & --- & ${n_i}$ \\\\\\\\"
    caption_old = (
        "The compute-matched SC cell is left\n"
        "empty: $N$ and accuracy are not measured yet."
    )
    caption_new = (
        f"Compute-matched SC is ${acc_f:.3f}$ "
        f"({samples_s}, $n{{=}}{n_i}$) from \\texttt{{{tex_name}}}; "
        "we do not claim debate beats SC."
    )
    replaced_any = False
    for tex in TEX_TARGETS:
        if not tex.exists():
            print(f"skip missing {tex}")
            continue
        text = tex.read_text(encoding="utf-8")
        new_text, n_row = re.subn(PENDING_ROW, row, text, count=1)
        if n_row == 0:
            if row in text:
                print(f"already inserted in {tex}")
            else:
                print(f"WARNING: pending SC row not found in {tex}", file=sys.stderr)
            continue
        if caption_old in new_text:
            new_text = new_text.replace(caption_old, caption_new, 1)
        tex.write_text(new_text, encoding="utf-8")
        print(f"Wrote SC row {acc_f:.3f} (n={n_i}, {samples_s}) into {tex}")
        replaced_any = True
    if not replaced_any:
        raise SystemExit("No tex file was updated.")
    print("Recompile: cd paper && make")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
