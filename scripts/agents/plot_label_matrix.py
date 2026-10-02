"""Figure 1: how PQA-L's gold label is produced (BRAKI A1, D).

Renders the context-only x sees-conclusion x final_decision cross-tab computed by
`audit_pqal_labels.py` into a standalone SVG, and optionally into a LaTeX table.

Pure standard library on purpose: this repo's analysis scripts avoid numpy, and neither
matplotlib nor numpy is installed in `.venv`. SVG is written directly; convert to PDF for
the paper with whatever the submission box has (`rsvg-convert`, `inkscape`, `cairosvg`).

Reading the figure: rows are the annotator who saw only question + abstract (the model's
information set), columns the annotator who also saw the authors' conclusion. On the
diagonal the two agreed and `final_decision` follows by construction. Off the diagonal the
label comes out of a discussion between those same two people, and items they could not
settle were dropped from the dataset — so the off-diagonal is where the label is made.

Usage:
  python scripts/agents/plot_label_matrix.py
  python scripts/agents/plot_label_matrix.py --format latex
  python scripts/agents/plot_label_matrix.py --audit path/to/pqal_protocol_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.pqal_official import LABELS  # noqa: E402

ANALYSIS = PROJECT_ROOT / "reports/debate/analysis"
AUDIT = ANALYSIS / "pqal_protocol_audit.json"
FIGURES = PROJECT_ROOT / "reports/debate/analysis/figures"

# `maybe` is the subject of the paper, so it gets the one saturated colour; yes/no stay
# cool and neutral. Distinguishable in greyscale by lightness order (blue < grey < red).
FINAL_COLOURS = {"yes": "#4C72B0", "no": "#9A9A9A", "maybe": "#C44E52"}

CELL = 148.0
GAP = 8.0
MARGIN_LEFT = 150.0
MARGIN_TOP = 140.0
BAR_H = 20.0
MIN_WIDTH = 660.0


def _cell(audit: dict, context_only: str, sees_conclusion: str) -> dict:
    key = f"context_only={context_only}|sees_conclusion={sees_conclusion}"
    try:
        return audit["label_matrix"]["cells"][key]
    except KeyError as exc:  # pragma: no cover - guards a stale audit file
        raise SystemExit(
            f"{key} missing from the audit file — regenerate it with "
            "scripts/agents/audit_pqal_labels.py"
        ) from exc


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(audit: dict) -> str:
    matrix = audit["label_matrix"]
    n_items = matrix["n_items"]
    grid_w = 3 * CELL + 2 * GAP
    width = max(MARGIN_LEFT + grid_w + 24, MIN_WIDTH)
    height = MARGIN_TOP + grid_w + 118

    subtitle = [
        "Bars show the <tspan font-style=\"italic\">final_decision</tspan> split inside each "
        "cell. On the diagonal the annotators agreed,",
        "so the label follows by construction. Off-diagonal cells are settled by discussion "
        "between the",
        "same two annotators; items they could not settle were removed from the dataset.",
    ]
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
        f'<text x="24" y="30" font-size="15" font-weight="600" fill="#111111">'
        f"How PQA-L&#8217;s gold label is produced ({n_items} items)</text>",
    ]
    for k, line in enumerate(subtitle):
        out.append(
            f'<text x="24" y="{52 + k * 16:.0f}" font-size="11.5" fill="#444444">{line}</text>'
        )
    out.append(
        # Column group header
        f'<text x="{MARGIN_LEFT + grid_w / 2:.1f}" y="{MARGIN_TOP - 36:.1f}" font-size="12" '
        f'font-weight="600" text-anchor="middle" fill="#111111">'
        f"annotator 1 &#8212; saw question + abstract + conclusion</text>"
    )

    for j, concl in enumerate(LABELS):
        cx = MARGIN_LEFT + j * (CELL + GAP) + CELL / 2
        total = matrix["marginals"]["sees_conclusion"].get(concl, 0)
        out.append(
            f'<text x="{cx:.1f}" y="{MARGIN_TOP - 14:.1f}" font-size="12.5" '
            f'text-anchor="middle" fill="#111111">{concl} '
            f'<tspan fill="#777777">(n={total})</tspan></text>'
        )

    # Row group header, rotated
    ry = MARGIN_TOP + grid_w / 2
    out.append(
        f'<text transform="translate(26,{ry:.1f}) rotate(-90)" font-size="12" '
        f'font-weight="600" text-anchor="middle" fill="#111111">'
        f"annotator 2 &#8212; abstract only (model&#8217;s information)</text>"
    )

    for i, ctx in enumerate(LABELS):
        cy = MARGIN_TOP + i * (CELL + GAP)
        total = matrix["marginals"]["context_only"].get(ctx, 0)
        out.append(
            f'<text x="{MARGIN_LEFT - 14:.1f}" y="{cy + CELL / 2:.1f}" font-size="12.5" '
            f'text-anchor="end" fill="#111111">{ctx} '
            f'<tspan fill="#777777">(n={total})</tspan></text>'
        )

        for j, concl in enumerate(LABELS):
            cell = _cell(audit, ctx, concl)
            x = MARGIN_LEFT + j * (CELL + GAP)
            agree = ctx == concl
            fill = "#F2F5F9" if agree else "#FBFBFB"
            stroke = "#B8C6D8" if agree else "#E2E2E2"
            out.append(
                f'<rect x="{x:.1f}" y="{cy:.1f}" width="{CELL:.1f}" height="{CELL:.1f}" '
                f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            )
            n = cell["n"]
            out.append(
                f'<text x="{x + 12:.1f}" y="{cy + 34:.1f}" font-size="26" font-weight="600" '
                f'fill="#111111">{n}</text>'
            )
            share = n / n_items if n_items else 0.0
            out.append(
                f'<text x="{x + 12:.1f}" y="{cy + 52:.1f}" font-size="10.5" fill="#777777">'
                f"{share:.1%} of items</text>"
            )
            if agree:
                out.append(
                    f'<text x="{x + CELL - 12:.1f}" y="{cy + 20:.1f}" font-size="9.5" '
                    f'text-anchor="end" fill="#8A9AAD">agree</text>'
                )

            # Stacked bar of the final_decision split.
            bar_x, bar_w = x + 12, CELL - 24
            bar_y = cy + CELL - BAR_H - 30
            if n:
                cursor = bar_x
                for lab in LABELS:
                    count = cell["final_decision"].get(lab, 0)
                    if not count:
                        continue
                    seg = bar_w * count / n
                    out.append(
                        f'<rect x="{cursor:.2f}" y="{bar_y:.1f}" width="{seg:.2f}" '
                        f'height="{BAR_H:.1f}" fill="{FINAL_COLOURS[lab]}"/>'
                    )
                    if seg >= 26:
                        out.append(
                            f'<text x="{cursor + seg / 2:.2f}" y="{bar_y + 14:.1f}" '
                            f'font-size="10.5" text-anchor="middle" fill="#ffffff" '
                            f'font-weight="600">{count}</text>'
                        )
                    cursor += seg
                parts = ", ".join(
                    f"{cell['final_decision'].get(lab, 0)} {lab}"
                    for lab in LABELS
                    if cell["final_decision"].get(lab, 0)
                )
                out.append(
                    f'<text x="{bar_x:.1f}" y="{bar_y + BAR_H + 15:.1f}" font-size="10" '
                    f'fill="#555555">{_esc(parts)}</text>'
                )
            else:
                out.append(
                    f'<text x="{bar_x:.1f}" y="{bar_y + 14:.1f}" font-size="10.5" '
                    f'fill="#AAAAAA">no items</text>'
                )

    # Legend
    ly = MARGIN_TOP + grid_w + 34
    out.append(
        f'<text x="24" y="{ly:.1f}" font-size="11" font-weight="600" fill="#111111">'
        f"final_decision:</text>"
    )
    lx = 124.0
    for lab in LABELS:
        out.append(
            f'<rect x="{lx:.1f}" y="{ly - 10:.1f}" width="13" height="13" rx="2" '
            f'fill="{FINAL_COLOURS[lab]}"/>'
        )
        out.append(
            f'<text x="{lx + 19:.1f}" y="{ly:.1f}" font-size="11" fill="#333333">{lab}</text>'
        )
        lx += 74.0

    invented = matrix["final_label_neither_annotator_proposed"]
    out.append(
        f'<text x="24" y="{ly + 24:.1f}" font-size="10.5" fill="#555555">'
        f"In {invented['n']} items the agreed label is one neither annotator proposed "
        f"({_esc(', '.join(f'{v} {k}' for k, v in sorted(invented['by_final_label'].items())))})."
        f"</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def render_latex(audit: dict) -> str:
    matrix = audit["label_matrix"]
    n_items = matrix["n_items"]
    lines = [
        "% Generated by scripts/agents/plot_label_matrix.py --format latex",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "annotator 2 & annotator 1 & $n$ & \\multicolumn{3}{c}{\\texttt{final\\_decision}} \\\\",
        "(abstract only) & (+ conclusion) & & yes & no & maybe \\\\",
        "\\midrule",
    ]
    for ctx in LABELS:
        for concl in LABELS:
            cell = _cell(audit, ctx, concl)
            fd = cell["final_decision"]
            mark = "\\textit{" if ctx == concl else "{"
            lines.append(
                f"{mark}{ctx}}} & {mark}{concl}}} & {cell['n']} & "
                f"{fd.get('yes', 0)} & {fd.get('no', 0)} & {fd.get('maybe', 0)} \\\\"
            )
        if ctx != LABELS[-1]:
            lines.append("\\addlinespace")
    lines += [
        "\\midrule",
        "\\multicolumn{2}{l}{total} & "
        f"{n_items} & "
        + " & ".join(str(matrix["marginals"]["final_decision"].get(lab, 0)) for lab in LABELS)
        + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--format", choices=["svg", "latex"], default="svg")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.audit.exists():
        raise SystemExit(
            f"{args.audit} is missing — run scripts/agents/audit_pqal_labels.py first."
        )
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if "label_matrix" not in audit:
        raise SystemExit(
            f"{args.audit} predates the label_matrix section — regenerate it with "
            "scripts/agents/audit_pqal_labels.py."
        )

    if args.format == "svg":
        content, suffix = render_svg(audit), "svg"
    else:
        content, suffix = render_latex(audit), "tex"

    out = args.out or FIGURES / f"fig1_label_matrix.{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
