"""Risk-coverage figure for BioLinkBERT on PQA-L 500 (BRAKI B1, D).

Draws the curve stored by ``analyze_b1_selective_prediction.py`` (``risk_by_k``): the error
rate among answered questions as the system answers more of them, least uncertain first.
Two references frame it: a random order, whose risk is the error rate at every coverage,
and an oracle that ranks every error last. The marked point is the out-of-fold operating
point from the cost table.

Two outputs, like Figure 1: a standalone SVG for preview and a pgfplots ``.tex`` that
compiles inside the paper (``\\usepackage{pgfplots}``), so no SVG-to-PDF tool is needed.
Pure standard library. The first 5% of coverage is not drawn: with fewer than 25 answered
questions the risk is noise.

Usage:
  python scripts/agents/plot_risk_coverage.py
  python scripts/agents/plot_risk_coverage.py --signal debate_dissent_u_score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT_ROOT / "reports/debate/analysis"
B1 = ANALYSIS / "b1_selective_prediction.json"
FIGURES = ANALYSIS / "figures"

SIGNAL = "biolinkbert_1_minus_confidence"
LABELS = {
    "biolinkbert_1_minus_confidence": "BioLinkBERT 1 − confidence",
    "self_consistency_1_minus_agreement": "Self-consistency 1 − agreement",
    "debate_dissent_panel_split": "Debate panel split",
    "debate_dissent_u_score": "Debate u-score",
}
MIN_COVERAGE = 0.05

# One data series, so one hue; references and all text stay in neutral ink.
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

WIDTH, HEIGHT = 560.0, 350.0
LEFT, RIGHT, TOP, BOTTOM = 62.0, 24.0, 22.0, 52.0
Y_MAX = 0.32
Y_TICKS = (0.0, 0.1, 0.2, 0.3)
X_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def curve_points(risk_by_k: list[float], min_coverage: float = MIN_COVERAGE) -> list[tuple[float, float]]:
    """(coverage, risk) for every k from ``min_coverage`` to full coverage."""
    n = len(risk_by_k)
    first = max(int(round(min_coverage * n)), 1)
    return [(k / n, risk_by_k[k - 1]) for k in range(first, n + 1)]


def oracle_points(n: int, errors: int, min_coverage: float = MIN_COVERAGE) -> list[tuple[float, float]]:
    """Risk when every error is answered last: zero until the correct answers run out."""
    first = max(int(round(min_coverage * n)), 1)
    return [(k / n, max(0, k - (n - errors)) / k) for k in range(first, n + 1)]


def operating_point(signal: dict) -> tuple[float, float]:
    op = signal["operating_point"]
    return 1.0 - op["abstention_rate"], 1.0 - op["selective_accuracy"]


def _x(coverage: float) -> float:
    return LEFT + coverage * (WIDTH - LEFT - RIGHT)


def _y(risk: float) -> float:
    return TOP + (1.0 - risk / Y_MAX) * (HEIGHT - TOP - BOTTOM)


def _path(points: list[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{_x(c):.1f},{_y(r):.1f}" for c, r in points)


def render_svg(signal: dict, label: str) -> str:
    n, errors, error_rate = signal["n"], signal["errors"], signal["always_answer_cost"]
    curve = curve_points(signal["risk_by_k"])
    oracle = oracle_points(n, errors)
    op_cov, op_risk = operating_point(signal)
    op = signal["operating_point"]
    text = f'font-family="Helvetica,Arial,sans-serif" font-size="11.5"'

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH:.0f}" height="{HEIGHT:.0f}" '
        f'viewBox="0 0 {WIDTH:.0f} {HEIGHT:.0f}" role="img" '
        f'aria-label="Risk-coverage curve, {label}, PQA-L 500">',
        f'<rect width="{WIDTH:.0f}" height="{HEIGHT:.0f}" fill="{SURFACE}"/>',
    ]
    for tick in Y_TICKS:
        y = _y(tick)
        if tick:
            out.append(f'<line x1="{LEFT:.1f}" y1="{y:.1f}" x2="{WIDTH - RIGHT:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{LEFT - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" fill="{INK_SECONDARY}" {text}>{tick:.1f}</text>')
    for tick in X_TICKS:
        x = _x(tick)
        out.append(f'<line x1="{x:.1f}" y1="{_y(0):.1f}" x2="{x:.1f}" y2="{_y(0) + 4:.1f}" stroke="{INK_SECONDARY}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{_y(0) + 18:.1f}" text-anchor="middle" fill="{INK_SECONDARY}" {text}>{tick:.1f}</text>')
    out.append(f'<line x1="{LEFT:.1f}" y1="{_y(0):.1f}" x2="{WIDTH - RIGHT:.1f}" y2="{_y(0):.1f}" stroke="{INK_SECONDARY}" stroke-width="1"/>')
    out.append(
        f'<text x="{(LEFT + WIDTH - RIGHT) / 2:.1f}" y="{HEIGHT - 12:.1f}" text-anchor="middle" fill="{INK}" {text}>'
        "coverage (share of questions answered)</text>"
    )
    out.append(
        f'<text transform="translate(16,{(TOP + HEIGHT - BOTTOM) / 2:.1f}) rotate(-90)" text-anchor="middle" fill="{INK}" {text}>'
        "risk (error rate among answered)</text>"
    )

    # References: random order (flat at the error rate) and oracle.
    out.append(
        f'<line x1="{_x(MIN_COVERAGE):.1f}" y1="{_y(error_rate):.1f}" x2="{_x(1):.1f}" y2="{_y(error_rate):.1f}" '
        f'stroke="{INK_SECONDARY}" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    out.append(
        f'<text x="{_x(0.07):.1f}" y="{_y(error_rate) - 7:.1f}" fill="{INK_SECONDARY}" {text}>'
        f"random order: {error_rate:.3f} at every coverage</text>"
    )
    out.append(f'<path d="{_path(oracle)}" fill="none" stroke="{INK_SECONDARY}" stroke-width="1.5" stroke-dasharray="2 3"/>')
    out.append(
        f'<text x="{_x((n - errors) / n) - 8:.1f}" y="{_y(0.0) - 9:.1f}" text-anchor="end" fill="{INK_SECONDARY}" {text}>'
        f"oracle: every error answered last (AURC {signal['aurc_oracle']:.3f})</text>"
    )

    # The signal.
    out.append(f'<path d="{_path(curve)}" fill="none" stroke="{SERIES}" stroke-width="2" stroke-linejoin="round"/>')
    out.append(
        f'<text x="{_x(0.16):.1f}" y="{_y(0.232):.1f}" fill="{INK}" {text} font-weight="600">'
        f"{label} (AURC {signal['aurc']['value']:.3f})</text>"
    )

    # Operating point: ring in the surface colour so the marker stays legible on the line.
    out.append(f'<circle cx="{_x(op_cov):.1f}" cy="{_y(op_risk):.1f}" r="5" fill="{SERIES}" stroke="{SURFACE}" stroke-width="2"/>')
    for row, line in enumerate(("threshold chosen out of fold:", f"{op['abstention_rate']:.1%} abstained, risk {op_risk:.3f}")):
        out.append(
            f'<text x="{_x(op_cov):.1f}" y="{_y(op_risk) + 26 + 15 * row:.1f}" text-anchor="end" fill="{INK}" {text}>'
            f"{line}</text>"
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_pgfplots(signal: dict, label: str, step: int = 5) -> str:
    """The same figure as a pgfplots picture; every ``step``-th point keeps the file small."""
    n, errors, error_rate = signal["n"], signal["errors"], signal["always_answer_cost"]
    curve = curve_points(signal["risk_by_k"])
    thinned = curve[::step] + ([curve[-1]] if (len(curve) - 1) % step else [])
    oracle = [(c, r) for c, r in oracle_points(n, errors) if r > 0]
    oracle_coordinates = f"({MIN_COVERAGE},0) ({(n - errors) / n:.3f},0) " + " ".join(
        f"({c:.3f},{r:.4f})" for c, r in oracle[::step] + [oracle[-1]]
    )
    op_cov, op_risk = operating_point(signal)
    op = signal["operating_point"]
    coordinates = " ".join(f"({c:.3f},{r:.4f})" for c, r in thinned)
    tex_label = label.replace("−", "$-$")
    return rf"""% Generated by scripts/agents/plot_risk_coverage.py from b1_selective_prediction.json. Do not edit by hand.
% Needs \usepackage{{pgfplots}} and \pgfplotsset{{compat=1.17}} in the preamble.
\definecolor{{rcseries}}{{HTML}}{{{SERIES.lstrip('#').upper()}}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=\columnwidth, height=0.68\columnwidth,
  xmin=0, xmax=1, ymin=0, ymax={Y_MAX},
  xtick={{0,0.2,0.4,0.6,0.8,1}}, ytick={{0,0.1,0.2,0.3}},
  xlabel={{coverage}}, ylabel={{risk}},
  ymajorgrids, grid style={{gray!20}},
  axis lines=left, tick align=outside,
  label style={{font=\footnotesize}}, tick label style={{font=\scriptsize}},
  clip=false,
]
\addplot[gray!70!black, dashed, thick, domain={MIN_COVERAGE}:1, samples=2] {{{error_rate:.3f}}};
\addplot[gray!70!black, densely dotted, thick] coordinates {{{oracle_coordinates}}};
\addplot[rcseries, very thick, line join=round] coordinates {{{coordinates}}};
\addplot[only marks, mark=*, mark size=2pt, rcseries, mark options={{draw=white, line width=0.6pt}}] coordinates {{({op_cov:.3f},{op_risk:.4f})}};
\node[font=\scriptsize, anchor=south west, gray!70!black] at (axis cs:0.06,{error_rate:.3f}) {{random order ({error_rate:.3f})}};
\node[font=\scriptsize, anchor=south east, gray!70!black] at (axis cs:{(n - errors) / n - 0.01:.3f},0.004) {{oracle}};
\node[font=\scriptsize, anchor=south west] at (axis cs:0.16,0.215) {{{tex_label}}};
\node[font=\scriptsize, anchor=north east] at (axis cs:{op_cov - 0.02:.3f},{op_risk - 0.012:.3f}) {{{op['abstention_rate'] * 100:.1f}\% abstained}};
\end{{axis}}
\end{{tikzpicture}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1", type=Path, default=B1)
    parser.add_argument("--signal", choices=sorted(LABELS), default=SIGNAL)
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    signal = json.loads(args.b1.read_text(encoding="utf-8"))["signals"][args.signal]
    if "risk_by_k" not in signal:
        raise SystemExit(f"{args.b1} has no risk_by_k; rerun scripts/agents/analyze_b1_selective_prediction.py")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_risk_coverage" if args.signal == SIGNAL else f"fig_risk_coverage_{args.signal}"
    svg_path, tex_path = args.out_dir / f"{stem}.svg", args.out_dir / f"{stem}.tex"
    svg_path.write_text(render_svg(signal, LABELS[args.signal]), encoding="utf-8")
    tex_path.write_text(render_pgfplots(signal, LABELS[args.signal]), encoding="utf-8")
    print(f"wrote {svg_path}\nwrote {tex_path}")


if __name__ == "__main__":
    main()
