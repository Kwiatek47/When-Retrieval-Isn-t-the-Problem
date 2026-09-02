#!/usr/bin/env python3
"""Generate ML4H v1 figures from released analysis JSON (no new experiments)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "reports" / "debate" / "analysis"
OUT = Path(__file__).resolve().parent
STATS = json.loads((ANALYSIS / "statistics.json").read_text())
RISK = json.loads((ANALYSIS / "risk_coverage.json").read_text())

# Display names and source keys — values come only from statistics.json.
FOREST_ROWS = [
    ("$u$-score (combined)", "debate_balanced90_ollama_r2_uncertainty", "uncertainty_score"),
    ("panel maybe-conf.", "debate_balanced90_ollama_r2_uncertainty", "panel_uncertainty_conf"),
    ("maybe fraction", "debate_balanced90_ollama_r2_uncertainty", "maybe_fraction"),
    ("inconclusive frac.", "debate_balanced90_ollama_r2_uncertainty", "inconclusive_fraction"),
    ("label entropy", "debate_balanced90_ollama_r2_uncertainty", "label_entropy"),
    ("mean disagreement", "debate_balanced90_ollama_r2_uncertainty", "mean_disagreement_with_mode"),
    ("flip rate", "debate_balanced90_ollama_r2_uncertainty", "flip_rate"),
    ("qwen2.5:7b audit", "audit_qwen7b_balanced90", "audit_score"),
    ("qwen2.5:14b audit", "audit_qwen14b_balanced90", "audit_score"),
    ("deepseek-r1:14b", "audit_r1_14b_balanced90", "audit_score"),
    ("gpt-5 abstract", "audit_gpt5_balanced90", "audit_score"),
    ("DeBERTa-NLI (C1)", "nli_abstract_balanced90", "audit_score"),
    ("NLI oracle (C2)", "nli_oracle_balanced90", "audit_score"),
    ("gpt-5 oracle (C2)", "audit_gpt5_oracle_balanced90", "audit_score"),
    ("qwen2.5:14b oracle", "audit_qwen14b_oracle_balanced90", "audit_score"),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def forest_auroc() -> None:
    ci = STATS["signal_auroc_ci"]
    labels, means, lows, highs, chance = [], [], [], [], []
    for name, method, key in FOREST_ROWS:
        row = ci[method][key]
        labels.append(name)
        means.append(row["auroc"])
        lows.append(row["ci_low"])
        highs.append(row["ci_high"])
        chance.append(row["brackets_chance"])

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for i, (m, lo, hi, ch) in enumerate(zip(means, lows, highs, chance)):
        color = "#7a7a7a" if ch else "#1f4e79"
        ax.plot([lo, hi], [i, i], color=color, lw=1.4, solid_capstyle="round")
        ax.plot(m, i, "o", color=color, ms=4.2, zorder=3)
    ax.axvline(0.5, color="#c0392b", ls="--", lw=0.9, label="chance")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("maybe-vs-rest AUROC (balanced90)")
    ax.set_xlim(0.30, 0.80)
    ax.set_title("Uncertainty signals vs. chance (95% bootstrap CI)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(OUT / "auroc_forest.pdf")
    fig.savefig(OUT / "auroc_forest.png")
    plt.close(fig)


def risk_coverage() -> None:
    block = RISK["debate_balanced90_ollama_r2_uncertainty"]
    pts = block["points"]
    cov = [p["coverage"] for p in pts]
    risk = [p["risk"] for p in pts]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(cov, risk, color="#1f4e79", lw=1.6, marker="o", ms=3.2)
    ax.axhline(1 - 0.640, color="#7a7a7a", ls=":", lw=0.9, label="full-coverage risk")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.45)
    ax.set_title(f"Risk–coverage (AURC={block['aurc']:.3f})")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(OUT / "risk_coverage.pdf")
    fig.savefig(OUT / "risk_coverage.png")
    plt.close(fig)


def stage_bars() -> None:
    pq = STATS["pqal500"]
    labels = ["hit@1", "citation", "decision acc.", "maybe recall"]
    vals = [
        pq["retrieval_hit_at_1"]["mean"],
        pq["citation_pass"]["mean"],
        pq["decision_overall_accuracy"]["mean"],
        pq["decision_per_class_recall"]["maybe"]["mean"],
    ]
    lows = [
        pq["retrieval_hit_at_1"]["ci_low"],
        pq["citation_pass"]["ci_low"],
        pq["decision_overall_accuracy"]["ci_low"],
        pq["decision_per_class_recall"]["maybe"]["ci_low"],
    ]
    highs = [
        pq["retrieval_hit_at_1"]["ci_high"],
        pq["citation_pass"]["ci_high"],
        pq["decision_overall_accuracy"]["ci_high"],
        pq["decision_per_class_recall"]["maybe"]["ci_high"],
    ]
    yerr = np.vstack([np.array(vals) - np.array(lows), np.array(highs) - np.array(vals)])
    colors = ["#2d6a4f", "#2d6a4f", "#1f4e79", "#c0392b"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(labels, vals, color=colors, yerr=yerr, capsize=3, width=0.62, error_kw={"lw": 0.9})
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("PQA-L 500")
    ax.set_title("Retrieval $\\neq$ decision")
    for i, v in enumerate(vals):
        ax.text(i, min(v + 0.06, 1.02), f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.savefig(OUT / "stage_separated.pdf")
    fig.savefig(OUT / "stage_separated.png")
    plt.close(fig)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    forest_auroc()
    risk_coverage()
    stage_bars()
    print(f"Wrote figures in {OUT}")


if __name__ == "__main__":
    main()
