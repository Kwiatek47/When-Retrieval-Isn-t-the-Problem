#!/usr/bin/env python3
"""Apples-to-apples debate vs SC vs BioLinkBERT on the same case IDs.

Reads only:
  reports/debate/debate_balanced90_ml4h_v1.json
  reports/debate/sc_balanced90_ml4h_v1.json

Does not invent numbers. Does not call a GPU.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEBATE = ROOT / "reports/debate/debate_balanced90_ml4h_v1.json"
SC = ROOT / "reports/debate/sc_balanced90_ml4h_v1.json"
OUT = ROOT / "reports/debate/analysis/apples_to_apples_ml4h_v1.json"


def _binom_sf_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value under p=0.5 (McNemar exact)."""
    if n == 0:
        return float("nan")
    # P(X >= max(k, n-k)) * 2, capped at 1
    tail = max(k, n - k)
    # sum_{i=tail}^{n} C(n,i) / 2^n
    log2 = math.log(2.0)
    s = 0.0
    for i in range(tail, n + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * log2)
    return min(1.0, 2.0 * s)


def _recall(pairs: list[tuple[str, str]], gold_label: str) -> dict:
    gold_n = sum(1 for g, _ in pairs if g == gold_label)
    tp = sum(1 for g, p in pairs if g == gold_label and p == gold_label)
    return {
        "support": gold_n,
        "correct": tp,
        "recall": (tp / gold_n) if gold_n else float("nan"),
    }


def _acc(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    correct = sum(1 for g, p in pairs if g == p)
    return {
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else float("nan"),
        "per_label_recall": {lab: _recall(pairs, lab) for lab in ("yes", "no", "maybe")},
        "pred_counts": dict(Counter(p for _, p in pairs)),
    }


def _mcnemar(a_ok: list[bool], b_ok: list[bool]) -> dict:
    assert len(a_ok) == len(b_ok)
    both = sum(1 for a, b in zip(a_ok, b_ok) if a and b)
    a_only = sum(1 for a, b in zip(a_ok, b_ok) if a and not b)
    b_only = sum(1 for a, b in zip(a_ok, b_ok) if b and not a)
    neither = sum(1 for a, b in zip(a_ok, b_ok) if not a and not b)
    disc = a_only + b_only
    return {
        "n": len(a_ok),
        "both_correct": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "discordant": disc,
        "exact_p": _binom_sf_two_sided(a_only, disc) if disc else 1.0,
    }


def _subset(rows: list[dict], ids: set[str]) -> list[dict]:
    return [r for r in rows if r["id"] in ids]


def _pairs(rows: list[dict], pred_key: str) -> list[tuple[str, str]]:
    out = []
    for r in rows:
        p = r.get(pred_key)
        if p is None:
            raise SystemExit(f"missing {pred_key} on {r['id']}")
        out.append((r["gold"], p))
    return out


def main() -> None:
    debate = json.loads(DEBATE.read_text(encoding="utf-8"))
    sc = json.loads(SC.read_text(encoding="utf-8"))
    d_sum = debate["summary"]
    s_sum = sc["summary"]
    routing = d_sum.get("uncertainty_routing") or {}
    report_ids = set(routing.get("report_ids") or [])
    calib_ids = set(routing.get("calib_ids") or [])

    d_cases = {c["id"]: c for c in debate["cases"]}
    s_cases = {c["id"]: c for c in sc["cases"]}
    d_ids = set(d_cases)
    s_ids = set(s_cases)
    common = sorted(d_ids & s_ids)
    missing_in_sc = sorted(d_ids - s_ids)
    missing_in_debate = sorted(s_ids - d_ids)

    rows = []
    for cid in common:
        d = d_cases[cid]
        s = s_cases[cid]
        gold = d["expected_label"]
        if s["expected_label"] != gold:
            raise SystemExit(f"gold mismatch on {cid}: debate={gold} sc={s['expected_label']}")
        base = d.get("base_label")
        if not base:
            raise SystemExit(f"no base_label on {cid}")
        bert = d.get("biolinkbert_label")
        if not bert:
            raise SystemExit(f"no biolinkbert_label on {cid}")
        sc_pred = s.get("predicted_label")
        if not sc_pred:
            raise SystemExit(f"no SC predicted_label on {cid}")
        rows.append(
            {
                "id": cid,
                "gold": gold,
                "debate_routed": d["predicted_label"],
                "debate_ungated": base,
                "bert": bert,
                "sc": sc_pred,
                "split": "held_out" if cid in report_ids else ("calib" if cid in calib_ids else "unknown"),
                "routed_to_maybe": bool(d.get("routed_to_maybe")),
                "debate_llm_calls": d.get("llm_calls"),
                "sc_llm_calls": s.get("llm_calls"),
            }
        )

    held = _subset(rows, report_ids)
    full = rows

    def pack(subset: list[dict]) -> dict:
        return {
            "n": len(subset),
            "debate_ungated": _acc(_pairs(subset, "debate_ungated")),
            "debate_routed": _acc(_pairs(subset, "debate_routed")),
            "bert": _acc(_pairs(subset, "bert")),
            "sc": _acc(_pairs(subset, "sc")),
            "mcnemar": {
                "debate_ungated_vs_bert": _mcnemar(
                    [r["debate_ungated"] == r["gold"] for r in subset],
                    [r["bert"] == r["gold"] for r in subset],
                ),
                "debate_ungated_vs_sc": _mcnemar(
                    [r["debate_ungated"] == r["gold"] for r in subset],
                    [r["sc"] == r["gold"] for r in subset],
                ),
                "sc_vs_bert": _mcnemar(
                    [r["sc"] == r["gold"] for r in subset],
                    [r["bert"] == r["gold"] for r in subset],
                ),
                "debate_routed_vs_ungated": _mcnemar(
                    [r["debate_routed"] == r["gold"] for r in subset],
                    [r["debate_ungated"] == r["gold"] for r in subset],
                ),
            },
            "mean_llm_calls": {
                "debate": sum(r["debate_llm_calls"] or 0 for r in subset) / len(subset) if subset else None,
                "sc": sum(r["sc_llm_calls"] or 0 for r in subset) / len(subset) if subset else None,
            },
            "routed_to_maybe_count": sum(1 for r in subset if r["routed_to_maybe"]),
        }

    result = {
        "source": {
            "debate": str(DEBATE.relative_to(ROOT)),
            "sc": str(SC.relative_to(ROOT)),
        },
        "id_audit": {
            "debate_n_cases": len(d_ids),
            "sc_n_cases": len(s_ids),
            "common_n": len(common),
            "missing_in_sc": missing_in_sc,
            "missing_in_debate": missing_in_debate,
            "held_out_n": len(report_ids),
            "calib_n": len(calib_ids),
            "held_out_in_common": len(report_ids & set(common)),
            "unknown_split": sum(1 for r in rows if r["split"] == "unknown"),
        },
        "json_summaries_as_written": {
            "debate": {
                "summary.cases": d_sum.get("cases"),
                "summary.label_accuracy": d_sum.get("label_accuracy"),
                "summary.round1_accuracy": d_sum.get("round1_accuracy"),
                "summary.biolinkbert_accuracy": d_sum.get("biolinkbert_accuracy"),
                "summary.per_label_accuracy": d_sum.get("per_label_accuracy"),
                "routing.reported_on": routing.get("reported_on"),
                "routing.base_label_accuracy": routing.get("base_label_accuracy"),
                "routing.routed_to_maybe_count": routing.get("routed_to_maybe_count"),
                "routing.threshold": routing.get("threshold"),
                "cost.measured_cases": (d_sum.get("cost") or {}).get("measured_cases"),
                "cost.mean_llm_calls_per_case": (d_sum.get("cost") or {}).get("mean_llm_calls_per_case"),
            },
            "sc": {
                "summary.cases": s_sum.get("cases"),
                "summary.label_accuracy": s_sum.get("label_accuracy"),
                "summary.per_label_accuracy": s_sum.get("per_label_accuracy"),
                "summary.samples": s_sum.get("samples"),
                "cost.measured_cases": (s_sum.get("cost") or {}).get("measured_cases"),
                "cost.mean_llm_calls_per_case": (s_sum.get("cost") or {}).get("mean_llm_calls_per_case"),
            },
            "how_insert_sc_row_chose_numbers": (
                "insert_sc_row.py reads sc summary.label_accuracy and summary.cases "
                "verbatim (0.522, n=90). Debate 0.622 is summary.label_accuracy of the "
                "debate JSON, which _summarize() restricts to held-out report_ids (n=45, routed)."
            ),
        },
        "full_n90": pack(full),
        "held_out_n45": pack(held),
        "primary_choice": {
            "metric": "ungated_bert_gate debate vs SC vs BioLinkBERT on the same 90 IDs",
            "why": (
                "Both reports contain all 90 cases. Routed 0.622 is held-out n=45 only "
                "(threshold fit on the other 45). Applying the routed labels to all 90 "
                "mixes calib (ungated) + held-out (routed). SC has no router. "
                "Routing on held-out drops ungated 0.644 -> routed 0.622, so ungated "
                "is the honest debate arm for a compute-matched comparison."
            ),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def line(name: str, block: dict, key: str) -> str:
        a = block[key]
        mr = a["per_label_recall"]["maybe"]
        return (
            f"  {name:18s}  acc={a['accuracy']:.3f}  "
            f"({a['correct']}/{a['n']})  "
            f"maybe={mr['recall']:.3f} ({mr['correct']}/{mr['support']})  "
            f"yes={a['per_label_recall']['yes']['recall']:.3f}  "
            f"no={a['per_label_recall']['no']['recall']:.3f}"
        )

    print("=== ID audit ===")
    print(json.dumps(result["id_audit"], indent=2))
    print("\n=== JSON summaries as written ===")
    print(json.dumps(result["json_summaries_as_written"], indent=2))
    for title, block in (("FULL n=90 (same IDs)", result["full_n90"]), ("HELD-OUT n=45", result["held_out_n45"])):
        print(f"\n=== {title} ===")
        print(f"  mean_llm_calls debate={block['mean_llm_calls']['debate']}  sc={block['mean_llm_calls']['sc']}")
        print(f"  routed_to_maybe_count={block['routed_to_maybe_count']}")
        print(line("BERT", block, "bert"))
        print(line("debate ungated", block, "debate_ungated"))
        print(line("debate routed", block, "debate_routed"))
        print(line("SC N=8", block, "sc"))
        print("  McNemar exact p:")
        for k, v in block["mcnemar"].items():
            print(
                f"    {k}: a_only={v['a_only']} b_only={v['b_only']} "
                f"disc={v['discordant']} p={v['exact_p']:.4f}"
            )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
