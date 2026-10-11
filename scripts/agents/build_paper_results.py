"""Collect the results behind the current thesis into ``paper/results/`` and check every number.

``paper/results/claims.json`` is the hand-edited list: each claim names a source file, a key path in
it and the number as the paper writes it (string, so the decimals say how it is rounded). This
script

  1. copies each source JSON from ``reports/debate/analysis/`` to ``paper/results/data/`` and each
     figure to ``paper/results/figures/``;
  2. checks every claim against the copy: the value (and the interval, when given) must equal the
     stored number rounded to the claim's decimals;
  3. writes ``paper/results/MANIFEST.md``: one table per section with the number, its source,
     script, registration commit and sha256 of the copied file.

``--check`` copies nothing and writes nothing: it checks the claims against the copies already in
``paper/results/`` and reports any source in ``reports/`` that has changed since it was copied.
Exit code 1 if any claim fails or any copy is stale. Stdlib only (system python 3.8 is enough).

Usage:
  python3 scripts/agents/build_paper_results.py           # copy, check, write MANIFEST.md
  python3 scripts/agents/build_paper_results.py --check   # check only
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "paper/results"
CLAIMS = RESULTS / "claims.json"
VALUE_KEYS = ("value", "mean", "d_auroc", "auroc")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lookup(doc: dict, path: str):
    node = doc
    for key in path.strip("/").split("/"):
        node = node[key]
    return node


def same(actual: float, expected: str) -> bool:
    """True if ``expected`` is ``actual`` rounded to ``expected``'s decimals.

    Within half a unit, inclusive: the files already store values rounded to 4 decimals, so 0.1725
    may be 0.172 or 0.173 and both are accepted.
    """
    half = Decimal(5).scaleb(Decimal(expected).as_tuple().exponent - 1)
    return abs(Decimal(repr(actual)) - Decimal(expected)) <= half


def claim_values(claim: dict, doc: dict) -> tuple[float, tuple[float, float] | None]:
    if "sum" in claim:
        return sum(lookup(doc, p) for p in claim["sum"]), None
    node = lookup(doc, claim["path"])
    if not isinstance(node, dict):
        return node, None
    value = next(node[k] for k in VALUE_KEYS if k in node)
    ci = (node["ci_low"], node["ci_high"]) if "ci_low" in node else None
    return value, ci


def check_claim(claim: dict, doc: dict) -> list[str]:
    """Problems with one claim; empty if the number and interval match the file."""
    value, ci = claim_values(claim, doc)
    problems = []
    if not same(value, claim["value"]):
        problems.append(f"value {claim['value']} but file has {value}")
    if "ci" in claim:
        if ci is None:
            problems.append("claim gives an interval but the file has none")
        else:
            for side, want, got in zip(("low", "high"), claim["ci"], ci):
                if not same(got, want):
                    problems.append(f"ci_{side} {want} but file has {got}")
    if "verdict" in claim and isinstance(lookup(doc, claim.get("path", "/")), dict):
        got = lookup(doc, claim["path"]).get("verdict")
        if got is not None and got != claim["verdict"]:
            problems.append(f"verdict {claim['verdict']!r} but file has {got!r}")
    return problems


def copy_name(source: dict) -> str:
    return Path(source["from"]).name


def check_all(spec: dict, root: Path = PROJECT_ROOT, results: Path = RESULTS) -> tuple[list[str], list[str]]:
    """(claim failures, stale copies) against the copies in ``results/data``."""
    docs = {name: json.loads((results / "data" / copy_name(s)).read_text(encoding="utf-8"))
            for name, s in spec["sources"].items()}
    failures = []
    for section in spec["sections"]:
        for claim in section["claims"]:
            try:
                problems = check_claim(claim, docs[claim["source"]])
            except (KeyError, StopIteration) as exc:
                problems = [f"key not found: {exc}"]
            failures += [f"{claim['id']}: {p}" for p in problems]
    pairs = [(results / "data" / copy_name(s), root / s["from"]) for s in spec["sources"].values()]
    pairs += [(results / "figures" / name, root / src) for name, src in spec["figures"].items()]
    stale = [str(src.relative_to(root)) for copy, src in pairs if src.exists() and sha256(src) != sha256(copy)]
    return failures, stale


def copy_all(spec: dict, root: Path = PROJECT_ROOT, results: Path = RESULTS) -> None:
    (results / "data").mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    for source in spec["sources"].values():
        shutil.copy2(root / source["from"], results / "data" / copy_name(source))
    for name, src in spec["figures"].items():
        shutil.copy2(root / src, results / "figures" / name)


def fmt(claim: dict) -> str:
    text = claim["value"]
    if "ci" in claim:
        text += f" [{claim['ci'][0]}, {claim['ci'][1]}]"
    return text


def manifest(spec: dict, results: Path = RESULTS) -> str:
    sources = spec["sources"]
    lines = [
        "# Results behind the current thesis",
        "",
        "Generated by `scripts/agents/build_paper_results.py` from `claims.json`. Do not edit by hand:",
        "change `claims.json` and rerun `make paper-results`. `make paper-results-check` verifies every",
        "number against the copies in `data/` and flags copies older than their source in `reports/`.",
        "",
        f"**Thesis.** {spec['thesis']}",
        "",
        f"**Thesis version.** {spec['thesis_version']}",
        "",
        "Intervals are 95% bootstrap CIs over questions (5000 resamples, seed 47) unless the source says",
        "otherwise. Status: *registered* = criterion committed before the run; *exploratory* = not",
        "registered or on questions already inspected; *descriptive* = a count, not a test.",
        "",
    ]
    for section in spec["sections"]:
        lines += [f"## {section['id']}. {section['title']}", "",
                  "| ID | Claim | Number | Verdict | Source | Status |", "|---|---|---|---|---|---|"]
        for claim in section["claims"]:
            src = sources[claim["source"]]
            key = claim.get("path") or " + ".join(claim["sum"])
            lines.append(
                f"| {claim['id']} | {claim['text']} | {fmt(claim)} | {claim.get('verdict', '')} "
                f"| `data/{copy_name(src)}` `{key}` | {src['status']} |"
            )
        lines.append("")
    lines += ["## Sources", "", "| File | Script | Status | Registration | Questions | sha256 |",
              "|---|---|---|---|---|---|"]
    for src in sources.values():
        reg = f"`{src['registration']}`" if src["registration"] and " " not in src["registration"] else (src["registration"] or "—")
        lines.append(
            f"| `data/{copy_name(src)}` | `{src['script']}` | {src['status']} | {reg} | {src['n']} "
            f"| `{sha256(results / 'data' / copy_name(src))[:12]}` |"
        )
    lines += ["", "## Figures", "", "| File | Copied from | sha256 |", "|---|---|---|"]
    for name, src in spec["figures"].items():
        lines.append(f"| `figures/{name}` | `{src}` | `{sha256(results / 'figures' / name)[:12]}` |")
    lines += ["", "## Left out on purpose", ""] + [f"- {item}" for item in spec["left_out"]] + [""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="check only; copy and write nothing")
    args = parser.parse_args()

    spec = json.loads(CLAIMS.read_text(encoding="utf-8"))
    if not args.check:
        copy_all(spec)
    failures, stale = check_all(spec)
    n_claims = sum(len(s["claims"]) for s in spec["sections"])
    for line in failures:
        print(f"FAIL {line}")
    for path in stale:
        print(f"STALE {path} changed since it was copied; rerun without --check")
    if not args.check and not failures:
        (RESULTS / "MANIFEST.md").write_text(manifest(spec), encoding="utf-8")
        print(f"wrote {RESULTS / 'MANIFEST.md'}")
    print(f"{n_claims - len({f.split(':')[0] for f in failures})}/{n_claims} claims match; {len(stale)} stale copies")
    sys.exit(1 if failures or stale else 0)


if __name__ == "__main__":
    main()
