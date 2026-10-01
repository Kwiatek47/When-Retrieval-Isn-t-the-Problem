"""Ask one LLM for yes / no / maybe on PQA-L, with or without the authors' conclusion.

One prompt from ``scripts/agents/probe_prompts.py`` and one input condition per run:
  - ``context``: the abstract without its conclusion, as models and annotator 2 saw it;
  - ``context+conclusion``: the abstract plus the conclusion, as annotator 1 saw it.
The same prompt under both conditions is the direct H1 test with our own models (does
seeing the conclusion raise the maybe rate the way it did for the annotators?). Comparing
answers with ``reasoning_required_pred`` as well as ``final_decision`` is H2; running
``label-minimal@1`` / ``label-defined@1`` / ``label-defined-prior@1`` is RQ10 / RQ9a.

The model never sees the gold label or the annotators' labels. Resumable; every row records
model, prompt id and hash, and every invocation snapshots its prompt next to the output.
The closing summary is descriptive only; the tests that use these runs live in their own
analysis scripts and are registered before the runs they read.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.probe_prompts import (  # noqa: E402
    PromptSpec,
    get_prompt,
    ollama_chat_json,
    snapshot_run,
)

PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
TEST_SET = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/eval.json"
OUT_DIR = PROJECT_ROOT / "reports/debate/analysis/label_probe"
INPUTS = ("context", "context+conclusion")
LABELS = ("yes", "no", "maybe")


def evidence_text(item: dict, condition: str) -> str:
    """The abstract as the model sees it: labelled sections, plus the conclusion if asked."""
    sections = "\n".join(f"{lab}: {text}" for text, lab in zip(item["CONTEXTS"], item["LABELS"]))
    text = f"Abstract (without its conclusion):\n{sections}"
    if condition == "context+conclusion":
        text += f"\n\nAuthors' conclusion:\n{item['LONG_ANSWER']}"
    elif condition != "context":
        raise ValueError(f"unknown input condition {condition!r}")
    return text


def parse_label(content: str) -> tuple[str | None, int | None, str]:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None, None, ""
    if not isinstance(data, dict) or data.get("label") not in LABELS:
        return None, None, ""
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
        confidence = None
    rationale = data.get("rationale", "")
    return data["label"], confidence, rationale if isinstance(rationale, str) else ""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-")


THINK = {"default": None, "on": True, "off": False}
# Token budget of one reply. A long rationale was cut off at the prompt's default of 200,
# and thinking spends the same budget, so the runner sets its own limits.
DECODING = {
    "default": {"num_predict": 600},
    "off": {"num_predict": 600},
    "on": {"num_predict": 4096, "num_ctx": 8192},
}


def default_out(spec: PromptSpec, condition: str, model: str, split: str, think: str = "default") -> Path:
    suffix = "" if think == "default" else f".think-{think}"
    return OUT_DIR / f"{spec.name}-v{spec.version}.{_slug(condition)}.{_slug(model)}{suffix}.{split}.jsonl"


def done_keys(path: Path, model: str, spec: PromptSpec, condition: str, think: str = "default") -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        found = (row.get("prompt_sha"), row.get("model"), row.get("input"), row.get("think", "default"))
        if found != (spec.sha, model, condition, think):
            raise SystemExit(f"{path} holds another prompt, model, input condition or think mode; use a new --out.")
        if row.get("label") is not None:
            keys.add((row["pmid"], row["repeat"]))
    return keys


def select_pmids(data: dict, split: str, test_path: Path = TEST_SET) -> list[str]:
    in_test = {str(p) for case in json.loads(test_path.read_text(encoding="utf-8")) for p in case["relevant_pmids"]}
    if split == "all":
        return sorted(data)
    return sorted(p for p in data if (p in in_test) == (split == "test"))


def ask_one(
    task: dict, spec: PromptSpec, base_url: str, model: str, timeout: float, think: str = "default", attempts: int = 3
) -> dict:
    messages = spec.messages(question=task["question"], evidence=task["evidence"])
    label, confidence, rationale, error = None, None, "", ""
    for attempt in range(attempts):
        try:
            label, confidence, rationale = parse_label(
                ollama_chat_json(
                    base_url,
                    model,
                    spec,
                    messages,
                    seed=task["repeat"] + 1,
                    timeout=timeout,
                    think=THINK[think],
                    options=DECODING[think],
                )
            )
            if label is not None:
                break
            error = "unparseable reply"
        except Exception as exc:  # network or server error: retry, then record the failure
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    return {
        "pmid": task["pmid"],
        "input": task["input"],
        "repeat": task["repeat"],
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
        "error": "" if label is not None else error,
        "model": model,
        "think": think,
        "prompt_id": spec.id,
        "prompt_sha": spec.sha,
    }


def summarize(out: Path, data: dict) -> dict:
    """Descriptive counts: predicted labels and maybe recall against final and annotator 2."""
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("label") is not None and r["repeat"] == 0]
    summary: dict = {"n": len(rows), "predicted": dict(Counter(r["label"] for r in rows))}
    for key in ("final_decision", "reasoning_required_pred", "reasoning_free_pred"):
        gold = [data[r["pmid"]][key] for r in rows]
        pred = [r["label"] for r in rows]
        n_maybe = sum(g == "maybe" for g in gold)
        hit = sum(g == p == "maybe" for g, p in zip(gold, pred))
        summary[key] = {
            "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(rows), 4) if rows else None,
            "maybe_recall": f"{hit}/{n_maybe}",
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="label prompt id, e.g. label-defined@1")
    parser.add_argument("--input", choices=INPUTS, default="context")
    parser.add_argument("--split", choices=("test", "cv", "all"), default="test")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--think", choices=sorted(THINK), default="default", help="reasoning models: thinking on / off")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None, help="ask only the first N questions (smoke test)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = get_prompt(args.prompt)
    if "evidence" not in spec.fields:
        raise SystemExit(f"{spec.id} is not a label prompt")
    out = args.out or default_out(spec, args.input, args.model, args.split, args.think)
    data = json.loads(PQAL.read_text(encoding="utf-8"))
    skip = done_keys(out, args.model, spec, args.input, args.think)
    tasks = [
        {
            "pmid": pmid,
            "input": args.input,
            "repeat": repeat,
            "question": data[pmid]["QUESTION"],
            "evidence": evidence_text(data[pmid], args.input),
        }
        for pmid in select_pmids(data, args.split)[: args.limit]
        for repeat in range(args.repeats)
        if (pmid, repeat) not in skip
    ]

    snapshot_run(
        out,
        [spec],
        script="scripts/agents/run_label_probe.py",
        model=args.model,
        meta={
            "input": args.input,
            "split": args.split,
            "think": args.think,
            "decoding_overrides": DECODING[args.think],
            "repeats": args.repeats,
            "limit": args.limit,
            "questions_to_do": len(tasks),
        },
    )
    print(
        f"prompt {spec.id} ({spec.sha}), input {args.input}, model {args.model}, think {args.think}: "
        f"{len(tasks)} to do -> {out}",
        flush=True,
    )

    failed = 0
    started = time.time()
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(ask_one, t, spec, args.base_url, args.model, args.timeout, args.think) for t in tasks]
        for n, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            failed += row["label"] is None
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 100 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)} done, {failed} failed, {n / (time.time() - started):.2f}/s", flush=True)
    print(json.dumps(summarize(out, data), indent=2))


if __name__ == "__main__":
    main()
