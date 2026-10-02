"""Hypothesis H1b: rate how conditional a passage's answer to its PubMedQA question is.

H1 (hedging words) failed: hedging is register in scientific conclusions, not answer
uncertainty. Jin et al. define ``maybe`` as an answer that holds under some conditions
and not others, so H1b asks whether that conditionality is stated in the conclusion
(hidden from models) rather than in the RESULTS section (visible to models).

For every PQA-L question a local LLM rates two passages with the same frozen prompt
(``scripts/agents/probe_prompts.py``): the authors' conclusion and the RESULTS section of
the context. The rater sees the question and one passage only, never the label, the other
passage or the annotators. Each passage is rated ``--repeats`` times with fixed seeds at
temperature 0 (single ratings were not stable in a pilot); the analysis averages the repeats.

``--prompt conditionality@1`` is the pre-registered prompt and writes to
``h1b_conditionality_ratings.jsonl``; any other prompt writes to its own file.
``--only-calibration`` rates just the 50 passages of the human calibration sheet, which is
how candidate prompts are compared before a full run.

Resumable: finished (pmid, part, repeat) keys are skipped on restart. Every row records the
model, prompt id and prompt hash, and every invocation snapshots its prompts next to the output.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.probe_prompts import (  # noqa: E402
    CONDITIONALITY_V1,
    PromptSpec,
    get_prompt,
    ollama_chat_json,
    snapshot_run,
)

PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
ANALYSIS = PROJECT_ROOT / "reports/debate/analysis"
OUT = ANALYSIS / "h1b_conditionality_ratings.jsonl"
CALIBRATION_KEY = ANALYSIS / "h1b_calibration_key.json"

# Kept for callers that predate the prompt registry; equals CONDITIONALITY_V1.sha.
PROMPT_SHA = CONDITIONALITY_V1.sha


def default_out(spec: PromptSpec, only_calibration: bool) -> Path:
    """The registered prompt keeps its original file; other prompts get their own."""
    if spec.id == CONDITIONALITY_V1.id:
        base = OUT
    else:
        base = ANALYSIS / f"h1b_conditionality_ratings.{spec.name}-v{spec.version}.jsonl"
    return base.with_name(base.stem + ".calibration.jsonl") if only_calibration else base


def passages(item: dict) -> dict[str, str]:
    """The two passages rated for one question; RESULTS is empty when the abstract has none."""
    results = " ".join(
        c for c, lab in zip(item["CONTEXTS"], item["LABELS"]) if "RESULT" in lab.upper()
    )
    return {"conclusion": item["LONG_ANSWER"], "results": results}


def parse_rating(content: str) -> tuple[int | None, str]:
    """Rating and evidence from the model's JSON; ``None`` if the reply is unusable."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None, ""
    rating = data.get("conditionality") if isinstance(data, dict) else None
    if isinstance(rating, bool) or rating not in (0, 1, 2):
        return None, ""
    evidence = data.get("evidence", "")
    return rating, evidence if isinstance(evidence, str) else ""


def done_keys(path: Path, model: str, spec: PromptSpec = CONDITIONALITY_V1) -> set[tuple[str, str, int]]:
    """(pmid, part, repeat) already rated by ``model`` with ``spec``."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a partial last line from an interrupted run
        if row.get("prompt_sha") != spec.sha or row.get("model") != model:
            raise SystemExit(
                f"{path} holds ratings from another prompt or model "
                f"({row.get('prompt_sha')}, {row.get('model')}); use a new --out."
            )
        if row.get("conditionality") is not None:
            keys.add((row["pmid"], row["part"], row["repeat"]))
    return keys


def rate_one(
    task: dict, spec: PromptSpec, base_url: str, model: str, timeout: float, attempts: int = 3
) -> dict:
    messages = spec.messages(question=task["question"], passage=task["passage"])
    rating, evidence, error = None, "", ""
    for attempt in range(attempts):
        try:
            rating, evidence = parse_rating(
                ollama_chat_json(base_url, model, spec, messages, seed=task["repeat"] + 1, timeout=timeout)
            )
            if rating is not None:
                break
            error = "unparseable reply"
        except Exception as exc:  # network or server error: retry, then record the failure
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    return {
        "pmid": task["pmid"],
        "part": task["part"],
        "repeat": task["repeat"],
        "conditionality": rating,
        "evidence": evidence,
        "error": "" if rating is not None else error,
        "model": model,
        "prompt_id": spec.id,
        "prompt_sha": spec.sha,
    }


def calibration_items(path: Path = CALIBRATION_KEY) -> set[tuple[str, str]]:
    return {(k["pmid"], k["part"]) for k in json.loads(path.read_text(encoding="utf-8"))}


def build_tasks(
    data: dict,
    repeats: int,
    skip: set[tuple[str, str, int]],
    limit: int | None,
    only: set[tuple[str, str]] | None = None,
) -> list[dict]:
    tasks = []
    for pmid in sorted(data)[:limit]:
        item = data[pmid]
        for part, text in passages(item).items():
            if not text or (only is not None and (pmid, part) not in only):
                continue
            for repeat in range(repeats):
                if (pmid, part, repeat) not in skip:
                    tasks.append(
                        {"pmid": pmid, "part": part, "repeat": repeat, "question": item["QUESTION"], "passage": text}
                    )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=CONDITIONALITY_V1.id, help="prompt id from probe_prompts.PROMPTS")
    parser.add_argument("--pqal", type=Path, default=PQAL)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None, help="rate only the first N questions (smoke test)")
    parser.add_argument("--only-calibration", action="store_true", help="rate only the calibration sheet passages")
    args = parser.parse_args()

    spec = get_prompt(args.prompt)
    if spec.name != "conditionality":
        raise SystemExit(f"{spec.id} is not a conditionality prompt")
    out = args.out or default_out(spec, args.only_calibration)
    data = json.loads(args.pqal.read_text(encoding="utf-8"))
    only = calibration_items() if args.only_calibration else None
    tasks = build_tasks(data, args.repeats, done_keys(out, args.model, spec), args.limit, only)

    snapshot_run(
        out,
        [spec],
        script="scripts/agents/rate_conditionality.py",
        model=args.model,
        meta={
            "repeats": args.repeats,
            "limit": args.limit,
            "only_calibration": args.only_calibration,
            "ratings_to_do": len(tasks),
        },
    )
    print(f"prompt {spec.id} ({spec.sha}, {spec.status}), model {args.model}: {len(tasks)} ratings to do -> {out}", flush=True)

    failed = 0
    started = time.time()
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(rate_one, t, spec, args.base_url, args.model, args.timeout) for t in tasks]
        for n, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            failed += row["conditionality"] is None
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 100 == 0 or n == len(tasks):
                rate = n / (time.time() - started)
                print(f"{n}/{len(tasks)} done, {failed} failed, {rate:.2f}/s", flush=True)


if __name__ == "__main__":
    main()
