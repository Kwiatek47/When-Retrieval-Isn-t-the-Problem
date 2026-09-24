"""Hypothesis H1b: rate how conditional a passage's answer to its PubMedQA question is.

H1 (hedging words) failed: hedging is register in scientific conclusions, not answer
uncertainty. Jin et al. define ``maybe`` as an answer that holds under some conditions
and not others, so H1b asks whether that conditionality is stated in the conclusion
(hidden from models) rather than in the RESULTS section (visible to models).

For every PQA-L question a local LLM rates two passages with the same frozen prompt:
the authors' conclusion and the RESULTS section of the context. The rater sees the
question and one passage only, never the label, the other passage or the annotators.
Each passage is rated ``--repeats`` times with fixed seeds at temperature 0 (single
ratings were not stable in a pilot); the analysis averages the repeats.

Resumable: ratings are appended to ``h1b_conditionality_ratings.jsonl`` and finished
(pmid, part, repeat) keys are skipped on restart. Every row records the model and the
prompt hash, so a changed prompt cannot silently mix into one run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
OUT = PROJECT_ROOT / "reports/debate/analysis/h1b_conditionality_ratings.jsonl"

SYSTEM_PROMPT = (
    "You annotate biomedical research abstracts. You judge whether a passage answers a "
    "research question unconditionally or only conditionally. You do not judge whether "
    "the answer is yes or no, and you do not judge how cautious the wording is."
)

USER_TEMPLATE = """Research question: {question}

Passage:
{passage}

Does this passage, taken on its own, give a CONDITIONAL answer to the research question?
An answer is conditional when it differs across subgroups, outcomes, settings, doses or time
points, holds only for some of them, or when the passage reports both supporting and
non-supporting findings for the question.

Rate conditionality:
0 = not conditional: one answer that holds as stated, or the passage does not address the question
1 = partly: one main answer with a minor qualification or restriction
2 = conditional: the answer clearly depends on a subgroup or condition, or findings point in different directions

Ignore hedging words such as "may", "suggest" or "further studies are needed"; they do not
make an answer conditional by themselves.
Return JSON: {{"conditionality": 0, 1 or 2, "evidence": "<quote of at most 20 words, or empty>"}}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "conditionality": {"type": "integer", "enum": [0, 1, 2]},
        "evidence": {"type": "string"},
    },
    "required": ["conditionality", "evidence"],
}

PROMPT_SHA = hashlib.sha256(
    (SYSTEM_PROMPT + "\n" + USER_TEMPLATE + json.dumps(RESPONSE_SCHEMA, sort_keys=True)).encode("utf-8")
).hexdigest()[:12]

def passages(item: dict) -> dict[str, str]:
    """The two passages rated for one question; RESULTS is empty when the abstract has none."""
    results = " ".join(
        c for c, lab in zip(item["CONTEXTS"], item["LABELS"]) if "RESULT" in lab.upper()
    )
    return {"conclusion": item["LONG_ANSWER"], "results": results}


def build_messages(question: str, passage: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, passage=passage)},
    ]


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


def done_keys(path: Path, model: str) -> set[tuple[str, str, int]]:
    """(pmid, part, repeat) already rated by ``model`` with the current prompt."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a partial last line from an interrupted run
        if row.get("prompt_sha") != PROMPT_SHA or row.get("model") != model:
            raise SystemExit(
                f"{path} holds ratings from another prompt or model "
                f"({row.get('prompt_sha')}, {row.get('model')}); use a new --out."
            )
        if row.get("conditionality") is not None:
            keys.add((row["pmid"], row["part"], row["repeat"]))
    return keys


def _chat(base_url: str, model: str, messages: list[dict], seed: int, timeout: float) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "60m",
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": 0, "seed": seed, "num_ctx": 4096, "num_predict": 120},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["message"]["content"]


def rate_one(task: dict, base_url: str, model: str, timeout: float, attempts: int = 3) -> dict:
    messages = build_messages(task["question"], task["passage"])
    rating, evidence, error = None, "", ""
    for attempt in range(attempts):
        try:
            rating, evidence = parse_rating(
                _chat(base_url, model, messages, seed=task["repeat"] + 1, timeout=timeout)
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
        "prompt_sha": PROMPT_SHA,
    }


def build_tasks(data: dict, repeats: int, skip: set[tuple[str, str, int]], limit: int | None) -> list[dict]:
    tasks = []
    for pmid in sorted(data)[:limit]:
        item = data[pmid]
        for part, text in passages(item).items():
            if not text:
                continue
            for repeat in range(repeats):
                if (pmid, part, repeat) not in skip:
                    tasks.append(
                        {"pmid": pmid, "part": part, "repeat": repeat, "question": item["QUESTION"], "passage": text}
                    )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pqal", type=Path, default=PQAL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None, help="rate only the first N questions (smoke test)")
    args = parser.parse_args()

    data = json.loads(args.pqal.read_text(encoding="utf-8"))
    tasks = build_tasks(data, args.repeats, done_keys(args.out, args.model), args.limit)
    print(f"prompt {PROMPT_SHA}, model {args.model}: {len(tasks)} ratings to do", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    started = time.time()
    with args.out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(rate_one, t, args.base_url, args.model, args.timeout) for t in tasks]
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
