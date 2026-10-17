"""Replay only the discussion of the mixed-model protocol runs, from their own round-0 labels.

Source runs (``run_annotation_protocol.py``): QM (qwen3:30b with the conclusion, mistral-small3.2:24b
without) and MQ (the reverse). Every question the two agents disputed at round 0 is discussed
again, by the same models in the same seats, starting from the same round-0 labels and reasons:

  old       negotiate@1, the discussion prompt of the source runs (a rerun: measures how much the
            outcome moves on its own);
  evidence  justify@1 then negotiate-evidence@1: each side first quotes the passage behind its
            label, both see each other's quotes, and both are told to weigh quoted evidence rather
            than how certain the colleague sounds.

Needs the app's dependencies: run with ``llm_env``. Resumable; snapshots its prompts next to the output.

Usage:
  python scripts/agents/run_negotiation_replay.py --source QM --arm evidence --source-file pilot.jsonl --out /tmp/p.jsonl
  python scripts/agents/run_negotiation_replay.py --source QM --arm old
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.annotation_protocol import ANNOTATORS, CONTEXT_ONLY, WITH_CONCLUSION, AnnotationProtocol, Turn  # noqa: E402
from app.agents.backends import OllamaInferenceBackend  # noqa: E402
from app.providers.ollama import OllamaProvider  # noqa: E402
from scripts.agents.analyze_annotation_protocol import PILOT_PMIDS  # noqa: E402
from scripts.agents.probe_prompts import get_prompt, snapshot_run  # noqa: E402
from scripts.agents.run_annotation_protocol import PQAL, done_pmids, think_for  # noqa: E402
from scripts.agents.run_label_probe import evidence_text  # noqa: E402

RUN_DIR = PROJECT_ROOT / "reports/debate/analysis/annotation_protocol"
SOURCES = {
    "QM": RUN_DIR / "protocol-qwen3-30b+mistral-small3.2-24b.think-off.all.jsonl",
    "MQ": RUN_DIR / "protocol-mistral-small3.2-24b+qwen3-30b.think-off.all.jsonl",
}
ARMS = {"old": (None, "negotiate@1"), "evidence": ("justify@1", "negotiate-evidence@1")}
OUT_DIR = PROJECT_ROOT / "reports/debate/analysis/negotiation_replay"


def disputes(path: Path, skip_pilot: bool = True) -> list[dict]:
    """Source rows the agents disputed at round 0, in pmid order; pilot questions left out unless asked."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["status"] in ("negotiated", "removed") and not (skip_pilot and row["pmid"] in PILOT_PMIDS):
            rows[row["pmid"]] = row
    return [rows[p] for p in sorted(rows)]


def first_turns(row: dict) -> list[Turn]:
    by_seat = {t["annotator"]: t for t in row["turns"] if t["round"] == 0 and t.get("kind", "label") == "label"}
    return [Turn(a, 0, by_seat[a]["label"], by_seat[a]["confidence"], by_seat[a]["rationale"]) for a in ANNOTATORS]


async def run_all(args: argparse.Namespace) -> None:
    justify_id, negotiate_id = ARMS[args.arm]
    label_spec, negotiate_spec = get_prompt("label-defined@1"), get_prompt(negotiate_id)
    justify_spec = get_prompt(justify_id) if justify_id else None
    data = json.loads(PQAL.read_text(encoding="utf-8"))
    source_file = args.source_file or SOURCES[args.source]
    source = disputes(source_file, skip_pilot=args.source_file is None)
    if not source:
        print(f"{args.source}/{args.arm}: no disputes in {source_file}", flush=True)
        return
    models = source[0]["models"]
    skip = done_pmids(args.out)
    todo = [r for r in source[: args.limit] if r["pmid"] not in skip]

    specs = [s for s in (justify_spec, negotiate_spec) if s is not None]
    snapshot_run(
        args.out, specs, script="scripts/agents/run_negotiation_replay.py",
        model=f"{models[WITH_CONCLUSION]}|{models[CONTEXT_ONLY]}",
        meta={"source": args.source, "source_file": str(source_file), "arm": args.arm,
              "max_rounds": args.max_rounds, "temperature": 0.0, "num_predict": args.num_predict, "num_ctx": args.num_ctx,
              "questions_to_do": len(todo)},
    )
    print(f"{args.source}/{args.arm}: {len(todo)} disputes to replay -> {args.out}", flush=True)

    provider = OllamaProvider(base_url=args.base_url, timeout=args.timeout, keep_alive="60m", num_ctx=args.num_ctx)
    backends = {
        seat: OllamaInferenceBackend(provider, model=models[seat], temperature=0.0, max_retries=2,
                                     think=think_for(models[seat], "off"))
        for seat in ANNOTATORS
    }
    protocol = AnnotationProtocol(backends=backends, label_spec=label_spec, negotiate_spec=negotiate_spec,
                                  justify_spec=justify_spec, max_rounds=args.max_rounds, num_predict=args.num_predict)
    gate = asyncio.Semaphore(args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0

    async def one(row: dict) -> dict:
        item = data[row["pmid"]]
        evidence = {WITH_CONCLUSION: evidence_text(item, "context+conclusion"), CONTEXT_ONLY: evidence_text(item, "context")}
        async with gate:
            result = await protocol.negotiate(row["pmid"], item["QUESTION"], evidence, first_turns(row))
        return {**result.to_json(), "source": args.source, "arm": args.arm, "models": models,
                "negotiate_prompt_sha": negotiate_spec.sha, "justify_prompt_sha": justify_spec.sha if justify_spec else None}

    with args.out.open("a", encoding="utf-8") as fh:
        for task in asyncio.as_completed([one(r) for r in todo]):
            row = await task
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} done, {done / (time.time() - started):.2f} q/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--source-file", type=Path, default=None, help="pilot only: another protocol run; keeps pilot questions")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-predict", type=int, default=600)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.out = args.out or OUT_DIR / f"{args.source}.{args.arm}.jsonl"
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
