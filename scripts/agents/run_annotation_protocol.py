"""Run the PubMedQA annotation-protocol replay (``app/agents/annotation_protocol.py``) over PQA-L.

Two qwen agents play PQA-L's two annotators: one sees the question, the abstract and the
authors' conclusion, the other only the question and the abstract. They label independently
(prompt ``label-defined@1``, the same as in the earlier label probes), and when they disagree
they discuss (``negotiate@1``) for up to ``--max-rounds`` rounds; unsettled questions are
removed, as Jin et al. (2019) did.

Needs the app's dependencies (pydantic, httpx): run with ``llm_env``. Resumable; every row is
one question; every invocation snapshots both prompts next to the output.

Usage:
  python scripts/agents/run_annotation_protocol.py --limit 5 --out /tmp/pilot.jsonl   # pilot
  python scripts/agents/run_annotation_protocol.py                                    # all 1000
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

from app.agents.annotation_protocol import CONTEXT_ONLY, WITH_CONCLUSION, AnnotationProtocol  # noqa: E402
from app.agents.backends import OllamaInferenceBackend  # noqa: E402
from app.providers.ollama import OllamaProvider  # noqa: E402
from scripts.agents.probe_prompts import get_prompt, snapshot_run  # noqa: E402
from scripts.agents.run_label_probe import evidence_text, select_pmids  # noqa: E402

PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
OUT = PROJECT_ROOT / "reports/debate/analysis/annotation_protocol/protocol-qwen3-30b.think-off.all.jsonl"
LABEL_PROMPT = "label-defined@1"
NEGOTIATE_PROMPT = "negotiate@1"


def done_pmids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("status") != "failed":
            done.add(row["pmid"])
    return done


async def run_all(args: argparse.Namespace) -> None:
    label_spec, negotiate_spec = get_prompt(LABEL_PROMPT), get_prompt(NEGOTIATE_PROMPT)
    data = json.loads(PQAL.read_text(encoding="utf-8"))
    skip = done_pmids(args.out)
    pmids = [p for p in select_pmids(data, args.split)[: args.limit] if p not in skip]

    snapshot_run(
        args.out,
        [label_spec, negotiate_spec],
        script="scripts/agents/run_annotation_protocol.py",
        model=args.model,
        meta={
            "think": args.think,
            "max_rounds": args.max_rounds,
            "split": args.split,
            "limit": args.limit,
            "temperature": 0.0,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "questions_to_do": len(pmids),
        },
    )
    print(f"{len(pmids)} questions to do -> {args.out}", flush=True)

    provider = OllamaProvider(base_url=args.base_url, timeout=args.timeout, keep_alive="60m", num_ctx=args.num_ctx)
    backend = OllamaInferenceBackend(
        provider, model=args.model, temperature=0.0, max_retries=2, think={"on": True, "off": False}[args.think]
    )
    protocol = AnnotationProtocol(
        backend,
        label_spec=label_spec,
        negotiate_spec=negotiate_spec,
        max_rounds=args.max_rounds,
        num_predict=args.num_predict,
    )
    gate = asyncio.Semaphore(args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0

    async def one(pmid: str) -> dict:
        item = data[pmid]
        evidence = {
            WITH_CONCLUSION: evidence_text(item, "context+conclusion"),
            CONTEXT_ONLY: evidence_text(item, "context"),
        }
        async with gate:
            result = await protocol.run(pmid, item["QUESTION"], evidence)
        return {**result.to_json(), "model": args.model, "think": args.think, "max_rounds": args.max_rounds,
                "label_prompt_sha": label_spec.sha, "negotiate_prompt_sha": negotiate_spec.sha}

    with args.out.open("a", encoding="utf-8") as fh:
        for task in asyncio.as_completed([one(p) for p in pmids]):
            row = await task
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done += 1
            if done % 50 == 0 or done == len(pmids):
                print(f"{done}/{len(pmids)} done, {done / (time.time() - started):.2f} q/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3:30b")
    parser.add_argument("--think", choices=("on", "off"), default="off")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--split", choices=("test", "cv", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-predict", type=int, default=600)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--out", type=Path, default=OUT)
    asyncio.run(run_all(parser.parse_args()))


if __name__ == "__main__":
    main()
