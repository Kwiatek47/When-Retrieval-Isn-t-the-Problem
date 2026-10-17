"""Run the PubMedQA annotation-protocol replay (``app/agents/annotation_protocol.py``) over PQA-L.

Two qwen agents play PQA-L's two annotators: one sees the question, the abstract and the
authors' conclusion, the other only the question and the abstract. They label independently
(prompt ``label-defined@1``, the same as in the earlier label probes), and when they disagree
they discuss (``negotiate@1``) for up to ``--max-rounds`` rounds; unsettled questions are
removed, as Jin et al. (2019) did.

The two seats can be different models (``--model-with`` / ``--model-without``, both default to
``--model``). ``--think`` is sent only to qwen3 models; other models get no thinking switch.

Needs the app's dependencies (pydantic, httpx): run with ``llm_env``. Resumable; every row is
one question; every invocation snapshots both prompts next to the output.

Usage:
  python scripts/agents/run_annotation_protocol.py --limit 5 --out /tmp/pilot.jsonl   # pilot
  python scripts/agents/run_annotation_protocol.py                                    # all 1000
  python scripts/agents/run_annotation_protocol.py --model-with qwen3:30b --model-without mistral-small3.2:24b
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
OUT_DIR = PROJECT_ROOT / "reports/debate/analysis/annotation_protocol"
LABEL_PROMPT = "label-defined@1"
NEGOTIATE_PROMPT = "negotiate@1"


def model_tag(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def default_out(model_with: str, model_without: str, think: str, split: str) -> Path:
    """One model in both seats keeps the original name; a mixed pair names both, with-conclusion first."""
    pair = model_tag(model_with) if model_with == model_without else f"{model_tag(model_with)}+{model_tag(model_without)}"
    return OUT_DIR / f"protocol-{pair}.think-{think}.{split}.jsonl"


def think_for(model: str, think: str) -> bool | None:
    return {"on": True, "off": False}[think] if model.startswith("qwen3") else None


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
        model=f"{args.model_with}|{args.model_without}",
        meta={
            "model_with_conclusion": args.model_with,
            "model_context_only": args.model_without,
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
    backends = {
        seat: OllamaInferenceBackend(
            provider, model=model, temperature=0.0, max_retries=2, think=think_for(model, args.think)
        )
        for seat, model in ((WITH_CONCLUSION, args.model_with), (CONTEXT_ONLY, args.model_without))
    }
    protocol = AnnotationProtocol(
        backends=backends,
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
        return {**result.to_json(), "model": args.model_with if args.model_with == args.model_without else None,
                "models": {WITH_CONCLUSION: args.model_with, CONTEXT_ONLY: args.model_without}, "think": args.think, "max_rounds": args.max_rounds,
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
    parser.add_argument("--model", default="qwen3:30b", help="model for both seats unless overridden")
    parser.add_argument("--model-with", default=None, help="model of the annotator who sees the conclusion")
    parser.add_argument("--model-without", default=None, help="model of the annotator who sees only the context")
    parser.add_argument("--think", choices=("on", "off"), default="off")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--split", choices=("test", "cv", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-predict", type=int, default=600)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.model_with = args.model_with or args.model
    args.model_without = args.model_without or args.model
    args.out = args.out or default_out(args.model_with, args.model_without, args.think, args.split)
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
