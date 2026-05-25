from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = Path("/private/tmp/ori_pqal.json")
DEFAULT_CORPUS_OUT = PROJECT_ROOT / "data" / "benchmarks" / "pubmedqa" / "pubmedqa_benchmark_corpus.json"
DEFAULT_EVAL_OUT = PROJECT_ROOT / "data" / "benchmarks" / "pubmedqa" / "eval_pubmedqa_benchmark.json"


def main() -> None:
    args = _parse_args()
    raw_items = _load_pubmedqa(args.input)
    selected_items = _balanced_sample(raw_items, per_label=args.per_label)
    corpus = [_to_corpus_item(item, strict=args.strict) for item in selected_items]
    eval_cases = [_to_eval_case(item, strict=args.strict) for item in selected_items]

    _write_json(args.corpus_out, corpus)
    _write_json(args.eval_out, eval_cases)
    label_counts = _label_counts(eval_cases)
    print(
        "Wrote PubMedQA benchmark dataset "
        f"corpus={args.corpus_out} eval={args.eval_out} cases={len(eval_cases)} "
        f"labels={label_counts} strict={args.strict}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced PubMedQA benchmark corpus and eval dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--per-label", type=int, default=30)
    parser.add_argument("--corpus-out", type=Path, default=DEFAULT_CORPUS_OUT)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_OUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Build a paper-like corpus without LONG_ANSWER/conclusion text or final-label metadata.",
    )
    return parser.parse_args()


def _load_pubmedqa(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected PubMedQA dict in {path}.")

    items = []
    for pmid, item in data.items():
        label = str(item.get("final_decision") or "").strip().lower()
        question = str(item.get("QUESTION") or "").strip()
        contexts = [str(value).strip() for value in item.get("CONTEXTS", []) if str(value).strip()]
        long_answer = str(item.get("LONG_ANSWER") or "").strip()
        if label not in {"yes", "no", "maybe"} or not question or not contexts or not long_answer:
            continue
        items.append({"pmid": str(pmid), **item, "final_decision": label})
    return sorted(items, key=lambda item: int(item["pmid"]) if item["pmid"].isdigit() else item["pmid"])


def _balanced_sample(items: list[dict[str, Any]], *, per_label: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_label[str(item["final_decision"])].append(item)

    selected = []
    for label in ("yes", "no", "maybe"):
        label_items = by_label[label]
        if len(label_items) < per_label:
            raise RuntimeError(f"Only {len(label_items)} `{label}` PubMedQA cases available; need {per_label}.")
        selected.extend(label_items[:per_label])
    return sorted(selected, key=lambda item: (item["final_decision"], item["pmid"]))


def _to_corpus_item(item: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    pmid = str(item["pmid"])
    question = str(item["QUESTION"]).strip()
    context = " ".join(str(value).strip() for value in item.get("CONTEXTS", []) if str(value).strip())
    long_answer = str(item.get("LONG_ANSWER") or "").strip()
    meshes = [str(value).strip() for value in item.get("MESHES", []) if str(value).strip()]
    labels = [str(value).strip() for value in item.get("LABELS", []) if str(value).strip()]
    content_parts = [
        f"PubMedQA {'strict ' if strict else ''}benchmark source PMID {pmid}.",
        f"Research question: {question}",
        f"Abstract context: {context}",
    ]
    if not strict:
        content_parts.append(f"Conclusion: {long_answer}")

    metadata = {
        "pmid": pmid,
        "journal": "PubMedQA",
        "year": _to_int(item.get("YEAR")),
        "topic": " ".join(meshes),
        "publicationTypes": ["PubMedQA Benchmark"],
        "corpusType": "pubmedqa_strict_benchmark" if strict else "pubmedqa_benchmark",
        "sourceAuthority": "PubMedQA",
        "meshTerms": meshes,
    }
    if not strict:
        metadata["pubmedqaFinalDecision"] = str(item["final_decision"])
        metadata["pubmedqaLabels"] = labels

    return {
        "id": f"pubmedqa-{pmid}",
        "title": question,
        "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "score": 0.0,
        "content": " ".join(content_parts),
        "metadata": metadata,
    }


def _to_eval_case(item: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    pmid = str(item["pmid"])
    question = str(item["QUESTION"]).strip()
    label = str(item["final_decision"]).strip().lower()
    return {
        "id": f"pubmedqa-{pmid}",
        "benchmark": "PubMedQA PQA-L strict" if strict else "PubMedQA PQA-L",
        "question": f"Answer yes, no, or maybe based on retrieved evidence: {question}",
        "benchmark_question": question,
        "expected_label": label,
        "expected_status": "grounded",
        "relevant_document_ids": [f"pubmedqa-{pmid}"],
        "relevant_pmids": [pmid],
        "source_urls": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"],
        "strict": strict,
    }


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _label_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"yes": 0, "no": 0, "maybe": 0}
    for item in cases:
        counts[str(item["expected_label"])] += 1
    return counts


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
