from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import httpx


LABELS = ("yes", "no", "maybe")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


SYSTEM_PROMPT = (
    "You are a biomedical evidence classifier for PubMedQA. "
    "Use only the provided abstract evidence. Choose exactly one option: yes, no, or maybe. "
    "Return compact JSON with one key `answer` whose value is exactly `yes`, `no`, or `maybe`."
)


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args.dataset, corpus_path=args.corpus)
    if args.max_cases:
        cases = cases[: args.max_cases]
    report = asyncio.run(_run(cases, args=args))
    _write_json(args.json_out, report)
    _write_markdown(args.md_out, report)
    print(f"Wrote LLM judge baseline JSON: {args.json_out}")
    print(f"Wrote LLM judge baseline Markdown: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path("reports") / "classifier"
    parser = argparse.ArgumentParser(description="Run deterministic PubMedQA yes/no/maybe LLM judge baseline.")
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/benchmarks/pubmedqa/official_pqal_test/corpus.json"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=report_dir / f"llm_judge_{timestamp}.json")
    parser.add_argument("--md-out", type=Path, default=report_dir / f"llm_judge_{timestamp}.md")
    return parser.parse_args()


async def _run(cases: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    async with httpx.AsyncClient(base_url=args.ollama_url, timeout=args.timeout) as client:
        for index, case in enumerate(cases, start=1):
            prediction, raw_response = await _judge_case(client, model=args.model, case=case)
            rows.append(
                {
                    "id": case["id"],
                    "pmid": case["pmid"],
                    "expected_label": case["label"],
                    "predicted_label": prediction,
                    "pass": prediction == case["label"],
                    "raw_response": raw_response,
                }
            )
            if index % 25 == 0:
                print(f"judged {index}/{len(cases)} cases")

    labels = [LABEL_TO_ID[row["expected_label"]] for row in rows]
    predictions = [LABEL_TO_ID.get(str(row["predicted_label"]), -1) for row in rows]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "corpus": str(args.corpus),
        "model": args.model,
        "temperature": 0.0,
        "prompt": SYSTEM_PROMPT,
        "case_count": len(rows),
        "metrics": _classification_metrics(labels=labels, predictions=predictions),
        "predicted_labels": _ordered_counts(Counter(row["predicted_label"] for row in rows)),
        "cases": rows,
    }


async def _judge_case(client: httpx.AsyncClient, *, model: str, case: dict[str, Any]) -> tuple[str | None, str]:
    prompt = (
        f"Question:\n{case['question']}\n\n"
        f"Abstract evidence:\n{case['evidence']}\n\n"
        "Options: yes, no, maybe\n"
        "Return JSON only."
    )
    response = await client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        },
    )
    response.raise_for_status()
    data = response.json()
    content = str((data.get("message") or {}).get("content") or "")
    return _parse_label(content), content


def _parse_label(content: str) -> str | None:
    try:
        parsed = json.loads(content)
        label = str(parsed.get("answer") or parsed.get("label") or "").strip().lower()
        if label in LABELS:
            return label
    except json.JSONDecodeError:
        pass
    normalized = re.sub(r"[^a-z]+", " ", content.lower()).strip()
    for label in LABELS:
        if re.search(rf"\b{label}\b", normalized):
            return label
    return None


def _load_cases(dataset_path: Path, *, corpus_path: Path) -> list[dict[str, Any]]:
    corpus = _corpus_by_pmid(corpus_path)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = []
    for item in data:
        pmid = str((item.get("relevant_pmids") or [""])[0])
        evidence = _paper_evidence_text(str(corpus[pmid].get("content") or ""))
        question = str(item.get("benchmark_question") or item.get("question") or "").strip()
        question = re.sub(r"^Answer yes, no, or maybe based on retrieved evidence:\s*", "", question).strip()
        cases.append(
            {
                "id": str(item["id"]),
                "pmid": pmid,
                "question": question,
                "evidence": evidence,
                "label": str(item["expected_label"]).lower(),
            }
        )
    return cases


def _corpus_by_pmid(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str((item.get("metadata") or {}).get("pmid")): item for item in data}


def _paper_evidence_text(content: str) -> str:
    if "Abstract context:" in content:
        content = content.split("Abstract context:", 1)[1]
    return re.sub(r"\s+", " ", content).strip()


def _classification_metrics(*, labels: list[int], predictions: list[int]) -> dict[str, Any]:
    labels_with_none = (*LABELS, "none")
    confusion = {gold: {pred: 0 for pred in labels_with_none} for gold in labels_with_none}
    for gold, pred in zip(labels, predictions):
        gold_label = ID_TO_LABEL.get(gold, "none")
        pred_label = ID_TO_LABEL.get(pred, "none")
        confusion[gold_label][pred_label] += 1

    per_label = {}
    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels_with_none if gold != label)
        fn = sum(confusion[label][pred] for pred in labels_with_none if pred != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    correct = sum(int(gold == pred) for gold, pred in zip(labels, predictions))
    return {
        "accuracy": correct / max(len(labels), 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def _ordered_counts(counter: Counter[str | None]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in (*LABELS, None)}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    lines = [
        "# PubMedQA LLM Judge Baseline",
        "",
        f"- Model: `{report['model']}`",
        f"- Temperature: `{report['temperature']}`",
        f"- Cases: {report['case_count']}",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Macro F1: {metrics['macro_f1']:.3f}",
        "",
        "| Label | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in metrics["per_label"].items():
        lines.append(
            f"| {label} | {values['support']} | {values['precision']:.3f} | "
            f"{values['recall']:.3f} | {values['f1']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
