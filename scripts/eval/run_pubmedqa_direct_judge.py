from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean
from time import perf_counter
from typing import Any

import httpx


LABELS = ("yes", "no", "maybe")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


PROMPTS = {
    "compact": {
        "system": (
            "You are a biomedical evidence classifier for PubMedQA. "
            "Choose exactly one answer: yes, no, or maybe. Return compact JSON only."
        ),
        "instruction": (
            "Return JSON with keys: answer, confidence, evidence_sufficient, rationale. "
            "answer must be exactly yes, no, or maybe. confidence must be a number from 0 to 1."
        ),
    },
    "definitions": {
        "system": (
            "You are a biomedical evidence classifier for PubMedQA. "
            "Use strict PubMedQA labels and return compact JSON only."
        ),
        "instruction": (
            "Definitions: yes = the abstract directly supports the proposition in the question; "
            "no = the abstract directly refutes it or shows no effect/association; "
            "maybe = the abstract is inconclusive, indirect, mixed, suggestive, or insufficient. "
            "Return JSON with keys: answer, confidence, evidence_sufficient, rationale."
        ),
    },
    "cite_then_answer": {
        "system": (
            "You are a biomedical evidence classifier for PubMedQA. "
            "First identify the decisive evidence sentence, then classify the answer. Return JSON only."
        ),
        "instruction": (
            "Return JSON with keys: decisive_evidence, answer, confidence, evidence_sufficient, rationale. "
            "answer must be exactly yes, no, or maybe. If no decisive evidence exists, answer maybe."
        ),
    },
    "sufficiency_first": {
        "system": (
            "You are an evidence sufficiency detector and PubMedQA classifier. "
            "Decide whether the evidence is sufficient before choosing the label. Return JSON only."
        ),
        "instruction": (
            "Step 1: decide evidence_sufficient. Step 2: choose answer. "
            "If evidence is only suggestive, partial, mixed, or not directly answering the question, choose maybe. "
            "Return JSON with keys: evidence_sufficient, answer, confidence, rationale."
        ),
    },
}


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args.dataset, corpus_path=args.corpus, evidence_mode=args.evidence_mode)
    if args.max_cases:
        cases = cases[: args.max_cases]
    report = asyncio.run(_run(cases, args=args))
    _write_json(args.json_out, report)
    _write_markdown(args.md_out, report)
    print(f"Wrote direct judge JSON: {args.json_out}")
    print(f"Wrote direct judge Markdown: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Run direct PubMedQA yes/no/maybe judge without RAG API.")
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/benchmarks/pubmedqa/official_pqal_test/corpus.json"))
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--evidence-mode", choices=("none", "oracle"), default="oracle")
    parser.add_argument("--prompt-style", choices=tuple(PROMPTS), default="compact")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=Path("reports") / f"direct_judge_{timestamp}.json")
    parser.add_argument("--md-out", type=Path, default=Path("reports") / f"direct_judge_{timestamp}.md")
    return parser.parse_args()


async def _run(cases: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    async with httpx.AsyncClient(base_url=args.ollama_url, timeout=args.timeout) as client:
        for index, case in enumerate(cases, start=1):
            started_at = perf_counter()
            parsed, raw_response = await _judge_case(client, model=args.model, case=case, args=args)
            latency_ms = (perf_counter() - started_at) * 1000.0
            predicted_label = parsed.get("answer")
            confidence = _float_or_none(parsed.get("confidence"))
            if confidence is not None:
                confidence = min(max(confidence, 0.0), 1.0)
            rows.append(
                {
                    "id": case["id"],
                    "pmid": case.get("pmid"),
                    "expected_label": case["label"],
                    "predicted_label": predicted_label,
                    "label_pass": predicted_label == case["label"],
                    "confidence": confidence,
                    "evidence_sufficient": _bool_or_none(parsed.get("evidence_sufficient")),
                    "rationale": str(parsed.get("rationale") or ""),
                    "decisive_evidence": str(parsed.get("decisive_evidence") or ""),
                    "latency_ms": latency_ms,
                    "raw_response": raw_response,
                }
            )
            if index % 25 == 0:
                print(f"direct_judge {index}/{len(cases)} cases")

    labels = [LABEL_TO_ID[row["expected_label"]] for row in rows]
    predictions = [LABEL_TO_ID.get(str(row["predicted_label"]), -1) for row in rows]
    metrics = _classification_metrics(labels=labels, predictions=predictions)
    confidence_metrics = _confidence_metrics(rows)
    report = {
        "kind": "direct_judge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "corpus": str(args.corpus),
        "model": args.model,
        "temperature": args.temperature,
        "evidence_mode": args.evidence_mode,
        "prompt_style": args.prompt_style,
        "case_count": len(rows),
        "metrics": metrics,
        "confidence_metrics": confidence_metrics,
        "sufficiency": _sufficiency_summary(rows),
        "predicted_labels": _ordered_counts(Counter(row["predicted_label"] for row in rows)),
        "prompt": PROMPTS[args.prompt_style],
        "cases": rows,
    }
    return report


async def _judge_case(
    client: httpx.AsyncClient,
    *,
    model: str,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    prompt = _build_prompt(case, evidence_mode=args.evidence_mode, prompt_style=args.prompt_style)
    response = await client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": PROMPTS[args.prompt_style]["system"]},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": args.temperature},
        },
    )
    response.raise_for_status()
    data = response.json()
    content = str((data.get("message") or {}).get("content") or "")
    return _parse_response(content), content


def _build_prompt(case: dict[str, Any], *, evidence_mode: str, prompt_style: str) -> str:
    pieces = [
        f"Question:\n{case['question']}",
        "",
    ]
    if evidence_mode == "oracle":
        pieces.extend([f"Abstract evidence:\n{case['evidence']}", ""])
    else:
        pieces.extend(["Abstract evidence:\n<none provided>", ""])
    pieces.extend(
        [
            "Options: yes, no, maybe",
            PROMPTS[prompt_style]["instruction"],
            "Return JSON only.",
        ]
    )
    return "\n".join(pieces)


def _parse_response(content: str) -> dict[str, Any]:
    parsed = _json_from_text(content)
    if parsed is None:
        parsed = {}
    label = str(parsed.get("answer") or parsed.get("label") or "").strip().lower()
    if label not in LABELS:
        label = _parse_label(content)
    parsed["answer"] = label
    if "evidence_sufficient" not in parsed:
        parsed["evidence_sufficient"] = _infer_sufficiency(content)
    return parsed


def _json_from_text(content: str) -> dict[str, Any] | None:
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_label(content: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", " ", content.lower()).strip()
    for label in LABELS:
        if re.search(rf"\b{label}\b", normalized):
            return label
    return None


def _infer_sufficiency(content: str) -> bool | None:
    normalized = content.lower()
    if re.search(r"\b(insufficient|inconclusive|unclear|not enough|cannot determine|no decisive)\b", normalized):
        return False
    if re.search(r"\b(sufficient|directly supports|directly refutes|decisive)\b", normalized):
        return True
    return None


def _load_cases(dataset_path: Path, *, corpus_path: Path, evidence_mode: str) -> list[dict[str, Any]]:
    corpus = _corpus_by_pmid(corpus_path) if evidence_mode == "oracle" else {}
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = []
    for item in data:
        pmid = str((item.get("relevant_pmids") or [""])[0])
        question = str(item.get("benchmark_question") or item.get("question") or "").strip()
        question = re.sub(r"^Answer yes, no, or maybe based on retrieved evidence:\s*", "", question).strip()
        evidence = ""
        if evidence_mode == "oracle":
            if pmid not in corpus:
                raise RuntimeError(f"PMID {pmid} from {dataset_path} not found in {corpus_path}.")
            evidence = _paper_evidence_text(str(corpus[pmid].get("content") or ""))
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


def _confidence_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_confidence = [row for row in rows if row.get("confidence") is not None]
    if not with_confidence:
        return {
            "coverage": 0.0,
            "mean_confidence": None,
            "mean_confidence_correct": None,
            "mean_confidence_wrong": None,
            "high_confidence_error_rate": None,
            "ece": None,
        }
    correct = [row for row in with_confidence if row["label_pass"]]
    wrong = [row for row in with_confidence if not row["label_pass"]]
    high_conf = [row for row in with_confidence if float(row["confidence"]) >= 0.80]
    high_conf_wrong = [row for row in high_conf if not row["label_pass"]]
    return {
        "coverage": len(with_confidence) / max(len(rows), 1),
        "mean_confidence": mean(float(row["confidence"]) for row in with_confidence),
        "mean_confidence_correct": mean(float(row["confidence"]) for row in correct) if correct else None,
        "mean_confidence_wrong": mean(float(row["confidence"]) for row in wrong) if wrong else None,
        "high_confidence_error_rate": len(high_conf_wrong) / max(len(high_conf), 1) if high_conf else 0.0,
        "high_confidence_count": len(high_conf),
        "high_confidence_error_count": len(high_conf_wrong),
        "ece": _ece(with_confidence),
    }


def _ece(rows: list[dict[str, Any]], bins: int = 10) -> float:
    total = len(rows)
    if total == 0:
        return 0.0
    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = []
        for row in rows:
            confidence = float(row["confidence"])
            if index == bins - 1:
                in_bucket = low <= confidence <= high
            else:
                in_bucket = low <= confidence < high
            if in_bucket:
                bucket.append(row)
        if not bucket:
            continue
        acc = sum(1 for row in bucket if row["label_pass"]) / len(bucket)
        conf = mean(float(row["confidence"]) for row in bucket)
        ece += (len(bucket) / total) * abs(acc - conf)
    return ece


def _sufficiency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, dict[str, int]] = {}
    for label in LABELS:
        label_rows = [row for row in rows if row["expected_label"] == label]
        by_label[label] = {
            "count": len(label_rows),
            "sufficient_true": sum(1 for row in label_rows if row.get("evidence_sufficient") is True),
            "sufficient_false": sum(1 for row in label_rows if row.get("evidence_sufficient") is False),
            "sufficient_unknown": sum(1 for row in label_rows if row.get("evidence_sufficient") is None),
        }
    return {
        "counts": {
            "true": sum(1 for row in rows if row.get("evidence_sufficient") is True),
            "false": sum(1 for row in rows if row.get("evidence_sufficient") is False),
            "unknown": sum(1 for row in rows if row.get("evidence_sufficient") is None),
        },
        "by_expected_label": by_label,
    }


def _ordered_counts(counter: Counter[str | None]) -> dict[str, int]:
    return {str(label): int(counter.get(label, 0)) for label in (*LABELS, None)}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "sufficient"}:
        return True
    if normalized in {"false", "no", "0", "insufficient"}:
        return False
    return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    confidence = report["confidence_metrics"]
    lines = [
        "# PubMedQA Direct Judge",
        "",
        f"- Model: `{report['model']}`",
        f"- Evidence mode: `{report['evidence_mode']}`",
        f"- Prompt style: `{report['prompt_style']}`",
        f"- Cases: {report['case_count']}",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Macro F1: {metrics['macro_f1']:.3f}",
        f"- ECE: {_format_optional(confidence.get('ece'))}",
        f"- High-confidence error rate: {_format_optional(confidence.get('high_confidence_error_rate'))}",
        "",
        "| Label | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in metrics["per_label"].items():
        lines.append(
            f"| {label} | {values['support']} | {values['precision']:.3f} | "
            f"{values['recall']:.3f} | {values['f1']:.3f} |"
        )
    lines.extend(["", "## Confusion Matrix", "", "| True \\ Pred | yes | no | maybe | none |", "|---|---:|---:|---:|---:|"])
    for true_label, row in metrics["confusion_matrix"].items():
        lines.append(
            f"| {true_label} | {row.get('yes', 0)} | {row.get('no', 0)} | "
            f"{row.get('maybe', 0)} | {row.get('none', 0)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
