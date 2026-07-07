from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


LABELS = ("yes", "no", "maybe")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

OPTION_DEFINITIONS = {
    "yes": "The evidence directly supports the proposition in the question.",
    "no": "The evidence directly refutes the proposition or shows no relevant effect.",
    "maybe": "The evidence is inconclusive, mixed, indirect, limited, or insufficient.",
}


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args)
    if args.max_cases:
        cases = cases[: args.max_cases]

    report: dict[str, Any] = {
        "kind": "pubmedqa_option_ranker_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset or args.jsonl),
        "corpus": str(args.corpus) if args.corpus else "",
        "model_dir": str(args.model_dir),
        "case_count": len(cases),
        "labels": _ordered_counts(Counter(case["label"] for case in cases)),
        "majority_baseline": _baseline_report(cases),
        "option_ranker": _ranker_report(cases, args=args),
    }

    _write_json(args.json_out, report)
    _write_markdown(args.md_out, report)
    print(f"Wrote option-ranker eval JSON: {args.json_out}")
    print(f"Wrote option-ranker eval Markdown: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_report_dir = Path("reports") / "classifier"
    parser = argparse.ArgumentParser(description="Evaluate a PubMedQA option-ranker yes/no/maybe model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=Path, help="PubMedQA eval JSON with relevant_pmids.")
    group.add_argument("--jsonl", type=Path, help="Prepared classifier JSONL split.")
    parser.add_argument("--corpus", type=Path, default=Path("data/benchmarks/pubmedqa/official_pqal_test/corpus.json"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=default_report_dir / f"option_ranker_eval_{timestamp}.json")
    parser.add_argument("--md-out", type=Path, default=default_report_dir / f"option_ranker_eval_{timestamp}.md")
    return parser.parse_args()


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.jsonl:
        return _load_jsonl_cases(args.jsonl)
    assert args.dataset is not None
    return _load_eval_cases(args.dataset, corpus_path=args.corpus)


def _load_jsonl_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            label = str(item.get("label") or "").lower()
            label_id = int(item.get("label_id", LABEL_TO_ID.get(label, -1)))
            if label_id not in ID_TO_LABEL:
                raise RuntimeError(f"Invalid label/label_id in {path}: {label!r}/{item.get('label_id')!r}")
            cases.append(
                {
                    "id": str(item["id"]),
                    "pmid": str(item.get("pmid") or ""),
                    "question": _strip_instruction(str(item["question"])),
                    "evidence": str(item["evidence"]),
                    "label": ID_TO_LABEL[label_id],
                }
            )
    if not cases:
        raise RuntimeError(f"No cases loaded from {path}.")
    return cases


def _load_eval_cases(dataset_path: Path, *, corpus_path: Path) -> list[dict[str, Any]]:
    corpus_by_pmid = _corpus_by_pmid(corpus_path)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = []
    for item in data:
        pmid = str((item.get("relevant_pmids") or [""])[0])
        corpus_item = corpus_by_pmid.get(pmid)
        if corpus_item is None:
            raise RuntimeError(f"Missing corpus item for PMID {pmid}.")
        question = str(item.get("benchmark_question") or item.get("question") or "").strip()
        cases.append(
            {
                "id": str(item["id"]),
                "pmid": pmid,
                "question": _strip_instruction(question),
                "evidence": _paper_evidence_text(str(corpus_item.get("content") or "")),
                "label": str(item["expected_label"]).lower(),
            }
        )
    if not cases:
        raise RuntimeError(f"No cases loaded from {dataset_path}.")
    return cases


def _ranker_report(cases: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Option-ranker eval requires torch and transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = _resolve_device(torch, args.device)
    model.to(device)
    model.eval()

    dataset = _OptionEvalDataset(cases, tokenizer=tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    labels: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    rows: list[dict[str, Any]] = []
    cursor = 0
    temperature = max(float(args.temperature), 1e-6)

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            label_tensor = batch.pop("labels")
            flat_batch = _flatten_option_batch(batch)
            with _autocast(torch, device=device, amp=args.amp):
                scores = model(**flat_batch).logits.view(label_tensor.shape[0], len(LABELS))
            probs = torch.softmax(scores / temperature, dim=-1).detach().cpu()
            scores_cpu = scores.detach().cpu()
            for row_index, probs_row in enumerate(probs):
                pred_id = int(probs_row.argmax().item())
                gold_id = int(label_tensor[row_index].detach().cpu())
                labels.append(gold_id)
                predictions.append(pred_id)
                confidences.append(float(probs_row[pred_id]))
                case = cases[cursor]
                rows.append(
                    {
                        "id": case["id"],
                        "pmid": case.get("pmid", ""),
                        "expected_label": ID_TO_LABEL[gold_id],
                        "predicted_label": ID_TO_LABEL[pred_id],
                        "confidence": float(probs_row[pred_id]),
                        "scores": {label: float(scores_cpu[row_index][index]) for index, label in enumerate(LABELS)},
                        "probabilities": {label: float(probs_row[index]) for index, label in enumerate(LABELS)},
                        "pass": gold_id == pred_id,
                    }
                )
                cursor += 1

    return {
        "metrics": _classification_metrics(labels=labels, predictions=predictions, confidences=confidences),
        "predicted_labels": _ordered_counts(Counter(ID_TO_LABEL[prediction] for prediction in predictions)),
        "temperature": temperature,
        "cases": rows,
    }


class _OptionEvalDataset:
    def __init__(self, cases: list[dict[str, Any]], *, tokenizer: Any, max_length: int) -> None:
        first_sequences: list[str] = []
        second_sequences: list[str] = []
        for case in cases:
            for option in LABELS:
                first_sequences.append(_option_query(case["question"], option))
                second_sequences.append(case["evidence"])
        self.encoded = tokenizer(
            first_sequences,
            second_sequences,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        import torch

        self.labels = torch.tensor([LABEL_TO_ID[case["label"]] for case in cases], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = index * len(LABELS)
        end = start + len(LABELS)
        item = {key: value[start:end] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        return item


def _option_query(question: str, option: str) -> str:
    return (
        f"Question: {question}\n"
        f"Candidate answer: {option}\n"
        f"Candidate meaning: {OPTION_DEFINITIONS[option]}"
    )


def _flatten_option_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.view(value.shape[0] * value.shape[1], *value.shape[2:]) for key, value in batch.items()}


def _classification_metrics(*, labels: list[int], predictions: list[int], confidences: list[float]) -> dict[str, Any]:
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    for gold, pred in zip(labels, predictions):
        confusion[gold][pred] += 1

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, label in ID_TO_LABEL.items():
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(LABELS)) if row != index)
        fn = sum(confusion[index][col] for col in range(len(LABELS)) if col != index)
        support = sum(confusion[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        per_label[label] = {
            "support": support,
            "accuracy": tp / support if support else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(int(gold == pred) for gold, pred in zip(labels, predictions))
    return {
        "accuracy": correct / max(len(labels), 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
        "per_label": per_label,
        "confusion_matrix": confusion,
        "ece": _ece(labels=labels, predictions=predictions, confidences=confidences),
    }


def _baseline_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(case["label"] for case in cases)
    majority = max(LABELS, key=lambda label: counts.get(label, 0))
    labels = [LABEL_TO_ID[case["label"]] for case in cases]
    predictions = [LABEL_TO_ID[majority] for _case in cases]
    return {
        "label": majority,
        "metrics": _classification_metrics(labels=labels, predictions=predictions, confidences=[1.0] * len(cases)),
    }


def _ece(*, labels: list[int], predictions: list[int], confidences: list[float], bins: int = 10) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if lower < confidence <= upper or (bin_index == 0 and confidence == 0.0)
        ]
        if not indices:
            continue
        accuracy = sum(int(labels[index] == predictions[index]) for index in indices) / len(indices)
        confidence = sum(confidences[index] for index in indices) / len(indices)
        ece += len(indices) / total * abs(accuracy - confidence)
    return ece


def _corpus_by_pmid(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_pmid = {}
    for item in data:
        metadata = item.get("metadata") or {}
        pmid = str(metadata.get("pmid") or "")
        if pmid:
            by_pmid[pmid] = item
    return by_pmid


def _paper_evidence_text(content: str) -> str:
    marker = "Abstract context:"
    if marker in content:
        content = content.split(marker, 1)[1]
    content = re.sub(r"\bConclusion:\s*.*$", "", content, flags=re.I | re.S)
    return re.sub(r"\s+", " ", content).strip()


def _strip_instruction(question: str) -> str:
    question = re.sub(r"^Answer yes, no, or maybe based on retrieved evidence:\s*", "", question).strip()
    return re.sub(r"\s+", " ", question)


def _ordered_counts(counter: Counter[str]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in LABELS}


def _resolve_device(torch: Any, requested: str) -> Any:
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast(torch: Any, *, device: Any, amp: str) -> Any:
    enabled = amp != "off" and getattr(device, "type", "") == "cuda"
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PubMedQA Option-Ranker Eval",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Cases: {report['case_count']}",
        f"- Labels: `{report['labels']}`",
        f"- Model: `{report['model_dir']}`",
        "",
        "## Baselines",
        "",
        "| Method | Label | Accuracy | Macro F1 | Maybe Recall | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    majority = report["majority_baseline"]
    majority_metrics = majority["metrics"]
    lines.append(
        f"| majority | {majority['label']} | {majority_metrics['accuracy']:.3f} | "
        f"{majority_metrics['macro_f1']:.3f} | "
        f"{majority_metrics['per_label'].get('maybe', {}).get('recall', 0.0):.3f} | "
        f"{majority_metrics['ece']:.3f} |"
    )
    metrics = report["option_ranker"]["metrics"]
    lines.append(
        f"| option_ranker | - | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} | "
        f"{metrics['per_label'].get('maybe', {}).get('recall', 0.0):.3f} | {metrics['ece']:.3f} |"
    )
    lines.extend(
        [
            "",
            "## Per Label",
            "",
            "| Label | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, values in metrics["per_label"].items():
        lines.append(
            f"| {label} | {values['support']} | {values['precision']:.3f} | "
            f"{values['recall']:.3f} | {values['f1']:.3f} |"
        )
    lines.extend(["", "## Confusion Matrix", "", "`rows=true labels, columns=predicted labels yes/no/maybe`", ""])
    lines.append(f"`{metrics['confusion_matrix']}`")
    lines.extend(["", "## Predicted Labels", "", f"`{report['option_ranker']['predicted_labels']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
