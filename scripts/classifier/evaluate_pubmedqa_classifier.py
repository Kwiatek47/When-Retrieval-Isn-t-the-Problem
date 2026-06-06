from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys_path = str(PROJECT_ROOT)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from app.rag.evidence_classifier import LABELS


LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def main() -> None:
    args = _parse_args()
    cases = _load_eval_cases(args.dataset, corpus_path=args.corpus)
    if args.max_cases:
        cases = cases[: args.max_cases]

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "corpus": str(args.corpus),
        "case_count": len(cases),
        "labels": _ordered_counts(Counter(case["label"] for case in cases)),
        "baselines": {
            "majority": _baseline_report(cases, label=_majority_label(cases)),
        },
    }
    if args.model_dir:
        report["classifier"] = _classifier_report(cases, args=args)

    _write_json(args.json_out, report)
    _write_markdown(args.md_out, report)
    print(f"Wrote classifier eval JSON: {args.json_out}")
    print(f"Wrote classifier eval Markdown: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_report_dir = Path("reports") / "classifier"
    parser = argparse.ArgumentParser(description="Evaluate a yes/no/maybe evidence classifier without the RAG writer.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json"),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/benchmarks/pubmedqa/official_pqal_test/corpus.json"),
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--temperature-path", type=Path, default=None)
    parser.add_argument("--thresholds-path", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="off")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=default_report_dir / f"classifier_eval_{timestamp}.json")
    parser.add_argument("--md-out", type=Path, default=default_report_dir / f"classifier_eval_{timestamp}.md")
    return parser.parse_args()


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
        question = re.sub(r"^Answer yes, no, or maybe based on retrieved evidence:\s*", "", question).strip()
        evidence = _paper_evidence_text(str(corpus_item.get("content") or ""))
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


def _baseline_report(cases: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    labels = [LABEL_TO_ID[case["label"]] for case in cases]
    predictions = [LABEL_TO_ID[label] for _case in cases]
    confidences = [1.0 for _case in cases]
    return {
        "label": label,
        "metrics": _classification_metrics(labels=labels, predictions=predictions, confidences=confidences),
    }


def _classifier_report(cases: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Classifier eval requires torch and transformers.") from exc

    assert args.model_dir is not None
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = _resolve_device(torch, args.device)
    model.to(device)
    model.eval()
    temperature = _load_temperature(args.temperature_path or (args.model_dir / "calibration.json"))
    thresholds = _load_thresholds(args.thresholds_path or (args.model_dir / "decision_thresholds.json"))

    dataset = _EvalDataset(cases, tokenizer=tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    labels: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    rows: list[dict[str, Any]] = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            label_tensor = batch.pop("labels")
            with _autocast(torch, device=device, amp=args.amp):
                logits = model(**batch).logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            for row_index, row in enumerate(probs):
                probabilities = {ID_TO_LABEL[index]: float(row[index]) for index in range(len(LABELS))}
                pred_label = _select_label(probabilities, thresholds=thresholds)
                gold_id = int(label_tensor[row_index].detach().cpu())
                pred_id = LABEL_TO_ID[pred_label]
                labels.append(gold_id)
                predictions.append(pred_id)
                confidences.append(probabilities[pred_label])
                case = cases[cursor]
                rows.append(
                    {
                        "id": case["id"],
                        "pmid": case["pmid"],
                        "expected_label": ID_TO_LABEL[gold_id],
                        "predicted_label": pred_label,
                        "confidence": probabilities[pred_label],
                        "probabilities": probabilities,
                        "pass": gold_id == pred_id,
                    }
                )
                cursor += 1

    return {
        "model_dir": str(args.model_dir),
        "temperature": temperature,
        "thresholds": thresholds,
        "metrics": _classification_metrics(labels=labels, predictions=predictions, confidences=confidences),
        "predicted_labels": _ordered_counts(Counter(ID_TO_LABEL[prediction] for prediction in predictions)),
        "cases": rows,
    }


class _EvalDataset:
    def __init__(self, cases: list[dict[str, Any]], *, tokenizer: Any, max_length: int) -> None:
        self.cases = cases
        self.encoded = tokenizer(
            [case["question"] for case in cases],
            [case["evidence"] for case in cases],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        import torch

        self.labels = torch.tensor([LABEL_TO_ID[case["label"]] for case in cases], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        return item


def _classification_metrics(*, labels: list[int], predictions: list[int], confidences: list[float]) -> dict[str, Any]:
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    for gold, pred in zip(labels, predictions):
        confusion[gold][pred] += 1

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for index, label in ID_TO_LABEL.items():
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(LABELS)) if row != index)
        fn = sum(confusion[index][col] for col in range(len(LABELS)) if col != index)
        support = sum(confusion[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(int(gold == pred) for gold, pred in zip(labels, predictions))
    return {
        "accuracy": correct / max(len(labels), 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "per_label": per_label,
        "confusion_matrix": confusion,
        "ece": _ece(labels=labels, predictions=predictions, confidences=confidences),
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


def _load_temperature(path: Path) -> float:
    if not path.exists():
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return max(float(data.get("temperature", 1.0)), 1e-6)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 1.0


def _load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("thresholds", data)
    return {label: max(float(raw[label]), 1e-6) for label in LABELS if label in raw}


def _select_label(probabilities: dict[str, float], *, thresholds: dict[str, float]) -> str:
    if len(thresholds) != len(LABELS):
        return max(probabilities, key=probabilities.get)
    return max(
        probabilities,
        key=lambda label: probabilities[label] / max(thresholds.get(label, 1.0), 1e-6),
    )


def _majority_label(cases: list[dict[str, Any]]) -> str:
    counts = Counter(case["label"] for case in cases)
    return max(LABELS, key=lambda label: counts.get(label, 0))


def _ordered_counts(counter: Counter[str]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in LABELS}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PubMedQA Classifier Eval",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Cases: {report['case_count']}",
        f"- Labels: `{report['labels']}`",
        "",
        "## Baselines",
        "",
        "| Method | Label | Accuracy | Macro F1 | ECE |",
        "|---|---|---:|---:|---:|",
    ]
    majority = report["baselines"]["majority"]
    majority_metrics = majority["metrics"]
    lines.append(
        f"| majority | {majority['label']} | {majority_metrics['accuracy']:.3f} | "
        f"{majority_metrics['macro_f1']:.3f} | {majority_metrics['ece']:.3f} |"
    )
    if "classifier" in report:
        metrics = report["classifier"]["metrics"]
        lines.append(
            f"| classifier | - | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} | {metrics['ece']:.3f} |"
        )
        lines.extend(["", "## Classifier Per Label", "", "| Label | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"])
        for label, values in metrics["per_label"].items():
            lines.append(
                f"| {label} | {values['support']} | {values['precision']:.3f} | "
                f"{values['recall']:.3f} | {values['f1']:.3f} |"
            )
        lines.extend(["", "## Classifier Confusion Matrix", "", "`rows=true labels, columns=predicted labels yes/no/maybe`", ""])
        lines.append(f"`{metrics['confusion_matrix']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
