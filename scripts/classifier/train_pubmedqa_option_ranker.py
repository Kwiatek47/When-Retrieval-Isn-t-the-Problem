from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
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


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    evidence: str
    label_id: int
    source_dataset: str = ""


def main() -> None:
    args = _parse_args()
    _seed_everything(args.seed)

    try:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception as exc:
        raise RuntimeError("Option-ranker training requires torch and transformers.") from exc

    train_examples = _load_examples(args.train_jsonl)
    dev_examples = _load_examples(args.dev_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = _OptionRankerDataset(train_examples, tokenizer=tokenizer, max_length=args.max_length)
    dev_dataset = _OptionRankerDataset(dev_examples, tokenizer=tokenizer, max_length=args.max_length)
    device = _resolve_device(torch, args.device)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    sampler = (
        WeightedRandomSampler(
            weights=_sample_weights(train_examples),
            num_samples=len(train_examples),
            replacement=True,
        )
        if args.balanced_sampling
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) * args.epochs) // args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16" and device.type == "cuda")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "run_config.json", _jsonable_args(args))
    best_dir = args.out_dir / "best"
    best_metric = -1.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            torch=torch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp=args.amp,
            gradient_accumulation=args.gradient_accumulation,
            log_every=args.log_every,
        )
        dev_metrics = _evaluate(torch=torch, model=model, loader=dev_loader, device=device, amp=args.amp)
        metric = _selection_metric(dev_metrics, args.selection_metric)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"dev_accuracy={dev_metrics['accuracy']:.4f} dev_macro_f1={dev_metrics['macro_f1']:.4f} "
            f"selection_{args.selection_metric}={metric:.4f}"
        )

        best_updated = metric > best_metric
        if best_updated:
            best_metric = metric
            bad_epochs = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            _write_json(best_dir / "label_map.json", LABEL_TO_ID)
            _write_json(best_dir / "option_ranker_config.json", _jsonable_args(args))
            _write_json(best_dir / "dev_metrics.json", dev_metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stopping_patience:
                print(f"Early stopping after epoch={epoch}.")
                break

        _append_jsonl(
            args.out_dir / "train_log.jsonl",
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_metrics": dev_metrics,
                "selection_metric": args.selection_metric,
                "selection_value": metric,
                "best_metric": best_metric,
                "best_updated": best_updated,
            },
        )

    model = AutoModelForSequenceClassification.from_pretrained(best_dir)
    model.to(device)
    final_metrics = _evaluate(torch=torch, model=model, loader=dev_loader, device=device, amp=args.amp)
    _write_json(best_dir / "dev_metrics_final.json", final_metrics)
    print(f"Wrote best option-ranker: {best_dir}")
    print(json.dumps({"dev_metrics": final_metrics}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PubMedQA option-ranker evidence decision model.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="michiyasunaga/BioLinkBERT-large")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1", "accuracy", "accuracy_macro_f1", "balanced_accuracy"),
        default="accuracy_macro_f1",
    )
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-mismatched-sizes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            label = str(item.get("label") or "").lower()
            label_id = int(item.get("label_id", LABEL_TO_ID.get(label, -1)))
            if label_id not in ID_TO_LABEL:
                raise RuntimeError(f"Invalid label/label_id in {path}: {item.get('label')!r}/{item.get('label_id')!r}")
            examples.append(
                Example(
                    id=str(item["id"]),
                    question=_strip_instruction(str(item["question"])),
                    evidence=str(item["evidence"]),
                    label_id=label_id,
                    source_dataset=str(item.get("source_dataset") or ""),
                )
            )
    if not examples:
        raise RuntimeError(f"No examples loaded from {path}.")
    return examples


class _OptionRankerDataset:
    def __init__(self, examples: list[Example], *, tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        first_sequences: list[str] = []
        second_sequences: list[str] = []
        for example in examples:
            for option in LABELS:
                first_sequences.append(_option_query(example.question, option))
                second_sequences.append(example.evidence)
        self.encoded = tokenizer(
            first_sequences,
            second_sequences,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        import torch

        self.labels = torch.tensor([example.label_id for example in examples], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.examples)

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


def _train_epoch(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    device: Any,
    amp: str,
    gradient_accumulation: int,
    log_every: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        labels = batch.pop("labels")
        flat_batch = _flatten_option_batch(batch)
        with _autocast(torch, device=device, amp=amp):
            logits = model(**flat_batch).logits.view(labels.shape[0], len(LABELS))
            raw_loss = torch.nn.functional.cross_entropy(logits, labels)
        loss = raw_loss / gradient_accumulation
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        losses.append(float(raw_loss.detach().cpu()))
        if step % gradient_accumulation == 0 or step == len(loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if log_every > 0 and step % log_every == 0:
            print(f"  train_step={step}/{len(loader)} loss={sum(losses[-log_every:]) / min(len(losses), log_every):.4f}")
    return sum(losses) / max(len(losses), 1)


def _evaluate(*, torch: Any, model: Any, loader: Any, device: Any, amp: str) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    score_rows: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            label_tensor = batch.pop("labels")
            flat_batch = _flatten_option_batch(batch)
            with _autocast(torch, device=device, amp=amp):
                scores = model(**flat_batch).logits.view(label_tensor.shape[0], len(LABELS))
            probs = torch.softmax(scores, dim=-1)
            conf, pred = probs.max(dim=-1)
            labels.extend(int(value) for value in label_tensor.detach().cpu())
            predictions.extend(int(value) for value in pred.detach().cpu())
            confidences.extend(float(value) for value in conf.detach().cpu())
            score_rows.extend([[float(value) for value in row] for row in scores.detach().cpu()])
    metrics = _classification_metrics(labels=labels, predictions=predictions, confidences=confidences)
    metrics["mean_scores"] = {
        label: sum(row[index] for row in score_rows) / max(len(score_rows), 1)
        for index, label in enumerate(LABELS)
    }
    return metrics


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
    predicted_counts = Counter(ID_TO_LABEL[prediction] for prediction in predictions)
    return {
        "accuracy": correct / max(len(labels), 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
        "per_label": per_label,
        "confusion_matrix": confusion,
        "predicted_labels": {label: int(predicted_counts.get(label, 0)) for label in LABELS},
        "ece": _ece(labels=labels, predictions=predictions, confidences=confidences),
    }


def _selection_metric(metrics: dict[str, Any], name: str) -> float:
    if name == "macro_f1":
        return float(metrics["macro_f1"])
    if name == "accuracy":
        return float(metrics["accuracy"])
    if name == "accuracy_macro_f1":
        return (float(metrics["accuracy"]) + float(metrics["macro_f1"])) / 2
    if name == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    raise ValueError(f"Unknown selection metric: {name}")


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


def _sample_weights(examples: list[Example]) -> list[float]:
    counts = Counter(example.label_id for example in examples)
    return [1.0 / max(counts[example.label_id], 1) for example in examples]


def _strip_instruction(question: str) -> str:
    question = re.sub(r"^Answer yes, no, or maybe based on retrieved evidence:\s*", "", question).strip()
    return re.sub(r"\s+", " ", question)


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    output = vars(args).copy()
    for key, value in list(output.items()):
        if isinstance(value, Path):
            output[key] = str(value)
    output["labels"] = list(LABELS)
    output["option_definitions"] = OPTION_DEFINITIONS
    output["architecture"] = "question+candidate option paired with abstract evidence; scalar score per option; argmax over yes/no/maybe"
    return output


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
