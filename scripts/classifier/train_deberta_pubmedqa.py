from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
import random
import re
from typing import Any


LABELS = ("yes", "no", "maybe")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    evidence: str
    label_id: int
    long_answer: str = ""
    source_dataset: str = ""


def main() -> None:
    args = _parse_args()
    _seed_everything(args.seed)

    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torch.utils.data.distributed import DistributedSampler
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception as exc:
        raise RuntimeError(
            "Training requires torch and transformers. Install project dependencies first: "
            ".venv/bin/pip install -r requirements-dev.txt"
        ) from exc

    distributed = _is_distributed()
    rank = _rank()
    world_size = _world_size()
    local_rank = _local_rank()
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed DeBERTa training requires CUDA GPUs.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.ddp_backend)
    is_main = rank == 0

    train_examples = _load_examples(args.train_jsonl)
    dev_examples = _load_examples(args.dev_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    aux_vocab = (
        _build_aux_vocab(
            train_examples,
            vocab_size=args.aux_long_answer_bow_vocab_size,
            min_df=args.aux_long_answer_min_df,
        )
        if args.aux_long_answer_bow
        else []
    )
    train_dataset = _PairDataset(train_examples, tokenizer, args.max_length, aux_vocab=aux_vocab)
    dev_dataset = _PairDataset(dev_examples, tokenizer, args.max_length, aux_vocab=aux_vocab)
    train_class_counts = _class_counts(train_examples)

    device = torch.device("cuda", local_rank) if distributed else _resolve_device(torch, args.device)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={index: label for index, label in ID_TO_LABEL.items()},
        label2id=LABEL_TO_ID,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    aux_head = None
    if aux_vocab:
        hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise RuntimeError("LONG_ANSWER BOW auxiliary objective requires model.config.hidden_size.")
        aux_head = torch.nn.Linear(hidden_size, len(aux_vocab)).to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if aux_head is not None:
            aux_head = DistributedDataParallel(
                aux_head,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if args.balanced_sampling and is_main:
            print("balanced_sampling is disabled under DDP; use --class-weighted-loss for class balance.")
    elif args.balanced_sampling:
        sampler = WeightedRandomSampler(
            weights=_sample_weights(train_examples),
            num_samples=len(train_examples),
            replacement=True,
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
    optimizer_parameters = list(model.parameters())
    if aux_head is not None:
        optimizer_parameters.extend(aux_head.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16" and device.type == "cuda")
    total_steps = max(1, (len(train_loader) * args.epochs) // args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_metric = -1.0
    bad_epochs = 0
    best_dir = args.out_dir / "best"
    if is_main:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.out_dir / "run_config.json", _jsonable_args(args, distributed=distributed))
        if aux_vocab:
            _write_json(args.out_dir / "aux_long_answer_vocab.json", aux_vocab)
    class_weights = (
        torch.tensor(_class_weights(train_class_counts), dtype=torch.float, device=device)
        if args.class_weighted_loss
        else None
    )

    for epoch in range(1, args.epochs + 1):
        if distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        train_loss = _train_epoch(
            torch=torch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation=args.gradient_accumulation,
            class_weights=class_weights,
            focal_loss_gamma=args.focal_loss_gamma,
            aux_head=aux_head,
            aux_loss_weight=args.aux_long_answer_bow_weight,
            amp=args.amp,
            scaler=scaler,
            log_every=args.log_every,
            is_main=is_main,
        )
        if distributed:
            dist.barrier()

        should_stop = False
        if is_main:
            dev_metrics = _evaluate(torch=torch, model=_unwrap_model(model), loader=dev_loader, device=device, amp=args.amp)
            metric = _selection_metric(dev_metrics, args.selection_metric)
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"dev_accuracy={dev_metrics['accuracy']:.4f} dev_macro_f1={dev_metrics['macro_f1']:.4f} "
                f"selection_{args.selection_metric}={metric:.4f}"
            )

            if metric > best_metric:
                best_metric = metric
                bad_epochs = 0
                best_dir.mkdir(parents=True, exist_ok=True)
                _unwrap_model(model).save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                _write_json(best_dir / "label_map.json", LABEL_TO_ID)
                _write_json(best_dir / "training_config.json", _jsonable_args(args, distributed=distributed))
                if aux_vocab:
                    _write_json(best_dir / "aux_long_answer_vocab.json", aux_vocab)
                _write_json(best_dir / "dev_metrics.json", dev_metrics)
                best_updated = True
            else:
                bad_epochs += 1
                best_updated = False
                if bad_epochs >= args.early_stopping_patience:
                    print(f"Early stopping after epoch={epoch}.")
                    should_stop = True
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

        if distributed:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            if int(stop_tensor.item()) == 1:
                break
        elif should_stop:
            break

    if distributed:
        dist.barrier()
    if is_main:
        model = AutoModelForSequenceClassification.from_pretrained(best_dir)
        model.to(device)
        calibration = _fit_temperature(torch=torch, model=model, loader=dev_loader, device=device, amp=args.amp)
        _write_json(best_dir / "calibration.json", calibration)
        final_metrics = _evaluate(
            torch=torch,
            model=model,
            loader=dev_loader,
            device=device,
            temperature=calibration["temperature"],
            amp=args.amp,
        )
        _write_json(best_dir / "dev_metrics_calibrated.json", final_metrics)
        if args.tune_thresholds:
            labels, probabilities = _predict_probabilities(
                torch=torch,
                model=model,
                loader=dev_loader,
                device=device,
                temperature=calibration["temperature"],
                amp=args.amp,
            )
            threshold_result = _tune_decision_thresholds(
                labels=labels,
                probabilities=probabilities,
                metric=args.threshold_metric,
                minimum=args.threshold_min,
                maximum=args.threshold_max,
                step=args.threshold_step,
            )
            _write_json(best_dir / "decision_thresholds.json", threshold_result)
            _write_json(best_dir / "dev_metrics_threshold_tuned.json", threshold_result["metrics"])
        print(f"Wrote best model: {best_dir}")
        print(json.dumps({"calibration": calibration, "dev_metrics": final_metrics}, indent=2))
    if distributed:
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeBERTa PubMedQA yes/no/maybe evidence classifier.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/interim/classifier/pubmedqa_deberta/train.jsonl"))
    parser.add_argument("--dev-jsonl", type=Path, default=Path("data/interim/classifier/pubmedqa_deberta/dev.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/classifier/pubmedqa_deberta"))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weighted-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--focal-loss-gamma", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1", "accuracy", "accuracy_macro_f1", "balanced_accuracy"),
        default="macro_f1",
    )
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="off")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aux-long-answer-bow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aux-long-answer-bow-weight", type=float, default=0.10)
    parser.add_argument("--aux-long-answer-bow-vocab-size", type=int, default=512)
    parser.add_argument("--aux-long-answer-min-df", type=int, default=3)
    parser.add_argument("--tune-thresholds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--threshold-metric",
        choices=("macro_f1", "accuracy", "balanced_accuracy"),
        default="macro_f1",
    )
    parser.add_argument("--threshold-min", type=float, default=0.25)
    parser.add_argument("--threshold-max", type=float, default=0.75)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ddp-backend", default="nccl")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def _load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            label = item.get("label")
            label_id = int(item.get("label_id", LABEL_TO_ID.get(label, -1)))
            if label_id not in ID_TO_LABEL:
                raise RuntimeError(f"Invalid label_id={label_id} in {path}.")
            examples.append(
                Example(
                    id=str(item["id"]),
                    question=str(item["question"]),
                    evidence=str(item["evidence"]),
                    label_id=label_id,
                    long_answer=str(item.get("long_answer") or item.get("LONG_ANSWER") or ""),
                    source_dataset=str(item.get("source_dataset") or ""),
                )
            )
    if not examples:
        raise RuntimeError(f"No examples loaded from {path}.")
    return examples


class _PairDataset:
    def __init__(self, examples: list[Example], tokenizer: Any, max_length: int, *, aux_vocab: list[str]) -> None:
        self.examples = examples
        self.aux_vocab = aux_vocab
        questions = [example.question for example in examples]
        evidences = [example.evidence for example in examples]
        self.encoded = tokenizer(
            questions,
            evidences,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        import torch

        self.labels = torch.tensor([example.label_id for example in examples], dtype=torch.long)
        self.aux_bow = (
            torch.tensor([_long_answer_bow(example.long_answer, aux_vocab) for example in examples], dtype=torch.float)
            if aux_vocab
            else None
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        if self.aux_bow is not None:
            item["aux_bow"] = self.aux_bow[index]
        return item


def _resolve_device(torch: Any, requested: str) -> Any:
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_distributed() -> bool:
    return _world_size() > 1


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def _class_counts(examples: list[Example]) -> Counter[int]:
    return Counter(example.label_id for example in examples)


def _class_weights(counts: Counter[int]) -> list[float]:
    total = sum(counts.values())
    weights: list[float] = []
    for label_id in range(len(LABELS)):
        count = max(counts.get(label_id, 0), 1)
        weights.append(total / (len(LABELS) * count))
    return weights


def _sample_weights(examples: list[Example]) -> list[float]:
    counts = _class_counts(examples)
    return [1.0 / max(counts[example.label_id], 1) for example in examples]


def _train_epoch(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    gradient_accumulation: int,
    class_weights: Any | None,
    focal_loss_gamma: float,
    aux_head: Any | None,
    aux_loss_weight: float,
    amp: str,
    scaler: Any,
    log_every: int,
    is_main: bool,
) -> float:
    model.train()
    if aux_head is not None:
        aux_head.train()
    optimizer.zero_grad(set_to_none=True)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    aux_loss_fn = torch.nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        labels = batch.pop("labels")
        aux_bow = batch.pop("aux_bow", None)
        with _autocast(torch, device=device, amp=amp):
            outputs = model(**batch, output_hidden_states=aux_head is not None)
            logits = outputs.logits
            raw_loss = _classification_loss(
                torch=torch,
                logits=logits,
                labels=labels,
                loss_fn=loss_fn,
                class_weights=class_weights,
                focal_loss_gamma=focal_loss_gamma,
            )
            if aux_head is not None and aux_bow is not None:
                cls_hidden = outputs.hidden_states[-1][:, 0]
                aux_logits = aux_head(cls_hidden)
                raw_loss = raw_loss + aux_loss_weight * aux_loss_fn(aux_logits, aux_bow)
        loss = raw_loss / gradient_accumulation
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        losses.append(float(raw_loss.detach().cpu()))
        if step % gradient_accumulation == 0 or step == len(loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_unwrap_model(model).parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if is_main and log_every > 0 and step % log_every == 0:
            print(f"  train_step={step}/{len(loader)} loss={sum(losses[-log_every:]) / min(len(losses), log_every):.4f}")
    return sum(losses) / max(len(losses), 1)


def _evaluate(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    temperature: float = 1.0,
    amp: str = "off",
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            label_tensor = batch.pop("labels")
            batch.pop("aux_bow", None)
            with _autocast(torch, device=device, amp=amp):
                logits = model(**batch).logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            predictions.extend(int(value) for value in pred.detach().cpu())
            labels.extend(int(value) for value in label_tensor.detach().cpu())
            confidences.extend(float(value) for value in conf.detach().cpu())

    return _classification_metrics(labels=labels, predictions=predictions, confidences=confidences)


def _classification_loss(
    *,
    torch: Any,
    logits: Any,
    labels: Any,
    loss_fn: Any,
    class_weights: Any | None,
    focal_loss_gamma: float,
) -> Any:
    if focal_loss_gamma <= 0:
        return loss_fn(logits, labels)
    ce = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** focal_loss_gamma) * ce).mean()


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
            "accuracy": tp / support if support else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(int(gold == pred) for gold, pred in zip(labels, predictions))
    return {
        "accuracy": correct / max(len(labels), 1),
        "macro_f1": sum(f1_values) / len(f1_values),
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


def _selection_metric(metrics: dict[str, Any], name: str) -> float:
    if name == "macro_f1":
        return float(metrics["macro_f1"])
    if name == "accuracy":
        return float(metrics["accuracy"])
    if name == "accuracy_macro_f1":
        return (float(metrics["accuracy"]) + float(metrics["macro_f1"])) / 2
    if name == "balanced_accuracy":
        per_label = metrics.get("per_label", {})
        recalls = [float(per_label.get(label, {}).get("recall", 0.0)) for label in LABELS]
        return sum(recalls) / max(len(recalls), 1)
    raise ValueError(f"Unknown selection metric: {name}")


def _predict_probabilities(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    temperature: float = 1.0,
    amp: str = "off",
) -> tuple[list[int], list[list[float]]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            label_tensor = batch.pop("labels")
            batch.pop("aux_bow", None)
            with _autocast(torch, device=device, amp=amp):
                logits = model(**batch).logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            labels.extend(int(value) for value in label_tensor.detach().cpu())
            probabilities.extend([[float(value) for value in row] for row in probs])
    return labels, probabilities


def _tune_decision_thresholds(
    *,
    labels: list[int],
    probabilities: list[list[float]],
    metric: str,
    minimum: float,
    maximum: float,
    step: float,
) -> dict[str, Any]:
    values = _threshold_grid(minimum=minimum, maximum=maximum, step=step)
    best_score = -1.0
    best_thresholds = [1.0 for _ in LABELS]
    best_predictions: list[int] = []
    best_confidences: list[float] = []
    for thresholds in product(values, repeat=len(LABELS)):
        predictions, confidences = _predict_with_thresholds(probabilities, list(thresholds))
        metrics = _classification_metrics(labels=labels, predictions=predictions, confidences=confidences)
        score = _selection_metric(metrics, metric if metric != "balanced_accuracy" else "balanced_accuracy")
        if score > best_score:
            best_score = score
            best_thresholds = list(thresholds)
            best_predictions = predictions
            best_confidences = confidences

    best_metrics = _classification_metrics(labels=labels, predictions=best_predictions, confidences=best_confidences)
    return {
        "method": "probability_divided_by_per_label_threshold",
        "metric": metric,
        "score": best_score,
        "thresholds": {label: best_thresholds[index] for index, label in enumerate(LABELS)},
        "metrics": best_metrics,
    }


def _threshold_grid(*, minimum: float, maximum: float, step: float) -> list[float]:
    values = []
    current = minimum
    while current <= maximum + 1e-9:
        values.append(round(current, 6))
        current += max(step, 1e-6)
    return values or [1.0]


def _predict_with_thresholds(
    probabilities: list[list[float]],
    thresholds: list[float],
) -> tuple[list[int], list[float]]:
    predictions = []
    confidences = []
    for row in probabilities:
        scores = [
            float(probability) / max(float(thresholds[index]), 1e-6)
            for index, probability in enumerate(row)
        ]
        prediction = max(range(len(scores)), key=lambda index: scores[index])
        predictions.append(prediction)
        confidences.append(float(row[prediction]))
    return predictions, confidences


def _build_aux_vocab(examples: list[Example], *, vocab_size: int, min_df: int) -> list[str]:
    document_frequency: Counter[str] = Counter()
    for example in examples:
        tokens = set(_long_answer_tokens(example.long_answer))
        document_frequency.update(tokens)
    candidates = [
        (token, count)
        for token, count in document_frequency.items()
        if count >= min_df and token not in _AUX_STOPWORDS
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [token for token, _count in candidates[: max(vocab_size, 0)]]


def _long_answer_bow(long_answer: str, vocab: list[str]) -> list[float]:
    if not vocab:
        return []
    tokens = set(_long_answer_tokens(long_answer))
    return [1.0 if token in tokens else 0.0 for token in vocab]


def _long_answer_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9]{2,}", text.lower())
        if token not in _AUX_STOPWORDS
    ]


_AUX_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "were",
    "was",
    "are",
    "our",
    "from",
    "have",
    "has",
    "had",
    "not",
    "but",
    "can",
    "may",
    "these",
    "those",
    "study",
    "results",
    "suggest",
    "suggests",
    "conclusion",
    "conclusions",
}


def _fit_temperature(*, torch: Any, model: Any, loader: Any, device: Any, amp: str = "off") -> dict[str, float]:
    logits_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop("labels")
            batch.pop("aux_bow", None)
            with _autocast(torch, device=device, amp=amp):
                logits = model(**batch).logits
            logits_list.append(logits)
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure() -> Any:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = loss_fn(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).detach().cpu()[0])
    return {"temperature": max(temperature, 1e-6)}


def _autocast(torch: Any, *, device: Any, amp: str) -> Any:
    enabled = amp != "off" and getattr(device, "type", "") == "cuda"
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def _jsonable_args(args: argparse.Namespace, *, distributed: bool = False) -> dict[str, Any]:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload["distributed"] = distributed
    payload["world_size"] = _world_size()
    return payload


if __name__ == "__main__":
    main()
