from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
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
    train_dataset = _PairDataset(train_examples, tokenizer, args.max_length)
    dev_dataset = _PairDataset(dev_examples, tokenizer, args.max_length)
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
    if distributed:
        model = DistributedDataParallel(
            model,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
            metric = (dev_metrics["accuracy"] + dev_metrics["macro_f1"]) / 2
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"dev_accuracy={dev_metrics['accuracy']:.4f} dev_macro_f1={dev_metrics['macro_f1']:.4f}"
            )

            if metric > best_metric:
                best_metric = metric
                bad_epochs = 0
                best_dir.mkdir(parents=True, exist_ok=True)
                _unwrap_model(model).save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                _write_json(best_dir / "label_map.json", LABEL_TO_ID)
                _write_json(best_dir / "training_config.json", _jsonable_args(args, distributed=distributed))
                _write_json(best_dir / "dev_metrics.json", dev_metrics)
            else:
                bad_epochs += 1
                if bad_epochs >= args.early_stopping_patience:
                    print(f"Early stopping after epoch={epoch}.")
                    should_stop = True

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
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="off")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
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
                )
            )
    if not examples:
        raise RuntimeError(f"No examples loaded from {path}.")
    return examples


class _PairDataset:
    def __init__(self, examples: list[Example], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
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

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
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
    amp: str,
    scaler: Any,
    log_every: int,
    is_main: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    losses: list[float] = []
    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        labels = batch.pop("labels")
        with _autocast(torch, device=device, amp=amp):
            logits = model(**batch).logits
            raw_loss = loss_fn(logits, labels)
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
            with _autocast(torch, device=device, amp=amp):
                logits = model(**batch).logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            predictions.extend(int(value) for value in pred.detach().cpu())
            labels.extend(int(value) for value in label_tensor.detach().cpu())
            confidences.extend(float(value) for value in conf.detach().cpu())

    return _classification_metrics(labels=labels, predictions=predictions, confidences=confidences)


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


def _fit_temperature(*, torch: Any, model: Any, loader: Any, device: Any, amp: str = "off") -> dict[str, float]:
    logits_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop("labels")
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
