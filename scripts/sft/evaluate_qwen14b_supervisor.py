#!/usr/bin/env python3
"""Evaluate base or fine-tuned Qwen supervisor outputs on leakage-safe splits."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.backends import OllamaInferenceBackend
from app.agents.models import SupervisorDirectorOutput, SupervisorModerationOutput
from app.schemas import ChatMessage


LABELS = ("yes", "no", "maybe")


def _extract_json(raw: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", (raw or "").strip(), flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_director_metrics(
    rows: list[dict[str, Any]],
    predictions: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")
    confusion = {gold: {pred: 0 for pred in (*LABELS, "invalid")} for gold in LABELS}
    valid_json = 0
    valid_schema = 0
    correct = 0
    details: list[dict[str, Any]] = []
    for row, raw in zip(rows, predictions):
        gold = str(row.get("gold_label") or "").lower()
        data = _extract_json(raw)
        if data is not None:
            valid_json += 1
        predicted = "invalid"
        error = None
        if data is not None:
            try:
                output = SupervisorDirectorOutput.model_validate(data)
                predicted = output.final_label
                valid_schema += 1
            except Exception as exc:
                error = str(exc)
        if gold in LABELS:
            confusion[gold][predicted] += 1
            correct += int(predicted == gold)
        details.append(
            {
                "id": row.get("id"),
                "gold_label": gold,
                "predicted_label": predicted,
                "valid_json": data is not None,
                "schema_error": error,
                "raw": raw,
            }
        )

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = confusion[label][label]
        fn = sum(confusion[label][pred] for pred in (*LABELS, "invalid") if pred != label)
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
        f1_values.append(f1)
    count = len(rows)
    metrics = {
        "count": count,
        "accuracy": _safe_ratio(correct, count),
        "macro_f1": sum(f1_values) / len(f1_values),
        "json_validity": _safe_ratio(valid_json, count),
        "pydantic_validity": _safe_ratio(valid_schema, count),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }
    return metrics, details


def compute_moderator_metrics(
    rows: list[dict[str, Any]],
    predictions: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")
    valid_json = 0
    valid_schema = 0
    author_correct = 0
    instructions_complete = 0
    conflicts_present = 0
    details: list[dict[str, Any]] = []
    for row, raw in zip(rows, predictions):
        data = _extract_json(raw)
        if data is not None:
            valid_json += 1
        output = None
        error = None
        if data is not None:
            try:
                output = SupervisorModerationOutput.model_validate(data)
                valid_schema += 1
            except Exception as exc:
                error = str(exc)
        if output is not None:
            author_correct += int(output.author_conclusion == row.get("gold_label"))
            instructions_complete += int(bool(output.round_instructions))
            conflicts_present += int(bool(output.contradictions))
        details.append(
            {
                "id": row.get("id"),
                "gold_label": row.get("gold_label"),
                "valid_json": data is not None,
                "schema_error": error,
                "raw": raw,
            }
        )
    count = len(rows)
    metrics = {
        "count": count,
        "json_validity": _safe_ratio(valid_json, count),
        "pydantic_validity": _safe_ratio(valid_schema, count),
        "author_conclusion_accuracy": _safe_ratio(author_correct, count),
        "instruction_completeness": _safe_ratio(instructions_complete, count),
        "contradiction_presence_rate": _safe_ratio(conflicts_present, count),
    }
    return metrics, details


def checkpoint_qualifies(
    candidate: dict[str, float],
    *,
    baseline: dict[str, float],
    baseline_moderator_pydantic_validity: float,
    accuracy_allowed_drop: float = 0.01,
) -> bool:
    return bool(
        candidate.get("macro_f1", 0.0) > baseline.get("macro_f1", 0.0)
        and candidate.get("accuracy", 0.0)
        >= baseline.get("accuracy", 0.0) - accuracy_allowed_drop
        and candidate.get("json_validity", 0.0) >= 0.99
        and candidate.get("moderator_pydantic_validity", 0.0)
        >= baseline_moderator_pydantic_validity
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


async def _ollama_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    base_url: str,
    num_predict: int,
) -> list[str]:
    from app.core.config import get_settings
    from app.providers.ollama import OllamaProvider

    settings = get_settings()
    provider = OllamaProvider(
        base_url=base_url.rstrip("/"),
        timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
        num_predict=num_predict,
        num_ctx=8192,
    )
    backend = OllamaInferenceBackend(provider, model=model_name, temperature=0.0)
    predictions = []
    for index, row in enumerate(rows, start=1):
        messages = [
            ChatMessage(role=item["role"], content=item["content"])
            for item in row["messages"][:-1]
        ]
        predictions.append(await backend.complete(messages, temperature=0.0))
        print(f"[{index}/{len(rows)}] id={row.get('id')}")
    return predictions


def _hf_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    max_seq_length: int,
    max_new_tokens: int,
) -> list[str]:
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    predictions: list[str] = []
    for index, row in enumerate(rows, start=1):
        input_ids = tokenizer.apply_chat_template(
            row["messages"][:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        answer_ids = generated[0, input_ids.shape[-1] :]
        predictions.append(tokenizer.decode(answer_ids, skip_special_tokens=True).strip())
        print(f"[{index}/{len(rows)}] id={row.get('id')}")
    return predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="HF/Unsloth base model or adapter path.")
    source.add_argument("--ollama-model")
    source.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11437")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = _read_jsonl(args.dataset)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if args.predictions_jsonl:
        prediction_rows = _read_jsonl(args.predictions_jsonl)
        predictions = [str(item.get("raw") or item.get("prediction") or "") for item in prediction_rows]
    elif args.ollama_model:
        predictions = asyncio.run(
            _ollama_predictions(
                rows,
                model_name=args.ollama_model,
                base_url=args.ollama_base_url,
                num_predict=args.max_new_tokens,
            )
        )
    else:
        predictions = _hf_predictions(
            rows,
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
        )
    task_types = {str(row.get("task_type") or "") for row in rows}
    if task_types == {"director"}:
        metrics, details = compute_director_metrics(rows, predictions)
    elif task_types == {"moderator"}:
        metrics, details = compute_moderator_metrics(rows, predictions)
    else:
        raise SystemExit(f"Dataset must contain exactly one task type, found: {sorted(task_types)}")
    report = {
        "dataset": str(args.dataset),
        "model": args.model or args.ollama_model or str(args.predictions_jsonl),
        "task_type": next(iter(task_types)),
        "metrics": metrics,
        "cases": details,
    }
    from app.agents.prompt_versioning import snapshot_prompts_for_run

    prompt_snapshot = snapshot_prompts_for_run(
        args.output.parent,
        run_label=args.output.stem,
        script="scripts/sft/evaluate_qwen14b_supervisor.py",
        extra_meta={
            "dataset": str(args.dataset),
            "model": report["model"],
            "task_type": report["task_type"],
        },
    )
    report["prompt_versioning"] = {
        "prompt_version": prompt_snapshot["prompt_version"],
        "prompt_sha256": prompt_snapshot["prompt_sha256"],
        "prompts_py_sha256": prompt_snapshot.get("prompts_py_sha256"),
        "captured_at": prompt_snapshot.get("captured_at"),
        "snapshot_path": prompt_snapshot.get("snapshot_path"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
