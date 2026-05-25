#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request

from tqdm import tqdm


LETTER_RE = re.compile(r"\b([A-G])\b")
CORRECT_RE = re.compile(r"correct answer\s*[:\-]?\s*([A-G])", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MCQ accuracy on prepared MedQA chat JSONL.")
    parser.add_argument("--dataset", default="data/sft/medqa/test.jsonl")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api/chat")
    parser.add_argument("--model", default="medgemma")
    parser.add_argument("--prompt-version", default="v3")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--output-dir", default="eval/reports")
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Write checkpoint every N evaluated samples (0 disables periodic checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint JSON path (default: <output-dir>/medqa_mcq_checkpoint.json).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    if not rows:
        raise ValueError(f"Dataset {path} is empty.")
    return rows


def call_chat(
    *,
    api_url: str,
    model: str,
    prompt_version: str,
    question: str,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> str:
    payload = {
        "model": model,
        "temperature": 0.0,
        "prompt_version": prompt_version,
        "messages": [{"role": "user", "content": question}],
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def format_http_error(exc: error.HTTPError) -> str:
        details = [f"HTTP {exc.code}"]
        try:
            raw = exc.read().decode("utf-8").strip()
            if raw:
                details.append(raw)
        except Exception:
            pass
        return ": ".join(details)

    attempt = 0
    while True:
        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            return str((parsed.get("message") or {}).get("content") or "").strip()
        except (error.HTTPError, error.URLError) as exc:
            attempt += 1
            is_http_error = isinstance(exc, error.HTTPError)
            status = getattr(exc, "code", None) if is_http_error else None
            retryable_statuses = {429, 500, 502, 503, 504}
            retryable = (status in retryable_statuses) if is_http_error else True
            if attempt > max_retries or not retryable:
                message = format_http_error(exc) if is_http_error else str(exc)
                raise RuntimeError(f"API request failed after {attempt} attempt(s): {message}") from exc
            delay = retry_backoff ** attempt
            time.sleep(delay)


def extract_user_question(sample: dict[str, Any]) -> str:
    for turn in sample.get("messages", []):
        if turn.get("role") == "user":
            content = str(turn.get("content") or "").strip()
            if content:
                return content
    return ""


def extract_gold_letter(sample: dict[str, Any]) -> str | None:
    meta = sample.get("meta") or {}
    raw = meta.get("gold_answer_letter")
    if raw is None:
        return None
    value = str(raw).strip().upper()
    return value if value in ["A", "B", "C", "D", "E", "F", "G"] else None


def parse_predicted_letter(answer_text: str) -> str | None:
    match = CORRECT_RE.search(answer_text)
    if match:
        return match.group(1).upper()
    generic = LETTER_RE.search(answer_text.upper())
    if generic:
        return generic.group(1).upper()
    return None


def build_payload(
    *,
    dataset_path: Path,
    args: argparse.Namespace,
    total_samples: int,
    rows: list[dict[str, Any]],
    correct: int,
    parsed: int,
    is_partial: bool,
) -> dict[str, Any]:
    completed = len(rows)
    accuracy_full = correct / max(1, total_samples)
    parsed_rate_full = parsed / max(1, total_samples)
    accuracy_completed = correct / max(1, completed)
    parsed_rate_completed = parsed / max(1, completed)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "api_url": args.api_url,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "num_samples": total_samples,
        "completed_samples": completed,
        "is_partial": is_partial,
        "accuracy": accuracy_full,
        "parsed_rate": parsed_rate_full,
        "accuracy_full": accuracy_full,
        "parsed_rate_full": parsed_rate_full,
        "accuracy_completed": accuracy_completed,
        "parsed_rate_completed": parsed_rate_completed,
        "correct_count": correct,
        "parsed_count": parsed,
        "results": rows,
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else output_dir / "medqa_mcq_checkpoint.json"
    )

    samples = load_dataset(dataset_path)

    correct = 0
    parsed = 0
    rows: list[dict[str, Any]] = []

    interrupted = False
    try:
        with tqdm(total=len(samples), desc="Evaluating", unit="q") as pbar:
            for sample in samples:
                user_question = extract_user_question(sample)
                gold = extract_gold_letter(sample)
                qid = str(sample.get("id") or "?")
                if not user_question or gold is None:
                    rows.append(
                        {
                            "id": qid,
                            "gold": gold,
                            "pred": None,
                            "correct": False,
                            "note": "missing user question or gold label",
                        }
                    )
                    pbar.update(1)
                    continue

                answer = call_chat(
                    api_url=args.api_url,
                    model=args.model,
                    prompt_version=args.prompt_version,
                    question=user_question,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                )
                pred = parse_predicted_letter(answer)
                is_correct = pred == gold
                if pred is not None:
                    parsed += 1
                if is_correct:
                    correct += 1
                rows.append(
                    {
                        "id": qid,
                        "gold": gold,
                        "pred": pred,
                        "correct": is_correct,
                        "answer": answer,
                    }
                )
                pbar.update(1)
                pbar.set_postfix(acc=f"{correct / max(1, len(rows)):.3f}")

                if args.save_every > 0 and len(rows) % args.save_every == 0:
                    checkpoint_payload = build_payload(
                        dataset_path=dataset_path,
                        args=args,
                        total_samples=len(samples),
                        rows=rows,
                        correct=correct,
                        parsed=parsed,
                        is_partial=True,
                    )
                    checkpoint_path.write_text(
                        json.dumps(checkpoint_payload, ensure_ascii=True, indent=2),
                        encoding="utf-8",
                    )
    except KeyboardInterrupt:
        interrupted = True

    payload = build_payload(
        dataset_path=dataset_path,
        args=args,
        total_samples=len(samples),
        rows=rows,
        correct=correct,
        parsed=parsed,
        is_partial=interrupted or len(rows) < len(samples),
    )
    if payload["is_partial"]:
        checkpoint_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = output_dir / f"medqa_mcq_report_{timestamp}.json"
    latest_json = output_dir / "medqa_mcq_latest.json"
    report_md = output_dir / f"medqa_mcq_report_{timestamp}.md"
    latest_md = output_dir / "medqa_mcq_latest.md"

    report_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = [
        "# MedQA MCQ Evaluation",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Model: `{args.model}`",
        f"- Prompt version: `{args.prompt_version}`",
        f"- Samples: `{len(samples)}`",
        f"- Completed: `{payload['completed_samples']}`",
        f"- Accuracy (full): `{payload['accuracy_full']:.4f}`",
        f"- Parsed answer rate (full): `{payload['parsed_rate_full']:.4f}`",
        f"- Accuracy (completed): `{payload['accuracy_completed']:.4f}`",
        f"- Parsed answer rate (completed): `{payload['parsed_rate_completed']:.4f}`",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "accuracy_full": payload["accuracy_full"],
                "accuracy_completed": payload["accuracy_completed"],
                "parsed_rate_full": payload["parsed_rate_full"],
                "parsed_rate_completed": payload["parsed_rate_completed"],
                "report": str(report_json),
                "checkpoint": str(checkpoint_path) if payload["is_partial"] else None,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
