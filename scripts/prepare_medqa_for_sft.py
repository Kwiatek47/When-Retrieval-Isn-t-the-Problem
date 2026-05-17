#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import random
import re
from typing import Any


LETTER_CANDIDATES = ["A", "B", "C", "D", "E", "F", "G"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MedQA-style JSONL MCQ files into chat-format JSONL for SFT."
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help=(
            "Glob pattern for source MedQA JSONL files, e.g. "
            "'data/raw/medqa/**/*.jsonl' (use quotes in shell)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data/sft/medqa",
        help="Output directory for train/dev/test JSONL files.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a cautious medical assistant. For multiple-choice items, return the correct "
            "option and a concise rationale. Add a short safety disclaimer that this is educational "
            "content and not medical advice."
        ),
        help="System prompt injected into each chat sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic fallback split.",
    )
    parser.add_argument(
        "--fallback-split",
        default="0.9,0.05,0.05",
        help="Used only when filenames do not expose train/dev/test split.",
    )
    parser.add_argument(
        "--include-rationale-if-present",
        action="store_true",
        help="If source row has rationale/explanation fields, append them in assistant answer.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def normalize_option(option: Any, idx: int) -> tuple[str, str]:
    letter = LETTER_CANDIDATES[idx]
    if isinstance(option, str):
        return letter, option.strip()
    if isinstance(option, dict):
        raw_label = option.get("label") or option.get("key") or option.get("id")
        if isinstance(raw_label, str) and raw_label.strip():
            maybe_letter = raw_label.strip().upper().replace(".", "")
            if maybe_letter in LETTER_CANDIDATES:
                letter = maybe_letter
        text = (
            option.get("text")
            or option.get("content")
            or option.get("option")
            or option.get("value")
            or ""
        )
        return letter, str(text).strip()
    return letter, str(option).strip()


def extract_options(row: dict[str, Any]) -> list[tuple[str, str]]:
    options_raw = (
        row.get("options")
        or row.get("choices")
        or row.get("answer_options")
        or row.get("candidates")
    )
    if options_raw is None:
        # Fallback for flattened schema like "A": "...", "B": "..."
        pairs: list[tuple[str, str]] = []
        for letter in LETTER_CANDIDATES:
            value = row.get(letter) or row.get(letter.lower())
            if isinstance(value, str) and value.strip():
                pairs.append((letter, value.strip()))
        return pairs

    if isinstance(options_raw, dict):
        pairs = []
        for key, value in options_raw.items():
            letter = str(key).strip().upper().replace(".", "")
            if letter not in LETTER_CANDIDATES:
                continue
            pairs.append((letter, str(value).strip()))
        return sorted(pairs, key=lambda x: x[0])

    if isinstance(options_raw, list):
        return [normalize_option(opt, i) for i, opt in enumerate(options_raw)]

    return []


def extract_question(row: dict[str, Any]) -> str:
    question = (
        row.get("question")
        or row.get("query")
        or row.get("stem")
        or row.get("question_text")
        or row.get("body")
        or ""
    )
    return str(question).strip()


def normalize_answer_letter(raw: Any, option_count: int) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if text in LETTER_CANDIDATES:
        return text
    if text.endswith(".") and text[:-1] in LETTER_CANDIDATES:
        return text[:-1]
    if re.fullmatch(r"\d+", text):
        idx = int(text)
        if 0 <= idx < option_count:
            return LETTER_CANDIDATES[idx]
        if 1 <= idx <= option_count:
            return LETTER_CANDIDATES[idx - 1]
    return None


def extract_answer_letter(row: dict[str, Any], options: list[tuple[str, str]]) -> str | None:
    raw_candidates = [
        row.get("answer_idx"),
        row.get("answer_id"),
        row.get("correct_idx"),
        row.get("correct_answer_idx"),
        row.get("answer"),
        row.get("label"),
        row.get("correct"),
        row.get("target"),
    ]
    for raw in raw_candidates:
        letter = normalize_answer_letter(raw, len(options))
        if letter is not None:
            return letter
    return None


def extract_rationale(row: dict[str, Any]) -> str | None:
    for key in ("rationale", "explanation", "analysis", "solution"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def render_user_prompt(question: str, options: list[tuple[str, str]]) -> str:
    lines = [f"Question: {question}", "", "Options:"]
    for letter, text in options:
        lines.append(f"{letter}) {text}")
    lines.append("")
    lines.append("Return the single best answer.")
    return "\n".join(lines)


def render_assistant_response(
    *,
    answer_letter: str,
    answer_text: str,
    rationale: str | None,
    include_rationale_if_present: bool,
) -> str:
    chunks = [f"Correct answer: {answer_letter}) {answer_text}"]
    if include_rationale_if_present and rationale:
        chunks.append(f"Rationale: {rationale}")
    chunks.append("Safety: This is educational content and not medical advice.")
    return "\n".join(chunks)


def infer_split_from_name(path: Path) -> str | None:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "valid" in name or "val" in name:
        return "dev"
    if "test" in name:
        return "test"
    return None


def infer_split_from_row(row: dict[str, Any]) -> str | None:
    candidates = [
        row.get("meta_info"),
        row.get("split"),
        row.get("subset"),
        row.get("source_split"),
        row.get("partition"),
    ]
    for value in candidates:
        if not isinstance(value, str):
            continue
        text = value.strip().lower()
        if not text:
            continue
        if "train" in text:
            return "train"
        if "dev" in text or "valid" in text or "val" in text:
            return "dev"
        if "test" in text:
            return "test"
    return None


def parse_split_ratio(raw: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--fallback-split must have exactly 3 comma-separated numbers.")
    train_ratio, dev_ratio, test_ratio = [float(p) for p in parts]
    total = train_ratio + dev_ratio + test_ratio
    if total <= 0:
        raise ValueError("Fallback split sum must be positive.")
    return train_ratio / total, dev_ratio / total, test_ratio / total


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = sorted(glob.glob(args.input_glob, recursive=True))
    if not matches:
        raise ValueError(f"No input files matched: {args.input_glob}")

    rng = random.Random(args.seed)
    split_ratio = parse_split_ratio(args.fallback_split)
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}

    skipped = 0
    converted = 0

    for filename in matches:
        src_path = Path(filename)
        rows = read_jsonl(src_path)
        inferred_split = infer_split_from_name(src_path)
        for i, row in enumerate(rows):
            question = extract_question(row)
            options = extract_options(row)
            answer_letter = extract_answer_letter(row, options)

            if not question or len(options) < 2 or answer_letter is None:
                skipped += 1
                continue

            option_map = {letter: text for letter, text in options}
            answer_text = option_map.get(answer_letter)
            if not answer_text:
                skipped += 1
                continue

            user_prompt = render_user_prompt(question, options)
            assistant_text = render_assistant_response(
                answer_letter=answer_letter,
                answer_text=answer_text,
                rationale=extract_rationale(row),
                include_rationale_if_present=args.include_rationale_if_present,
            )

            qid = row.get("id") or row.get("qid") or row.get("question_id") or f"{src_path.stem}_{i}"
            sample = {
                "id": str(qid),
                "messages": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ],
                "meta": {
                    "source_file": str(src_path),
                    "gold_answer_letter": answer_letter,
                    "gold_answer_text": answer_text,
                },
            }

            split = inferred_split or infer_split_from_row(row)
            if split is None:
                r = rng.random()
                train_r, dev_r, _ = split_ratio
                if r < train_r:
                    split = "train"
                elif r < train_r + dev_r:
                    split = "dev"
                else:
                    split = "test"

            buckets[split].append(sample)
            converted += 1

    for split_name, records in buckets.items():
        write_jsonl(out_dir / f"{split_name}.jsonl", records)

    manifest = {
        "input_glob": args.input_glob,
        "output_dir": str(out_dir),
        "converted": converted,
        "skipped": skipped,
        "split_counts": {k: len(v) for k, v in buckets.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
