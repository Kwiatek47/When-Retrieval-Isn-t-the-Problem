#!/usr/bin/env python3
"""Merge a supervisor adapter, export Q4_K_M GGUF, and register it in Ollama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


def render_modelfile(gguf_path: Path, *, num_ctx: int = 8192) -> str:
    return (
        f"FROM {gguf_path.resolve()}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        "PARAMETER temperature 0\n"
        "PARAMETER stop <|im_end|>\n"
    )


def select_smoke_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one yes/no/maybe example for Director and Moderator."""
    selected: list[dict[str, Any]] = []
    for task_type in ("director", "moderator"):
        for label in ("yes", "no", "maybe"):
            match = next(
                (
                    row
                    for row in records
                    if row.get("task_type") == task_type
                    and row.get("gold_label") == label
                ),
                None,
            )
            if match is None:
                raise ValueError(f"Missing smoke record for {task_type}/{label}")
            selected.append(match)
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def export_gguf(
    *,
    adapter_path: Path,
    output_dir: Path,
    max_seq_length: int,
) -> Path:
    from unsloth import FastLanguageModel

    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )
    candidates = sorted(
        output_dir.rglob("*.gguf"),
        key=lambda path: ("q4_k_m" not in path.name.lower(), -path.stat().st_size),
    )
    if not candidates:
        raise RuntimeError(f"Unsloth did not create a GGUF file under {output_dir}")
    return candidates[0]


def register_ollama_model(
    *,
    gguf_path: Path,
    output_dir: Path,
    ollama_model: str,
    num_ctx: int,
) -> Path:
    modelfile = output_dir / "Modelfile"
    modelfile.write_text(
        render_modelfile(gguf_path, num_ctx=num_ctx),
        encoding="utf-8",
    )
    subprocess.run(
        ["ollama", "create", ollama_model, "-f", str(modelfile)],
        check=True,
    )
    return modelfile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ollama-model", default="qwen2.5-supervisor-14b-ft")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Export GGUF and Modelfile without running ollama create.",
    )
    parser.add_argument(
        "--smoke-dataset",
        type=Path,
        help="Optional mixed JSONL; writes six selected smoke records.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gguf_path = export_gguf(
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
    )
    modelfile = args.output_dir / "Modelfile"
    modelfile.write_text(
        render_modelfile(gguf_path, num_ctx=args.num_ctx),
        encoding="utf-8",
    )
    if not args.no_register:
        register_ollama_model(
            gguf_path=gguf_path,
            output_dir=args.output_dir,
            ollama_model=args.ollama_model,
            num_ctx=args.num_ctx,
        )
    smoke_path = None
    if args.smoke_dataset is not None:
        smoke_records = select_smoke_records(_read_jsonl(args.smoke_dataset))
        smoke_path = args.output_dir / "smoke6.jsonl"
        with smoke_path.open("w", encoding="utf-8") as handle:
            for record in smoke_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "gguf": str(gguf_path),
                "modelfile": str(modelfile),
                "ollama_model": None if args.no_register else args.ollama_model,
                "smoke_dataset": str(smoke_path) if smoke_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
