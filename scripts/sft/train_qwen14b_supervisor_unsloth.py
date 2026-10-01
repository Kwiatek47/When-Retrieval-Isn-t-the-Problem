#!/usr/bin/env python3
"""Single-GPU Unsloth QLoRA trainer for the PubMedQA supervisor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


IGNORE_INDEX = -100
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    max_seq_length: int = 4096
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    seed: int = 47
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01


def encode_chat_example(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize one chat record and mask every non-assistant target token."""
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Example must contain a non-empty messages list.")
    if messages[-1].get("role") != "assistant":
        raise ValueError("The final chat turn must have role='assistant'.")

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template produced a non-prefix assistant prompt.")
    if len(prompt_ids) >= max_seq_length:
        raise ValueError("Prompt consumes the full context; no assistant answer tokens remain.")

    input_ids = full_ids[:max_seq_length]
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("Assistant answer was truncated completely.")
    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("Assistant answer has no trainable tokens.")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def validate_single_a40(torch_module: Any, *, allow_non_a40: bool = False) -> dict[str, Any]:
    """Fail early when the process is not isolated to one suitable training GPU."""
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen2.5-14B QLoRA training.")
    count = int(torch_module.cuda.device_count())
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA device, found {count}. "
            "Set CUDA_VISIBLE_DEVICES to one A40."
        )
    props = torch_module.cuda.get_device_properties(0)
    name = str(props.name)
    memory_gib = float(props.total_memory) / (1024**3)
    if not allow_non_a40 and ("A40" not in name.upper() or memory_gib < 40.0):
        raise RuntimeError(
            f"Expected an NVIDIA A40 with at least 40 GiB, found {name} ({memory_gib:.1f} GiB). "
            "Use --allow-non-a40 only after checking memory requirements."
        )
    return {"device_name": name, "memory_gib": round(memory_gib, 2), "device_count": count}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"No training records found in {path}")
    return records


def _parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("director", "multitask"), required=True)
    parser.add_argument("--base-model", default=defaults.base_model)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--allow-non-a40", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the visible GPU and exit before loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TrainingConfig(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )

    import torch

    gpu = validate_single_a40(torch, allow_non_a40=args.allow_non_a40)
    print(json.dumps({"gpu_preflight": gpu}, indent=2))
    if args.preflight_only:
        return
    if args.stage == "multitask" and args.adapter_path is None:
        raise SystemExit("--adapter-path is required for the multitask curriculum stage.")
    if not args.train_file.exists() or not args.eval_file.exists():
        raise FileNotFoundError("Both --train-file and --eval-file must exist.")

    # Unsloth must be imported before Transformers so its kernels are patched.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model_source = str(args.adapter_path) if args.adapter_path else config.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if args.adapter_path is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=list(TARGET_MODULES),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def encode_records(path: Path) -> Dataset:
        encoded = [
            encode_chat_example(
                record,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
            )
            for record in _read_jsonl(path)
        ]
        return Dataset.from_list(encoded)

    train_dataset = encode_records(args.train_file)
    eval_dataset = encode_records(args.eval_file)
    learning_rate = args.learning_rate or (1e-4 if args.stage == "director" else 5e-5)
    epochs = args.epochs or (2.0 if args.stage == "director" else 1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **asdict(config),
        "stage": args.stage,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "gpu": gpu,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=False,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        padding=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    adapter_dir = args.output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = trainer.evaluate()
    (args.output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
