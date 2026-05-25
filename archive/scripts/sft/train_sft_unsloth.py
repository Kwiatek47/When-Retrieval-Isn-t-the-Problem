#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a chat model with Unsloth SFT (LoRA/QLoRA).")
    parser.add_argument("--train-file", default="data/sft/medqa/train.jsonl")
    parser.add_argument("--eval-file", default="data/sft/medqa/dev.jsonl")
    parser.add_argument("--base-model", default="unsloth/Llama-3.1-8B-bnb-4bit")
    parser.add_argument("--output-dir", default="artifacts/sft-medqa-lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation during training.")
    return parser.parse_args()


def render_messages_fallback(messages, eos_token: str) -> str:
    chunks: list[str] = []
    for turn in messages:
        role = str(turn.get("role") or "user").strip().lower()
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            chunks.append(f"<|system|>\n{content}")
        elif role == "assistant":
            chunks.append(f"<|assistant|>\n{content}")
        else:
            chunks.append(f"<|user|>\n{content}")
    text = "\n\n".join(chunks).strip()
    if eos_token and not text.endswith(eos_token):
        text = f"{text}{eos_token}"
    return text


def format_chat_dataset(path: str, tokenizer, max_seq_length: int):
    dataset = load_dataset("json", data_files={"data": path}, split="data")
    if "messages" not in dataset.column_names:
        raise ValueError(
            f"Dataset {path} must contain a 'messages' field with chat turns "
            "(e.g. system/user/assistant)."
        )

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    eos_token = tokenizer.eos_token or ""

    def to_tokenized(example):
        if has_chat_template:
            try:
                text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except ValueError:
                # Some base tokenizers expose no usable chat template.
                text = render_messages_fallback(example["messages"], eos_token=eos_token)
        else:
            text = render_messages_fallback(example["messages"], eos_token=eos_token)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized.get("attention_mask"),
        }

    return dataset.map(
        to_tokenized,
        batched=False,
        remove_columns=dataset.column_names,
        desc="Pre-tokenizing dataset",
    )


def main() -> None:
    args = parse_args()

    train_file = Path(args.train_file)
    eval_file = Path(args.eval_file)
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not args.no_eval and not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    # Import Unsloth before TRL/Transformers for proper patching.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_dataset = format_chat_dataset(str(train_file), tokenizer, args.max_seq_length)
    eval_dataset = None if args.no_eval else format_chat_dataset(str(eval_file), tokenizer, args.max_seq_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "seed": args.seed,
        "optim": "adamw_8bit",
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "report_to": "none",
    }
    strategy_value = "no" if args.no_eval else "steps"
    training_args = None

    # Prefer SFTConfig when available (new TRL), so we can force single-process
    # dataset preparation and avoid pickling issues in multiprocess map.
    try:
        from trl import SFTConfig  # type: ignore

        sft_signature = inspect.signature(SFTConfig.__init__)
        sft_kwargs = {k: v for k, v in training_kwargs.items() if k in sft_signature.parameters}
        if "evaluation_strategy" in sft_signature.parameters:
            sft_kwargs["evaluation_strategy"] = strategy_value
        elif "eval_strategy" in sft_signature.parameters:
            sft_kwargs["eval_strategy"] = strategy_value
        if "dataset_num_proc" in sft_signature.parameters:
            sft_kwargs["dataset_num_proc"] = None
        if "dataset_kwargs" in sft_signature.parameters:
            # Data is already tokenized above; skip internal dataset.map in TRL.
            sft_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
        if "max_seq_length" in sft_signature.parameters:
            sft_kwargs["max_seq_length"] = args.max_seq_length
        elif "max_length" in sft_signature.parameters:
            sft_kwargs["max_length"] = args.max_seq_length
        if "packing" in sft_signature.parameters:
            sft_kwargs["packing"] = False

        training_args = SFTConfig(**sft_kwargs)
    except Exception:
        signature = inspect.signature(TrainingArguments.__init__)
        if "evaluation_strategy" in signature.parameters:
            training_kwargs["evaluation_strategy"] = strategy_value
        elif "eval_strategy" in signature.parameters:
            training_kwargs["eval_strategy"] = strategy_value
        training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_signature = inspect.signature(SFTTrainer.__init__)
    trainer_params = trainer_signature.parameters

    # TRL API changed across versions: tokenizer -> processing_class.
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    # Older TRL versions accepted these directly in SFTTrainer.__init__.
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    model.save_pretrained(str(output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapter"))

    print(f"Training complete. Adapter saved to: {output_dir / 'lora_adapter'}")


if __name__ == "__main__":
    main()
