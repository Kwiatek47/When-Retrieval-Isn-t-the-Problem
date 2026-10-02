"""RQ8 / RQ9a: can BioLinkBERT detect ``maybe`` when it is trained for it?

The deployed classifier was trained on PQA-A plus the non-test half of PQA-L. PQA-A has no
``maybe`` label, so 0.13% of its training examples were ``maybe`` against 11% at test time,
and it recovers 4 of 55. Here BioLinkBERT-large is fine-tuned from its pretrained weights
on the 500 non-test PQA-L questions only (55 ``maybe``, the natural 11%), in a 2 x 2 design:

  head:      ``binary`` (maybe vs not)        | ``three_class`` (yes / no / maybe)
  sampling:  ``natural`` (each example once)  | ``balanced`` (classes drawn equally often)

Everything else is fixed in advance and identical across conditions: input format
(question, context; 512 tokens), learning rate, epochs, batch size, seeds. There is no
model selection: no dev split, no early stopping, no threshold. Each run writes the
probability of ``maybe`` for every test question; ``scripts/agents/analyze_rq8_maybe_detector.py``
scores them. No checkpoint is kept.

``--reference`` scores the deployed 3-class checkpoint the same way, without training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
TEST_SET = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/eval.json"
OUT_DIR = PROJECT_ROOT / "reports/debate/analysis/maybe_detector"
REFERENCE_CHECKPOINT = PROJECT_ROOT / "artifacts/classifier/pubmedqa_biolinkbert_seed47/best"

MODEL_NAME = "michiyasunaga/BioLinkBERT-large"
MODEL_REVISION = "1eb6d81c5fc1c42d3a43c71956b0e526558ae053"
HEADS = {"binary": ("not_maybe", "maybe"), "three_class": ("yes", "no", "maybe")}
SAMPLINGS = ("natural", "balanced")
SEEDS = (11, 23, 42, 47, 101)
HPARAMS = {
    "max_length": 512,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}


def load_split(pqal_path: Path = PQAL, test_path: Path = TEST_SET) -> tuple[list[dict], list[dict]]:
    """(train, test) examples: the 500 non-test and the 500 official test PQA-L questions."""
    data = json.loads(pqal_path.read_text(encoding="utf-8"))
    test_pmids = {str(p) for case in json.loads(test_path.read_text(encoding="utf-8")) for p in case["relevant_pmids"]}
    train, test = [], []
    for pmid in sorted(data):
        item = data[pmid]
        example = {
            "pmid": pmid,
            "question": item["QUESTION"],
            "evidence": " ".join(str(c).strip() for c in item["CONTEXTS"] if str(c).strip()),
            "label": item["final_decision"],
        }
        (test if pmid in test_pmids else train).append(example)
    if {e["pmid"] for e in train} & {e["pmid"] for e in test}:
        raise RuntimeError("train and test share questions")
    return train, test


def label_ids(examples: list[dict], head: str) -> list[int]:
    classes = HEADS[head]
    if head == "binary":
        return [int(e["label"] == "maybe") for e in examples]
    return [classes.index(e["label"]) for e in examples]


def sampling_weights(ids: list[int], sampling: str) -> list[float] | None:
    """Per-example draw weights; ``None`` means plain shuffling of each example once."""
    if sampling == "natural":
        return None
    if sampling != "balanced":
        raise ValueError(f"unknown sampling {sampling!r}")
    counts = Counter(ids)
    return [1.0 / counts[i] for i in ids]


def run_name(head: str, sampling: str, seed: int) -> str:
    return f"{head}.{sampling}.seed{seed}"


def _encode(tokenizer, examples: list[dict], max_length: int):
    return tokenizer(
        [e["question"] for e in examples],
        [e["evidence"] for e in examples],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


def _score(model, encoded, device, batch_size: int = 32):
    """Softmax probabilities for every encoded example."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, encoded["input_ids"].shape[0], batch_size):
            batch = {k: v[start : start + batch_size].to(device) for k, v in encoded.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out)


def _write_scores(path: Path, test: list[dict], probs, classes: tuple[str, ...], meta: dict) -> None:
    maybe_index = classes.index("maybe")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for example, row in zip(test, probs.tolist()):
            fh.write(
                json.dumps(
                    {
                        "pmid": example["pmid"],
                        "p_maybe": row[maybe_index],
                        "probs": dict(zip(classes, row)),
                    }
                )
                + "\n"
            )


def train_and_score(head: str, sampling: str, seed: int, train: list[dict], test: list[dict], out_dir: Path) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    out_path = out_dir / f"{run_name(head, sampling, seed)}.jsonl"
    if out_path.exists():
        print(f"skip {out_path.name} (exists)", flush=True)
        return

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = HEADS[head]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, num_labels=len(classes)
    ).to(device)

    ids = label_ids(train, head)
    encoded = _encode(tokenizer, train, HPARAMS["max_length"])
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"], torch.tensor(ids))
    weights = sampling_weights(ids, sampling)
    generator = torch.Generator().manual_seed(seed)
    if weights is None:
        loader = DataLoader(dataset, batch_size=HPARAMS["batch_size"], shuffle=True, generator=generator)
    else:
        sampler = WeightedRandomSampler(weights, num_samples=len(ids), replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=HPARAMS["batch_size"], sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=HPARAMS["learning_rate"], weight_decay=HPARAMS["weight_decay"])
    total_steps = len(loader) * HPARAMS["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(HPARAMS["warmup_ratio"] * total_steps), total_steps)

    started = time.time()
    for epoch in range(HPARAMS["epochs"]):
        model.train()
        running = 0.0
        for input_ids, attention_mask, token_type_ids, labels in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    token_type_ids=token_type_ids.to(device),
                    labels=labels.to(device),
                ).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += float(loss)
        print(f"{out_path.stem} epoch {epoch + 1}/{HPARAMS['epochs']} loss {running / len(loader):.4f}", flush=True)

    probs = _score(model, _encode(tokenizer, test, HPARAMS["max_length"]), device)
    meta = {
        "head": head,
        "sampling": sampling,
        "seed": seed,
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "hparams": HPARAMS,
        "train_examples": len(train),
        "train_labels": dict(Counter(e["label"] for e in train)),
        "train_seconds": round(time.time() - started, 1),
    }
    _write_scores(out_path, test, probs, classes, meta)
    print(f"wrote {out_path}", flush=True)


def score_reference(test: list[dict], out_dir: Path, checkpoint: Path = REFERENCE_CHECKPOINT) -> None:
    """Score the deployed 3-class checkpoint (trained with PQA-A) on the test questions."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    label_map = json.loads((checkpoint / "label_map.json").read_text(encoding="utf-8"))
    classes = tuple(sorted(label_map, key=label_map.get))
    probs = _score(model, _encode(tokenizer, test, HPARAMS["max_length"]), device)
    out_path = out_dir / "reference_deployed_three_class.jsonl"
    _write_scores(out_path, test, probs, classes, {"checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)), "temperature": 1.0})
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", nargs="+", choices=sorted(HEADS), default=sorted(HEADS))
    parser.add_argument("--samplings", nargs="+", choices=SAMPLINGS, default=list(SAMPLINGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--reference", action="store_true", help="only score the deployed checkpoint")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    train, test = load_split()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"train {len(train)} {dict(Counter(e['label'] for e in train))}; test {len(test)}", flush=True)
    if args.reference:
        score_reference(test, args.out_dir)
        return
    for head in args.heads:
        for sampling in args.samplings:
            for seed in args.seeds:
                train_and_score(head, sampling, seed, train, test, args.out_dir)


if __name__ == "__main__":
    main()
