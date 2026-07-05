from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


LABELS = ("yes", "no", "maybe")


def main() -> None:
    args = _parse_args()
    report = _build_audit(args)
    _write_json(args.json_out, report)
    _write_markdown(args.md_out, report)
    print(f"Wrote PubMedQA classifier data audit JSON: {args.json_out}")
    print(f"Wrote PubMedQA classifier data audit Markdown: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_out_dir = Path("reports") / "classifier"
    parser = argparse.ArgumentParser(description="Audit PubMedQA classifier train/dev data and held-out leakage.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/pubmedqa_official/data"))
    parser.add_argument(
        "--official-eval",
        type=Path,
        default=Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json"),
    )
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/interim/classifier/pubmedqa_deberta/train.jsonl"))
    parser.add_argument("--dev-jsonl", type=Path, default=Path("data/interim/classifier/pubmedqa_deberta/dev.jsonl"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/classifier/pubmedqa_deberta/best"),
    )
    parser.add_argument("--json-out", type=Path, default=default_out_dir / f"pubmedqa_data_audit_{timestamp}.json")
    parser.add_argument("--md-out", type=Path, default=default_out_dir / f"pubmedqa_data_audit_{timestamp}.md")
    return parser.parse_args()


def _build_audit(args: argparse.Namespace) -> dict[str, Any]:
    official_eval = _load_json(args.official_eval)
    official_pmids = {
        str(pmid)
        for item in official_eval
        for pmid in item.get("relevant_pmids", [])
    }

    source_paths = {
        "PQA-A": args.raw_dir / "ori_pqaa.json",
        "PQA-U": args.raw_dir / "ori_pqau.json",
        "PQA-L": args.raw_dir / "ori_pqal.json",
        "PQA-L official test labels": args.raw_dir / "test_ground_truth.json",
    }
    sources = {
        name: _source_summary(path, official_pmids=official_pmids)
        for name, path in source_paths.items()
    }

    train_split = _jsonl_split_summary(args.train_jsonl, official_pmids=official_pmids)
    dev_split = _jsonl_split_summary(args.dev_jsonl, official_pmids=official_pmids)
    model_artifacts = _artifact_summary(args.model_dir)
    long_answer_flow = _long_answer_flow(sources=sources, train_split=train_split, dev_split=dev_split)

    leakage = {
        "official_eval_cases": len(official_eval),
        "official_eval_pmids": len(official_pmids),
        "official_eval_labels": _label_counts(item.get("expected_label") for item in official_eval),
        "train_official_pmid_overlap_count": train_split["heldout_overlap_count"],
        "train_official_pmid_overlap_examples": train_split["heldout_overlap_examples"],
        "dev_official_pmid_overlap_count": dev_split["heldout_overlap_count"],
        "dev_official_pmid_overlap_examples": dev_split["heldout_overlap_examples"],
        "passed": train_split["heldout_overlap_count"] == 0 and dev_split["heldout_overlap_count"] == 0,
    }

    maybe_analysis = {
        "raw_pqa_a_maybe": sources.get("PQA-A", {}).get("labels", {}).get("maybe", 0),
        "raw_pqa_l_nonheldout_maybe": sources.get("PQA-L", {}).get("trainable_nonheldout_labels", {}).get("maybe", 0),
        "current_train_maybe": train_split["labels"].get("maybe", 0),
        "current_dev_maybe": dev_split["labels"].get("maybe", 0),
        "why_current_split_has_50_train_5_dev": (
            "PQA-A contains no `maybe` labels. After removing the official PQA-L 500 held-out PMIDs, "
            "only 55 PQA-L `maybe` examples remain. With dev_fraction=0.10, the current builder assigns "
            "5 to dev and 50 to train."
        ),
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "raw_dir": str(args.raw_dir),
            "official_eval": _file_info(args.official_eval),
            "train_jsonl": _file_info(args.train_jsonl),
            "dev_jsonl": _file_info(args.dev_jsonl),
            "model_dir": str(args.model_dir),
        },
        "sources": sources,
        "splits": {
            "train": train_split,
            "dev": dev_split,
        },
        "leakage": leakage,
        "maybe_analysis": maybe_analysis,
        "long_answer_flow": long_answer_flow,
        "model_artifacts": model_artifacts,
        "recommendations": [
            "Keep official PQA-L 500 locked out of training, threshold tuning, prompt tuning, and early stopping.",
            "Use PQA-A as weak/supervised yes/no data, but preserve all non-held-out PQA-L examples so scarce `maybe` labels are not drowned by PQA-A.",
            "Increase PQA-L dev `maybe` support with a larger dev fraction or repeated seed/cross-validation; report `maybe` recall/F1 separately.",
            "Store `LONG_ANSWER` only in train/dev examples for auxiliary supervision; never concatenate it into official held-out inference evidence.",
            "Treat stale calibrated metrics/calibration as invalid whenever model artifacts or uncalibrated metrics are newer.",
        ],
    }


def _source_summary(path: Path, *, official_pmids: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": _file_info(path),
            "items": 0,
            "unique_pmids": 0,
            "labels": _ordered_counts(Counter()),
            "with_contexts": 0,
            "with_long_answer": 0,
            "maybe_with_long_answer": 0,
            "heldout_removed_labels": _ordered_counts(Counter()),
            "trainable_nonheldout_labels": _ordered_counts(Counter()),
            "trainable_nonheldout_with_long_answer": 0,
            "missing_trainable_fields": {},
            "note": "Source file not found locally.",
        }
    data = _load_json(path)
    pmids: list[str] = []
    labels = Counter()
    heldout_labels = Counter()
    trainable_labels = Counter()
    with_contexts = 0
    with_long_answer = 0
    trainable_with_long_answer = 0
    maybe_with_long_answer = 0
    missing = Counter()

    for raw_id, item in _iter_items(data):
        pmid = _pmid(raw_id, item)
        pmids.append(pmid)
        label = _label(item)
        if label in LABELS:
            labels[label] += 1
        if not isinstance(item, dict):
            continue
        question = str(item.get("QUESTION") or item.get("question") or "").strip()
        contexts = item.get("CONTEXTS") or item.get("contexts") or item.get("context") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        has_context = isinstance(contexts, list) and any(str(value).strip() for value in contexts)
        has_long_answer = bool(str(item.get("LONG_ANSWER") or item.get("long_answer") or "").strip())
        if has_context:
            with_contexts += 1
        if has_long_answer:
            with_long_answer += 1
            if label == "maybe":
                maybe_with_long_answer += 1
        if label not in LABELS:
            missing["label"] += 1
            continue
        if not question:
            missing["question"] += 1
            continue
        if not has_context:
            missing["context"] += 1
            continue
        if pmid in official_pmids:
            heldout_labels[label] += 1
            continue
        trainable_labels[label] += 1
        if has_long_answer:
            trainable_with_long_answer += 1

    return {
        "path": _file_info(path),
        "items": len(data),
        "unique_pmids": len(set(pmids)),
        "labels": _ordered_counts(labels),
        "with_contexts": with_contexts,
        "with_long_answer": with_long_answer,
        "maybe_with_long_answer": maybe_with_long_answer,
        "heldout_removed_labels": _ordered_counts(heldout_labels),
        "trainable_nonheldout_labels": _ordered_counts(trainable_labels),
        "trainable_nonheldout_with_long_answer": trainable_with_long_answer,
        "missing_trainable_fields": dict(sorted(missing.items())),
    }


def _jsonl_split_summary(path: Path, *, official_pmids: set[str]) -> dict[str, Any]:
    labels = Counter()
    source_files = Counter()
    heldout_overlap: list[str] = []
    ids: set[str] = set()
    pmids: set[str] = set()
    with_long_answer = 0
    total = 0
    if not path.exists():
        return {
            "path": _file_info(path),
            "count": 0,
            "unique_ids": 0,
            "unique_pmids": 0,
            "labels": _ordered_counts(labels),
            "source_files": {},
            "with_long_answer_field": 0,
            "heldout_overlap_count": 0,
            "heldout_overlap_examples": [],
        }

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1
            ids.add(str(item.get("id") or ""))
            pmid = str(item.get("pmid") or "")
            pmids.add(pmid)
            label = str(item.get("label") or "").lower()
            if label in LABELS:
                labels[label] += 1
            source_file = Path(str(item.get("source_file") or "")).name
            source_files[source_file] += 1
            if str(item.get("long_answer") or item.get("LONG_ANSWER") or "").strip():
                with_long_answer += 1
            if pmid in official_pmids and len(heldout_overlap) < 20:
                heldout_overlap.append(pmid)

    return {
        "path": _file_info(path),
        "count": total,
        "unique_ids": len(ids),
        "unique_pmids": len(pmids),
        "labels": _ordered_counts(labels),
        "source_files": dict(sorted(source_files.items())),
        "with_long_answer_field": with_long_answer,
        "heldout_overlap_count": len([pmid for pmid in pmids if pmid in official_pmids]),
        "heldout_overlap_examples": sorted(heldout_overlap),
    }


def _artifact_summary(model_dir: Path) -> dict[str, Any]:
    files = [
        "model.safetensors",
        "pytorch_model.bin",
        "dev_metrics.json",
        "dev_metrics_calibrated.json",
        "calibration.json",
        "decision_thresholds.json",
        "training_config.json",
    ]
    info = {name: _file_info(model_dir / name) for name in files}
    reference_paths = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
        model_dir / "dev_metrics.json",
    ]
    stale = {}
    for name in ("dev_metrics_calibrated.json", "calibration.json", "decision_thresholds.json"):
        path = model_dir / name
        stale[name] = _is_stale(path, reference_paths)
    return {
        "model_dir": str(model_dir),
        "files": info,
        "stale_against_model_or_dev_metrics": stale,
        "dev_metrics": _load_json_if_exists(model_dir / "dev_metrics.json"),
        "dev_metrics_calibrated": _load_json_if_exists(model_dir / "dev_metrics_calibrated.json"),
    }


def _long_answer_flow(
    *,
    sources: dict[str, dict[str, Any]],
    train_split: dict[str, Any],
    dev_split: dict[str, Any],
) -> dict[str, Any]:
    trainable_with_long_answer = sum(
        int(source.get("trainable_nonheldout_with_long_answer", 0))
        for source in sources.values()
    )
    split_with_long_answer = int(train_split["with_long_answer_field"]) + int(dev_split["with_long_answer_field"])
    return {
        "trainable_nonheldout_source_examples_with_long_answer": trainable_with_long_answer,
        "current_train_dev_examples_with_long_answer_field": split_with_long_answer,
        "currently_dropped": trainable_with_long_answer > 0 and split_with_long_answer == 0,
        "drop_location": (
            "`scripts/classifier/prepare_pubmedqa_deberta_dataset.py::_to_example` previously built "
            "`question + evidence + label` only; `LONG_ANSWER` was not serialized into train/dev JSONL."
        ),
        "heldout_policy": "Official PQA-L 500 inference must use only `QUESTION + CONTEXTS`/retrieved paper evidence.",
    }


def _iter_items(data: Any) -> list[tuple[str, Any]]:
    if isinstance(data, dict):
        return [(str(key), value) for key, value in data.items()]
    if isinstance(data, list):
        return [(str(index), value) for index, value in enumerate(data)]
    return []


def _label(item: Any) -> str:
    if isinstance(item, dict):
        return str(
            item.get("final_decision")
            or item.get("final_decision_label")
            or item.get("label")
            or item.get("expected_label")
            or ""
        ).strip().lower()
    return str(item).strip().lower()


def _pmid(raw_id: str, item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("pmid") or item.get("PMID") or raw_id).strip()
    return raw_id


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return {"error": f"could not read {path}"}


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_stale(path: Path, reference_paths: list[Path]) -> bool:
    if not path.exists():
        return False
    path_mtime = path.stat().st_mtime
    for reference_path in reference_paths:
        if reference_path == path or not reference_path.exists():
            continue
        if reference_path.stat().st_mtime > path_mtime:
            return True
    return False


def _label_counts(values: Any) -> dict[str, int]:
    return _ordered_counts(Counter(str(value).lower() for value in values))


def _ordered_counts(counter: Counter[str]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in LABELS}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    leakage = report["leakage"]
    maybe = report["maybe_analysis"]
    long_answer = report["long_answer_flow"]
    lines = [
        "# PubMedQA Classifier Data Audit",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Held-out Check",
        "",
        f"- Official eval cases: {leakage['official_eval_cases']}",
        f"- Official eval labels: `{leakage['official_eval_labels']}`",
        f"- Train PMID overlap with official eval: {leakage['train_official_pmid_overlap_count']}",
        f"- Dev PMID overlap with official eval: {leakage['dev_official_pmid_overlap_count']}",
        f"- Leakage check passed: `{leakage['passed']}`",
        "",
        "## Source Distributions",
        "",
        "| Source | Items | Labels | Trainable non-held-out labels | With LONG_ANSWER | Held-out removed |",
        "|---|---:|---|---|---:|---|",
    ]
    for name, source in report["sources"].items():
        lines.append(
            f"| {name} | {source['items']} | `{source['labels']}` | "
            f"`{source['trainable_nonheldout_labels']}` | {source['with_long_answer']} | "
            f"`{source['heldout_removed_labels']}` |"
        )

    lines.extend(
        [
            "",
            "## Current Split",
            "",
            "| Split | Count | Labels | Source files | Held-out overlap | LONG_ANSWER fields |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for split_name, split in report["splits"].items():
        lines.append(
            f"| {split_name} | {split['count']} | `{split['labels']}` | `{split['source_files']}` | "
            f"{split['heldout_overlap_count']} | {split['with_long_answer_field']} |"
        )

    lines.extend(
        [
            "",
            "## Maybe Class",
            "",
            f"- Raw PQA-A maybe examples: {maybe['raw_pqa_a_maybe']}",
            f"- Raw PQA-L non-held-out maybe examples: {maybe['raw_pqa_l_nonheldout_maybe']}",
            f"- Current train/dev maybe examples: {maybe['current_train_maybe']} / {maybe['current_dev_maybe']}",
            f"- Explanation: {maybe['why_current_split_has_50_train_5_dev']}",
            "",
            "## LONG_ANSWER Flow",
            "",
            f"- Trainable non-held-out examples with LONG_ANSWER: {long_answer['trainable_nonheldout_source_examples_with_long_answer']}",
            f"- Current train/dev examples with LONG_ANSWER serialized: {long_answer['current_train_dev_examples_with_long_answer_field']}",
            f"- Currently dropped: `{long_answer['currently_dropped']}`",
            f"- Drop location: {long_answer['drop_location']}",
            f"- Held-out policy: {long_answer['heldout_policy']}",
            "",
            "## Classifier Artifact Freshness",
            "",
            f"- Stale files: `{report['model_artifacts']['stale_against_model_or_dev_metrics']}`",
            "",
            "## Recommendations",
            "",
        ]
    )
    for recommendation in report["recommendations"]:
        lines.append(f"- {recommendation}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
