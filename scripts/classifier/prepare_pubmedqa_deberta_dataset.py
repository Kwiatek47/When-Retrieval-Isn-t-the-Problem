from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


LABELS = ("yes", "no", "maybe")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
OFFICIAL_REPO_URL = "https://github.com/pubmedqa/pubmedqa.git"
OFFICIAL_PQAA_GOOGLE_DRIVE_ID = "15v1x6aQDlZymaHGP7cZJZZYFfeJt2NdS"


@dataclass(frozen=True)
class ClassifierExample:
    id: str
    pmid: str
    question: str
    evidence: str
    label: str
    source_file: str
    source_dataset: str
    long_answer: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pmid": self.pmid,
            "question": self.question,
            "evidence": self.evidence,
            "long_answer": self.long_answer,
            "label": self.label,
            "label_id": LABEL_TO_ID[self.label],
            "source_file": self.source_file,
            "source_dataset": self.source_dataset,
        }


def main() -> None:
    args = _parse_args()
    source_dir = args.source_dir
    if args.download and not source_dir.exists():
        _clone_official_repo(args.repo_url, source_dir)
    if args.download_pqaa:
        _download_google_drive_file(
            file_id=args.pqaa_google_drive_id,
            output_path=source_dir / "data" / "ori_pqaa.json",
        )

    heldout_pmids = _heldout_pmids(args.heldout_eval)
    source_files = _source_files(source_dir, args.source_json)
    if not source_files:
        raise RuntimeError(
            f"No PubMedQA source JSON files found in {source_dir}. "
            "Run with --download or pass --source-json pointing at official PubMedQA JSON files."
        )

    examples = _load_examples(source_files, heldout_pmids=heldout_pmids)
    if not examples:
        raise RuntimeError("No trainable PubMedQA examples found after held-out filtering.")

    train, dev = _split_examples(
        examples,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        max_train_per_label=args.max_train_per_label,
        max_dev_per_label=args.max_dev_per_label,
        min_dev_per_label=args.min_dev_per_label,
        balance_train=args.balance_train,
        balance_dev=args.balance_dev,
        priority_source_names=set(args.priority_source_name),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    dev_path = args.out_dir / "dev.jsonl"
    manifest_path = args.out_dir / "manifest.json"
    label_map_path = args.out_dir / "label_map.json"

    _write_jsonl(train_path, train)
    _write_jsonl(dev_path, dev)
    _write_json(label_map_path, LABEL_TO_ID)
    _write_json(
        manifest_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_repo_url": args.repo_url,
            "source_repo_commit": _git_commit(source_dir),
            "source_dir": str(source_dir),
            "source_files": [
                {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in source_files
            ],
            "license_note": (
                "PubMedQA is used from the official project repository. "
                "Check the upstream repository/license before redistributing generated datasets."
            ),
            "heldout_eval": str(args.heldout_eval),
            "heldout_pmids_count": len(heldout_pmids),
            "label_map": LABEL_TO_ID,
            "split": {
                "seed": args.seed,
                "dev_fraction": args.dev_fraction,
                "max_train_per_label": args.max_train_per_label,
                "max_dev_per_label": args.max_dev_per_label,
                "min_dev_per_label": args.min_dev_per_label,
                "balance_train": args.balance_train,
                "balance_dev": args.balance_dev,
                "priority_source_names": args.priority_source_name,
                "train_count": len(train),
                "dev_count": len(dev),
                "train_labels": _label_counts(train),
                "dev_labels": _label_counts(dev),
                "train_sources": _source_counts(train),
                "dev_sources": _source_counts(dev),
                "train_with_long_answer": sum(1 for example in train if example.long_answer),
                "dev_with_long_answer": sum(1 for example in dev if example.long_answer),
            },
        },
    )

    print(f"Wrote train split: {train_path} examples={len(train)} labels={_label_counts(train)}")
    print(f"Wrote dev split: {dev_path} examples={len(dev)} labels={_label_counts(dev)}")
    print(f"Wrote manifest: {manifest_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official PubMedQA evidence-classifier data for DeBERTa.")
    parser.add_argument("--repo-url", default=OFFICIAL_REPO_URL)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/pubmedqa_official"))
    parser.add_argument("--source-json", type=Path, action="append", default=[])
    parser.add_argument("--download", action="store_true", help="Clone the official PubMedQA repo if source-dir is absent.")
    parser.add_argument(
        "--download-pqaa",
        action="store_true",
        help="Download official PQA-A from the Google Drive link published in the PubMedQA README.",
    )
    parser.add_argument("--pqaa-google-drive-id", default=OFFICIAL_PQAA_GOOGLE_DRIVE_ID)
    parser.add_argument("--heldout-eval", type=Path, default=Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim/classifier/pubmedqa_deberta"))
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--max-train-per-label", type=int, default=6000)
    parser.add_argument("--max-dev-per-label", type=int, default=500)
    parser.add_argument(
        "--min-dev-per-label",
        type=int,
        default=1,
        help="Minimum dev examples per label when available. Leaves at least one train example per label.",
    )
    parser.add_argument(
        "--priority-source-name",
        action="append",
        default=[],
        help=(
            "Source filename to keep before filling per-label train caps. Use `ori_pqal.json` so scarce PQA-L "
            "examples, especially `maybe`, are not drowned by PQA-A."
        ),
    )
    parser.add_argument("--balance-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--balance-dev", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _clone_official_repo(repo_url: str, source_dir: Path) -> None:
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(source_dir)], check=True)


def _download_google_drive_file(*, file_id: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = _open_google_drive_download(file_id)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    first_non_ws = b""
    with response, temp_path.open("wb") as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            if not first_non_ws:
                first_non_ws = chunk.lstrip()[:1]
            file.write(chunk)
    if first_non_ws not in {b"{", b"["}:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded PQA-A does not look like JSON. "
            "If Google Drive blocks automated download, manually place official ori_pqaa.json "
            f"at {output_path} and rerun this command."
        )
    temp_path.replace(output_path)


def _open_google_drive_download(file_id: str) -> Any:
    url = "https://drive.google.com/uc?" + urlencode({"export": "download", "id": file_id})
    response = urlopen(url, timeout=120)
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return response

    html = response.read().decode("utf-8", "replace")
    form_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>', html, re.S)
    if not form_match:
        return urlopen(url, timeout=120)

    action = form_match.group(1).replace("&amp;", "&")
    form_html = form_match.group(2)
    params = {
        key.replace("&amp;", "&"): value.replace("&amp;", "&")
        for key, value in re.findall(r'name="([^"]+)" value="([^"]*)"', form_html)
    }
    return urlopen(action + "?" + urlencode(params), timeout=120)


def _source_files(source_dir: Path, explicit_files: list[Path]) -> list[Path]:
    if explicit_files:
        return [path for path in explicit_files if path.exists()]
    if not source_dir.exists():
        return []
    candidates = sorted(source_dir.rglob("*.json"))
    return [
        path
        for path in candidates
        if not any(part.startswith(".") for part in path.parts)
        and path.name not in {"README.json", "manifest.json"}
    ]


def _heldout_pmids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pmids: set[str] = set()
    for item in data:
        for value in item.get("relevant_pmids", []):
            pmids.add(str(value))
    return pmids


def _load_examples(source_files: list[Path], *, heldout_pmids: set[str]) -> list[ClassifierExample]:
    examples: list[ClassifierExample] = []
    seen_ids: set[str] = set()
    for path in source_files:
        for raw_id, item in _iter_pubmedqa_items(path):
            example = _to_example(raw_id, item, source_file=str(path))
            if example is None:
                continue
            if example.pmid in heldout_pmids:
                continue
            if example.id in seen_ids:
                continue
            seen_ids.add(example.id)
            examples.append(example)
    return examples


def _iter_pubmedqa_items(path: Path) -> list[tuple[str, dict[str, Any]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        items = []
        for key, value in data.items():
            if isinstance(value, dict):
                items.append((str(key), value))
        return items
    if isinstance(data, list):
        return [
            (str(item.get("pmid") or item.get("id") or index), item)
            for index, item in enumerate(data)
            if isinstance(item, dict)
        ]
    return []


def _to_example(raw_id: str, item: dict[str, Any], *, source_file: str) -> ClassifierExample | None:
    label = str(
        item.get("final_decision")
        or item.get("final_decision_label")
        or item.get("label")
        or item.get("expected_label")
        or ""
    ).strip().lower()
    if label not in LABEL_TO_ID:
        return None

    question = str(item.get("QUESTION") or item.get("question") or "").strip()
    if not question:
        return None

    contexts = item.get("CONTEXTS") or item.get("contexts") or item.get("context") or []
    if isinstance(contexts, str):
        contexts = [contexts]
    if not isinstance(contexts, list):
        return None
    evidence_parts = [str(part).strip() for part in contexts if str(part).strip()]
    if not evidence_parts:
        return None

    pmid = str(item.get("pmid") or item.get("PMID") or raw_id).strip()
    if not pmid:
        return None
    example_id = f"pubmedqa-{pmid}"
    long_answer = str(item.get("LONG_ANSWER") or item.get("long_answer") or "").strip()
    return ClassifierExample(
        id=example_id,
        pmid=pmid,
        question=question,
        evidence=" ".join(evidence_parts),
        label=label,
        source_file=source_file,
        source_dataset=_source_dataset_name(source_file),
        long_answer=long_answer,
    )


def _split_examples(
    examples: list[ClassifierExample],
    *,
    seed: int,
    dev_fraction: float,
    max_train_per_label: int,
    max_dev_per_label: int,
    min_dev_per_label: int,
    balance_train: bool,
    balance_dev: bool,
    priority_source_names: set[str],
) -> tuple[list[ClassifierExample], list[ClassifierExample]]:
    rng = random.Random(seed)
    by_label: dict[str, list[ClassifierExample]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    train_by_label: dict[str, list[ClassifierExample]] = {}
    dev_by_label: dict[str, list[ClassifierExample]] = {}
    for label in LABELS:
        label_examples = list(by_label.get(label, []))
        rng.shuffle(label_examples)
        if len(label_examples) <= 1:
            dev_count = len(label_examples)
        else:
            requested_dev = max(int(len(label_examples) * dev_fraction), min_dev_per_label, 1)
            dev_count = min(requested_dev, max_dev_per_label, len(label_examples) - 1)
        dev_by_label[label] = label_examples[:dev_count]
        train_pool = label_examples[dev_count:]
        if priority_source_names:
            priority = [
                example
                for example in train_pool
                if Path(example.source_file).name in priority_source_names
            ]
            non_priority = [
                example
                for example in train_pool
                if Path(example.source_file).name not in priority_source_names
            ]
            train_pool = [*priority, *non_priority]
        train_by_label[label] = train_pool[:max_train_per_label]

    if balance_train:
        train_target = min((len(train_by_label[label]) for label in LABELS), default=0)
        train_by_label = {label: examples[:train_target] for label, examples in train_by_label.items()}
    if balance_dev:
        dev_target = min((len(dev_by_label[label]) for label in LABELS), default=0)
        dev_by_label = {label: examples[:dev_target] for label, examples in dev_by_label.items()}

    train = [example for label in LABELS for example in train_by_label[label]]
    dev = [example for label in LABELS for example in dev_by_label[label]]

    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def _write_jsonl(path: Path, examples: list[ClassifierExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _label_counts(examples: list[ClassifierExample]) -> dict[str, int]:
    counts = Counter(example.label for example in examples)
    return {label: counts.get(label, 0) for label in LABELS}


def _source_counts(examples: list[ClassifierExample]) -> dict[str, int]:
    counts = Counter(example.source_dataset for example in examples)
    return dict(sorted(counts.items()))


def _source_dataset_name(source_file: str) -> str:
    filename = Path(source_file).name.lower()
    if "pqaa" in filename:
        return "pqa_a"
    if "pqal" in filename:
        return "pqa_l"
    if "ground_truth" in filename:
        return "pqa_l_official_test_labels"
    return Path(source_file).stem or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


if __name__ == "__main__":
    main()
