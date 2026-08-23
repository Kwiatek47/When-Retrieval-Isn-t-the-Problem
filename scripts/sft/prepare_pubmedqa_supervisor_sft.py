#!/usr/bin/env python3
"""Prepare leakage-safe PubMedQA source splits for supervisor SFT."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.classifier.prepare_pubmedqa_deberta_dataset import (
    ClassifierExample,
    _heldout_pmids,
    _load_examples,
)
from app.agents.aggregation import (
    build_full_debate_transcript,
    build_panel_vote_summary,
    opinion_label,
)
from app.agents.models import (
    AgentRoundOpinion,
    SupervisorDirectorOutput,
    SupervisorModerationOutput,
)
from app.agents.prompts import (
    SUPERVISOR_DIRECTOR_PROMPT,
    SUPERVISOR_MODERATOR_PROMPT,
    format_opinions_for_supervisor,
)


SPLIT_NAMES = ("train", "dev", "internal_test")
LABELS = ("yes", "no", "maybe")
JSON_ONLY_SYSTEM = "Return only valid JSON matching the requested schema."


def load_safe_examples(
    *,
    pqal_path: Path,
    pqaa_path: Path,
    heldout_eval: Path,
) -> tuple[list[ClassifierExample], set[str]]:
    """Load only labeled PQA-L/PQA-A records after held-out PMID filtering."""
    for path in (pqal_path, pqaa_path, heldout_eval):
        if not path.exists():
            raise FileNotFoundError(path)
    blocked = _heldout_pmids(heldout_eval)
    examples = _load_examples([pqal_path, pqaa_path], heldout_pmids=blocked)
    examples = [
        example
        for example in examples
        if example.source_dataset in {"pqa_l", "pqa_a"}
        and example.label in LABELS
    ]
    if not examples:
        raise ValueError("No safe labeled PQA-L/PQA-A examples were loaded.")
    return examples, blocked


def _partition_group(
    values: list[ClassifierExample],
    *,
    train_fraction: float,
    dev_fraction: float,
) -> tuple[list[ClassifierExample], list[ClassifierExample], list[ClassifierExample]]:
    count = len(values)
    if count == 0:
        return [], [], []
    if count < 3:
        train_count = max(1, count - 1)
        dev_count = count - train_count
    else:
        train_count = max(1, int(count * train_fraction))
        dev_count = max(1, int(count * dev_fraction))
        if train_count + dev_count >= count:
            train_count = count - 2
            dev_count = 1
    return (
        values[:train_count],
        values[train_count : train_count + dev_count],
        values[train_count + dev_count :],
    )


def split_examples_three_way(
    examples: Iterable[ClassifierExample],
    *,
    seed: int,
    train_fraction: float,
    dev_fraction: float,
    max_pqaa_per_label: int,
) -> dict[str, list[ClassifierExample]]:
    """Stratify by source and label, cap PQA-A, and split before augmentation."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("train_fraction + dev_fraction must be below 1")
    if max_pqaa_per_label < 1:
        raise ValueError("max_pqaa_per_label must be positive")

    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[ClassifierExample]] = defaultdict(list)
    for example in examples:
        grouped[(example.source_dataset, example.label)].append(example)

    splits: dict[str, list[ClassifierExample]] = {name: [] for name in SPLIT_NAMES}
    for source_dataset in ("pqa_l", "pqa_a"):
        for label in LABELS:
            group = list(grouped.get((source_dataset, label), []))
            rng.shuffle(group)
            if source_dataset == "pqa_a":
                group = group[:max_pqaa_per_label]
            train, dev, internal_test = _partition_group(
                group,
                train_fraction=train_fraction,
                dev_fraction=dev_fraction,
            )
            splits["train"].extend(train)
            splits["dev"].extend(dev)
            splits["internal_test"].extend(internal_test)

    for rows in splits.values():
        rng.shuffle(rows)
    _assert_disjoint(splits)
    return splits


def _assert_disjoint(splits: dict[str, list[ClassifierExample]]) -> None:
    seen: dict[str, str] = {}
    for split_name, examples in splits.items():
        for example in examples:
            prior = seen.get(example.pmid)
            if prior is not None:
                raise ValueError(
                    f"PMID {example.pmid} appears in both {prior} and {split_name}"
                )
            seen[example.pmid] = split_name


def assert_no_heldout_overlap(
    splits: dict[str, list[ClassifierExample]],
    blocked_pmids: set[str],
) -> None:
    overlaps = sorted(
        {
            example.pmid
            for examples in splits.values()
            for example in examples
            if example.pmid in blocked_pmids
        }
    )
    if overlaps:
        raise ValueError(f"Held-out PMID overlap detected: {', '.join(overlaps[:20])}")


def _final_entries(debate: dict) -> list[AgentRoundOpinion]:
    raw = debate.get("final_opinions") or []
    return [AgentRoundOpinion.model_validate(item) for item in raw]


def _role_labels(entries: list[AgentRoundOpinion]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entry in entries:
        label = opinion_label(entry.opinion)
        if label:
            labels[entry.agent_id or entry.persona] = label
    return labels


def _label_counts_from_roles(labels: dict[str, str]) -> Counter[str]:
    return Counter(label for label in labels.values() if label in LABELS)


def _clean_long_answer(debate: dict) -> str:
    text = " ".join(str(debate.get("long_answer") or "").split())
    if text:
        return text[:1200]
    return (
        f"Gold PubMedQA decision: {str(debate.get('gold_label') or 'maybe').lower()}. "
        "No long-form author conclusion was available."
    )


def _conflict_summary(labels: dict[str, str]) -> list[str]:
    counts = _label_counts_from_roles(labels)
    if not counts:
        return []
    plurality = counts.most_common(1)[0][0]
    contradictions = [
        f"{role} votes {label} while the panel plurality is {plurality}."
        for role, label in sorted(labels.items())
        if label != plurality
    ]
    return contradictions


def build_director_sft_record(debate: dict) -> dict:
    """Build a deterministic gold-aligned Director chat example."""
    entries = _final_entries(debate)
    labels = _role_labels(entries)
    counts = _label_counts_from_roles(labels)
    gold = str(debate.get("gold_label") or "").strip().lower()
    if gold not in LABELS:
        raise ValueError(f"Invalid gold label: {gold!r}")
    total = max(1, sum(counts.values()))
    top_count = max(counts.values(), default=0)
    conflict_level = "low" if len(counts) <= 1 else ("high" if top_count / total < 0.6 else "medium")
    advocate_label = labels.get("uncertainty_advocate")
    coverage = "partial" if advocate_label == "maybe" else "full"
    uncertainty_text = _clean_long_answer(debate).lower()
    authors_uncertain = gold == "maybe" or any(
        phrase in uncertainty_text
        for phrase in ("inconclusive", "unclear", "cannot", "insufficient", "further research")
    )
    unresolved = _conflict_summary(labels)
    if top_count == total and gold != "maybe":
        consensus_type = "consensus"
    elif gold == "maybe" and advocate_label == "maybe":
        consensus_type = "escalation"
    else:
        consensus_type = "differential"
    target = SupervisorDirectorOutput(
        debate_conflict_level=conflict_level,
        conclusiveness_score=max(1, min(10, round(10 * top_count / total))),
        unresolved_contradictions=unresolved,
        final_label=gold,
        consensus_type=consensus_type,
        rationale=_clean_long_answer(debate),
        primary_endpoint_answers_question=advocate_label != "maybe",
        findings_decisive_for_question=gold != "maybe" and len(counts) <= 2,
        authors_state_uncertainty=authors_uncertain,
        question_coverage=coverage,
    )
    history_entries = [
        [AgentRoundOpinion.model_validate(item) for item in round_entries]
        for round_entries in (debate.get("history") or [])
    ]
    shared_raw = (debate.get("debate_brief") or {}).get("shared_report")
    shared_report = None
    if isinstance(shared_raw, dict) and shared_raw:
        from app.agents.models import SharedDebateReport

        shared_report = SharedDebateReport.model_validate(shared_raw)
    transcript_text = build_full_debate_transcript(
        history_entries, shared_report=shared_report
    )
    if not transcript_text:
        transcript_text = json.dumps(debate.get("debate_brief") or {}, ensure_ascii=False)
    hint_raw = debate.get("biolinkbert_hint") or {}
    hint_text = json.dumps(hint_raw, ensure_ascii=False)
    bert_label = hint_raw.get("label") if isinstance(hint_raw, dict) else None
    prompt = SUPERVISOR_DIRECTOR_PROMPT.format(
        patient_case=str(debate.get("patient_case") or ""),
        full_debate_transcript=transcript_text,
        panel_vote_summary=build_panel_vote_summary(
            history_entries, biolinkbert_label=bert_label
        ),
        biolinkbert_hint=hint_text,
    )
    prompt += (
        "\n\nPydantic JSON schema (Director):\n"
        + json.dumps(SupervisorDirectorOutput.model_json_schema(), ensure_ascii=False)
    )
    return {
        "id": f"{debate.get('id')}:director",
        "pmid": str(debate.get("pmid") or ""),
        "task_type": "director",
        "gold_label": gold,
        "source_dataset": str(debate.get("source_dataset") or ""),
        "messages": [
            {"role": "system", "content": JSON_ONLY_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target.model_dump_json()},
        ],
        "meta": {
            "rounds_completed": len(debate.get("history") or []),
            "role_labels": labels,
        },
    }


def build_moderator_sft_record(debate: dict) -> dict:
    """Build a deterministic, grounded Moderator chat example."""
    history = debate.get("history") or []
    if not history:
        raise ValueError("Moderator example requires at least one debate round.")
    round_one = [AgentRoundOpinion.model_validate(item) for item in history[0]]
    labels = _role_labels(round_one)
    counts = _label_counts_from_roles(labels)
    gold = str(debate.get("gold_label") or "").strip().lower()
    if gold not in LABELS:
        raise ValueError(f"Invalid gold label: {gold!r}")
    agreements = [
        f"{count} agents agree on label {label}."
        for label, count in sorted(counts.items())
        if count >= 2
    ]
    contradictions = _conflict_summary(labels)
    instructions = [
        "generalist: cite the exact abstract sentence that supports the proposed final label.",
        "evidence_skeptic: distinguish a decisive primary result from ordinary study limitations.",
    ]
    if labels.get("uncertainty_advocate") == "maybe":
        instructions.append(
            "uncertainty_advocate: name the coverage gap or internal contradiction that blocks yes/no."
        )
    if contradictions:
        instructions.append(
            "All agents: resolve the listed label contradictions using only explicit abstract evidence."
        )
    residual = []
    if gold == "maybe":
        residual.append("The gold author conclusion remains inconclusive.")
    if labels.get("uncertainty_advocate") == "maybe":
        residual.append("Coverage or internal consistency remains disputed.")
    target = SupervisorModerationOutput(
        agreements=agreements,
        contradictions=contradictions,
        round_instructions=instructions,
        primary_endpoint_result=_clean_long_answer(debate),
        author_conclusion=gold,
        residual_uncertainty=residual,
    )
    rendered = format_opinions_for_supervisor(round_one, peer_context="nl")
    prompt = SUPERVISOR_MODERATOR_PROMPT.format(
        patient_case=str(debate.get("patient_case") or ""),
        previous_round_opinions=rendered,
    )
    return {
        "id": f"{debate.get('id')}:moderator",
        "pmid": str(debate.get("pmid") or ""),
        "task_type": "moderator",
        "gold_label": gold,
        "source_dataset": str(debate.get("source_dataset") or ""),
        "messages": [
            {"role": "system", "content": JSON_ONLY_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target.model_dump_json()},
        ],
        "meta": {"round": 1, "role_labels": labels},
    }


def validate_sft_record(
    record: dict,
    *,
    blocked_pmids: set[str],
    max_estimated_tokens: int = 4096,
) -> None:
    pmid = str(record.get("pmid") or "")
    if pmid in blocked_pmids:
        raise ValueError(f"Record contains held-out PMID {pmid}.")
    messages = record.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("SFT record must contain system/user/assistant messages.")
    estimated_tokens = sum(len(str(item.get("content") or "")) for item in messages) // 4
    if estimated_tokens > max_estimated_tokens:
        raise ValueError(
            f"Estimated context {estimated_tokens} exceeds {max_estimated_tokens} tokens."
        )
    task_type = str(record.get("task_type") or "")
    if task_type == "director":
        target = SupervisorDirectorOutput.model_validate_json(messages[-1]["content"])
        if target.final_label != record.get("gold_label"):
            raise ValueError("Director target label differs from gold label.")
    elif task_type == "moderator":
        target = SupervisorModerationOutput.model_validate_json(messages[-1]["content"])
        if target.author_conclusion != record.get("gold_label"):
            raise ValueError("Moderator author conclusion differs from gold label.")
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")


def _read_jsonl_dicts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _write_dict_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def class_aware_director_replay(
    records: list[dict],
    *,
    min_maybe_share: float = 0.20,
) -> list[dict]:
    """Deterministically replay scarce gold-maybe Director examples."""
    if not 0.0 < min_maybe_share < 1.0:
        raise ValueError("min_maybe_share must be between 0 and 1")
    result = list(records)
    maybe_records = [record for record in records if record.get("gold_label") == "maybe"]
    if not maybe_records:
        return result
    maybe_count = len(maybe_records)
    replay_index = 0
    while maybe_count / len(result) < min_maybe_share:
        source = maybe_records[replay_index % len(maybe_records)]
        result.append(
            {
                **source,
                "id": f"{source['id']}:class-replay{replay_index}",
                "meta": {
                    **dict(source.get("meta") or {}),
                    "class_replay": replay_index,
                },
            }
        )
        maybe_count += 1
        replay_index += 1
    random.Random(47).shuffle(result)
    return result


def materialize_sft_files(
    *,
    debate_file: Path,
    output_dir: Path,
    split_name: str,
    heldout_eval: Path,
    max_estimated_tokens: int = 4096,
) -> dict:
    blocked = _heldout_pmids(heldout_eval)
    debates = _read_jsonl_dicts(debate_file)
    directors: list[dict] = []
    moderators: list[dict] = []
    rejected: list[dict] = []
    for debate in debates:
        try:
            director = build_director_sft_record(debate)
            moderator = build_moderator_sft_record(debate)
            validate_sft_record(
                director,
                blocked_pmids=blocked,
                max_estimated_tokens=max_estimated_tokens,
            )
            validate_sft_record(
                moderator,
                blocked_pmids=blocked,
                max_estimated_tokens=max_estimated_tokens,
            )
        except (ValueError, TypeError) as exc:
            rejected.append({"id": debate.get("id"), "reason": str(exc)})
            continue
        directors.append(director)
        moderators.append(moderator)

    output_dir.mkdir(parents=True, exist_ok=True)
    director_path = output_dir / f"director_{split_name}.jsonl"
    moderator_path = output_dir / f"moderator_{split_name}.jsonl"
    multitask_path = output_dir / f"multitask_{split_name}.jsonl"
    train_directors = (
        class_aware_director_replay(directors)
        if split_name == "train"
        else directors
    )
    _write_dict_jsonl(director_path, train_directors)
    _write_dict_jsonl(moderator_path, moderators)
    # Two Director copies to one Moderator gives a deterministic 2:1 replay mix.
    multitask = [
        {**record, "id": f"{record['id']}:replay{repeat}"}
        for repeat in range(2)
        for record in train_directors
    ] + moderators
    random.Random(47).shuffle(multitask)
    _write_dict_jsonl(multitask_path, multitask)
    report = {
        "split": split_name,
        "debates": len(debates),
        "director_records": len(train_directors),
        "director_unique_records": len(directors),
        "moderator_records": len(moderators),
        "multitask_records": len(multitask),
        "rejected": rejected,
        "heldout_overlap_count": 0,
    }
    (output_dir / f"sft_{split_name}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, examples: list[ClassifierExample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")


def _counts(examples: list[ClassifierExample], attribute: str) -> dict[str, int]:
    counts = Counter(str(getattr(example, attribute)) for example in examples)
    return dict(sorted(counts.items()))


def build_safe_source_splits(
    *,
    pqal_path: Path,
    pqaa_path: Path,
    heldout_eval: Path,
    output_dir: Path,
    seed: int = 47,
    train_fraction: float = 0.70,
    dev_fraction: float = 0.15,
    max_pqaa_per_label: int = 1500,
) -> dict:
    examples, blocked = load_safe_examples(
        pqal_path=pqal_path,
        pqaa_path=pqaa_path,
        heldout_eval=heldout_eval,
    )
    splits = split_examples_three_way(
        examples,
        seed=seed,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
        max_pqaa_per_label=max_pqaa_per_label,
    )
    assert_no_heldout_overlap(splits, blocked)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        _write_jsonl(output_dir / f"source_{split_name}.jsonl", rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "train_fraction": train_fraction,
        "dev_fraction": dev_fraction,
        "internal_test_fraction": 1.0 - train_fraction - dev_fraction,
        "max_pqaa_per_label": max_pqaa_per_label,
        "sources": {
            "pqa_l": {"path": str(pqal_path), "sha256": _sha256(pqal_path)},
            "pqa_a": {"path": str(pqaa_path), "sha256": _sha256(pqaa_path)},
            "heldout_eval": {
                "path": str(heldout_eval),
                "sha256": _sha256(heldout_eval),
            },
        },
        "heldout_pmids": len(blocked),
        "heldout_overlap_count": 0,
        "splits": {
            name: {
                "count": len(rows),
                "labels": _counts(rows, "label"),
                "sources": _counts(rows, "source_dataset"),
                "path": str(output_dir / f"source_{name}.jsonl"),
            }
            for name, rows in splits.items()
        },
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debate-file",
        type=Path,
        help="Build SFT records from a generated debate JSONL instead of source splits.",
    )
    parser.add_argument(
        "--split-name",
        choices=SPLIT_NAMES,
        help="Required with --debate-file.",
    )
    parser.add_argument(
        "--pqal",
        type=Path,
        default=Path("data/raw/pubmedqa_official/data/ori_pqal.json"),
    )
    parser.add_argument(
        "--pqaa",
        type=Path,
        default=Path("data/raw/pubmedqa_official/data/ori_pqaa.json"),
    )
    parser.add_argument(
        "--heldout-eval",
        type=Path,
        default=Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/sft/pubmedqa_supervisor"),
    )
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--max-pqaa-per-label", type=int, default=1500)
    parser.add_argument("--max-estimated-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debate_file is not None:
        if args.split_name is None:
            raise SystemExit("--split-name is required with --debate-file.")
        report = materialize_sft_files(
            debate_file=args.debate_file,
            output_dir=args.output_dir,
            split_name=args.split_name,
            heldout_eval=args.heldout_eval,
            max_estimated_tokens=args.max_estimated_tokens,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    manifest = build_safe_source_splits(
        pqal_path=args.pqal,
        pqaa_path=args.pqaa,
        heldout_eval=args.heldout_eval,
        output_dir=args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        max_pqaa_per_label=args.max_pqaa_per_label,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
