"""Snapshot and version debate prompts for each evaluation / training run."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import app.agents.prompts as prompts_module

# Named string / mapping constants that define agent + supervisor behavior.
_PROMPT_ATTRS: tuple[str, ...] = (
    "CLINICAL_OPINION_SCHEMA",
    "DEFENSE_OPINION_SCHEMA",
    "SAFETY_OPINION_SCHEMA",
    "SUPERVISOR_MODERATOR_PROMPT",
    "SUPERVISOR_DIRECTOR_PROMPT",
    "EVIDENCE_SKEPTIC_PROMPT",
    "UNCERTAINTY_ADVOCATE_PROMPT",
    "PERSONA_INSTRUCTIONS",
    "PUBMEDQA_PERSONA_INSTRUCTIONS",
    "PUBMEDQA_LABEL_RULE",
    "PUBMEDQA_LABEL_RULE_ADVOCATE",
    "PUBMEDQA_COMPACT_SCHEMA",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def collect_prompt_bundle(module: Any = prompts_module) -> dict[str, Any]:
    """Collect versioned prompt texts from ``app.agents.prompts``."""
    bundle: dict[str, Any] = {}
    for name in _PROMPT_ATTRS:
        if hasattr(module, name):
            bundle[name] = _jsonable(getattr(module, name))
    # Always record uncertainty persona set for auditability.
    if hasattr(module, "PUBMEDQA_UNCERTAINTY_PERSONAS"):
        bundle["PUBMEDQA_UNCERTAINTY_PERSONAS"] = _jsonable(
            getattr(module, "PUBMEDQA_UNCERTAINTY_PERSONAS")
        )
    return bundle


def hash_prompt_bundle(bundle: Mapping[str, Any]) -> str:
    """Stable SHA-256 over canonical JSON of the prompt bundle."""
    payload = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt_version_record(
    *,
    run_label: str,
    script: str,
    extra_meta: Mapping[str, Any] | None = None,
    module: Any = prompts_module,
) -> dict[str, Any]:
    """Build a full prompt snapshot record for one run."""
    bundle = collect_prompt_bundle(module)
    digest = hash_prompt_bundle(bundle)
    prompts_path = Path(getattr(module, "__file__", "") or "")
    source_text = ""
    if prompts_path.is_file():
        source_text = prompts_path.read_text(encoding="utf-8")
    source_sha = (
        hashlib.sha256(source_text.encode("utf-8")).hexdigest() if source_text else None
    )
    record: dict[str, Any] = {
        "prompt_version": digest[:12],
        "prompt_sha256": digest,
        "prompts_py_sha256": source_sha,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "script": script,
        "source_file": str(prompts_path) if prompts_path else "app.agents.prompts",
        "prompts": bundle,
        "meta": dict(extra_meta or {}),
    }
    return record


def snapshot_prompts_for_run(
    report_dir: Path | str,
    *,
    run_label: str,
    script: str = "scripts/agents/evaluate_debate_pubmedqa.py",
    extra_meta: Mapping[str, Any] | None = None,
    registry_name: str = "prompt_versions.jsonl",
    also_copy_prompts_py: bool = True,
) -> dict[str, Any]:
    """
    Write per-run prompt snapshot next to the report and append to a registry.

    Artifacts:
    - ``{report_dir}/{run_label}.prompts.json`` — full bundle + hashes
    - ``{report_dir}/{run_label}.prompts.py`` — copy of ``prompts.py`` (optional)
    - ``{report_dir}/{registry_name}`` — append-only index (label, hashes, path)
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    record = build_prompt_version_record(
        run_label=run_label,
        script=script,
        extra_meta=extra_meta,
    )
    snapshot_path = report_dir / f"{run_label}.prompts.json"
    snapshot_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    record["snapshot_path"] = str(snapshot_path)

    if also_copy_prompts_py:
        src = Path(record["source_file"])
        if src.is_file():
            py_copy = report_dir / f"{run_label}.prompts.py"
            py_copy.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            record["prompts_py_copy"] = str(py_copy)

    registry_path = report_dir / registry_name
    index_row = {
        "captured_at": record["captured_at"],
        "run_label": run_label,
        "prompt_version": record["prompt_version"],
        "prompt_sha256": record["prompt_sha256"],
        "prompts_py_sha256": record["prompts_py_sha256"],
        "snapshot_path": str(snapshot_path),
        "script": script,
        "meta": record.get("meta") or {},
    }
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(index_row, ensure_ascii=False) + "\n")
    record["registry_path"] = str(registry_path)
    return record
