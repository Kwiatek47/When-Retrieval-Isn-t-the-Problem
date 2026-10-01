"""Versioned prompts for the hypothesis probes on PubMedQA's ``maybe``, and a per-run snapshot.

Every prompt is a frozen ``PromptSpec`` with an id ``name@version`` and a content hash over
its system text, user template and response schema. Rules:

  - a spec's text never changes once any run used it; a change is a new ``version``;
  - ``registered`` specs back a pre-registered test (the hash is pinned by a unit test);
    ``candidate`` specs are drafts, chosen on the human-calibrated sheet before use;
  - every run calls ``snapshot_run`` first: it writes ``{run_label}.prompts.json`` next to
    the run's output and appends one row to ``prompt_versions.jsonl`` in the same format
    as the debate runs (``app/agents/prompt_versioning.py``), plus model, decoding
    options, command line and git commit.

The module is stdlib-only so the probes run without the app's dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: int
    status: str  # "registered" | "candidate"
    purpose: str
    system: str
    user: str
    schema: dict
    fields: tuple[str, ...]
    notes: str = ""
    options: dict = field(default_factory=lambda: {"temperature": 0, "num_ctx": 4096})

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def sha(self) -> str:
        """12-char content hash; unchanged formula keeps ``conditionality@1`` at 77e624205b2a."""
        payload = self.system + "\n" + self.user + json.dumps(self.schema, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def messages(self, **values: str) -> list[dict]:
        missing = set(self.fields) - set(values)
        if missing:
            raise KeyError(f"{self.id} needs {sorted(missing)}")
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user.format(**values)},
        ]

    def record(self) -> dict:
        return {
            "id": self.id,
            "sha": self.sha,
            "status": self.status,
            "purpose": self.purpose,
            "notes": self.notes,
            "system": self.system,
            "user": self.user,
            "schema": self.schema,
            "options": self.options,
        }


# --- H1b: conditionality of a passage's answer -------------------------------------------

_CONDITIONALITY_SYSTEM = (
    "You annotate biomedical research abstracts. You judge whether a passage answers a "
    "research question unconditionally or only conditionally. You do not judge whether "
    "the answer is yes or no, and you do not judge how cautious the wording is."
)

CONDITIONALITY_V1 = PromptSpec(
    name="conditionality",
    version=1,
    status="registered",
    purpose="H1b: rate how conditional a passage's answer is (0-2), conclusion vs RESULTS.",
    notes="Pre-registered 2026-09-24 in commit 73cd9bf; full run on 1000 PQA-L questions done.",
    system=_CONDITIONALITY_SYSTEM,
    user="""Research question: {question}

Passage:
{passage}

Does this passage, taken on its own, give a CONDITIONAL answer to the research question?
An answer is conditional when it differs across subgroups, outcomes, settings, doses or time
points, holds only for some of them, or when the passage reports both supporting and
non-supporting findings for the question.

Rate conditionality:
0 = not conditional: one answer that holds as stated, or the passage does not address the question
1 = partly: one main answer with a minor qualification or restriction
2 = conditional: the answer clearly depends on a subgroup or condition, or findings point in different directions

Ignore hedging words such as "may", "suggest" or "further studies are needed"; they do not
make an answer conditional by themselves.
Return JSON: {{"conditionality": 0, 1 or 2, "evidence": "<quote of at most 20 words, or empty>"}}""",
    schema={
        "type": "object",
        "properties": {
            "conditionality": {"type": "integer", "enum": [0, 1, 2]},
            "evidence": {"type": "string"},
        },
        "required": ["conditionality", "evidence"],
    },
    fields=("question", "passage"),
    options={"temperature": 0, "num_ctx": 4096, "num_predict": 120},
)

CONDITIONALITY_V2 = PromptSpec(
    name="conditionality",
    version=2,
    status="candidate",
    purpose="H1b: same scale as v1, built to remove v1's weak points; choose v1 or v2 on the calibration sheet.",
    notes=(
        "Changes vs v1: judge only findings about this question and ignore passage length "
        "(v1 rated RESULTS as more conditional even for 'yes'); three synthetic examples, one "
        "per rating; the quote comes before the rating so the rating rests on it."
    ),
    system=_CONDITIONALITY_SYSTEM,
    user="""Research question: {question}

Passage:
{passage}

Task: decide whether the passage gives a CONDITIONAL answer to this research question.

Consider only what the passage says about the research question itself. Findings about other
questions do not count, and neither does the number of findings: a long passage with many
results is not conditional if they all point to the same answer.

An answer is CONDITIONAL when, for this question, it
- holds for some subgroups, outcomes, settings, doses or time points but not for others, or
- is supported by some reported findings and contradicted by others.
Hedging words ("may", "suggest", "further studies are needed") and study limitations do not
make an answer conditional.

Examples (invented, not from the data):
- Q: Does drug A lower blood pressure? "A lowered systolic pressure by 12 mmHg (p<0.001) in all age groups." -> 0
- Q: Does drug A lower blood pressure? "A lowered pressure overall; the effect was smaller in patients over 80." -> 1
- Q: Does drug A lower blood pressure? "A lowered pressure in women (p=0.01) but not in men (p=0.64)." -> 2

First quote the words that decide your rating (at most 20 words; empty if the passage does not
address the question). Then rate:
0 = not conditional, or the passage does not address the question
1 = one main answer with a minor qualification or restriction
2 = conditional
Return JSON: {{"evidence": "<quote>", "conditionality": 0, 1 or 2}}""",
    schema={
        "type": "object",
        "properties": {
            "evidence": {"type": "string"},
            "conditionality": {"type": "integer", "enum": [0, 1, 2]},
        },
        "required": ["evidence", "conditionality"],
    },
    fields=("question", "passage"),
    options={"temperature": 0, "num_ctx": 4096, "num_predict": 160},
)

# --- Label probes: yes / no / maybe from the text a model is given ------------------------
# Used by H1-direct (same prompt, abstract with vs without the conclusion), H2 (compare with
# the annotator who saw no conclusion) and RQ10 / RQ9a (how the prompt sets the maybe rate).

_LABEL_SYSTEM = (
    "You answer biomedical research questions from the text of one study abstract. "
    "Use only the text you are given, not outside knowledge."
)

_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "label": {"type": "string", "enum": ["yes", "no", "maybe"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["rationale", "label", "confidence"],
}

_LABEL_OUTPUT = """First give a short rationale (at most 40 words), then the answer, then your
confidence from 0 to 100 that the answer is correct.
Return JSON: {{"rationale": "...", "label": "yes" | "no" | "maybe", "confidence": 0-100}}"""

_MAYBE_DEFINITION = """Answer:
- yes: the findings support "yes" to the question as asked;
- no: the findings support "no" to the question as asked;
- maybe: the answer holds for some subgroups, outcomes or conditions but not for others, or
  the findings are mixed and do not settle the question.
Do not answer maybe only because the study has limitations or the wording is cautious."""

LABEL_MINIMAL_V1 = PromptSpec(
    name="label-minimal",
    version=1,
    status="candidate",
    purpose="Default behaviour: yes/no/maybe with no definition of maybe (RQ10 baseline).",
    system=_LABEL_SYSTEM,
    user="""Research question: {question}

{evidence}

Answer the research question with yes, no or maybe, based only on the text above.
""" + _LABEL_OUTPUT,
    schema=_LABEL_SCHEMA,
    fields=("question", "evidence"),
    options={"temperature": 0, "num_ctx": 4096, "num_predict": 200},
)

LABEL_DEFINED_V1 = PromptSpec(
    name="label-defined",
    version=1,
    status="candidate",
    purpose="Yes/no/maybe with PubMedQA's definition of maybe (Jin et al. 2019); main probe prompt.",
    notes="The last line of the definition follows H1: hedging is not what makes an answer maybe.",
    system=_LABEL_SYSTEM,
    user="""Research question: {question}

{evidence}

""" + _MAYBE_DEFINITION + "\n\n" + _LABEL_OUTPUT,
    schema=_LABEL_SCHEMA,
    fields=("question", "evidence"),
    options={"temperature": 0, "num_ctx": 4096, "num_predict": 200},
)

LABEL_DEFINED_PRIOR_V1 = PromptSpec(
    name="label-defined-prior",
    version=1,
    status="candidate",
    purpose="As label-defined@1 plus the PQA-L label base rates (RQ9a for LLMs).",
    notes="Rates from all 1000 PQA-L questions: 552 yes, 338 no, 110 maybe.",
    system=_LABEL_SYSTEM,
    user="""Research question: {question}

{evidence}

""" + _MAYBE_DEFINITION + """
In this collection about 55% of questions are answered yes, 34% no and 11% maybe.

""" + _LABEL_OUTPUT,
    schema=_LABEL_SCHEMA,
    fields=("question", "evidence"),
    options={"temperature": 0, "num_ctx": 4096, "num_predict": 200},
)

PROMPTS: dict[str, PromptSpec] = {
    spec.id: spec
    for spec in (
        CONDITIONALITY_V1,
        CONDITIONALITY_V2,
        LABEL_MINIMAL_V1,
        LABEL_DEFINED_V1,
        LABEL_DEFINED_PRIOR_V1,
    )
}


def get_prompt(prompt_id: str) -> PromptSpec:
    try:
        return PROMPTS[prompt_id]
    except KeyError:
        raise SystemExit(f"unknown prompt {prompt_id!r}; known: {sorted(PROMPTS)}") from None


# --- Model call ---------------------------------------------------------------------------


def ollama_chat_json(
    base_url: str,
    model: str,
    spec: PromptSpec,
    messages: list[dict],
    seed: int,
    timeout: float,
) -> str:
    """One schema-constrained Ollama chat call with the spec's decoding options."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "60m",
        "format": spec.schema,
        "options": {**spec.options, "seed": seed},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["message"]["content"]


# --- Per-run snapshot ---------------------------------------------------------------------


def _git_state() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def snapshot_run(
    out_path: Path,
    specs: list[PromptSpec],
    *,
    script: str,
    model: str,
    meta: dict[str, Any] | None = None,
    registry_name: str = "prompt_versions.jsonl",
) -> dict:
    """Write ``{run_label}.prompts.json`` next to ``out_path`` and append the registry row.

    ``run_label`` is the output file's stem, so the snapshot sits next to the data it explains.
    Called once per invocation, including resumed runs: each row records one execution.
    """
    report_dir = out_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    run_label = out_path.stem
    combined = hashlib.sha256("|".join(f"{s.id}:{s.sha}" for s in specs).encode("utf-8")).hexdigest()
    record = {
        "prompt_version": combined[:12],
        "prompt_sha256": combined,
        "prompts_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "script": script,
        "source_file": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "prompts": {s.id: s.record() for s in specs},
        "meta": {
            "model": model,
            "prompt_ids": [s.id for s in specs],
            "argv": sys.argv,
            "git": _git_state(),
            **(meta or {}),
        },
    }
    snapshot_path = report_dir / f"{run_label}.prompts.json"
    snapshot_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "captured_at": record["captured_at"],
        "run_label": run_label,
        "prompt_version": record["prompt_version"],
        "prompt_sha256": record["prompt_sha256"],
        "prompts_py_sha256": record["prompts_py_sha256"],
        "snapshot_path": str(snapshot_path),
        "script": script,
        "meta": {**record["meta"], "prompt_shas": {s.id: s.sha for s in specs}},
    }
    with (report_dir / registry_name).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return record
