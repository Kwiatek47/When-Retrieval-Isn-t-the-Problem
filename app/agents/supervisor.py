"""Supervisor LLM: two-layer request routing and rigor-check closure.

Two responsibilities, matching the supervisor pattern the baseline debate lacks:

1. **Routing** (`eligibility_gate` + `SupervisorAgent.route`). Under information
   asymmetry an agent that cannot see a fact must ask for it. Requests are routed
   in two layers: a deterministic gate decides which segments are *allowed* to
   answer (pure Python, no model call, no chance of the model inventing a holder
   for evidence nobody has), and only then does the LLM pick among the eligible
   ones. A router that was LLM-only could route a question to a segment that
   simply does not contain the answer.

2. **Closure** (`SupervisorAgent.moderate`). Instead of counting votes, the
   supervisor answers a fixed list of rigor checks and lets those determine the
   label. Vote counting is what lets a panel be confidently wrong together; the
   checks are what a majority cannot outvote.

Like `ClinicalAgent`, nothing here raises. A failed supervisor returns a verdict
carrying `error`, and the orchestrator falls back to majority voting with the
fallback recorded in the reported rule rather than hidden.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.agents.backends import InferenceBackend
from app.agents.prompts import build_router_messages, build_supervisor_messages
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)

_LABELS = ("yes", "no", "maybe")

RIGOR_CHECK_IDS = (
    "question_addressed",
    "direction_established",
    "opposite_reading_excluded",
    "hedging_absent",
)

# Words too common to discriminate between evidence segments; keeping them would
# make the eligibility gate match every segment and stop being a gate at all.
_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for from with without by as is are was were be been
    being do does did done have has had having this that these those it its they them their
    there here what which who whom whose when where why how not no nor so than then too very
    can could should would may might must will shall about into over under between among any
    all both each few more most other some such only own same s t just don now
    """.split()
)


@dataclass
class SupervisorVerdict:
    """Outcome of the supervisor's closing moderation call."""

    label: str | None = None
    confidence: float = 0.0
    rule: str = ""
    rationale: str = ""
    rigor_checks: dict[str, bool] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.label in _LABELS and not self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "rule": self.rule,
            "rationale": self.rationale,
            "rigor_checks": dict(self.rigor_checks),
            "error": self.error,
        }


def tokenize(text: str) -> set[str]:
    """Content words of a string, lowercased, stopwords and short tokens dropped."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def eligibility_gate(
    request: str,
    segments: dict[str, str],
    *,
    min_overlap: int = 1,
) -> list[str]:
    """Deterministic layer: which segments are *allowed* to answer this request.

    A segment qualifies when it shares at least `min_overlap` content words with
    the request. Pure string work, no model call, so a segment can never be
    nominated to answer a question about evidence it does not hold. Segments are
    returned best-overlap first so a downstream LLM sees the strongest candidates
    at the top.
    """
    request_terms = tokenize(request)
    if not request_terms:
        return []
    scored: list[tuple[int, str]] = []
    for segment_id, text in segments.items():
        overlap = len(request_terms & tokenize(text))
        if overlap >= min_overlap:
            scored.append((overlap, segment_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [segment_id for _, segment_id in scored]


class SupervisorAgent:
    """LLM layer of the supervisor: routes information requests and closes the debate."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        temperature: float = 0.0,
    ) -> None:
        self.backend = backend
        self.temperature = temperature

    async def moderate(
        self,
        *,
        question: str,
        arguments: list[dict[str, Any]],
        evidence_hint: Any = None,
    ) -> SupervisorVerdict:
        """Close the debate by rigor checks. Never raises."""
        messages = build_supervisor_messages(
            question=question,
            arguments=arguments,
            evidence_hint=evidence_hint,
        )
        raw = await self._complete_or_none(messages, self.temperature)
        verdict = _parse_verdict(raw) if raw is not None else None
        if verdict is not None:
            return verdict

        repair_messages = build_supervisor_messages(
            question=question,
            arguments=arguments,
            evidence_hint=evidence_hint,
            repair=True,
        )
        raw_retry = await self._complete_or_none(repair_messages, 0.0)
        verdict = _parse_verdict(raw_retry) if raw_retry is not None else None
        if verdict is not None:
            return verdict

        logger.warning("Supervisor moderation failed; caller should fall back to voting.")
        return SupervisorVerdict(error="Supervisor returned no usable verdict.")

    async def route(
        self,
        *,
        question: str,
        requests: dict[str, str],
        eligible: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Pick, among the deterministically eligible segments, who answers what.

        `eligible` is the gate's output and is authoritative: any segment the LLM
        names that the gate did not allow is dropped. On failure we fall back to
        the gate's own top choice, so routing degrades to deterministic rather
        than stopping.
        """
        routable = {rid: text for rid, text in requests.items() if eligible.get(rid)}
        if not routable:
            return {rid: [] for rid in requests}

        messages = build_router_messages(
            question=question,
            requests=[{"id": rid, "question": text} for rid, text in routable.items()],
            segments=[
                {"request_id": rid, "eligible_segment_ids": eligible[rid]} for rid in routable
            ],
        )
        raw = await self._complete_or_none(messages, self.temperature)
        parsed = _parse_routing(raw) if raw is not None else None
        if parsed is None:
            logger.warning("Supervisor routing failed; using deterministic gate order.")
            return {rid: eligible.get(rid, [])[:1] for rid in requests}

        routed: dict[str, list[str]] = {}
        for rid in requests:
            allowed = eligible.get(rid, [])
            chosen = [sid for sid in parsed.get(rid, []) if sid in allowed]
            routed[rid] = chosen
        return routed

    async def _complete_or_none(
        self, messages: list[ChatMessage], temperature: float
    ) -> str | None:
        try:
            return await self.backend.complete(messages, temperature=temperature)
        except Exception:
            logger.warning("Supervisor backend call failed.", exc_info=True)
            return None


def apply_decision_rule(checks: dict[str, bool], proposed: str | None) -> tuple[str, str]:
    """Enforce the documented decision rule in code, not just in the prompt.

    The prompt states the rule, but a model that answers the checks correctly can
    still contradict itself on the label. Recomputing here means the reported
    rigor checks always explain the reported label.
    """
    failed = [check for check in RIGOR_CHECK_IDS if checks.get(check) is False]
    if failed:
        return "maybe", f"rigor_check_failed:{failed[0]}"
    if proposed in ("yes", "no"):
        return proposed, "rigor_checks_passed"
    if proposed == "maybe":
        return "maybe", "rigor_checks_passed"
    return "maybe", "no_label_proposed"


def _parse_verdict(raw: str) -> SupervisorVerdict | None:
    payload = _loads_json_object(raw)
    if payload is None:
        return None

    checks_raw = payload.get("rigor_checks")
    checks: dict[str, bool] = {}
    if isinstance(checks_raw, dict):
        for key, value in checks_raw.items():
            if key in RIGOR_CHECK_IDS:
                checks[key] = _as_bool(value)

    proposed = str(payload.get("label") or "").strip().lower()
    proposed = proposed if proposed in _LABELS else None
    if proposed is None and not checks:
        return None

    label, rule = apply_decision_rule(checks, proposed)
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return SupervisorVerdict(
        label=label,
        confidence=max(0.0, min(1.0, confidence)),
        rule=f"supervisor_moderation:{rule}",
        rationale=str(payload.get("rationale") or "").strip(),
        rigor_checks=checks,
    )


def _parse_routing(raw: str) -> dict[str, list[str]] | None:
    payload = _loads_json_object(raw)
    if payload is None:
        return None
    routed: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            routed[str(key)] = [value]
        elif isinstance(value, list):
            routed[str(key)] = [str(item) for item in value]
    return routed


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _loads_json_object(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating markdown fences and surrounding prose."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None
