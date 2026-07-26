"""Evidence-anchored condition audit for the `maybe` class (NLI-style).

Motivation
----------
The debate agents' *self-reported* uncertainty is a weak signal on PubMedQA:
agents (and the BioLinkBERT classifier) commit confidently to yes/no even when
the abstract does not settle the question, and their opinions look
stylistically identical whether the evidence is conclusive or not (semantic
entropy over opinions barely separates the classes).

Following the abstention-aware verification literature (arXiv 2602.14189,
"Knowing When Not to Answer"), we instead anchor uncertainty in the *evidence*
itself. The research question is decomposed into minimal yes/no sub-conditions;
each is audited against the abstract with a natural-language-inference style
verdict: SUPPORTED, REFUTED, or SILENT (neither entailed nor contradicted).

A question whose conditions are partly SILENT/mixed is genuinely inconclusive,
which is exactly the definition of the PubMedQA ``maybe`` label. The
``audit_score`` in [0, 1] measures this inconclusiveness and feeds the routing
decision as an evidence-grounded feature — independent of what the debate
agents claim about themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from app.schemas import ChatMessage

_AUDIT_SYSTEM = """
You are an evidence auditor for biomedical yes/no/maybe questions. You do NOT
answer the question. You decompose it into 2-4 minimal, independently checkable
sub-conditions, then judge EACH sub-condition strictly against the provided
abstract using three verdicts:
- "supported": the abstract directly and clearly entails this sub-condition
- "refuted": the abstract directly and clearly contradicts it
- "silent": the abstract does not decisively settle it (not measured, only
  hinted, indirect, underpowered, or mixed)

Be strict: hedged author language ("may", "suggests", "trend") is "silent",
not "supported". Return ONLY a compact JSON object:
{"conditions":[{"claim":"...","verdict":"supported|refuted|silent"}]}
No markdown, no commentary.
""".strip()


@dataclass
class EvidenceAudit:
    conditions: list[dict[str, str]] = field(default_factory=list)
    supported: int = 0
    refuted: int = 0
    silent: int = 0
    audit_score: float = 0.0
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "conditions": self.conditions,
            "supported": self.supported,
            "refuted": self.refuted,
            "silent": self.silent,
            "audit_score": self.audit_score,
            "error": self.error,
        }


def _parse_audit(raw: str) -> list[dict[str, str]]:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    data = json.loads(text)
    conditions = data.get("conditions") or []
    parsed: list[dict[str, str]] = []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        verdict = str(cond.get("verdict", "")).strip().lower()
        if verdict not in {"supported", "refuted", "silent"}:
            verdict = "silent"
        parsed.append({"claim": str(cond.get("claim", "")).strip(), "verdict": verdict})
    return parsed


def _score_from_conditions(conditions: list[dict[str, str]]) -> tuple[int, int, int, float]:
    supported = sum(1 for c in conditions if c["verdict"] == "supported")
    refuted = sum(1 for c in conditions if c["verdict"] == "refuted")
    silent = sum(1 for c in conditions if c["verdict"] == "silent")
    total = len(conditions)
    if total == 0:
        return 0, 0, 0, 0.0

    # A question is conclusive when its conditions point coherently one way
    # (all supported -> yes; all refuted -> no). It is inconclusive when
    # conditions are silent OR mixed between supported and refuted. We combine
    # the silent fraction with a "directional conflict" term.
    silent_frac = silent / total
    decided = supported + refuted
    if decided > 0:
        conflict = 1.0 - abs(supported - refuted) / decided  # 0 if unanimous, 1 if evenly split
    else:
        conflict = 1.0
    audit_score = min(1.0, 0.65 * silent_frac + 0.35 * conflict)
    return supported, refuted, silent, audit_score


async def audit_evidence(
    backend,
    *,
    question: str,
    evidence: str,
    temperature: float = 0.0,
    max_evidence_chars: int = 3000,
) -> EvidenceAudit:
    """Run the LLM condition audit for one case; never raises.

    ``backend`` is any object exposing ``async complete(messages, temperature=...)``
    (the same interface the debate agents use).
    """
    evidence = (evidence or "").strip()
    if len(evidence) > max_evidence_chars:
        evidence = evidence[:max_evidence_chars] + "…"
    messages = [
        ChatMessage(role="system", content=_AUDIT_SYSTEM),
        ChatMessage(
            role="user",
            content=(
                f"RESEARCH QUESTION:\n{question.strip()}\n\n"
                f"ABSTRACT / EVIDENCE:\n{evidence}\n\n"
                "Decompose the question and audit each sub-condition."
            ),
        ),
    ]
    try:
        raw = await backend.complete(messages, temperature=temperature)
    except Exception as exc:  # pragma: no cover - defensive
        return EvidenceAudit(error=f"backend error: {exc}")
    try:
        conditions = _parse_audit(raw)
    except Exception as exc:
        return EvidenceAudit(error=f"parse error: {exc}")
    supported, refuted, silent, score = _score_from_conditions(conditions)
    return EvidenceAudit(
        conditions=conditions,
        supported=supported,
        refuted=refuted,
        silent=silent,
        audit_score=score,
    )
