"""External NLI evidence audit for the `maybe` class (fixed cross-encoder).

Motivation
----------
The generative evidence audit (``evidence_audit.py``) asks a chat LLM to judge
whether an abstract settles a yes/no question. On PubMedQA that signal is at
chance (AUROC ~0.50-0.56) across qwen-7b/14b and a reasoning model — the LLM
almost always says "supported".

This module is the *external-auditor* control that directly extends the
abstention-aware verification literature (arXiv 2602.14189), which used a fixed
DeBERTa NLI cross-encoder rather than a generative model as the auditor. We ask a
dedicated NLI model whether the abstract (premise) *entails* the hypothesis
"the answer to the research question is yes" (and, symmetrically, "no"). If the
abstract neither entails nor contradicts either polarity — i.e. the model is
mostly ``neutral`` — the evidence is inconclusive, which is exactly the
definition of the PubMedQA ``maybe`` label.

The resulting ``audit_score`` in [0, 1] is high when the abstract is neutral /
undecided about the question. It is computed *without any generative LLM*, so it
isolates whether the "maybe is unrecoverable" result is a property of the data
(the abstract genuinely lacks the signal) or merely of using a chat model as the
auditor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_MODEL: Any = None
_ID2LABEL: dict[int, str] = {0: "contradiction", 1: "entailment", 2: "neutral"}


def _load_model() -> Any:
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder(_MODEL_NAME)
        # Trust the model's own label order if it exposes one.
        cfg = getattr(_MODEL, "config", None)
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict) and id2label:
            globals()["_ID2LABEL"] = {int(k): str(v).lower() for k, v in id2label.items()}
    return _MODEL


def _softmax(row: list[float]) -> list[float]:
    import math

    m = max(row)
    exps = [math.exp(x - m) for x in row]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


@dataclass
class NLIEvidenceAudit:
    """External-NLI inconclusiveness signal for one case."""

    p_entail_yes: float = 0.0
    p_contradict_yes: float = 0.0
    p_neutral_yes: float = 0.0
    p_entail_no: float = 0.0
    p_contradict_no: float = 0.0
    p_neutral_no: float = 0.0
    audit_score: float = 0.0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_entail_yes": self.p_entail_yes,
            "p_contradict_yes": self.p_contradict_yes,
            "p_neutral_yes": self.p_neutral_yes,
            "p_entail_no": self.p_entail_no,
            "p_contradict_no": self.p_contradict_no,
            "p_neutral_no": self.p_neutral_no,
            "audit_score": self.audit_score,
            "error": self.error,
        }


def _label_probs(model: Any, premise: str, hypothesis: str) -> dict[str, float]:
    logits = model.predict([(premise, hypothesis)])
    row = list(map(float, logits[0]))
    probs = _softmax(row)
    return {_ID2LABEL.get(i, str(i)): probs[i] for i in range(len(probs))}


def audit_evidence_nli(
    *,
    question: str,
    evidence: str,
    max_evidence_chars: int = 3000,
) -> NLIEvidenceAudit:
    """Score inconclusiveness of ``evidence`` w.r.t. ``question`` via DeBERTa-NLI.

    We build two hypotheses from the question and measure how *undecided* the
    abstract is about each polarity. ``audit_score`` is the mean neutrality across
    the yes/no hypotheses, boosted when neither polarity is clearly entailed:
    a conclusive abstract entails exactly one polarity (low score); an
    inconclusive one stays neutral on both (high score).
    """
    evidence = (evidence or "").strip()
    if len(evidence) > max_evidence_chars:
        evidence = evidence[:max_evidence_chars]
    q = question.strip().rstrip("?")
    hyp_yes = f"The answer to the question '{q}?' is yes."
    hyp_no = f"The answer to the question '{q}?' is no."
    try:
        model = _load_model()
        py = _label_probs(model, evidence, hyp_yes)
        pn = _label_probs(model, evidence, hyp_no)
    except Exception as exc:  # pragma: no cover - defensive
        return NLIEvidenceAudit(error=f"nli error: {exc}")

    neutrality = (py.get("neutral", 0.0) + pn.get("neutral", 0.0)) / 2.0
    max_entail = max(py.get("entailment", 0.0), pn.get("entailment", 0.0))
    # High when the abstract is neutral and does not strongly entail either
    # polarity; low when one polarity is clearly entailed.
    audit_score = max(0.0, min(1.0, 0.5 * neutrality + 0.5 * (1.0 - max_entail)))
    return NLIEvidenceAudit(
        p_entail_yes=py.get("entailment", 0.0),
        p_contradict_yes=py.get("contradiction", 0.0),
        p_neutral_yes=py.get("neutral", 0.0),
        p_entail_no=pn.get("entailment", 0.0),
        p_contradict_no=pn.get("contradiction", 0.0),
        p_neutral_no=pn.get("neutral", 0.0),
        audit_score=audit_score,
    )
