"""Deterministic, LLM-free citation gate for the Supervised Ledger Debate.

Every :class:`~app.agents.sld.ledger.Claim` and
:class:`~app.agents.sld.ledger.Gap` carries ``sentence_ids`` pointing into the
``S1..Sn`` dictionary from :mod:`app.agents.sld.segmentation`. This module
checks that contract before a claim is allowed to reach the Supervisor,
Round 2, or the Director:

1. every cited ``sentence_id`` must actually exist;
2. ``findings_auditor`` may only cite ``RESULTS``-tagged sentences;
3. the claim's own text must be lexically grounded in the sentence(s) it
   cites (whitespace/case-normalized token-coverage threshold) — this is what
   catches a claim whose citation exists but whose content was fabricated.

A claim that fails any check is dropped (set to ``None``, or removed from a
list), not silently repaired. ``grounding_score`` / ``dropped_claims`` turn
that into the first real hallucination metric in this repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypeVar

from app.agents.sld.ledger import (
    Claim,
    ConclusionReconstructorContribution,
    EvidenceLedger,
    FindingsAuditorContribution,
    Gap,
    GapAuditorContribution,
    LedgerConflict,
    QuestionFramerContribution,
    RoundTwoOpinion,
)

DEFAULT_COVERAGE_THRESHOLD = 0.5

_WORD_RE = re.compile(r"[a-z0-9]+")

T = TypeVar(
    "T",
    QuestionFramerContribution,
    FindingsAuditorContribution,
    GapAuditorContribution,
    ConclusionReconstructorContribution,
    RoundTwoOpinion,
)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _token_coverage(claim_text: str, cited_text: str) -> float:
    """Fraction of ``claim_text`` tokens that also appear in ``cited_text``.

    Whitespace/case are normalized away by tokenizing; this is a coverage
    ratio, not exact substring matching, so light paraphrase survives while
    claims sharing no vocabulary with their citation do not.
    """
    claim_tokens = _tokenize(claim_text)
    if not claim_tokens:
        return 0.0
    cited_tokens = set(_tokenize(cited_text))
    matched = sum(1 for token in claim_tokens if token in cited_tokens)
    return matched / len(claim_tokens)


@dataclass
class VerificationResult:
    """Outcome of running the citation gate over one contribution or ledger."""

    value: object
    checked: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def grounding_score(result: VerificationResult) -> float:
    """Fraction of citation-bearing fields that survived verification.

    1.0 when nothing was checked (nothing to hallucinate) or when every
    checked field survived.
    """
    if not result.checked:
        return 1.0
    return 1.0 - (len(result.dropped) / len(result.checked))


def dropped_claims(result: VerificationResult) -> list[str]:
    return list(result.dropped)


def _verify_claim(
    claim: Claim | None,
    sentences: dict[str, str],
    *,
    label: str,
    allowed_sentence_ids: set[str] | None = None,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[Claim | None, str | None]:
    """Returns ``(verified_claim_or_None, drop_reason_or_None)``.

    ``None`` in, ``None`` out, no drop reason: an absent claim was never
    asserted, so there's nothing to verify or hallucinate.
    """
    if claim is None:
        return None, None
    if not claim.sentence_ids:
        return None, f"{label}: no sentence_ids cited"
    unknown_ids = [sid for sid in claim.sentence_ids if sid not in sentences]
    if unknown_ids:
        return None, f"{label}: cites unknown sentence_ids {unknown_ids}"
    if allowed_sentence_ids is not None:
        disallowed = [sid for sid in claim.sentence_ids if sid not in allowed_sentence_ids]
        if disallowed:
            return None, f"{label}: cites out-of-section sentence_ids {disallowed}"
    cited_text = " ".join(sentences[sid] for sid in claim.sentence_ids)
    coverage = _token_coverage(claim.text, cited_text)
    if coverage < coverage_threshold:
        return None, f"{label}: low token coverage ({coverage:.2f} < {coverage_threshold})"
    return claim, None


def _verify_gap(
    gap: Gap,
    sentences: dict[str, str],
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[Gap | None, str | None]:
    if not gap.sentence_ids:
        # A gap can legitimately describe an *absence* (e.g. no_comparator)
        # with nothing to cite; only reject gaps that cite and get it wrong.
        return gap, None
    unknown_ids = [sid for sid in gap.sentence_ids if sid not in sentences]
    if unknown_ids:
        return None, f"gap[{gap.gap_type}]: cites unknown sentence_ids {unknown_ids}"
    cited_text = " ".join(sentences[sid] for sid in gap.sentence_ids)
    coverage = _token_coverage(gap.description, cited_text)
    if coverage < coverage_threshold:
        return None, f"gap[{gap.gap_type}]: low token coverage ({coverage:.2f} < {coverage_threshold})"
    return gap, None


def verify_contribution(
    contribution: T,
    sentences: dict[str, str],
    *,
    section_tags: dict[str, str] | None = None,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> VerificationResult:
    """Verify one Round 1 panel contribution or Round 2 opinion.

    ``section_tags`` (from :func:`app.agents.sld.segmentation.tag_sections`)
    is required to enforce the findings_auditor RESULTS-only restriction; if
    omitted, that restriction is skipped (useful for tests / mock abstracts
    with no section tags).
    """
    results_ids: set[str] | None = None
    if section_tags is not None:
        results_ids = {sid for sid, tag in section_tags.items() if tag == "RESULTS"}

    checked: list[str] = []
    dropped: list[str] = []

    def _check(claim: Claim | None, label: str, *, restrict_to_results: bool = False) -> Claim | None:
        if claim is None:
            return None
        checked.append(label)
        verified, reason = _verify_claim(
            claim,
            sentences,
            label=label,
            allowed_sentence_ids=results_ids if restrict_to_results else None,
            coverage_threshold=coverage_threshold,
        )
        if reason is not None:
            dropped.append(reason)
        return verified

    if isinstance(contribution, QuestionFramerContribution):
        cleaned = contribution.model_copy(
            update={
                "target_population": _check(contribution.target_population, "target_population"),
                "target_exposure": _check(contribution.target_exposure, "target_exposure"),
                "target_outcome": _check(contribution.target_outcome, "target_outcome"),
            }
        )
    elif isinstance(contribution, FindingsAuditorContribution):
        cleaned = contribution.model_copy(
            update={
                "primary_endpoint": _check(
                    contribution.primary_endpoint, "primary_endpoint", restrict_to_results=True
                ),
                "significance": _check(
                    contribution.significance, "significance", restrict_to_results=True
                ),
                "effect_magnitude": _check(
                    contribution.effect_magnitude, "effect_magnitude", restrict_to_results=True
                ),
            }
        )
    elif isinstance(contribution, GapAuditorContribution):
        verified_gaps: list[Gap] = []
        for index, gap in enumerate(contribution.gaps):
            label = f"gap[{index}:{gap.gap_type}]"
            checked.append(label)
            verified, reason = _verify_gap(gap, sentences, coverage_threshold=coverage_threshold)
            if reason is not None:
                dropped.append(reason)
            if verified is not None:
                verified_gaps.append(verified)
        cleaned = contribution.model_copy(update={"gaps": verified_gaps})
    elif isinstance(contribution, ConclusionReconstructorContribution):
        cleaned = contribution.model_copy(
            update={
                "reconstructed_conclusion": _check(
                    contribution.reconstructed_conclusion, "reconstructed_conclusion"
                ),
            }
        )
    elif isinstance(contribution, RoundTwoOpinion):
        checked.append("citations")
        unknown_ids = [sid for sid in contribution.citations if sid not in sentences]
        surviving_citations = [sid for sid in contribution.citations if sid in sentences]
        if unknown_ids:
            dropped.append(f"citations: cites unknown sentence_ids {unknown_ids}")
        if surviving_citations:
            cited_text = " ".join(sentences[sid] for sid in surviving_citations)
            coverage = _token_coverage(contribution.rationale, cited_text)
            if coverage < coverage_threshold:
                dropped.append(
                    f"rationale: low token coverage ({coverage:.2f} < {coverage_threshold}) "
                    "against surviving citations"
                )
        cleaned = contribution.model_copy(update={"citations": surviving_citations})
    else:  # pragma: no cover - exhaustive over the PanelContribution union + RoundTwoOpinion
        raise TypeError(f"Unsupported contribution type: {type(contribution)!r}")

    return VerificationResult(value=cleaned, checked=checked, dropped=dropped)


def verify_ledger(
    ledger: EvidenceLedger,
    sentences: dict[str, str],
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> VerificationResult:
    """Verify an EvidenceLedger's own claims/gaps/conflicts.

    The Supervisor that writes the ledger is itself an LLM call synthesizing
    across contributions, so it can introduce its own fabrications even when
    every contribution it read from was already verified. This re-checks the
    ledger's citations independently, right after it's produced.
    """
    checked: list[str] = []
    dropped: list[str] = []

    def _check(claim: Claim | None, label: str) -> Claim | None:
        if claim is None:
            return None
        checked.append(label)
        verified, reason = _verify_claim(
            claim, sentences, label=label, coverage_threshold=coverage_threshold
        )
        if reason is not None:
            dropped.append(reason)
        return verified

    verified_gaps: list[Gap] = []
    for index, gap in enumerate(ledger.gaps):
        label = f"gap[{index}:{gap.gap_type}]"
        checked.append(label)
        verified, reason = _verify_gap(gap, sentences, coverage_threshold=coverage_threshold)
        if reason is not None:
            dropped.append(reason)
        if verified is not None:
            verified_gaps.append(verified)

    verified_conflicts: list[LedgerConflict] = []
    for index, conflict in enumerate(ledger.conflicts):
        label = f"conflict[{index}]"
        checked.append(label)
        unknown_ids = [sid for sid in conflict.sentence_ids if sid not in sentences]
        if unknown_ids:
            dropped.append(f"{label}: cites unknown sentence_ids {unknown_ids}")
            continue
        if conflict.sentence_ids:
            cited_text = " ".join(sentences[sid] for sid in conflict.sentence_ids)
            coverage = _token_coverage(conflict.description, cited_text)
            if coverage < coverage_threshold:
                dropped.append(f"{label}: low token coverage ({coverage:.2f} < {coverage_threshold})")
                continue
        verified_conflicts.append(conflict)

    cleaned = ledger.model_copy(
        update={
            "target_population": _check(ledger.target_population, "target_population"),
            "target_exposure": _check(ledger.target_exposure, "target_exposure"),
            "target_outcome": _check(ledger.target_outcome, "target_outcome"),
            "primary_endpoint": _check(ledger.primary_endpoint, "primary_endpoint"),
            "significance": _check(ledger.significance, "significance"),
            "effect_magnitude": _check(ledger.effect_magnitude, "effect_magnitude"),
            "reconstructed_conclusion": _check(
                ledger.reconstructed_conclusion, "reconstructed_conclusion"
            ),
            "gaps": verified_gaps,
            "conflicts": verified_conflicts,
        }
    )
    return VerificationResult(value=cleaned, checked=checked, dropped=dropped)
