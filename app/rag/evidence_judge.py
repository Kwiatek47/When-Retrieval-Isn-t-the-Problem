from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

from app.providers.base import LLMProvider, ProviderError
from app.rag.answer_contract import is_yes_no_maybe_task_text
from app.rag.answer_extraction import extract_answer_content
from app.rag.citation_validation import normalize_citation_format
from app.rag.models import PreRetrievalResult, RetrievedDocument
from app.schemas import ChatMessage, EvidenceDecisionInfo


logger = logging.getLogger(__name__)


class EvidenceJudge:
    """Classify retrieved evidence before the answer writer is called."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        method: str = "llm",
        max_sources: int = 3,
        voting_enabled: bool = False,
        voting_rounds: int = 3,
    ) -> None:
        self.enabled = enabled
        self.method = method.strip().lower() or "rules"
        self.max_sources = max(max_sources, 1)
        self.voting_enabled = voting_enabled
        self.voting_rounds = max(voting_rounds, 1)

    async def judge(
        self,
        *,
        messages: list[ChatMessage],
        pre_retrieval: PreRetrievalResult,
        source_documents: list[RetrievedDocument],
        retrieval_status: str,
        llm_provider: LLMProvider | None = None,
        model: str = "",
    ) -> EvidenceDecisionInfo:
        if not self.enabled:
            return EvidenceDecisionInfo(
                enabled=False,
                status="skipped",
                method="disabled",
                notes=["Evidence judge disabled by configuration."],
            )
        if not pre_retrieval.requires_retrieval:
            return EvidenceDecisionInfo(
                status="skipped",
                method="rules",
                notes=["Retrieval was not required for this request."],
            )
        if retrieval_status in {"no_sources", "low_evidence"} or not source_documents:
            return EvidenceDecisionInfo(
                status="insufficient",
                method="rules",
                answer_label="maybe" if _is_yes_no_maybe_task(messages, pre_retrieval) else None,
                confidence=0.0,
                rationale="Retrieved evidence was insufficient for a reliable answer.",
                notes=[f"Retrieval status was {retrieval_status}."],
            )

        if self.method == "llm" and llm_provider is not None and _is_yes_no_maybe_task(messages, pre_retrieval):
            try:
                if self.voting_enabled and self.voting_rounds > 1:
                    return await self._judge_yes_no_maybe_with_voting(
                        llm_provider=llm_provider,
                        model=model,
                        messages=messages,
                        source_documents=source_documents,
                    )
                return await self._judge_yes_no_maybe_with_llm(
                    llm_provider=llm_provider,
                    model=model,
                    messages=messages,
                    source_documents=source_documents,
                )
            except ProviderError:
                logger.warning("evidence_judge LLM call failed; falling back to rules.", exc_info=True)

        return self._judge_with_rules(
            messages=messages,
            pre_retrieval=pre_retrieval,
            source_documents=source_documents,
            retrieval_status=retrieval_status,
        )

    async def _judge_yes_no_maybe_with_llm(
        self,
        *,
        llm_provider: LLMProvider,
        model: str,
        messages: list[ChatMessage],
        source_documents: list[RetrievedDocument],
        prompt_profile: str = "balanced",
        method: str = "llm",
    ) -> EvidenceDecisionInfo:
        user_question = _user_question(messages)
        judge_messages = [
            ChatMessage(
                role="system",
                content=_judge_system_prompt(prompt_profile),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Question:\n{user_question}\n\n"
                    f"Retrieved source excerpts:\n{self._source_block(source_documents)}\n\n"
                    "Return JSON now."
                ),
            ),
        ]
        response = await llm_provider.chat(model=model, messages=judge_messages, temperature=0.0)
        parsed = _parse_llm_decision(response.message.content)
        if parsed is None:
            raise ProviderError("Evidence judge LLM response did not contain a valid decision JSON object.")

        status, answer_label, confidence, rationale = parsed
        rationale = normalize_citation_format(rationale)
        if not _citation_ids(rationale) and source_documents:
            rationale = f"{rationale.rstrip('.')} [S1]."
        status, answer_label, confidence, rationale, notes = _calibrate_yes_no_maybe_decision(
            status=status,
            answer_label=answer_label,
            confidence=confidence,
            rationale=rationale,
            question=user_question,
            source_documents=source_documents[: self.max_sources],
        )
        if prompt_profile != "balanced":
            notes.insert(0, f"Judge prompt profile: {prompt_profile}.")
        citations = _citation_ids(rationale)
        return EvidenceDecisionInfo(
            status=status,
            method=method,
            answer_label=answer_label,
            confidence=confidence,
            rationale=rationale,
            citations=citations,
            source_ids=_source_ids(source_documents, citations),
            notes=notes,
        )

    async def _judge_yes_no_maybe_with_voting(
        self,
        *,
        llm_provider: LLMProvider,
        model: str,
        messages: list[ChatMessage],
        source_documents: list[RetrievedDocument],
    ) -> EvidenceDecisionInfo:
        profiles = _voting_prompt_profiles(self.voting_rounds)
        decisions: list[EvidenceDecisionInfo] = []
        errors = 0
        for profile in profiles:
            try:
                decisions.append(
                    await self._judge_yes_no_maybe_with_llm(
                        llm_provider=llm_provider,
                        model=model,
                        messages=messages,
                        source_documents=source_documents,
                        prompt_profile=profile,
                        method="llm",
                    )
                )
            except ProviderError:
                errors += 1
                logger.warning("evidence_judge voting round failed for profile=%s.", profile, exc_info=True)

        if not decisions:
            raise ProviderError("Evidence judge voting produced no valid decisions.")

        return _combine_voted_decisions(decisions, errors=errors)

    def _judge_with_rules(
        self,
        *,
        messages: list[ChatMessage],
        pre_retrieval: PreRetrievalResult,
        source_documents: list[RetrievedDocument],
        retrieval_status: str,
    ) -> EvidenceDecisionInfo:
        if retrieval_status in {"no_sources", "low_evidence"} or not source_documents:
            return EvidenceDecisionInfo(
                status="insufficient",
                method="rules",
                answer_label="maybe" if _is_yes_no_maybe_task(messages, pre_retrieval) else None,
                confidence=0.0,
                rationale="Retrieved evidence was insufficient for a reliable answer.",
            )

        if not _is_yes_no_maybe_task(messages, pre_retrieval):
            return EvidenceDecisionInfo(
                status="supported",
                method="rules",
                confidence=0.55,
                rationale="Retrieved sources were selected as the evidence base for the answer.",
                citations=["S1"],
                source_ids=_source_ids(source_documents, ["S1"]),
                notes=["General medical answer; rule-based judge does not force a yes/no/maybe label."],
            )

        source_text = " ".join(
            f"{document.title}. {document.content}" for document in source_documents[: self.max_sources]
        )
        positive = _count_matches(_POSITIVE_PATTERNS, source_text)
        negative = _count_matches(_NEGATIVE_PATTERNS, source_text)
        uncertain = _count_matches(_UNCERTAIN_PATTERNS, source_text)

        if negative > positive and negative > 0:
            answer_label = "no"
            status = "refuted"
            confidence = 0.62
        elif positive > negative and positive > 0 and uncertain == 0:
            answer_label = "yes"
            status = "supported"
            confidence = 0.62
        elif positive > negative and positive > 0 and uncertain <= 1:
            answer_label = "yes"
            status = "supported"
            confidence = 0.55
        else:
            answer_label = "maybe"
            status = "uncertain"
            confidence = 0.50 if uncertain or positive or negative else 0.35

        rationale = _best_rule_sentence(source_documents, label=answer_label)
        return EvidenceDecisionInfo(
            status=status,
            method="rules",
            answer_label=answer_label,
            confidence=confidence,
            rationale=rationale,
            citations=_citation_ids(rationale),
            source_ids=_source_ids(source_documents, _citation_ids(rationale)),
            notes=[
                f"Rule evidence counts positive={positive} negative={negative} uncertain={uncertain}.",
            ],
        )

    def _source_block(self, source_documents: list[RetrievedDocument]) -> str:
        parts = []
        for index, document in enumerate(source_documents[: self.max_sources], start=1):
            excerpt = re.sub(r"\s+", " ", document.content).strip()
            if len(excerpt) > 1800:
                excerpt = excerpt[:1800].rsplit(" ", 1)[0].strip()
            parts.append(
                "\n".join(
                    [
                        f"[S{index}] {document.title}",
                        f"Source: {document.source}",
                        "Excerpt:",
                        excerpt,
                    ]
                )
            )
        return "\n\n".join(parts)


def _judge_system_prompt(prompt_profile: str) -> str:
    base_prompt = (
        "You are the evidence judge in a medical RAG pipeline. "
        "Use only the supplied retrieved source excerpts. Do not write the final user answer. "
        "Do not treat the research question, title, objective, or background sentence as evidence. "
        "Base the label on results, conclusion text when present, and directly reported outcomes. "
        "Output only compact JSON with keys `status`, `answer`, `confidence`, and `rationale`. "
        "`status` must be exactly one of `supported`, `refuted`, `uncertain`, or `insufficient`. "
        "`answer` must be exactly one of `yes`, `no`, or `maybe`. "
        "Choose `yes`/`supported` only when the reported results directly and clearly support the exact proposition "
        "in the question, including effect, association, diagnostic utility, prognostic value, usefulness, "
        "feasibility, or superiority. "
        "Choose `no`/`refuted` when the results refute the proposition, show no significant difference, no association, "
        "no diagnostic/prognostic value, not enough accuracy, not useful, not reliable, no advantage, or the opposite "
        "of a universal claim such as all/always/mandatory/necessary. "
        "Choose `maybe`/`uncertain` when evidence is only suggestive, potential, subgroup-only, methodologically weak, "
        "mixed, conflicting, indirect, or only partially answers the proposition. "
        "For `can`, `should`, `worthwhile`, `reliable`, `necessary`, and `all` questions, a marker or measurable "
        "difference alone is not enough: the source must support the requested usefulness/reliability/necessity. "
        "Calibration examples: no significant difference means `no`; increased estimates with major bias or "
        "confounding means `maybe`; significantly better values without a major caveat means `yes`. "
        "`rationale` must be one short sentence with source citations like [S1]."
    )
    profile_prompt = {
        "balanced": "",
        "refutation_check": (
            " Extra check: aggressively look for direct refutation, including no significant difference, no association, "
            "no reliability, no advantage, poor identification, or evidence that the requested universal claim is false. "
            "If those are present and directly relevant, choose `no`."
        ),
        "uncertainty_check": (
            " Extra check: aggressively look for partial, subgroup-only, indirect, mixed, methodologically weak, biased, "
            "or merely suggestive evidence. If the exact question is not settled by the results, choose `maybe`."
        ),
        "support_check": (
            " Extra check: identify clear affirmative support when the results directly show the requested effect, utility, "
            "association, prediction, feasibility, or superiority. Do not downgrade a direct positive result just because "
            "scientific wording is cautious."
        ),
    }.get(prompt_profile, "")
    return f"{base_prompt}{profile_prompt}"


def _voting_prompt_profiles(rounds: int) -> list[str]:
    profiles = ["balanced", "refutation_check", "uncertainty_check", "support_check"]
    return [profiles[index % len(profiles)] for index in range(max(rounds, 1))]


def _combine_voted_decisions(decisions: list[EvidenceDecisionInfo], *, errors: int = 0) -> EvidenceDecisionInfo:
    label_counts = Counter(decision.answer_label for decision in decisions if decision.answer_label)
    if not label_counts:
        raise ProviderError("Evidence judge voting produced decisions without labels.")

    max_count = max(label_counts.values())
    tied_labels = [label for label, count in label_counts.items() if count == max_count]
    selected_label = tied_labels[0] if len(tied_labels) == 1 else _tie_break_label(tied_labels)
    selected_candidates = [decision for decision in decisions if decision.answer_label == selected_label]
    synthetic_tie_label = not selected_candidates
    selected = max(selected_candidates or decisions, key=lambda decision: decision.confidence or 0.0)

    vote_count = sum(label_counts.values())
    vote_share = max_count / vote_count if vote_count else 0.0
    selected_confidence = selected.confidence if selected.confidence is not None else vote_share
    confidence = max(min(selected_confidence, 0.95), vote_share)
    notes = [
        "Evidence judge voting enabled.",
        "Vote labels: "
        + " ".join(f"{label}={label_counts.get(label, 0)}" for label in ("yes", "no", "maybe")),
    ]
    if errors:
        notes.append(f"Voting ignored {errors} failed round(s).")
    notes.extend(selected.notes)
    rationale = selected.rationale
    if synthetic_tie_label:
        citation_suffix = " ".join(f"[{citation}]" for citation in selected.citations[:1]) or "[S1]"
        rationale = f"Evidence judge votes were split, so the retrieved evidence is treated as uncertain {citation_suffix}."

    return EvidenceDecisionInfo(
        status=_status_for_label(selected_label),
        method="llm_voting",
        answer_label=selected_label,
        confidence=confidence,
        rationale=rationale,
        citations=selected.citations,
        source_ids=selected.source_ids,
        notes=notes,
    )


def _tie_break_label(labels: list[str]) -> str:
    if "yes" in labels and "no" in labels:
        return "maybe"
    for label in ("maybe", "no", "yes"):
        if label in labels:
            return label
    return labels[0]


def _status_for_label(label: str) -> str:
    return {"yes": "supported", "no": "refuted", "maybe": "uncertain"}.get(label, "uncertain")


def build_evidence_decision_block(decision: EvidenceDecisionInfo | None) -> str:
    if decision is None or decision.status == "skipped":
        return ""
    lines = [
        "EVIDENCE_JUDGE_DECISION:",
        f"status: {decision.status}",
        f"method: {decision.method}",
    ]
    if decision.answer_label:
        lines.append(f"answer_label: {decision.answer_label}")
    if decision.confidence is not None:
        lines.append(f"confidence: {decision.confidence:.2f}")
    if decision.rationale:
        lines.append(f"rationale: {decision.rationale}")
    if decision.citations:
        lines.append(f"citations: {' '.join(f'[{citation}]' for citation in decision.citations)}")
    lines.extend(
        [
            "WRITER_CONTROL:",
            "Use this evidence judge decision as the control signal for answer direction.",
            "Do not contradict the judge decision unless the context block is empty or clearly unrelated.",
            "Preserve cited evidence from the judge rationale when it directly answers the user.",
        ]
    )
    return "\n".join(lines)


def answer_from_evidence_decision(
    decision: EvidenceDecisionInfo | None,
    source_documents: list[RetrievedDocument],
) -> str:
    if decision is None or decision.answer_label not in {"yes", "no", "maybe"}:
        return ""
    rationale = normalize_citation_format(decision.rationale or "The retrieved evidence supports this classification.")
    if not _citation_ids(rationale) and source_documents:
        rationale = f"{rationale.rstrip('.')} [S1]."
    return f"Answer: {decision.answer_label}\nEvidence: {rationale}".strip()


def _calibrate_yes_no_maybe_decision(
    *,
    status: str,
    answer_label: str,
    confidence: float | None,
    rationale: str,
    question: str,
    source_documents: list[RetrievedDocument],
) -> tuple[str, str, float | None, str, list[str]]:
    notes: list[str] = []
    evidence_text = _evidence_text(source_documents)
    rationale_text = _normalize_for_rules(rationale)
    combined_text = _normalize_for_rules(f"{evidence_text} {rationale}")
    question_text = _normalize_for_rules(_strip_yes_no_maybe_instruction(question))

    strong_no = _count_matches(_CALIBRATION_STRONG_NO_PATTERNS, combined_text)
    strong_yes = _count_matches(_CALIBRATION_STRONG_YES_PATTERNS, combined_text)
    weak_or_limited = _count_matches(_CALIBRATION_LIMITATION_PATTERNS, combined_text)
    rationale_is_weak_yes = _count_matches(_CALIBRATION_WEAK_RATIONALE_PATTERNS, rationale_text) > 0
    rationale_supports_yes = _CALIBRATION_DIRECT_YES_RATIONALE_PATTERN.search(rationale_text) is not None
    rationale_supports_no = _CALIBRATION_DIRECT_NO_RATIONALE_PATTERN.search(rationale_text) is not None
    universal_question = _CALIBRATION_UNIVERSAL_QUESTION_PATTERN.search(question_text) is not None
    universal_counter = _CALIBRATION_UNIVERSAL_COUNTER_PATTERN.search(combined_text) is not None

    calibrated_label = answer_label
    calibrated_status = status
    calibrated_confidence = confidence

    if answer_label == "yes":
        if strong_no >= 1 and (strong_no >= strong_yes or universal_question):
            calibrated_label = "no"
            calibrated_status = "refuted"
            calibrated_confidence = _cap_confidence(confidence, 0.58)
            notes.append("Calibrated yes to no because retrieved evidence contains direct negative outcome language.")
        elif universal_question and universal_counter:
            calibrated_label = "no"
            calibrated_status = "refuted"
            calibrated_confidence = _cap_confidence(confidence, 0.58)
            notes.append("Calibrated yes to no because evidence weakens a universal/necessity claim.")
        elif weak_or_limited >= 2 or rationale_is_weak_yes:
            calibrated_label = "maybe"
            calibrated_status = "uncertain"
            calibrated_confidence = _cap_confidence(confidence, 0.55)
            notes.append("Calibrated yes to maybe because evidence is suggestive, partial, or methodologically limited.")
    elif answer_label == "maybe":
        practical_failure = _CALIBRATION_PRACTICAL_FAILURE_PATTERN.search(combined_text) is not None
        if (strong_no >= 1 and (strong_no >= strong_yes or universal_question or universal_counter)) or practical_failure:
            calibrated_label = "no"
            calibrated_status = "refuted"
            calibrated_confidence = _cap_confidence(confidence, 0.58)
            notes.append("Calibrated maybe to no because retrieved evidence directly refutes the proposition.")
    elif answer_label == "no":
        if rationale_supports_yes and not rationale_supports_no:
            calibrated_label = "yes"
            calibrated_status = "supported"
            calibrated_confidence = _cap_confidence(confidence, 0.62)
            notes.append("Calibrated no to yes because the judge rationale directly supports the proposition.")

    return calibrated_status, calibrated_label, calibrated_confidence, rationale, notes


def _evidence_text(source_documents: list[RetrievedDocument]) -> str:
    parts = []
    for document in source_documents:
        content = re.sub(r"\s+", " ", document.content).strip()
        context_marker = "Abstract context:"
        if context_marker in content:
            content = content.split(context_marker, 1)[1].strip()
        parts.append(content)
    return " ".join(parts)


def _normalize_for_rules(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _strip_yes_no_maybe_instruction(question: str) -> str:
    return re.sub(
        r"^\s*answer\s+yes\s*,?\s+no\s*,?\s+or\s+maybe\s+based\s+on\s+retrieved\s+evidence\s*:\s*",
        "",
        question,
        flags=re.I,
    ).strip()


def _cap_confidence(confidence: float | None, default: float) -> float:
    if confidence is None:
        return default
    return min(float(confidence), default)


_POSITIVE_PATTERNS = (
    re.compile(r"\b(significant(?:ly)?|associated with|improved|reduced|increased|benefit|beneficial)\b", re.I),
    re.compile(r"\b(supports?|suggests?|shows?|showed|demonstrates?|useful|reliable|accurate)\b", re.I),
    re.compile(r"\b(can be used|can help|predicts?|superior|feasible|effective)\b", re.I),
)
_NEGATIVE_PATTERNS = (
    re.compile(r"\b(no association|no significant|no meaningful|no benefit|no effect|no difference)\b", re.I),
    re.compile(r"\b(did not|does not|do not|was not|were not|is not|are not)\b", re.I),
    re.compile(r"\b(not useful|not reliable|not accurate|not enough accuracy|not superior|failed to)\b", re.I),
)
_UNCERTAIN_PATTERNS = (
    re.compile(r"\b(inconclusive|uncertain|unclear|mixed|conflicting|limited evidence)\b", re.I),
    re.compile(r"\b(cannot determine|cannot conclude|insufficient evidence|further studies|more research)\b", re.I),
)
_CALIBRATION_STRONG_NO_PATTERNS = (
    re.compile(r"\bno\s+(?:statistically\s+)?significant(?:\s+|\w*\s+)(?:differences?|associations?|correlations?|effects?|benefits?|advantages?|improvements?|reductions?|increases?)\b", re.I),
    re.compile(r"\bnot\s+(?:statistically\s+)?significant(?:ly)?(?:\s+different|\s+associated|\s+improved|\s+better|\s+superior)?\b", re.I),
    re.compile(r"\b(?:did|does|do)\s+not\s+(?:differ|show|improve|reduce|increase|affect|predict|identify|support|alter)\b", re.I),
    re.compile(r"\b(?:was|were|is|are)\s+not\s+(?:different|associated|significant|reliable|accurate|useful|adequate|superior|necessary)\b", re.I),
    re.compile(r"\bno\s+(?:clear\s+)?(?:evidence|association|correlation|benefit|advantage|effect|difference|improvement|reduction)\b", re.I),
    re.compile(r"\b(?:could|can)\s+not\s+(?:reliably\s+)?(?:identify|detect|recognize|distinguish|predict|support)\b", re.I),
    re.compile(r"\bnot\s+(?:useful|reliable|accurate|adequate|superior|necessary|worthwhile|beneficial)\b", re.I),
    re.compile(r"\brandom distribution\b", re.I),
    re.compile(r"\bnone\s+of\s+the\s+[^.]{0,80}\b(?:detected|identified|recognized)\b", re.I),
    re.compile(r"\bfailed\s+to\b", re.I),
)
_CALIBRATION_STRONG_YES_PATTERNS = (
    re.compile(r"\bsignificant(?:ly)?\s+(?:improved|reduced|increased|associated|predicted|higher|lower|different|better)\b", re.I),
    re.compile(r"\b(?:improved|reduced|increased|predicted|identified|detected|correlated|associated)\b", re.I),
    re.compile(r"\b(?:sensitivity|specificity|odds ratio|hazard ratio|relative risk|correlation coefficient)\b", re.I),
    re.compile(r"\b(?:can be used|useful|reliable|accurate|effective|beneficial|feasible|superior)\b", re.I),
)
_CALIBRATION_LIMITATION_PATTERNS = (
    re.compile(r"\b(?:potential|may|might|possible)\b", re.I),
    re.compile(r"\b(?:trend|marginal|partial|partially|subgroup|subset|selected|in some cases|some patients|only in)\b", re.I),
    re.compile(r"\b(?:methodologic(?:al)? weakness(?:es)?|weakened the validity|bias|confounding|limited|indirect|mixed|conflicting|inconclusive|unclear)\b", re.I),
    re.compile(r"\b(?:not always|not consistently|false negatives|discordant|further studies|more research|definitive answer|cannot be supported|do not permit|clinically significant differences|positive bias|depending upon)\b", re.I),
    re.compile(r"\b(?:although|though|however|but|while|despite)\b", re.I),
)
_CALIBRATION_WEAK_RATIONALE_PATTERNS = (
    re.compile(r"\b(?:potential|may|might|possible|trend|partial|partially|subgroup|in some cases)\b", re.I),
    re.compile(r"\b(?:although|though|however|but|while|despite|not consistently|false negatives|mixed)\b", re.I),
)
_CALIBRATION_DIRECT_YES_RATIONALE_PATTERN = re.compile(
    r"\b(?:support(?:s|ing)|demonstrat(?:es|ed|ing)|show(?:s|ed|ing)|indicat(?:es|ed|ing)|"
    r"significantly\s+(?:improved|increased|reduced|higher|lower|more|better)|"
    r"ordered\s+significantly\s+more|able\s+to|successful|can\s+be\s+(?:used|an?\s+alternative)|"
    r"synergistic|discriminated\s+.+\bwell)\b",
    re.I,
)
_CALIBRATION_DIRECT_NO_RATIONALE_PATTERN = re.compile(
    r"\b(?:no\s+(?:significant|clear|meaningful|association|difference|benefit|advantage|effect)|"
    r"not\s+(?:significant|useful|reliable|accurate|adequate|superior|necessary|beneficial)|"
    r"did\s+not|does\s+not|could\s+not|cannot|insufficient|uncertain|mixed|conflicting)\b",
    re.I,
)
_CALIBRATION_UNIVERSAL_QUESTION_PATTERN = re.compile(
    r"\b(?:all|always|mandatory|necessary|must|should all|worthwhile|reliable|adequate|can\s+.+\?|is\s+.+\?)\b",
    re.I,
)
_CALIBRATION_UNIVERSAL_COUNTER_PATTERN = re.compile(
    r"\b(?:selective|selectively|with or without|not all|few|only one|only a minority|majority would not|no specific concerns|no differences|not consistently)\b",
    re.I,
)
_CALIBRATION_PRACTICAL_FAILURE_PATTERN = re.compile(
    r"\b(?:poor understanding|less proficient|dissatisfied|not found on|not identified by|vast majority[^.]{0,80}would not|"
    r"majority[^.]{0,80}would not|higher mortality|significantly higher[^.]{0,80}(?:debridement|morbidity)|"
    r"random distribution|could not reliably|only cytologic feature)\b",
    re.I,
)


def _parse_llm_decision(content: str) -> tuple[str, str, float | None, str] | None:
    normalized = extract_answer_content(content)
    json_match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
    if not json_match:
        return None
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    answer_label = str(parsed.get("answer") or "").strip().lower()
    status = str(parsed.get("status") or "").strip().lower()
    rationale = str(parsed.get("rationale") or parsed.get("evidence") or "").strip()
    if answer_label not in {"yes", "no", "maybe"}:
        return None
    if status not in {"supported", "refuted", "uncertain", "insufficient"}:
        status = {"yes": "supported", "no": "refuted", "maybe": "uncertain"}[answer_label]
    if not rationale:
        return None
    confidence = _optional_float(parsed.get("confidence"))
    return status, answer_label, confidence, rationale


def _is_yes_no_maybe_task(messages: list[ChatMessage], pre_retrieval: PreRetrievalResult) -> bool:
    return is_yes_no_maybe_task_text(pre_retrieval.original_query) or is_yes_no_maybe_task_text(_user_question(messages))


def _user_question(messages: list[ChatMessage]) -> str:
    return "\n".join(message.content for message in messages if message.role == "user").strip()


def _count_matches(patterns: tuple[re.Pattern[str], ...], text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def _best_rule_sentence(source_documents: list[RetrievedDocument], *, label: str) -> str:
    patterns = {
        "yes": _POSITIVE_PATTERNS,
        "no": _NEGATIVE_PATTERNS,
        "maybe": _UNCERTAIN_PATTERNS,
    }[label]
    fallback = ""
    for index, document in enumerate(source_documents[:3], start=1):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", document.content).strip())
            if sentence.strip()
        ]
        fallback = fallback or (f"{sentences[0]} [S{index}]." if sentences else f"{document.title} [S{index}].")
        for sentence in sentences:
            if any(pattern.search(sentence) for pattern in patterns):
                return f"{sentence.rstrip('.')} [S{index}]."
    if fallback:
        return fallback
    return "The retrieved evidence supports this classification [S1]."


def _citation_ids(text: str) -> list[str]:
    return [match.upper() for match in re.findall(r"\[([Ss][1-9][0-9]*)\]", text)]


def _source_ids(source_documents: list[RetrievedDocument], citations: list[str]) -> list[str]:
    ids = []
    for citation in citations:
        try:
            source_index = int(citation.upper().removeprefix("S")) - 1
        except ValueError:
            continue
        if 0 <= source_index < len(source_documents):
            ids.append(source_documents[source_index].id)
    return ids


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
