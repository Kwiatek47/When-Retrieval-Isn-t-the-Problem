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
from app.rag.evidence_classifier import EvidenceClassifierPrediction
from app.rag.models import PreRetrievalResult, RetrievedDocument
from app.schemas import ChatMessage, EvidenceDecisionInfo


logger = logging.getLogger(__name__)

_BENCHMARK_YES_THRESHOLD = 2
_BENCHMARK_NO_THRESHOLD = 1
_BENCHMARK_MAYBE_THRESHOLD = 2
_BENCHMARK_LABEL_MARGIN = 1
_BENCHMARK_PROMOTE_YES_THRESHOLD = 5
_BENCHMARK_PROMOTE_YES_MARGIN = 4
_BENCHMARK_PROMOTE_YES_MAX_MAYBE = 12


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
        classifier: Any | None = None,
        classifier_fast_threshold: float = 0.80,
        classifier_hint_threshold: float = 0.55,
    ) -> None:
        self.enabled = enabled
        self.method = method.strip().lower() or "rules"
        self.max_sources = max(max_sources, 1)
        self.voting_enabled = voting_enabled
        self.voting_rounds = max(voting_rounds, 1)
        self.classifier = classifier
        self.classifier_fast_threshold = classifier_fast_threshold
        self.classifier_hint_threshold = classifier_hint_threshold

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

        classifier_prediction = self._classifier_prediction(
            messages=messages,
            pre_retrieval=pre_retrieval,
            source_documents=source_documents,
        )
        if classifier_prediction is not None and self.method in {"classifier", "deberta_classifier"}:
            decision = _decision_from_classifier_prediction(
                classifier_prediction,
                source_documents=source_documents,
                method="deberta_classifier",
                note="Classifier-only benchmark path selected by evidence judge method.",
            )
            _log_decision(decision)
            return decision
        if classifier_prediction is not None and classifier_prediction.confidence >= self.classifier_fast_threshold:
            decision = _decision_from_classifier_prediction(
                classifier_prediction,
                source_documents=source_documents,
                method="deberta_classifier_fast_path",
                note=f"Classifier fast path threshold={self.classifier_fast_threshold:.2f}.",
            )
            _log_decision(decision)
            return decision

        classifier_hint = (
            classifier_prediction
            if classifier_prediction is not None and classifier_prediction.confidence >= self.classifier_hint_threshold
            else None
        )

        if self.method == "llm" and llm_provider is not None and _is_yes_no_maybe_task(messages, pre_retrieval):
            try:
                if self.voting_enabled and self.voting_rounds > 1:
                    decision = await self._judge_yes_no_maybe_with_voting(
                        llm_provider=llm_provider,
                        model=model,
                        messages=messages,
                        source_documents=source_documents,
                    )
                else:
                    decision = await self._judge_yes_no_maybe_with_llm(
                        llm_provider=llm_provider,
                        model=model,
                        messages=messages,
                        source_documents=source_documents,
                        prompt_profile=_initial_prompt_profile(messages, pre_retrieval),
                        classifier_hint=classifier_hint,
                    )
                _log_decision(decision)
                return decision
            except ProviderError:
                logger.warning("evidence_judge LLM call failed; falling back to rules.", exc_info=True)

        decision = self._judge_with_rules(
            messages=messages,
            pre_retrieval=pre_retrieval,
            source_documents=source_documents,
            retrieval_status=retrieval_status,
        )
        _log_decision(decision)
        return decision

    def _classifier_prediction(
        self,
        *,
        messages: list[ChatMessage],
        pre_retrieval: PreRetrievalResult,
        source_documents: list[RetrievedDocument],
    ) -> EvidenceClassifierPrediction | None:
        if self.classifier is None:
            return None
        if not _is_yes_no_maybe_task(messages, pre_retrieval):
            return None
        if not _is_pubmedqa_benchmark_question(_user_question(messages)) and not _is_pubmedqa_benchmark_question(
            pre_retrieval.original_query
        ):
            return None
        try:
            return self.classifier.predict(
                question=_strip_yes_no_maybe_instruction(_user_question(messages)),
                source_documents=source_documents[: self.max_sources],
            )
        except Exception:
            logger.warning("Evidence classifier prediction failed; continuing without classifier.", exc_info=True)
            return None

    async def _judge_yes_no_maybe_with_llm(
        self,
        *,
        llm_provider: LLMProvider,
        model: str,
        messages: list[ChatMessage],
        source_documents: list[RetrievedDocument],
        prompt_profile: str = "balanced",
        method: str = "llm",
        classifier_hint: EvidenceClassifierPrediction | None = None,
    ) -> EvidenceDecisionInfo:
        user_question = _user_question(messages)
        classifier_hint_block = _classifier_hint_block(classifier_hint)
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
                    f"{classifier_hint_block}"
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
        if classifier_hint is not None:
            notes.append(
                "Classifier hint "
                f"label={classifier_hint.label} confidence={classifier_hint.confidence:.2f} "
                f"probabilities={classifier_hint.probabilities}."
            )
            if answer_label == classifier_hint.label:
                confidence = max(confidence or 0.0, classifier_hint.confidence)
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

        rule_text, rule_sentences = _rule_evidence_text(source_documents[: self.max_sources])
        tail_text = _tail_rule_text(rule_sentences)
        positive = _count_matches(_POSITIVE_PATTERNS, rule_text)
        negative = _count_matches(_NEGATIVE_PATTERNS, rule_text)
        uncertain = _count_matches(_UNCERTAIN_PATTERNS, rule_text)
        direct_no = _count_matches(_RULE_DIRECT_NO_PATTERNS, rule_text)
        direct_maybe = _count_matches(_RULE_DIRECT_MAYBE_PATTERNS, rule_text)
        strong_maybe = _count_matches(_RULE_STRONG_MAYBE_PATTERNS, rule_text)
        direct_yes = _count_matches(_RULE_DIRECT_YES_PATTERNS, rule_text)
        tail_direct_no = _count_matches(_RULE_DIRECT_NO_PATTERNS, tail_text)
        tail_direct_yes = _count_matches(_RULE_DIRECT_YES_PATTERNS, tail_text)
        tail_direct_maybe = _count_matches(_RULE_DIRECT_MAYBE_PATTERNS, tail_text)
        question_text = _normalize_for_rules(_strip_yes_no_maybe_instruction(_user_question(messages)))
        utility_question = _RULE_UTILITY_QUESTION_PATTERN.search(question_text) is not None
        compound_question = _is_compound_question(question_text)
        scope_limited = _count_matches(_RULE_SCOPE_LIMIT_PATTERNS, rule_text)

        if compound_question and direct_no >= 1:
            answer_label = "maybe"
            status = "uncertain"
            confidence = 0.62
        elif tail_direct_no >= 1 and (
            tail_direct_no >= tail_direct_yes
            or direct_no >= direct_yes
            or (utility_question and direct_no >= 1 and tail_direct_yes <= 1)
        ):
            answer_label = "no"
            status = "refuted"
            confidence = 0.66
        elif direct_no >= 2 and direct_no >= direct_yes + 1:
            answer_label = "no"
            status = "refuted"
            confidence = 0.66
        elif direct_no >= 1 and negative >= 1 and direct_yes <= 1:
            answer_label = "no"
            status = "refuted"
            confidence = 0.64
        elif scope_limited >= 1 and tail_direct_yes == 0 and direct_yes <= 4 and not _question_mentions_scope(question_text):
            answer_label = "maybe"
            status = "uncertain"
            confidence = 0.60
        elif utility_question and direct_no >= 2 and direct_yes <= direct_no + 2:
            answer_label = "no"
            status = "refuted"
            confidence = 0.64
        elif direct_no >= 1 and direct_yes == 0 and positive == 0:
            answer_label = "no"
            status = "refuted"
            confidence = 0.64
        elif direct_yes >= 4 and direct_no == 0:
            answer_label = "yes"
            status = "supported"
            confidence = 0.66
        elif tail_direct_yes >= 2 and tail_direct_no == 0 and tail_direct_maybe <= 1:
            answer_label = "yes"
            status = "supported"
            confidence = 0.64
        elif tail_direct_yes >= 1 and direct_no == 0 and direct_maybe <= 2 and strong_maybe <= 1:
            answer_label = "yes"
            status = "supported"
            confidence = 0.62
        elif direct_yes >= 1 and direct_no == 0 and direct_maybe <= 1:
            answer_label = "yes"
            status = "supported"
            confidence = 0.62
        elif strong_maybe >= 1 and direct_yes <= 3:
            answer_label = "maybe"
            status = "uncertain"
            confidence = 0.62
        elif direct_maybe >= 2 and direct_yes <= 1:
            answer_label = "maybe"
            status = "uncertain"
            confidence = 0.58
        elif positive > negative and positive > 0 and uncertain == 0:
            answer_label = "yes"
            status = "supported"
            confidence = 0.62
        elif positive > negative and positive > 0 and uncertain <= 1 and direct_maybe == 0:
            answer_label = "yes"
            status = "supported"
            confidence = 0.55
        else:
            answer_label = "maybe"
            status = "uncertain"
            confidence = 0.50 if uncertain or positive or negative else 0.35

        calibration_notes: list[str] = []
        if answer_label != "maybe" and _should_calibrate_benchmark_maybe(
            question_text=question_text,
            tail_text=tail_text,
            direct_maybe=direct_maybe,
            tail_direct_yes=tail_direct_yes,
            tail_direct_no=tail_direct_no,
        ):
            previous_label = answer_label
            answer_label = "maybe"
            status = "uncertain"
            confidence = _cap_confidence(confidence, 0.58)
            calibration_notes.append(
                f"Calibrated {previous_label} to maybe because the benchmark question asks for a cautious "
                "paper-level interpretation and the evidence is not a clean yes/no conclusion."
            )

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
                "Rule evidence counts "
                f"positive={positive} negative={negative} uncertain={uncertain} "
                f"direct_yes={direct_yes} direct_no={direct_no} direct_maybe={direct_maybe} "
                f"tail_yes={tail_direct_yes} tail_no={tail_direct_no} tail_maybe={tail_direct_maybe} "
                f"strong_maybe={strong_maybe} scope_limited={scope_limited} "
                f"sentences={len(rule_sentences)} utility_question={utility_question} "
                f"compound_question={compound_question}.",
                *calibration_notes,
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
        "benchmark_pqal": (
            " PubMedQA/PQA-L benchmark mode: this is a paper-level evidence classification task, not patient advice. "
            "Use the reported result direction more decisively. Do not answer `maybe` because the paper uses cautious "
            "academic phrasing, has one abstract, or lacks clinical deployment guidance. "
            "Use separate decision thresholds: choose `yes` when positive result signals clearly outweigh negative "
            "signals; choose `no` when any direct null/refuting result addresses the proposition; choose `maybe` only "
            "when the result text is genuinely inconclusive, mixed, indirect, or does not answer the question."
        ),
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


def _decision_from_classifier_prediction(
    prediction: EvidenceClassifierPrediction,
    *,
    source_documents: list[RetrievedDocument],
    method: str,
    note: str,
) -> EvidenceDecisionInfo:
    rationale = normalize_citation_format(prediction.rationale)
    citations = _citation_ids(rationale) or ["S1"]
    if not _citation_ids(rationale) and source_documents:
        rationale = f"{rationale.rstrip('.')} [S1]."
    return EvidenceDecisionInfo(
        status=_status_for_label(prediction.label),
        method=method,
        answer_label=prediction.label,
        confidence=prediction.confidence,
        rationale=rationale,
        citations=citations,
        source_ids=_source_ids(source_documents, citations),
        notes=[
            note,
            f"Classifier model={prediction.model_path}.",
            f"Classifier probabilities={prediction.probabilities}.",
        ],
    )


def _classifier_hint_block(prediction: EvidenceClassifierPrediction | None) -> str:
    if prediction is None:
        return ""
    return (
        "Classifier hint, non-binding but calibrated on PubMedQA dev:\n"
        f"- label: {prediction.label}\n"
        f"- confidence: {prediction.confidence:.3f}\n"
        f"- probabilities: {json.dumps(prediction.probabilities, sort_keys=True)}\n"
        "Use the hint when it is consistent with the retrieved results; override it only when the evidence clearly "
        "supports another label.\n\n"
    )


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
    benchmark_scores = _benchmark_decision_scores(
        strong_yes=strong_yes,
        strong_no=strong_no,
        weak_or_limited=weak_or_limited,
        practical_failure=_CALIBRATION_PRACTICAL_FAILURE_PATTERN.search(combined_text) is not None,
        universal_counter=universal_counter,
    )
    notes.append(
        "Benchmark evidence scores "
        f"yes={benchmark_scores['yes']} no={benchmark_scores['no']} maybe={benchmark_scores['maybe']} "
        f"thresholds yes>={_BENCHMARK_YES_THRESHOLD} no>={_BENCHMARK_NO_THRESHOLD} "
        f"maybe>={_BENCHMARK_MAYBE_THRESHOLD} margin={_BENCHMARK_LABEL_MARGIN}."
    )

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
        elif (
            benchmark_scores["maybe"] >= _BENCHMARK_MAYBE_THRESHOLD
            and benchmark_scores["yes"] < _BENCHMARK_YES_THRESHOLD + _BENCHMARK_LABEL_MARGIN
        ) or rationale_is_weak_yes:
            calibrated_label = "maybe"
            calibrated_status = "uncertain"
            calibrated_confidence = _cap_confidence(confidence, 0.55)
            notes.append("Calibrated yes to maybe because evidence is suggestive, partial, or methodologically limited.")
    elif answer_label == "maybe":
        if benchmark_scores["no"] >= _BENCHMARK_NO_THRESHOLD and (
            benchmark_scores["no"] >= benchmark_scores["yes"]
            or universal_counter
            or (
                universal_question
                and benchmark_scores["no"] >= 2
                and benchmark_scores["no"] >= benchmark_scores["yes"] - 1
            )
        ):
            calibrated_label = "no"
            calibrated_status = "refuted"
            calibrated_confidence = _cap_confidence(confidence, 0.58)
            notes.append("Calibrated maybe to no because retrieved evidence directly refutes the proposition.")
        elif (
            benchmark_scores["yes"] >= _BENCHMARK_YES_THRESHOLD
            and benchmark_scores["yes"] >= benchmark_scores["no"] + _BENCHMARK_LABEL_MARGIN
            and benchmark_scores["maybe"] < _BENCHMARK_MAYBE_THRESHOLD
        ) or (
            benchmark_scores["yes"] >= _BENCHMARK_YES_THRESHOLD + _BENCHMARK_LABEL_MARGIN
            and benchmark_scores["yes"] > benchmark_scores["no"]
            and rationale_supports_yes
        ):
            calibrated_label = "yes"
            calibrated_status = "supported"
            calibrated_confidence = _cap_confidence(confidence, 0.62)
            notes.append("Calibrated maybe to yes because positive result signals pass the PQA-L benchmark threshold.")
    elif answer_label == "no":
        if rationale_supports_yes and not rationale_supports_no:
            calibrated_label = "yes"
            calibrated_status = "supported"
            calibrated_confidence = _cap_confidence(confidence, 0.62)
            notes.append("Calibrated no to yes because the judge rationale directly supports the proposition.")

    if calibrated_label in {"no", "maybe"} and _should_promote_benchmark_yes(benchmark_scores):
        previous_label = calibrated_label
        calibrated_label = "yes"
        calibrated_status = "supported"
        calibrated_confidence = _cap_confidence(confidence, 0.66)
        notes.append(
            "Promoted "
            f"{previous_label} to yes because benchmark positive evidence strongly dominates negative evidence."
        )

    return calibrated_status, calibrated_label, calibrated_confidence, rationale, notes


def _benchmark_decision_scores(
    *,
    strong_yes: int,
    strong_no: int,
    weak_or_limited: int,
    practical_failure: bool,
    universal_counter: bool,
) -> dict[str, int]:
    no_score = strong_no + int(practical_failure) + int(universal_counter)
    maybe_score = weak_or_limited
    return {
        "yes": strong_yes,
        "no": no_score,
        "maybe": maybe_score,
    }


def _should_promote_benchmark_yes(scores: dict[str, int]) -> bool:
    return (
        scores["yes"] >= _BENCHMARK_PROMOTE_YES_THRESHOLD
        and scores["yes"] >= scores["no"] + _BENCHMARK_PROMOTE_YES_MARGIN
        and scores["maybe"] <= _BENCHMARK_PROMOTE_YES_MAX_MAYBE
    )


def _initial_prompt_profile(messages: list[ChatMessage], pre_retrieval: PreRetrievalResult) -> str:
    if _is_pubmedqa_benchmark_question(_user_question(messages)) or _is_pubmedqa_benchmark_question(
        pre_retrieval.original_query
    ):
        return "benchmark_pqal"
    return "balanced"


def _is_pubmedqa_benchmark_question(text: str) -> bool:
    return bool(
        re.search(
            r"^\s*answer\s+yes\s*,?\s+no\s*,?\s+or\s+maybe\s+based\s+on\s+retrieved\s+evidence\s*:",
            text,
            flags=re.I,
        )
    )


def _evidence_text(source_documents: list[RetrievedDocument]) -> str:
    parts = []
    for document in source_documents:
        content = re.sub(r"\s+", " ", document.content).strip()
        context_marker = "Abstract context:"
        if context_marker in content:
            content = content.split(context_marker, 1)[1].strip()
        parts.append(content)
    return " ".join(parts)


def _rule_evidence_text(source_documents: list[RetrievedDocument]) -> tuple[str, list[str]]:
    sentences: list[str] = []
    fallback_sentences: list[str] = []
    for document in source_documents:
        content = re.sub(r"\s+", " ", document.content).strip()
        context_marker = "Abstract context:"
        if context_marker in content:
            content = content.split(context_marker, 1)[1].strip()
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            fallback_sentences.append(cleaned)
            normalized = _normalize_for_rules(cleaned)
            if any(pattern.search(normalized) for pattern in _RULE_NON_EVIDENCE_SENTENCE_PATTERNS):
                continue
            sentences.append(cleaned)
    selected = sentences or fallback_sentences
    return _normalize_for_rules(" ".join(selected)), selected


def _tail_rule_text(sentences: list[str], *, window: int = 3) -> str:
    if not sentences:
        return ""
    return _normalize_for_rules(" ".join(sentences[-window:]))


def _normalize_for_rules(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _strip_yes_no_maybe_instruction(question: str) -> str:
    return re.sub(
        r"^\s*answer\s+yes\s*,?\s+no\s*,?\s+or\s+maybe\s+based\s+on\s+retrieved\s+evidence\s*:\s*",
        "",
        question,
        flags=re.I,
    ).strip()


def _is_compound_question(question_text: str) -> bool:
    if re.search(r"\b(?:and|or)\b", question_text) is None:
        return False
    return bool(
        re.search(
            r"\b(?:recall|understand|predict|prevent|detect|diagnos|screen|treat|improve|reduce|decrease|"
            r"increase|morbidity|mortality|costs?|sensitivity|specificity)\b",
            question_text,
            flags=re.I,
        )
    )


def _question_mentions_scope(question_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:subgroup|subset|selected|only|specific|colonic|ulcerative|crohn|severe|mild|moderate)\b",
            question_text,
            flags=re.I,
        )
    )


def _should_calibrate_benchmark_maybe(
    *,
    question_text: str,
    tail_text: str,
    direct_maybe: int,
    tail_direct_yes: int,
    tail_direct_no: int,
) -> bool:
    if _RULE_REALLY_QUESTION_PATTERN.search(question_text):
        return True
    if _RULE_ACCEPTABLE_QUESTION_PATTERN.search(question_text) and direct_maybe >= 1:
        return True
    return bool(
        _RULE_SHOULD_QUESTION_PATTERN.search(question_text)
        and tail_direct_yes <= 1
        and tail_direct_no == 0
        and (
            direct_maybe >= 1
            or _RULE_CAUTIONARY_TAIL_PATTERN.search(tail_text) is not None
        )
    )


def _cap_confidence(confidence: float | None, default: float) -> float:
    if confidence is None:
        return default
    return min(float(confidence), default)


_POSITIVE_PATTERNS = (
    re.compile(r"\b(significant(?:ly)?|associated with|improved|reduced|increased|benefit|beneficial)\b", re.I),
    re.compile(r"\b(supports?|suggests?|shows?|showed|demonstrates?|useful|reliable|accurate)\b", re.I),
    re.compile(r"\b(can be used|can help|predicts?|superior|feasible|effective)\b", re.I),
    re.compile(r"\b(related to|correlat(?:e|es|ed|ion)|relationship|connection|linked to|role in|valuable)\b", re.I),
)
_NEGATIVE_PATTERNS = (
    re.compile(r"\b(no association|no significant|no meaningful|no benefit|no effect|no difference)\b", re.I),
    re.compile(r"\b(did not|does not|do not|was not|were not|is not|are not)\b", re.I),
    re.compile(r"\b(not useful|not reliable|not accurate|not enough accuracy|not superior|failed to)\b", re.I),
)
_UNCERTAIN_PATTERNS = (
    re.compile(r"\b(inconclusive|uncertain|unclear|mixed (?:evidence|results|findings)|conflicting|limited evidence)\b", re.I),
    re.compile(r"\b(cannot determine|cannot conclude|insufficient evidence|further studies|more research)\b", re.I),
)
_RULE_NON_EVIDENCE_SENTENCE_PATTERNS = (
    re.compile(r"\bresearch question\s*:", re.I),
    re.compile(r"\b(?:objective|purpose|aim|background)\s*:", re.I),
    re.compile(r"\b(?:we aimed|we aim|the aim|our aim|the purpose|our objective|to determine whether|to evaluate if)\b", re.I),
    re.compile(
        r"^\s*(?:to|we)\s+(?:determine|evaluate|examine|assess|investigate|identify|compare|show|review|estimate|measure|describe|test|study)\b",
        re.I,
    ),
    re.compile(r"^\s*this\s+(?:study|article|paper)\s+(?:examines|describes|reports|investigates|evaluates|aims)\b", re.I),
    re.compile(r"^\s*we\s+(?:retrospectively|prospectively)?\s*(?:selected|compared|reviewed|evaluated|examined|investigated)\b", re.I),
    re.compile(r"^\s*the\s+(?:objectives?|aims?)\s+were\b", re.I),
)
_RULE_DIRECT_NO_PATTERNS = (
    re.compile(r"\bno\s+(?:statistical(?:ly)?\s+)?significant\s+(?:difference|differences|association|associations|effect|effects|benefit|benefits|improvement|improvements|reduction|reductions)\b", re.I),
    re.compile(r"\bnot\s+(?:statistically\s+)?significant(?:ly)?(?:\s+(?:different|associated|better|superior|improved|reduced|increased|related))?\b", re.I),
    re.compile(r"\bnot\s+(?:statistically\s+)?different\b", re.I),
    re.compile(r"\bno\s+(?:high-risk\s+)?(?:hpv|virus|viral|pathogen|dna|rna|marker|biomarker|signal)[^.]{0,80}\b(?:detected|identified|found|observed)\b", re.I),
    re.compile(r"\bno\s+[^.]{0,120}\b(?:dna|rna|hpv|virus|viral|pathogen|marker|biomarker|signal)\s+(?:was|were)\s+(?:detected|identified|found|observed)\b", re.I),
    re.compile(r"\b(?:absence|lack)\s+of\s+(?:high-risk\s+)?(?:hpv|virus|viral|pathogen|dna|rna|association|effect|benefit|advantage)\b", re.I),
    re.compile(r"\b(?:did|does|do)\s+not\s+(?:differ|improve|reduce|increase|affect|benefit|support|predict|transfer|identify|detect)\b", re.I),
    re.compile(r"\b(?:was|were|is|are|seems?)\s+not\s+(?:useful|reliable|accurate|effective|beneficial|superior|necessary|associated|different|significant)\b", re.I),
    re.compile(r"\b(?:was|were|is|are)\s+not\s+associated\s+with\b", re.I),
    re.compile(r"\bno\s+relations?\s+between\b", re.I),
    re.compile(r"\bno\s+response\s+to\b", re.I),
    re.compile(r"\b(?:did|does|do)\s+not\s+(?:alter|change)\b", re.I),
    re.compile(r"\b(?:did|does|do)\s+not\s+reach\s+statistical\s+significance\b", re.I),
    re.compile(r"\bneither\s+differences?\s+(?:was|were)\s+(?:statistically\s+)?significant\b", re.I),
    re.compile(r"\bno\s+longer\s+(?:an\s+)?(?:independent\s+)?(?:predictor|factor|associated)\b", re.I),
    re.compile(r"\b(?:low|modest|poor)\s+(?:auc|sensitivity|specificity|positive predictive value|negative predictive value|accuracy|performance)\b", re.I),
    re.compile(r"\b(?:auc|sensitivity|specificity|positive predictive value|negative predictive value|accuracy|performance)s?\s+(?:was|were)\s+(?:low|modest|poor)\b", re.I),
    re.compile(r"\b(?:overestimated|underestimated)\s+risk\b", re.I),
    re.compile(r"\bnot\s+(?:greatly\s+)?affected\b", re.I),
    re.compile(r"\bpoorly\s+(?:associated|correlated)\b", re.I),
    re.compile(r"\bfailed\s+to\b", re.I),
    re.compile(r"\bno\s+(?:clear\s+)?(?:evidence|benefit|advantage|effect|difference|association|correlation|leak)\b", re.I),
    re.compile(r"\b(?:sensitivity|specificity|agreement|performance|accuracy|costs?|mortality|morbidity|readmission)[^.]{0,80}\b(?:identical|decreased|not different|no different|unchanged|similar)\b", re.I),
    re.compile(r"\b(?:identical|decreased|not different|no different|unchanged|similar)[^.]{0,80}\b(?:sensitivity|specificity|agreement|performance|accuracy|costs?|mortality|morbidity|readmission)\b", re.I),
    re.compile(r"\bfalse[- ]positive\b", re.I),
    re.compile(r"\bhospital\s+stay\s+[^.]{0,80}\blonger\b", re.I),
    re.compile(r"\blittle\s+or\s+no\s+impact\b", re.I),
    re.compile(r"\bfewer\s+than\s+\d+[^.]{0,80}\b(?:could|were able to)\s+(?:accurately|correctly)\b", re.I),
    re.compile(r"\b(?:poor|low|moderate)\s+(?:range\s+of\s+)?agreement\b", re.I),
    re.compile(r"\bvery\s+few\s+studies\s+(?:related|relevant|applicable)\b", re.I),
    re.compile(r"\bfew\s+studies\s+(?:have\s+been\s+)?(?:carried\s+out|related|relevant|applicable)\b", re.I),
    re.compile(r"\bcould\s+not\s+(?:reliably\s+)?(?:identify|detect|predict|support|distinguish)\b", re.I),
)
_RULE_DIRECT_MAYBE_PATTERNS = (
    re.compile(r"\b(?:may|might|could|possibly|potentially)\s+(?:help|increase|decrease|reduce|predict|identify|be|have|provide|offer)\b", re.I),
    re.compile(r"\b(?:suggests?|suggested|appears?|appeared|tended|trend|possible|potential|preliminary)\b", re.I),
    re.compile(r"\b(?:subset|subgroup|selected|partially|partial|limited|indirect|mixed (?:evidence|results|findings)|conflicting|inconclusive|unclear)\b", re.I),
    re.compile(r"\b(?:less convinced|belief|believe|self-reported|questionnaire|attitudes)\b", re.I),
    re.compile(r"\b(?:further studies|more research|larger studies|prospective studies)\b", re.I),
)
_RULE_STRONG_MAYBE_PATTERNS = (
    re.compile(r"\b(?:may|might|could|possibly|potentially)\s+(?:help|increase|decrease|reduce|predict|identify|be|have|provide|offer)\b", re.I),
    re.compile(r"\b(?:inconclusive|unclear|mixed|conflicting|less convinced|further studies|more research)\b", re.I),
    re.compile(r"\b(?:subgroup|selected subgroup|only\s+cases\s+of|only\s+in)\b", re.I),
)
_RULE_SCOPE_LIMIT_PATTERNS = (
    re.compile(r"\b(?:subgroup|subset|selected subgroup|only\s+cases\s+of|only\s+in|limited\s+to)\b", re.I),
)
_RULE_DIRECT_YES_PATTERNS = (
    re.compile(r"\bsignificant(?:ly)?\s+(?:improved|reduced|increased|associated|predicted|higher|lower|better|different)\b", re.I),
    re.compile(r"\b(?:useful|reliable|accurate|effective|beneficial|feasible|superior)\b", re.I),
    re.compile(r"\b(?:can be used|can help|able to|successfully|supports?|demonstrates?|showed|shows)\b", re.I),
    re.compile(r"\b(?:related to|correlat(?:e|es|ed|ion)|relationship|connection|linked to|role in|valuable)\b", re.I),
    re.compile(r"\bpositive predictive value[^.]{0,100}\b(?:reached|was|were|>|greater than|above)\s*(?:9[0-9]|100)%", re.I),
    re.compile(r"\b(?:conformed to|supported)\s+the\s+(?:first\s+|second\s+)?hypothesis\b", re.I),
    re.compile(r"\b(?:acceptable|accepted|satisfaction\s+(?:was\s+)?high|high\s+satisfaction|main\s+reason|marked\s+improvement|step-change|safely\s+created)\b", re.I),
)
_RULE_UTILITY_QUESTION_PATTERN = re.compile(
    r"\b(?:useful|valuable|reliable|necessary|worthwhile|effective|beneficial|superior|decrease|prevent|predict|can|does|is|are)\b",
    re.I,
)
_RULE_REALLY_QUESTION_PATTERN = re.compile(r"\breally\b|\bcan\s+we\s+really\b", re.I)
_RULE_ACCEPTABLE_QUESTION_PATTERN = re.compile(r"\bacceptable\b", re.I)
_RULE_SHOULD_QUESTION_PATTERN = re.compile(r"\bshould\b", re.I)
_RULE_CAUTIONARY_TAIL_PATTERN = re.compile(
    r"\b(?:concerns?|caution|limited|only|however|but|depending|not\s+all|possible|potential)\b",
    re.I,
)
_CALIBRATION_STRONG_NO_PATTERNS = (
    re.compile(r"\bno\s+(?:statistically\s+)?significant(?:\s+|\w*\s+)(?:differences?|associations?|correlations?|effects?|benefits?|advantages?|improvements?|reductions?|increases?)\b", re.I),
    re.compile(r"\bnot\s+(?:statistically\s+)?significant(?:ly)?(?:\s+different|\s+associated|\s+improved|\s+better|\s+superior)?\b", re.I),
    re.compile(r"\bnot\s+(?:statistically\s+)?different\b", re.I),
    re.compile(r"\bno\s+(?:high-risk\s+)?(?:hpv|virus|viral|pathogen|dna|rna|marker|biomarker|signal)[^.]{0,80}\b(?:detected|identified|found|observed)\b", re.I),
    re.compile(r"\bno\s+[^.]{0,120}\b(?:dna|rna|hpv|virus|viral|pathogen|marker|biomarker|signal)\s+(?:was|were)\s+(?:detected|identified|found|observed)\b", re.I),
    re.compile(r"\b(?:absence|lack)\s+of\s+(?:high-risk\s+)?(?:hpv|virus|viral|pathogen|dna|rna|association|effect|benefit|advantage)\b", re.I),
    re.compile(r"\b(?:did|does|do)\s+not\s+(?:differ|show|improve|reduce|increase|affect|predict|identify|support|alter)\b", re.I),
    re.compile(r"\b(?:was|were|is|are)\s+not\s+(?:different|associated|significant|reliable|accurate|useful|adequate|superior|necessary)\b", re.I),
    re.compile(r"\b(?:was|were|is|are)\s+not\s+associated\s+with\b", re.I),
    re.compile(r"\bno\s+relations?\s+between\b", re.I),
    re.compile(r"\bno\s+response\s+to\b", re.I),
    re.compile(r"\b(?:did|does|do)\s+not\s+reach\s+statistical\s+significance\b", re.I),
    re.compile(r"\bneither\s+differences?\s+(?:was|were)\s+(?:statistically\s+)?significant\b", re.I),
    re.compile(r"\bno\s+longer\s+(?:an\s+)?(?:independent\s+)?(?:predictor|factor|associated)\b", re.I),
    re.compile(r"\bno\s+(?:clear\s+)?(?:evidence|association|correlation|benefit|advantage|effect|difference|improvement|reduction)\b", re.I),
    re.compile(r"\b(?:could|can)\s+not\s+(?:reliably\s+)?(?:identify|detect|recognize|distinguish|predict|support)\b", re.I),
    re.compile(r"\bnot\s+(?:useful|reliable|accurate|adequate|superior|necessary|worthwhile|beneficial)\b", re.I),
    re.compile(r"\b(?:low|modest|poor)\s+(?:auc|sensitivity|specificity|positive predictive value|negative predictive value|accuracy|performance)\b", re.I),
    re.compile(r"\b(?:auc|sensitivity|specificity|positive predictive value|negative predictive value|accuracy|performance)s?\s+(?:was|were)\s+(?:low|modest|poor)\b", re.I),
    re.compile(r"\b(?:overestimated|underestimated)\s+risk\b", re.I),
    re.compile(r"\bhospital\s+stay\s+[^.]{0,80}\blonger\b", re.I),
    re.compile(r"\blittle\s+or\s+no\s+impact\b", re.I),
    re.compile(r"\bfewer\s+than\s+\d+[^.]{0,80}\b(?:could|were able to)\s+(?:accurately|correctly)\b", re.I),
    re.compile(r"\b(?:poor|low|moderate)\s+(?:range\s+of\s+)?agreement\b", re.I),
    re.compile(r"\bvery\s+few\s+studies\s+(?:related|relevant|applicable)\b", re.I),
    re.compile(r"\bfew\s+studies\s+(?:have\s+been\s+)?(?:carried\s+out|related|relevant|applicable)\b", re.I),
    re.compile(r"\brandom distribution\b", re.I),
    re.compile(r"\bnone\s+of\s+the\s+[^.]{0,80}\b(?:detected|identified|recognized)\b", re.I),
    re.compile(r"\bfailed\s+to\b", re.I),
)
_CALIBRATION_STRONG_YES_PATTERNS = (
    re.compile(r"\bsignificant(?:ly)?\s+(?:improved|reduced|increased|associated|predicted|higher|lower|different|better)\b", re.I),
    re.compile(r"\b(?:improved|reduced|increased|predicted|identified|detected|correlated|associated)\b", re.I),
    re.compile(r"\b(?:sensitivity|specificity|odds ratio|hazard ratio|relative risk|correlation coefficient)\b", re.I),
    re.compile(r"\b(?:can be used|useful|reliable|accurate|effective|beneficial|feasible|superior)\b", re.I),
    re.compile(r"\b(?:acceptable|accepted|satisfaction\s+(?:was\s+)?high|high\s+satisfaction|main\s+reason|marked\s+improvement|step-change|safely\s+created)\b", re.I),
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


def _log_decision(decision: EvidenceDecisionInfo) -> None:
    logger.info(
        "evidence_judge decision method=%s status=%s answer_label=%s confidence=%s citations=%s notes=%s",
        decision.method,
        decision.status,
        decision.answer_label,
        f"{decision.confidence:.2f}" if decision.confidence is not None else "none",
        " ".join(decision.citations),
        " | ".join(decision.notes),
    )


def _is_yes_no_maybe_task(messages: list[ChatMessage], pre_retrieval: PreRetrievalResult) -> bool:
    return is_yes_no_maybe_task_text(pre_retrieval.original_query) or is_yes_no_maybe_task_text(_user_question(messages))


def _user_question(messages: list[ChatMessage]) -> str:
    return "\n".join(message.content for message in messages if message.role == "user").strip()


def _count_matches(patterns: tuple[re.Pattern[str], ...], text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def _best_rule_sentence(source_documents: list[RetrievedDocument], *, label: str) -> str:
    patterns = {
        "yes": (*_RULE_DIRECT_YES_PATTERNS, *_POSITIVE_PATTERNS),
        "no": (*_RULE_DIRECT_NO_PATTERNS, *_NEGATIVE_PATTERNS),
        "maybe": (*_RULE_DIRECT_MAYBE_PATTERNS, *_UNCERTAIN_PATTERNS),
    }[label]
    fallback = ""
    for index, document in enumerate(source_documents[:3], start=1):
        _, filtered_sentences = _rule_evidence_text([document])
        sentences = [
            sentence.strip()
            for sentence in filtered_sentences
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
