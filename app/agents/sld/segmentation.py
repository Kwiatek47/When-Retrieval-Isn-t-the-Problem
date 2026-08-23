"""Stage 0 of the Supervised Ledger Debate pipeline: deterministic, LLM-free.

Turns a raw abstract into a stable sentence dictionary (``S1..Sn``) that every
later stage (panel, verifier, ledger, director) cites into instead of quoting
free text. Also extracts a regex-only ``stats_profile`` and a coarse
``question_type`` — both free signal that today's debate never sees explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

SectionLabel = Literal["BACKGROUND", "METHODS", "RESULTS", "OTHER"]
QuestionType = Literal[
    "utility", "association", "causal", "comparison", "diagnostic_accuracy", "prevalence"
]

# The corpus wraps every abstract in benchmark boilerplate before the real text:
# "PubMedQA <benchmark> source PMID <id>. Research question: <question> Abstract context: <abstract>"
# Segmenting the boilerplate would poison S1/S2 with metadata instead of evidence.
_ABSTRACT_CONTEXT_MARKER = "Abstract context: "


def extract_abstract_text(raw_content: str) -> str:
    """Strip the corpus.json benchmark-metadata prefix, if present."""
    text = raw_content or ""
    idx = text.find(_ABSTRACT_CONTEXT_MARKER)
    if idx == -1:
        return text.strip()
    return text[idx + len(_ABSTRACT_CONTEXT_MARKER) :].strip()


# Sentence splitting on medical abstracts has to survive "p<0.01", "95% CI",
# "e.g.", "et al." etc. without breaking mid-number or mid-abbreviation.
_ABBREVIATIONS: tuple[str, ...] = (
    "e.g.",
    "i.e.",
    "et al.",
    "vs.",
    "cf.",
    "approx.",
    "Fig.",
    "fig.",
    "No.",
    "no.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "St.",
    "vol.",
    "pp.",
)
_DECIMAL_POINT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PLACEHOLDER = chr(1)  # sentinel unlikely to occur in abstract text


def split_sentences(text: str) -> list[tuple[str, str]]:
    """Split into ``(sentence_id, sentence_text)`` pairs with stable ``S1..Sn`` IDs."""
    stripped = (text or "").strip()
    if not stripped:
        return []

    protected = _DECIMAL_POINT_RE.sub(_PLACEHOLDER, stripped)
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _PLACEHOLDER))

    raw_sentences = _SENTENCE_BOUNDARY_RE.split(protected)

    sentences: list[tuple[str, str]] = []
    next_index = 1
    for raw in raw_sentences:
        restored = raw.replace(_PLACEHOLDER, ".").strip()
        if not restored:
            continue
        sentences.append((f"S{next_index}", restored))
        next_index += 1
    return sentences


# --- Section tagging -------------------------------------------------------
# No structured headers survive into corpus.json (verified: 0/500 official
# PQA-L docs contain "RESULTS:" etc.), so tagging is phrase-heuristic only.
_RESULTS_PHRASES: tuple[str, ...] = (
    "we found",
    "we observed",
    "results showed",
    "results indicated",
    "results demonstrated",
    "was significantly",
    "were significantly",
    "was not significantly",
    "were not significantly",
    "was associated with",
    "were associated with",
    "was independently associated",
    "no significant difference",
    "no significant association",
    "significantly higher",
    "significantly lower",
    "significantly increased",
    "significantly decreased",
    "compared with",
    "compared to",
    "than in",
    "resulted in",
    "led to a",
)
_METHODS_PHRASES: tuple[str, ...] = (
    "we performed",
    "we conducted",
    "we retrospectively",
    "we prospectively",
    "we randomly",
    "we used",
    "we analyzed",
    "we analysed",
    "we reviewed",
    "we compared",
    "we assessed",
    "we evaluated",
    "we measured",
    "we enrolled",
    "we recruited",
    "patients were",
    "subjects were",
    "participants were",
    "were randomly assigned",
    "were randomized",
    "were enrolled",
    "were recruited",
    "were included",
    "were excluded",
    "inclusion criteria",
    "exclusion criteria",
    "study design",
    "retrospective study",
    "prospective study",
    "cohort study",
    "case-control study",
    "questionnaire",
    "were divided into",
)
_BACKGROUND_PHRASES: tuple[str, ...] = (
    "is a common",
    "is one of the",
    "little is known",
    "remains unclear whether",
    "remains controversial",
    "aim of this",
    "purpose of this",
    "objective of this",
    "this study aimed",
    "this study aims",
    "we aimed",
    "we hypothesized",
    "we hypothesised",
    "we investigated whether",
    "previous studies",
    "prior studies",
    "it is unknown",
    "it remains unknown",
)

# Shared with extract_stats_profile: a RESULTS sentence very often carries a
# stat marker even without a finding-verb phrase ("Sensitivity was 82%.").
_P_VALUE_RE = re.compile(r"\bp\s*[<=>]\s*0?\.\d+", re.IGNORECASE)
_CI_RE = re.compile(r"\b\d{1,3}\s*%\s*ci\b|\bconfidence interval\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_SAMPLE_SIZE_RE = re.compile(r"\bn\s*=\s*\d+\b", re.IGNORECASE)
_EFFECT_SIZE_RE = re.compile(r"\b(?:OR|HR|RR)\s*[:=]?\s*\d+(?:\.\d+)?\b")
_DIAGNOSTIC_METRIC_RE = re.compile(r"\b(?:AUC|PPV|NPV)\b")


def _has_stats_marker(sentence: str) -> bool:
    return bool(
        _P_VALUE_RE.search(sentence)
        or _CI_RE.search(sentence)
        or _PERCENT_RE.search(sentence)
        or _SAMPLE_SIZE_RE.search(sentence)
        or _EFFECT_SIZE_RE.search(sentence)
        or _DIAGNOSTIC_METRIC_RE.search(sentence)
    )


def _contains_any(text_lower: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text_lower for phrase in phrases)


def tag_sections(sentences: list[tuple[str, str]]) -> dict[str, SectionLabel]:
    """Tag each sentence id with a coarse section label.

    Priority: RESULTS (stats marker or finding-verb phrase) > BACKGROUND
    (purpose/rationale phrase) > METHODS (procedure/recruitment phrase) >
    OTHER (no signal either way). BACKGROUND is checked before METHODS
    because purpose sentences ("The aim of this prospective study...")
    routinely name the study design in passing without being a methods
    sentence themselves.
    """
    tags: dict[str, SectionLabel] = {}
    for sentence_id, sentence in sentences:
        lower = sentence.lower()
        if _has_stats_marker(sentence) or _contains_any(lower, _RESULTS_PHRASES):
            tags[sentence_id] = "RESULTS"
        elif _contains_any(lower, _BACKGROUND_PHRASES):
            tags[sentence_id] = "BACKGROUND"
        elif _contains_any(lower, _METHODS_PHRASES):
            tags[sentence_id] = "METHODS"
        else:
            tags[sentence_id] = "OTHER"
    return tags


# --- stats_profile -----------------------------------------------------------


@dataclass
class StatsHit:
    sentence_id: str
    span: str


@dataclass
class StatsProfile:
    p_values: list[StatsHit] = field(default_factory=list)
    confidence_intervals: list[StatsHit] = field(default_factory=list)
    percentages: list[StatsHit] = field(default_factory=list)
    sample_sizes: list[StatsHit] = field(default_factory=list)
    effect_sizes: list[StatsHit] = field(default_factory=list)
    diagnostic_metrics: list[StatsHit] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.p_values
            or self.confidence_intervals
            or self.percentages
            or self.sample_sizes
            or self.effect_sizes
            or self.diagnostic_metrics
        )

    def sentence_ids(self) -> set[str]:
        """Every sentence id carrying at least one statistical marker."""
        ids: set[str] = set()
        for hits in (
            self.p_values,
            self.confidence_intervals,
            self.percentages,
            self.sample_sizes,
            self.effect_sizes,
            self.diagnostic_metrics,
        ):
            ids.update(hit.sentence_id for hit in hits)
        return ids


def extract_stats_profile(sentences: list[tuple[str, str]]) -> StatsProfile:
    """Regex-only extraction of quantitative claims, tagged with sentence IDs."""
    profile = StatsProfile()
    for sentence_id, sentence in sentences:
        for pattern, bucket in (
            (_P_VALUE_RE, profile.p_values),
            (_CI_RE, profile.confidence_intervals),
            (_PERCENT_RE, profile.percentages),
            (_SAMPLE_SIZE_RE, profile.sample_sizes),
            (_EFFECT_SIZE_RE, profile.effect_sizes),
            (_DIAGNOSTIC_METRIC_RE, profile.diagnostic_metrics),
        ):
            for match in pattern.finditer(sentence):
                bucket.append(StatsHit(sentence_id=sentence_id, span=match.group(0)))
    return profile


# --- question_type -----------------------------------------------------------
# Checked most-specific first: "Is X valuable in diagnosing Y" should classify
# as diagnostic_accuracy, not utility, even though "valuable" also fires.
_DIAGNOSTIC_ACCURACY_PHRASES: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "predictive value",
    "diagnos",
    "screening",
    "detect",
    "differentiat",
)
_PREVALENCE_PHRASES: tuple[str, ...] = (
    "prevalence",
    "incidence",
    "how common",
    "frequency of",
    "how frequent",
)
_CAUSAL_PHRASES: tuple[str, ...] = (
    "cause",
    "causes",
    "caused by",
    "lead to",
    "leads to",
    "result in",
    "results in",
    "risk factor for",
    "due to",
)
_COMPARISON_PHRASES: tuple[str, ...] = (
    "compared to",
    "compared with",
    "versus",
    " vs ",
    " vs.",
    "better than",
    "superior to",
    "inferior to",
    "differ from",
    "difference between",
)
_UTILITY_PHRASES: tuple[str, ...] = (
    "valuable",
    "useful",
    "effective",
    "efficacious",
    "efficacy",
    "beneficial",
    "benefit",
    "helpful",
    "worthwhile",
)


def classify_question_type(question: str) -> QuestionType:
    """Coarse question-type classification from surface phrasing.

    Order matters: more specific categories (diagnostic_accuracy, prevalence,
    causal, comparison) are checked before the broader utility/association
    fallback, since e.g. "valuable in diagnosing" should not be classified as
    plain utility.
    """
    lower = (question or "").lower()
    if _contains_any(lower, _DIAGNOSTIC_ACCURACY_PHRASES):
        return "diagnostic_accuracy"
    if _contains_any(lower, _PREVALENCE_PHRASES):
        return "prevalence"
    if _contains_any(lower, _CAUSAL_PHRASES):
        return "causal"
    if _contains_any(lower, _COMPARISON_PHRASES):
        return "comparison"
    if _contains_any(lower, _UTILITY_PHRASES):
        return "utility"
    return "association"
