"""Prompt builders for the Supervised Ledger Debate pipeline.

Every prompt is built from the typed ``S1..Sn`` sentence dictionary
(:mod:`app.agents.sld.segmentation`), never from a raw abstract string, so a
persona can only ever cite an ID that actually exists. Round 1 prompts never
mention "yes/no/maybe" — personas are label-blind (P2 in the design doc: label
priors baked into persona prompts is what produced a zero-information
``uncertainty_advocate`` in the legacy debate). Round 2 drops the legacy
"attack the peers" framing entirely; its instructions ask agents to complete
the ledger, not win an argument (P1 / ColMAD, design doc §3).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from app.agents.sld.ledger import (
    ConclusionReconstructorContribution,
    DirectorVerdict,
    EvidenceLedger,
    FindingsAuditorContribution,
    GapAuditorContribution,
    LedgerConflict,
    NeutralContribution,
    PanelContribution,
    QuestionFramerContribution,
    RoundTwoOpinion,
)
from app.agents.sld.segmentation import StatsProfile

SYSTEM_JSON_ONLY = "Return only valid JSON matching the requested schema. No markdown fences, no commentary."

# "~1600 tok" per the design doc (§4 P4 fix). Char-based estimate is
# deliberately coarse (real tokenization is the provider's job); the point is
# to fail loudly on a runaway prompt, not to silently truncate it the way the
# legacy Director prompt does today (design doc P4).
#
# R1/R2 prompts run once per persona per round (x4) so they keep the tight
# budget; Moderator/Director each run once per round but aggregate up to 4
# contributions/opinions, so they get more headroom — enforced in two layers,
# not one: render_contribution/render_round_two_opinion truncate individual
# free-text fields deterministically (real 7B models routinely ignore "be
# concise" instructions, so this can't be prompt-only), AND the aggregate
# call still gets a hard budget on top, so a pathological case fails loudly
# instead of being silently truncated by the provider.
MAX_PROMPT_TOKENS = 1600
AGGREGATE_MAX_PROMPT_TOKENS = 2800
CHARS_PER_TOKEN_ESTIMATE = 4
_TRUNCATED_FIELD_CHARS = 220


class PromptBudgetExceeded(ValueError):
    pass


def assert_token_budget(text: str, *, label: str, max_tokens: int = MAX_PROMPT_TOKENS) -> None:
    estimated_tokens = len(text) / CHARS_PER_TOKEN_ESTIMATE
    if estimated_tokens > max_tokens:
        raise PromptBudgetExceeded(
            f"{label}: prompt is ~{estimated_tokens:.0f} estimated tokens, "
            f"over the {max_tokens} budget ({len(text)} chars). "
            "Shorten the prompt instead of letting the provider truncate it."
        )


# --- A narrow local schema for the Moderator's synthesis-only call -----------
# Not part of the public Evidence Ledger schema (app.agents.sld.ledger): the
# Moderator LLM call only ever produces conflicts/open_questions/
# round_instructions; merging that into a full EvidenceLedger is
# supervisor.py's job, not something this call's schema needs to represent.


class ModeratorSynthesis(BaseModel):
    conflicts: list[LedgerConflict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    round_instructions: list[str] = Field(default_factory=list)


# --- Rendering helpers --------------------------------------------------------


def render_sentences(
    sentences: dict[str, str],
    *,
    section_tags: dict[str, str] | None = None,
    only_ids: set[str] | None = None,
) -> str:
    """Render the S-ID dictionary as ``S1 [RESULTS]: text`` lines.

    ``only_ids`` restricts which sentences are shown at all (e.g. RESULTS-only
    for findings_auditor); ``section_tags`` just annotates each line.
    """
    ids = sorted(
        (sid for sid in sentences if only_ids is None or sid in only_ids),
        key=lambda sid: int(sid[1:]),
    )
    lines = []
    for sid in ids:
        tag = f" [{section_tags[sid]}]" if section_tags and sid in section_tags else ""
        lines.append(f"{sid}{tag}: {sentences[sid]}")
    return "\n".join(lines)


def render_stats_profile(profile: StatsProfile) -> str:
    """Render regex-extracted statistical markers as a compact reference block.

    Free, hallucination-proof signal (design doc §4 Stage 0 step 3) that no
    persona sees explicitly today; surfacing it lets findings_auditor and
    gap_auditor reason about significance/power without re-deriving it from
    prose.
    """
    rows: list[str] = []
    for label, hits in (
        ("p-value", profile.p_values),
        ("CI", profile.confidence_intervals),
        ("percentage", profile.percentages),
        ("sample size", profile.sample_sizes),
        ("effect size", profile.effect_sizes),
        ("diagnostic metric", profile.diagnostic_metrics),
    ):
        for hit in hits:
            rows.append(f"- {label}: {hit.span} ({hit.sentence_id})")
    return "\n".join(rows) if rows else "(none found)"


def _with_schema(
    prompt: str, model_cls: type[BaseModel], *, max_tokens: int = MAX_PROMPT_TOKENS
) -> str:
    schema = model_cls.model_json_schema()
    full = f"{prompt}\n\nJSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
    assert_token_budget(full, label=model_cls.__name__, max_tokens=max_tokens)
    return full


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def render_contribution(contribution: PanelContribution, *, max_field_chars: int | None = None) -> str:
    """Compact human-readable rendering of one R1 contribution, for the
    Supervisor/Moderator, a Round 2 agent's own R1 note, or R1 peer notes.

    ``max_field_chars`` caps each free-text field's rendered length —
    callers that aggregate multiple contributions into one prompt (Moderator:
    up to 4; L4 peer notes: up to 3) pass this so the aggregate call can't
    blow its budget on a single verbose field; a single-contribution caller
    (a Round 2 agent's own note) leaves it uncapped.
    """
    t = lambda s: _truncate(s, max_field_chars)  # noqa: E731
    parts = [f"[{contribution.persona} / {contribution.agent_id}]"]
    persona = contribution.persona
    if persona == "question_framer":
        parts.append(f"question_type: {contribution.question_type}")
        if contribution.target_population:
            parts.append(f"target_population: {t(contribution.target_population.text)}")
        if contribution.target_exposure:
            parts.append(f"target_exposure: {t(contribution.target_exposure.text)}")
        if contribution.target_outcome:
            parts.append(f"target_outcome: {t(contribution.target_outcome.text)}")
        parts.append(f"yes_requires: {t(contribution.yes_requires)}")
        parts.append(f"no_requires: {t(contribution.no_requires)}")
    elif persona == "findings_auditor":
        parts.append(f"direction: {contribution.direction}")
        if contribution.primary_endpoint:
            parts.append(
                f"primary_endpoint: {t(contribution.primary_endpoint.text)} "
                f"({', '.join(contribution.primary_endpoint.sentence_ids)})"
            )
        if contribution.significance:
            parts.append(
                f"significance: {t(contribution.significance.text)} "
                f"({', '.join(contribution.significance.sentence_ids)})"
            )
        if contribution.effect_magnitude:
            parts.append(
                f"effect_magnitude: {t(contribution.effect_magnitude.text)} "
                f"({', '.join(contribution.effect_magnitude.sentence_ids)})"
            )
    elif persona == "gap_auditor":
        if not contribution.gaps:
            parts.append("gaps: (none found)")
        for gap in contribution.gaps:
            cites = f" ({', '.join(gap.sentence_ids)})" if gap.sentence_ids else ""
            parts.append(f"gap[{gap.gap_type}]: {t(gap.description)}{cites}")
    elif persona == "conclusion_reconstructor":
        parts.append(f"direction: {contribution.direction}, strength: {contribution.strength}")
        if contribution.reconstructed_conclusion:
            parts.append(
                f"reconstructed_conclusion: {t(contribution.reconstructed_conclusion.text)} "
                f"({', '.join(contribution.reconstructed_conclusion.sentence_ids)})"
            )
    elif persona == "neutral":
        parts.append(f"question_type: {contribution.question_type}")
        for field_name in ("target_population", "target_exposure", "target_outcome", "primary_endpoint"):
            claim = getattr(contribution, field_name)
            if claim:
                parts.append(f"{field_name}: {t(claim.text)} ({', '.join(claim.sentence_ids)})")
        parts.append(f"direction: {contribution.direction}")
        if contribution.significance:
            parts.append(
                f"significance: {t(contribution.significance.text)} "
                f"({', '.join(contribution.significance.sentence_ids)})"
            )
        if contribution.effect_magnitude:
            parts.append(
                f"effect_magnitude: {t(contribution.effect_magnitude.text)} "
                f"({', '.join(contribution.effect_magnitude.sentence_ids)})"
            )
        if not contribution.gaps:
            parts.append("gaps: (none found)")
        for gap in contribution.gaps:
            cites = f" ({', '.join(gap.sentence_ids)})" if gap.sentence_ids else ""
            parts.append(f"gap[{gap.gap_type}]: {t(gap.description)}{cites}")
        if contribution.reconstructed_conclusion:
            parts.append(
                f"reconstructed_conclusion: {t(contribution.reconstructed_conclusion.text)} "
                f"({', '.join(contribution.reconstructed_conclusion.sentence_ids)}), "
                f"conclusion_direction: {contribution.conclusion_direction}, "
                f"strength: {contribution.conclusion_strength}"
            )
    return "\n".join(parts)


def render_ledger(ledger: EvidenceLedger) -> str:
    """Always truncates its free-text fields (unlike render_contribution's
    optional cap): the ledger is rendered into every R2 prompt (x4, tight
    1600-token budget) and the Director prompt, so it can't be allowed to
    grow unbounded from a verbose Moderator conflict/open_question/instruction."""
    t = lambda s: _truncate(s, _TRUNCATED_FIELD_CHARS)  # noqa: E731
    lines: list[str] = []
    if ledger.question_type:
        lines.append(f"question_type: {ledger.question_type}")
    for field_name, claim in (
        ("target_population", ledger.target_population),
        ("target_exposure", ledger.target_exposure),
        ("target_outcome", ledger.target_outcome),
        ("primary_endpoint", ledger.primary_endpoint),
        ("significance", ledger.significance),
        ("effect_magnitude", ledger.effect_magnitude),
        ("reconstructed_conclusion", ledger.reconstructed_conclusion),
    ):
        if claim is not None:
            lines.append(f"{field_name}: {t(claim.text)} ({', '.join(claim.sentence_ids)})")
    if ledger.direction:
        lines.append(f"direction (findings): {ledger.direction}")
    if ledger.conclusion_direction:
        lines.append(
            f"conclusion_direction: {ledger.conclusion_direction} "
            f"(strength: {ledger.conclusion_strength})"
        )
    if ledger.gaps:
        for gap in ledger.gaps:
            cites = f" ({', '.join(gap.sentence_ids)})" if gap.sentence_ids else ""
            lines.append(f"gap[{gap.gap_type}]: {t(gap.description)}{cites}")
    if ledger.conflicts:
        for conflict in ledger.conflicts:
            lines.append(
                f"conflict: {t(conflict.description)} "
                f"[{', '.join(conflict.agent_ids)}] ({', '.join(conflict.sentence_ids)})"
            )
    if ledger.open_questions:
        lines.append("open_questions: " + "; ".join(t(q) for q in ledger.open_questions))
    if ledger.round_instructions:
        lines.append("round_instructions: " + "; ".join(t(i) for i in ledger.round_instructions))
    return "\n".join(lines) if lines else "(empty ledger)"


def render_round_two_opinion(opinion: RoundTwoOpinion, *, max_field_chars: int | None = None) -> str:
    t = lambda s: _truncate(s, max_field_chars)  # noqa: E731
    parts = [f"[{opinion.agent_id}] label={opinion.label}: {t(opinion.rationale)}"]
    if opinion.citations:
        parts.append(f"citations: {', '.join(opinion.citations)}")
    if opinion.complement:
        parts.append(f"complement: {t(opinion.complement)}")
    if opinion.self_audit:
        parts.append(f"self_audit: {t(opinion.self_audit)}")
    return "\n".join(parts)


# --- Round 1: label-blind panel -----------------------------------------------
# None of these prompts mention "yes", "no", or "maybe" anywhere by default (P2 fix).

_R1_PREAMBLE_BLIND = """You are one analyst on a panel reviewing a biomedical research abstract. \
Your job is narrow and factual: extract what the text actually says, citing sentence IDs \
for every claim. You do NOT decide or hint at a yes/no/maybe answer to the research question \
— that is a different agent's job later. Never cite a sentence ID that isn't listed below, \
and never assert something the cited sentence doesn't actually say."""

# Ablation (d) (design doc §7): removes the label-blind instruction and tells
# the analyst the eventual task up front, to measure the cost of the label
# prior it reintroduces (P2 in the design doc: this is exactly what produced
# a zero-information uncertainty_advocate in the legacy debate).
_R1_PREAMBLE_NOT_BLIND = """You are one analyst on a panel reviewing a biomedical research \
abstract. The panel's ultimate job is to answer the research question yes, no, or maybe; keep \
that in mind while you extract what the text actually says, citing sentence IDs for every \
claim. Never cite a sentence ID that isn't listed below, and never assert something the cited \
sentence doesn't actually say."""


def _r1_preamble(label_blind: bool) -> str:
    return _R1_PREAMBLE_BLIND if label_blind else _R1_PREAMBLE_NOT_BLIND


def build_question_framer_prompt(
    question: str, sentences: dict[str, str], *, label_blind: bool = True
) -> str:
    prompt = f"""{_r1_preamble(label_blind)}

ROLE: question_framer. Identify what the research question is actually asking, independent \
of what the abstract found.

RESEARCH QUESTION:
{question}

ABSTRACT SENTENCES:
{render_sentences(sentences)}

Produce:
- target_population: who/what was studied (cite sentence_ids)
- target_exposure: the intervention/exposure/test named in the question (cite sentence_ids)
- target_outcome: the outcome the question is actually asking about (cite sentence_ids)
- question_type: one of utility, association, causal, comparison, diagnostic_accuracy, prevalence
- yes_requires: in one sentence, what finding would make the answer "yes"
- no_requires: in one sentence, what finding would make the answer "no"
"""
    return _with_schema(prompt.strip(), QuestionFramerContribution)


def build_findings_auditor_prompt(
    question: str,
    sentences: dict[str, str],
    section_tags: dict[str, str],
    stats_profile: StatsProfile,
    *,
    label_blind: bool = True,
) -> str:
    results_ids = {sid for sid, tag in section_tags.items() if tag == "RESULTS"}
    prompt = f"""{_r1_preamble(label_blind)}

ROLE: findings_auditor. Report ONLY what the results actually showed. You may cite ONLY the \
RESULTS sentences listed below — citing any other sentence will get your claim rejected.

RESEARCH QUESTION:
{question}

RESULTS SENTENCES ONLY:
{render_sentences(sentences, section_tags=section_tags, only_ids=results_ids) or "(no RESULTS-tagged sentences found)"}

STATISTICAL MARKERS FOUND (for reference, already extracted by regex — cross-check, don't invent new ones):
{render_stats_profile(stats_profile)}

Produce:
- primary_endpoint: the main result relevant to the research question (cite sentence_ids, RESULTS only)
- direction: positive (supports a "yes" reading), negative (supports a "no" reading), or none (no clear direction)
- significance: what the text says about statistical significance, if anything (cite sentence_ids, RESULTS only)
- effect_magnitude: the size of the effect, if stated (cite sentence_ids, RESULTS only)
"""
    return _with_schema(prompt.strip(), FindingsAuditorContribution)


def build_gap_auditor_prompt(
    question: str,
    sentences: dict[str, str],
    stats_profile: StatsProfile,
    *,
    label_blind: bool = True,
) -> str:
    prompt = f"""{_r1_preamble(label_blind)}

ROLE: gap_auditor. Identify evidentiary gaps that would make the research question hard to \
answer confidently from this abstract alone. An empty list is a completely valid answer if you \
find no real gaps — do not invent a gap to have something to say.

RESEARCH QUESTION:
{question}

ABSTRACT SENTENCES:
{render_sentences(sentences)}

STATISTICAL MARKERS FOUND (for reference):
{render_stats_profile(stats_profile)}

For each gap you find, use one of these gap_type values:
- surrogate_outcome: the measured outcome is a stand-in for the outcome the question really asks about
- subgroup_only: the finding only applies to a subgroup, not the population the question asks about
- association_not_utility: an association is shown but the question asks about utility/benefit
- no_comparator: no control/comparison group was used
- underpowered: sample size is explicitly too small to support the claim
- contradictory_endpoints: different results point in different directions
- weak_discrimination: a diagnostic/test result barely distinguishes groups

Produce: gaps (a list; each with gap_type, description, and sentence_ids if the gap is grounded \
in a specific sentence — omit sentence_ids only if the gap is an *absence*, like no_comparator).
"""
    return _with_schema(prompt.strip(), GapAuditorContribution)


def build_conclusion_reconstructor_prompt(
    question: str, sentences: dict[str, str], *, label_blind: bool = True
) -> str:
    prompt = f"""{_r1_preamble(label_blind)}

ROLE: conclusion_reconstructor. This abstract has no CONCLUSIONS sentence — it was stripped \
from the source data. Reconstruct, in your own words, the single sentence the authors most \
likely would have written as their conclusion, based only on the METHODS and RESULTS given. \
Do NOT answer the research question yourself; only characterize the conclusion's direction and \
how strongly it would likely be worded.

RESEARCH QUESTION:
{question}

ABSTRACT SENTENCES:
{render_sentences(sentences)}

Produce:
- reconstructed_conclusion: the likely conclusion sentence, in your own words (cite the sentence_ids it's based on)
- direction: positive, negative, or none
- strength: definitive (authors would state it plainly), qualified (authors would hedge it, e.g. "may", "suggests"), or speculative (authors would flag it as needing further study)
"""
    return _with_schema(prompt.strip(), ConclusionReconstructorContribution)


# --- Ablation (c): neutral (non-specialized) R1 agent -------------------------
# Same total extraction surface as the four specialized personas combined,
# run by identical agents instead — isolates whether role specialization
# itself adds value (design doc §7c).


def build_neutral_prompt(
    question: str,
    sentences: dict[str, str],
    section_tags: dict[str, str],
    stats_profile: StatsProfile,
    *,
    label_blind: bool = True,
) -> str:
    prompt = f"""{_r1_preamble(label_blind)}

ROLE: none — you are one of several identical analysts independently doing the full extraction \
task below (there is no role specialization in this ablation run). Sentences are tagged with \
their section in brackets, e.g. "S6 [RESULTS]: ...".

RESEARCH QUESTION:
{question}

ABSTRACT SENTENCES:
{render_sentences(sentences, section_tags=section_tags)}

STATISTICAL MARKERS FOUND (for reference, already extracted by regex — cross-check, don't invent new ones):
{render_stats_profile(stats_profile)}

Produce:
- target_population: who/what was studied (cite sentence_ids)
- target_exposure: the intervention/exposure/test named in the question (cite sentence_ids)
- target_outcome: the outcome the question is actually asking about (cite sentence_ids)
- question_type: one of utility, association, causal, comparison, diagnostic_accuracy, prevalence
- primary_endpoint: the main result relevant to the research question (cite sentence_ids — [RESULTS] sentences only)
- direction: positive (supports a "yes" reading), negative (supports a "no" reading), or none (no clear direction)
- significance: what the text says about statistical significance, if anything (cite sentence_ids — [RESULTS] sentences only)
- effect_magnitude: the size of the effect, if stated (cite sentence_ids — [RESULTS] sentences only)
- gaps: evidentiary gaps that would make the research question hard to answer confidently \
(gap_type one of surrogate_outcome, subgroup_only, association_not_utility, no_comparator, \
underpowered, contradictory_endpoints, weak_discrimination; empty list is a valid answer)
- reconstructed_conclusion: this abstract has no CONCLUSIONS sentence (stripped from the source \
data) — reconstruct, in your own words, the sentence the authors most likely would have written \
(cite the sentence_ids it's based on)
- conclusion_direction: positive, negative, or none
- conclusion_strength: definitive (authors would state it plainly), qualified (authors would hedge \
it, e.g. "may", "suggests"), or speculative (authors would flag it as needing further study)
"""
    # Legitimately more work per call than any single specialized R1 prompt
    # (it does all four personas' extraction at once), so it needs more than
    # the standard R1 budget — unlike Moderator/Director this call still runs
    # x4 per round (once per neutral agent), so this is a real, not free,
    # compute trade-off inherent to the ablation, not a workaround for bloat.
    return _with_schema(prompt.strip(), NeutralContribution, max_tokens=AGGREGATE_MAX_PROMPT_TOKENS)


# --- Supervisor / Moderator ---------------------------------------------------
# Deterministic code already merges verified R1 claims into a draft ledger
# (see supervisor.py:merge_verified_contributions) — this call's only job is
# judgment: is there an actual conflict, what's still unknown, what should R2
# focus on. Keeping its output schema narrow (conflicts/open_questions/
# round_instructions only) means the LLM never re-asserts a claim that was
# already verified, so it can't reintroduce a hallucination gate #1 removed.


def build_moderator_prompt(
    question: str,
    verified_contributions: list[PanelContribution],
) -> str:
    # Deliberately a single source of truth: the per-agent contributions
    # below already carry every citation the draft ledger would be merged
    # from, so re-rendering the merged ledger and the full abstract on top
    # would be redundant content burning the token budget on a call whose job
    # is judgment (conflicts/gaps/instructions), not re-extraction.
    contributions_block = "\n\n".join(
        render_contribution(c, max_field_chars=_TRUNCATED_FIELD_CHARS) for c in verified_contributions
    )
    prompt = f"""You are the Supervisor moderating a panel that just reviewed a biomedical \
abstract independently (each analyst could not see the others' answers). Your job is judgment, \
not extraction: the facts below were already extracted and citation-checked by code — do not \
restate them, and do not invent new claims or new sentence_ids.

RESEARCH QUESTION:
{question}

VERIFIED PANEL CONTRIBUTIONS (already citation-checked; field names tell you which persona a \
fact came from — e.g. `direction` is findings_auditor's, `conclusion_direction` is \
conclusion_reconstructor's):
{contributions_block}

Produce:
- conflicts: places where two contributions disagree (e.g. findings_auditor's direction vs \
conclusion_reconstructor's direction) — for each, name the agent_ids involved, cite ONLY \
sentence_ids that already appear above, and describe the disagreement. Empty list if there is no \
real conflict.
- open_questions: what a reader would still need to know that isn't in the contributions above. Empty list if none.
- round_instructions: 1-3 short, concrete instructions for round 2, derived ONLY from the gaps/\
open_questions/conflicts above — not generic advice like "be more careful".
"""
    return _with_schema(prompt.strip(), ModeratorSynthesis, max_tokens=AGGREGATE_MAX_PROMPT_TOKENS)


# --- Round 2: cooperative debate on the ledger --------------------------------
# No "attack the peers" framing (P1 fix): the instruction is to complete the
# ledger, not defend a position.

_R2_PREAMBLE = """You are the same analyst from round 1, now looking at the shared evidence \
ledger the Supervisor assembled from the whole panel. Your goal is cooperative: use the ledger \
to fill in what your own round-1 note missed, and now commit to an actual answer. This is not a \
debate to win — if the ledger changes your mind, say so."""


def build_round_two_prompt(
    question: str,
    sentences: dict[str, str],
    ledger_rendering: str,
    own_r1_contribution: PanelContribution,
    round_instructions: list[str],
) -> str:
    instructions_block = (
        "\n".join(f"- {instruction}" for instruction in round_instructions)
        if round_instructions
        else "(none)"
    )
    prompt = f"""{_R2_PREAMBLE}

RESEARCH QUESTION:
{question}

YOUR OWN ROUND-1 NOTE:
{render_contribution(own_r1_contribution)}

EVIDENCE LEDGER (from the whole panel, Supervisor-verified):
{ledger_rendering}

SUPERVISOR INSTRUCTIONS FOR THIS ROUND:
{instructions_block}

ABSTRACT SENTENCES:
{render_sentences(sentences)}

Produce:
- label: yes, no, or maybe — your actual answer to the research question
- rationale: why, citing sentence_ids
- citations: the sentence_ids your rationale relies on
- complement: one fact from the ledger that your own round-1 note missed (omit/empty if none)
- self_audit: one plausible way your own reading of the evidence could be wrong (omit/empty if none)
"""
    return _with_schema(prompt.strip(), RoundTwoOpinion)


_R2_PEER_PREAMBLE = """You are the same analyst from round 1, now looking at your fellow \
panelists' round-1 notes directly (no Supervisor synthesis this time). Your goal is cooperative: \
use their notes to fill in what your own round-1 note missed, and now commit to an actual answer. \
This is not a debate to win — if a peer's note changes your mind, say so."""


def build_round_two_peer_prompt(
    question: str,
    sentences: dict[str, str],
    peer_notes_rendering: str,
    own_r1_contribution: PanelContribution,
    round_instructions: list[str],
) -> str:
    """L4 ablation arm (design doc §7): round 2 without the verified ledger —
    agents see raw, unverified peer notes instead, isolating the ledger's
    marginal value at an identical round/call count."""
    instructions_block = (
        "\n".join(f"- {instruction}" for instruction in round_instructions)
        if round_instructions
        else "(none)"
    )
    prompt = f"""{_R2_PEER_PREAMBLE}

RESEARCH QUESTION:
{question}

YOUR OWN ROUND-1 NOTE:
{render_contribution(own_r1_contribution)}

PEER ROUND-1 NOTES (unverified — cross-check citations yourself):
{peer_notes_rendering}

SUPERVISOR INSTRUCTIONS FOR THIS ROUND:
{instructions_block}

ABSTRACT SENTENCES:
{render_sentences(sentences)}

Produce:
- label: yes, no, or maybe — your actual answer to the research question
- rationale: why, citing sentence_ids
- citations: the sentence_ids your rationale relies on
- complement: one fact from a peer's note that your own round-1 note missed (omit/empty if none)
- self_audit: one plausible way your own reading of the evidence could be wrong (omit/empty if none)
"""
    return _with_schema(prompt.strip(), RoundTwoOpinion)


# --- Supervisor / Director ----------------------------------------------------


def build_director_prompt(
    question: str,
    ledger_rendering: str,
    round_two_opinions: list[RoundTwoOpinion],
) -> str:
    # Deliberately no raw abstract block here (same fix as build_moderator_prompt):
    # the ledger and the R2 opinions already carry every citation the Director
    # needs, each traceable to a sentence_id. Re-rendering the full abstract on
    # top of both burned the token budget on real R2 output (rationale/
    # complement/self_audit are longer in practice than short test fixtures) —
    # this is what PromptBudgetExceeded caught on the first real dev-90 run.
    opinions_block = "\n\n".join(
        render_round_two_opinion(o, max_field_chars=_TRUNCATED_FIELD_CHARS) for o in round_two_opinions
    )
    prompt = f"""You are the Director making the final structured verdict on a biomedical \
research question. You receive the verified evidence ledger and the panel's round-2 labeled \
opinions — not a full debate transcript. Base your verdict on the ledger; use the opinions as \
supporting signal, not as a vote to rubber-stamp. Every citation below already resolves to a \
real sentence_id; do not invent new ones.

RESEARCH QUESTION:
{question}

EVIDENCE LEDGER:
{ledger_rendering}

ROUND 2 OPINIONS:
{opinions_block}

Produce a verdict with:
- question_answered_by_endpoint: does the measured endpoint actually answer the research question \
(not a surrogate, not off-topic)?
- direction_determinate: is there a single clear direction, or do findings point different ways?
- findings_statistically_supported: are the findings backed by real statistical support (not just \
an unqualified numeric difference)?
- conclusion_would_be_hedged: would the authors likely have hedged their conclusion (e.g. "may", \
"suggests", "further study needed")?
- direction: positive, negative, or none
- label: your own yes/no/maybe answer
- rationale: why, citing sentence_ids
- citations: the sentence_ids you relied on
"""
    return _with_schema(prompt.strip(), DirectorVerdict, max_tokens=AGGREGATE_MAX_PROMPT_TOKENS)
