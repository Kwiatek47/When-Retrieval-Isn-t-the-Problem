"""Prompt builders for multi-agent clinical debate (inject point for LLM / BioLinkBERT hint)."""

from __future__ import annotations

import json
import random

from app.agents.backends import EvidenceHint
from app.agents.models import AgentRoundOpinion
from app.schemas import ChatMessage

# Both headers introduce the peer-context block. MockInferenceBackend detects
# "has peers" by looking for these literals, so keep the two in sync there.
PEER_CONTEXT_HEADER = "PEER OPINIONS SO FAR"
ANONYMOUS_CONTEXT_HEADER = "ARGUMENTS SUBMITTED FOR REVIEW"

CLINICAL_OPINION_SCHEMA = """
Return ONLY a single JSON object with exactly these fields:
- top_1_diagnosis (string)
- evidence_conclusiveness (string; one of "conclusive", "inconclusive")
- top_3_differential_diagnoses (array of 1-3 strings)
- pros (array of strings)
- cons (array of strings)
- required_further_tests (array of strings)
- confidence_level (number between 0.0 and 1.0)
- sources_used (array of strings)
- red_flags (array of strings)
- missing_information (string)
No markdown fences, no commentary outside JSON.
""".strip()


NEUTRAL_ANALYST_INSTRUCTION = (
    "You are a neutral analyst. You have no assigned role, specialty, seniority or "
    "viewpoint to defend, and neither does anyone else in this panel. Reason from the "
    "evidence in front of you: state what it supports, what it fails to settle, and how "
    "strongly. Weigh every argument you are shown by its evidential content alone - never "
    "by who made it, how confidently it was phrased, or how many others agree. Update your "
    "position when the evidence warrants it and hold it when it does not."
)

PERSONA_INSTRUCTIONS: dict[str, str] = {
    "generalist": (
        "You are a broad clinical generalist. Prioritize the most likely common diagnoses "
        "and a practical initial workup."
    ),
    "evidence_skeptic": (
        "You are an evidence-skeptical clinician. Challenge weak causal leaps, demand "
        "stronger supporting findings, and keep confidence conservative."
    ),
    "differential_expander": (
        "You expand the differential. Explicitly consider less common but plausible "
        "and serious alternatives that others may underweight."
    ),
    "safety_officer": (
        "You focus on patient safety: life-threatening red flags, can't-miss diagnoses, "
        "and urgent tests that rule out catastrophic causes."
    ),
    "neutral_analyst": NEUTRAL_ANALYST_INSTRUCTION,
}

PUBMEDQA_PERSONA_INSTRUCTIONS: dict[str, str] = {
    "generalist": (
        "You answer PubMedQA-style yes/no/maybe questions from abstracts. "
        "Prefer the label best supported by the stated results. Answer 'maybe' "
        "only when the abstract itself does not settle the question."
    ),
    "evidence_skeptic": (
        "You are skeptical of overclaiming. Prefer 'maybe' when evidence is weak, "
        "mixed, underpowered, correlational-only, or only suggestive. A single "
        "significant p-value does not by itself resolve a causal or predictive question."
    ),
    "differential_expander": (
        "You stress alternative readings of the same abstract and whether the "
        "research question is truly resolved by the reported findings. Explicitly "
        "consider whether the finding could support the opposite conclusion or "
        "is confounded, which would make 'maybe' the honest answer."
    ),
    "uncertainty_advocate": (
        "You are the designated uncertainty advocate (a 'catfish' voice against "
        "premature consensus). Your job is NOT to be contrarian for its own sake, "
        "but to make the strongest possible case that the evidence is inconclusive: "
        "hedged author language ('may', 'might', 'could', 'suggests'), small or "
        "single-cohort samples, surrogate endpoints, lack of direct comparison, or "
        "results that only partially answer the specific question asked. If, after "
        "genuinely trying, the evidence is clearly conclusive, say so and concede."
    ),
    "neutral_analyst": (
        f"{NEUTRAL_ANALYST_INSTRUCTION}\n"
        "Here the evidence is a biomedical abstract and the task is to decide whether it "
        "answers the research question yes, no, or maybe. Judge only what the reported "
        "findings establish about that exact question."
    ),
}

PUBMEDQA_LABEL_RULE = """
PubMedQA mode: top_1_diagnosis MUST be exactly one of: "yes", "no", "maybe".

Decide the label from the EVIDENCE, in two explicit steps:
1. First judge evidence_conclusiveness: is the abstract's evidence CONCLUSIVE for the
   exact question asked, or INCONCLUSIVE (mixed, hedged, indirect, underpowered,
   correlational-only, or only partially answering it)?
2. If INCONCLUSIVE, the correct label is "maybe" even if the results lean one way.
   Only choose "yes"/"no" when the evidence directly and clearly settles the question.

Set confidence_level to how strongly the evidence supports your chosen label
(NOT how confident you feel in general). If you answer "maybe", confidence_level
should reflect how sure you are that the evidence is genuinely inconclusive.
Keep the JSON compact: at most 1 short sentence in pros and cons each.
required_further_tests/red_flags may be empty arrays. missing_information may be "".
""".strip()

PUBMEDQA_COMPACT_SCHEMA = """
Return ONLY one compact JSON object (no markdown) with exactly:
{"top_1_diagnosis":"yes|no|maybe","evidence_conclusiveness":"conclusive|inconclusive","top_3_differential_diagnoses":["yes","no","maybe"],"pros":["one short reason"],"cons":["one short caveat"],"required_further_tests":[],"confidence_level":0.0,"sources_used":["abstract"],"red_flags":[],"missing_information":""}
""".strip()


INFORMATION_REQUEST_SCHEMA = """
Add one more field to the JSON object:
- information_requests (array of strings; each a specific, self-contained question
  about evidence outside your segment, e.g. "Was the control group matched for age?").
  Ask only for facts that would change your answer. Use [] if your segment already
  settles the question.
""".strip()

SUPERVISOR_MODERATION_SCHEMA = """
Return ONLY a single JSON object with exactly these fields:
- rigor_checks (object with boolean values for every check id you were given)
- label (string; one of "yes", "no", "maybe")
- confidence (number between 0.0 and 1.0)
- rationale (string; at most 2 sentences, citing the checks that decided it)
No markdown fences, no commentary outside JSON.
""".strip()

SUPERVISOR_MODERATION_SYSTEM = """
You are the supervisor closing a panel deliberation. You are NOT a voter: ignore how
many arguments landed on each side and ignore how confident any of them sounded. A
majority of the panel can be wrong together, and a lone argument can be right.

Work through the rigor checks below in order, answer each one true or false from the
evidence, and let the answers determine the label. Then state the label the checks
imply, even when that contradicts where most of the panel ended up.

RIGOR CHECKS:
- question_addressed: the evidence speaks to the exact question asked, not a related one.
- direction_established: the findings point one way rather than being mixed or null.
- opposite_reading_excluded: the same findings cannot reasonably support the opposite answer.
- hedging_absent: the reported findings are stated as established, not as suggestive.

DECISION RULE:
- If question_addressed is false, the label is "maybe".
- If direction_established is false, the label is "maybe".
- If opposite_reading_excluded is false, the label is "maybe".
- If hedging_absent is false, the label is "maybe".
- Otherwise the label is "yes" or "no", following the direction the evidence establishes.
""".strip()

SUPERVISOR_ROUTER_SYSTEM = """
You route information requests between analysts who each see only one segment of the
evidence. You are given each request and the list of segments eligible to answer it
(already filtered for you). Assign every request to the eligible segments most likely
to actually contain the answer; assign to none if no segment can answer it.

Return ONLY a single JSON object mapping each request id to an array of segment ids:
{"r1": ["s2"], "r2": []}
No markdown fences, no commentary outside JSON.
""".strip()


def build_supervisor_messages(
    *,
    question: str,
    arguments: list[dict],
    evidence_hint: EvidenceHint | None = None,
    repair: bool = False,
) -> list[ChatMessage]:
    """Build the supervisor's final moderation call (rigor checks, not a vote)."""
    system = f"{SUPERVISOR_MODERATION_SYSTEM}\n\n{SUPERVISOR_MODERATION_SCHEMA}"
    if repair:
        system += (
            "\n\nYour previous reply was invalid JSON. "
            "Respond again with ONLY a valid JSON object matching the schema."
        )

    parts = [f"RESEARCH QUESTION:\n{question.strip()}"]
    parts.append(
        "ARGUMENTS SUBMITTED FOR REVIEW (unattributed; judge each on its evidential "
        f"content alone):\n{json.dumps(arguments, ensure_ascii=False)}"
    )
    if evidence_hint is not None:
        # One more piece of evidence to weigh, explicitly NOT a veto - this is the
        # difference from the baseline's bert_gate, where a confident classifier
        # simply overrode the panel.
        parts.append(
            "ADDITIONAL SIGNAL - an evidence classifier independently predicted:\n"
            f"- label: {evidence_hint.label}\n"
            f"- confidence: {evidence_hint.confidence:.3f}\n"
            "Treat this as one more piece of evidence to weigh against the rigor "
            "checks. It does NOT override them."
        )
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content="\n\n".join(parts)),
    ]


def build_router_messages(
    *,
    question: str,
    requests: list[dict],
    segments: list[dict],
) -> list[ChatMessage]:
    """Build the supervisor's InfoNav routing call (LLM layer of the two-layer router)."""
    user = "\n\n".join(
        [
            f"RESEARCH QUESTION:\n{question.strip()}",
            f"SEGMENTS:\n{json.dumps(segments, ensure_ascii=False)}",
            f"REQUESTS:\n{json.dumps(requests, ensure_ascii=False)}",
        ]
    )
    return [
        ChatMessage(role="system", content=SUPERVISOR_ROUTER_SYSTEM),
        ChatMessage(role="user", content=user),
    ]


def build_messages(
    *,
    agent_id: str,
    persona: str,
    patient_case: str,
    context: list[AgentRoundOpinion] | None = None,
    evidence_hint: EvidenceHint | None = None,
    repair: bool = False,
    task_mode: str = "clinical",
    compact: bool = False,
    anonymize: bool = False,
    info_requests: list[str] | None = None,
    partial_evidence: bool = False,
    shuffle_seed: int | None = None,
) -> list[ChatMessage]:
    """Build chat messages for independent (round 1) or critique (round 2+) opinion generation.

    `context` is the round-robin discussion so far: the previous round's final
    opinions plus any peers who have already spoken in the current round, in
    speaking order. Each entry keeps its agent_id/persona/round so the model
    sees an actual discussion transcript rather than an anonymous opinion dump.

    With `anonymize=True` that identifying metadata is stripped and the entries
    are reordered, so an argument can only be judged on its content. `shuffle_seed`
    keeps the reordering reproducible.

    `partial_evidence` tells the agent it is looking at one slice of the case and
    should ask for what it cannot see; `info_requests` carries peers' questions
    that the supervisor routed to this agent.
    """
    mode = (task_mode or "clinical").strip().lower()
    if mode == "pubmedqa":
        persona_text = PUBMEDQA_PERSONA_INSTRUCTIONS.get(
            persona, PUBMEDQA_PERSONA_INSTRUCTIONS["generalist"]
        )
        schema_block = (
            f"{PUBMEDQA_COMPACT_SCHEMA}\n\n{PUBMEDQA_LABEL_RULE}"
            if compact
            else f"{CLINICAL_OPINION_SCHEMA}\n\n{PUBMEDQA_LABEL_RULE}"
        )
    else:
        persona_text = PERSONA_INSTRUCTIONS.get(persona, PERSONA_INSTRUCTIONS["generalist"])
        schema_block = CLINICAL_OPINION_SCHEMA

    # Only advertise the extra field where it is actually used, so the prompts of
    # the shared-context architectures stay byte-identical to the baseline's and
    # remain a fair comparison.
    if partial_evidence:
        schema_block += f"\n\n{INFORMATION_REQUEST_SCHEMA}"

    # Neutral analysts are meant to be interchangeable, so they get no persona
    # label in their own header either - naming a persona is the very anchor the
    # neutral panel exists to remove. The `agent_id=`/`task_mode=` marker lines
    # are machine-readable state parsed by MockInferenceBackend and must stay.
    if persona == "neutral_analyst":
        header = f"You are analysis unit `{agent_id}` in a panel of identical units."
    else:
        header = f"You are clinical debate agent `{agent_id}` with persona `{persona}`."
    system = (
        f"{header}\n"
        f"agent_id={agent_id}\n"
        f"task_mode={mode}\n"
        f"{persona_text}\n\n"
        f"{schema_block}"
    )
    if repair:
        system += (
            "\n\nYour previous reply was invalid JSON. "
            "Respond again with ONLY a valid JSON object matching the schema."
        )

    parts = [f"PATIENT CASE:\n{patient_case.strip()}"]

    # Injection point for BioLinkBERT / classifier signal (not a substitute for LLM reasoning).
    if evidence_hint is not None:
        parts.append(
            "EVIDENCE CLASSIFIER HINT (BioLinkBERT-style yes/no/maybe signal; use critically):\n"
            f"- label: {evidence_hint.label}\n"
            f"- confidence: {evidence_hint.confidence:.3f}\n"
            f"- model: {evidence_hint.model_path or 'unspecified'}"
        )

    if partial_evidence:
        parts.append(
            "SCOPE: the evidence above is only ONE SEGMENT of the case; peers hold the "
            "other segments and you cannot see theirs. Report what your segment does and "
            "does not establish, and list in `information_requests` the specific facts "
            "you would need from the rest of the case to settle the question. Do not "
            "guess at content you were not shown, and do not treat your segment as the "
            "whole case."
        )

    if info_requests:
        parts.append(
            "QUESTIONS ROUTED TO YOU (peers who cannot see your segment asked these; "
            "answer each from your segment in `pros`/`cons`, or say your segment does "
            f"not cover it):\n{json.dumps(list(info_requests), ensure_ascii=False)}"
        )

    if context:
        current_round = max(entry.round for entry in context)
        entries = list(context)
        if anonymize:
            # Reorder so position cannot become the new identity cue: with a fixed
            # order the "first argument" is always the same agent.
            random.Random(shuffle_seed).shuffle(entries)
        serialized = [
            _serialize_context_entry(
                entry,
                current_round=current_round,
                compact=compact,
                mode=mode,
                anonymize=anonymize,
                argument_id=f"A{position + 1}",
            )
            for position, entry in enumerate(entries)
        ]
        if anonymize:
            header = (
                f"{ANONYMOUS_CONTEXT_HEADER} (unattributed and unordered; you cannot tell "
                "who made which argument, including which one is yours - weigh each purely "
                "on its evidential content):\n"
            )
        else:
            header = (
                f"{PEER_CONTEXT_HEADER} (previous round's final opinions, plus anyone who "
                "has already spoken this round, in speaking order; critique weak "
                "arguments, update hypotheses, and resolve contradictions where "
                "possible):\n"
            )
        parts.append(header + json.dumps(serialized, ensure_ascii=False))
        parts.append(
            "Produce an UPDATED ClinicalOpinion that reflects what you accept, "
            "reject, or still find uncertain after reviewing peers."
        )
    elif not partial_evidence:
        parts.append(
            "This is an independent first-round opinion. Do not assume peer input. "
            "Reason only from the patient case (and classifier hint if present)."
        )

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content="\n\n".join(parts)),
    ]


def _serialize_context_entry(
    entry: AgentRoundOpinion,
    *,
    current_round: int,
    compact: bool,
    mode: str,
    anonymize: bool = False,
    argument_id: str = "",
) -> dict:
    """Render one peer turn as a discussion entry, identified or anonymous.

    When `anonymize` is set, every field that could anchor the reader to an
    identity is withheld - agent id, persona, round number, and whether the turn
    is recent. Only the argument itself survives, under an opaque label.
    """
    opinion = entry.opinion
    rendered: dict
    if anonymize:
        rendered = {"argument_id": argument_id}
    else:
        rendered = {
            "agent_id": entry.agent_id,
            "persona": entry.persona,
            "round": entry.round,
            "already_spoken_this_round": entry.round == current_round,
        }
    if compact or mode == "pubmedqa":
        rendered.update(
            {
                "label": opinion.top_1_diagnosis,
                "evidence_conclusiveness": opinion.evidence_conclusiveness,
                "confidence": opinion.confidence_level,
                "pros": opinion.pros[:1],
                "cons": opinion.cons[:1],
            }
        )
    else:
        rendered["opinion"] = json.loads(opinion.model_dump_json())
    return rendered
