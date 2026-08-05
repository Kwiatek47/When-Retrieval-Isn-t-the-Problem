"""Prompt builders for multi-agent clinical debate (inject point for LLM / BioLinkBERT hint)."""

from __future__ import annotations

import json

from app.agents.backends import EvidenceHint
from app.agents.models import AgentRoundOpinion
from app.schemas import ChatMessage

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


SAFETY_OPINION_SCHEMA = """
Return ONLY a single JSON object with exactly these fields:
- safety_passed (boolean)
- red_flags_detected (array of strings)
- immediate_intervention_required (boolean)
- reasoning (string)

No markdown fences, no commentary outside JSON.
""".strip()


SUPERVISOR_MODERATOR_PROMPT = """
You are a Clinical Supervisor moderating a 4-agent PubMedQA debate and maintaining a shared reasoning report.

Inputs:
- patient_case:
{patient_case}
- previous_round_opinions:
{previous_round_opinions}

Task:
1. From the ABSTRACT (not just agent chatter), fill the shared report:
   - primary_endpoint_result: what the primary endpoint/result actually showed (1-2 sentences)
   - author_conclusion: "yes" | "no" | "maybe" | "unclear" for the research question
   - residual_uncertainty: concrete unresolved issues (empty if none)
2. Analyze previous_round_opinions for evidence-grounded agreements and contradictions.
3. Create round_instructions: concrete requirements for the next round.
4. Do NOT invent claims unsupported by the abstract or agent evidence signals.

Output ONLY JSON:
{{
  "primary_endpoint_result": "...",
  "author_conclusion": "yes" | "no" | "maybe" | "unclear",
  "residual_uncertainty": ["..."],
  "agreements": ["..."],
  "contradictions": ["..."],
  "round_instructions": ["..."]
}}
""".strip()

SUPERVISOR_DIRECTOR_PROMPT = """
You are a Clinical Director synthesizing a multi-agent debate to answer a PubMedQA research question.

Inputs:
- patient_case (original abstract and question — YOUR GROUND TRUTH):
{patient_case}
- shared_report (MedAgents-style summary from the moderator; prefer this over raw chatter):
{shared_report}
- debate_brief (compact final-round agent votes with confidence; discount fallbacks):
{debate_brief}
- biolinkbert_hint (classifier signal; use critically, do not rubber-stamp):
{biolinkbert_hint}

Task:
Decide whether the ABSTRACT fully settles the research question as written.
Do not rubber-stamp BioLinkBERT. Do not grade study quality for its own sake.

PubMedQA label semantics (critical):
- "yes"/"no": the abstract provides a decisive answer to the posed question (authors' primary conclusion clearly affirms or rejects it).
- "maybe": the abstract does NOT fully settle the question — even if authors report some directional finding.
  Typical maybe patterns:
  * mixed/partial results across endpoints or subgroups that the question asks about globally
  * study answers only a narrow surrogate / related outcome, not the full clinical question
  * findings conflict or are statistically null for the asked comparison
  * authors leave the practical question open
  Do NOT use maybe only for boilerplate "further research" / small-n when the primary answer is clear.

Fill question_coverage:
- "full": primary evidence fully answers the research question as written
- "partial": evidence speaks to part of the question / subgroup / surrogate only
- "none": evidence does not answer the question

MAYBE-AWARE GATE:
1. primary_endpoint_answers_question
2. findings_decisive_for_question
3. authors_state_uncertainty (explicit open/mixed language — not routine limitations)

Rules:
- If question_coverage is "partial" or "none", prefer final_label="maybe" unless the authors state an unambiguous global yes/no to the exact question.
- If consensus among agents is split or consensus_type would be differential, do not force a binary label.
- Discount is_fallback=true opinions.
- Dual-read: reconcile author_conclusion_reader vs uncertainty_auditor; auditor wins only with abstract support for unresolved coverage.
- BioLinkBERT is a hint, not a veto.

Output ONLY valid JSON:
{{
  "final_label": "yes" | "no" | "maybe",
  "consensus_type": "consensus" | "differential" | "escalation",
  "rationale": "Brief abstract-grounded rationale.",
  "primary_endpoint_answers_question": true | false,
  "findings_decisive_for_question": true | false,
  "authors_state_uncertainty": true | false,
  "question_coverage": "full" | "partial" | "none"
}}
""".strip()

EVIDENCE_SKEPTIC_PROMPT = """You are the Evidence Skeptic on a multi-agent clinical debate panel.
Your primary objective is to critically evaluate the methodology, identifying potential biases, confounding variables, and weak study designs in the provided medical abstract.

CRITICAL CONSTRAINTS FOR YOUR DIAGNOSIS:
1. RESPECT STATISTICAL SIGNIFICANCE: Distinguish standard academic limitations (small sample, retrospective design, limited follow-up) from fatal methodological flaws.
2. DO NOT DEFAULT TO 'maybe': If authors report statistically significant primary findings, top_1_diagnosis MUST be 'yes' or 'no' matching their conclusion; put caveats in cons/red_flags.
3. Only choose 'maybe' when methodology so thoroughly invalidates results that no direction remains.

Respond with ClinicalOpinion JSON only. top_1_diagnosis must be exactly 'yes', 'no', or 'maybe'.
""".strip()

AUTHOR_CONCLUSION_READER_PROMPT = """You are the AuthorConclusionReader (generalist dual-read role).
Your ONLY job is to read what the AUTHORS conclude about the research question from the abstract.

- Choose "yes" if authors report a positive/affirmative primary conclusion.
- Choose "no" if authors report a null/negative primary conclusion.
- Choose "maybe" ONLY if authors themselves leave the question unresolved or results are mixed for the asked question.
Ignore peer pressure to invent uncertainty that authors did not express.

Respond with ClinicalOpinion JSON only. top_1_diagnosis must be exactly 'yes', 'no', or 'maybe'.
""".strip()

UNCERTAINTY_ADVOCATE_PROMPT = """You are the UncertaintyAuditor (uncertainty_advocate dual-read role).
Audit whether the abstract FULLY settles the research question as written.

Choose 'maybe' when:
1. Primary results are insignificant or mixed across key endpoints the question asks about.
2. The study answers only a subgroup / surrogate / related outcome (partial coverage).
3. Authors leave the practical question open, or findings conflict.
4. A directional finding exists but does not fully answer the posed question.

Do NOT choose 'maybe' solely for boilerplate limitations if the abstract clearly and fully answers the question with robust primary findings.
If coverage is partial or mixed, advocate for 'maybe' with confidence reflecting that audit.

Respond with ClinicalOpinion JSON only. top_1_diagnosis must be exactly 'yes', 'no', or 'maybe'.
""".strip()

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
        "You are the safety_officer. Your ONLY job is a safety audit, not diagnosis generation.\n"
        "- Search the debate context for red flags, contraindications, missed critical symptoms,\n"
        "  and any safety-critical claims that could lead to harm.\n"
        "- Focus specifically on what generalist and differential_expander said: which symptoms/signs\n"
        "  they might have ignored or underweighted, and whether that creates a safety risk.\n"
        "- Decide safety_passed: true only if there are no unresolved critical red flags.\n"
        "- Decide immediate_intervention_required: true if urgent can't-miss safety action is needed now.\n"
    ),
}

PUBMEDQA_PERSONA_INSTRUCTIONS: dict[str, str] = {
    "generalist": AUTHOR_CONCLUSION_READER_PROMPT,
    "evidence_skeptic": EVIDENCE_SKEPTIC_PROMPT,
    "differential_expander": (
        "You stress alternative readings. Could the data actually imply the opposite conclusion? "
        "Argue for the counter-hypothesis (if generalist says 'yes', you argue for 'no')."
    ),
    "uncertainty_advocate": UNCERTAINTY_ADVOCATE_PROMPT,
}

PUBMEDQA_LABEL_RULE = """
PubMedQA mode: top_1_diagnosis MUST be exactly one of: "yes", "no", "maybe".

Decide the label based on the CORE DIRECTION of the findings:
- Choose "yes" if the findings support the hypothesis or show an effect.
- Choose "no" if the findings reject the hypothesis or show no significant effect.
- Choose "maybe" ONLY if the findings are completely mixed, inherently contradictory, or fail to lean in any direction.

WARNING: Do not choose "maybe" just because the authors use cautious words (e.g., "suggests", "potential", "might"). In science, these words accompany solid "yes" or "no" findings. Look at the actual results, not just the cautious tone.

Set confidence_level to how strongly the text supports your chosen label.
""".strip()

# Advocate-specific rule: no WARNING that suppresses genuine uncertainty / maybe.
PUBMEDQA_LABEL_RULE_ADVOCATE = """
PubMedQA mode: top_1_diagnosis MUST be exactly one of: "yes", "no", "maybe".

Decide the label based on the CORE DIRECTION of the findings:
- Choose "yes" if the findings support the hypothesis or show an effect.
- Choose "no" if the findings reject the hypothesis or show no significant effect.
- Choose "maybe" if the findings are mixed, contradictory, or express genuine uncertainty.

Set confidence_level to how strongly the text supports your chosen label.
""".strip()

PUBMEDQA_COMPACT_SCHEMA = """
Return ONLY one compact JSON object (no markdown) with exactly:
{"top_1_diagnosis":"yes|no|maybe","evidence_conclusiveness":"conclusive|inconclusive","top_3_differential_diagnoses":["yes","no","maybe"],"pros":["one short reason"],"cons":["one short caveat"],"required_further_tests":[],"confidence_level":0.0,"sources_used":["abstract"],"red_flags":[],"missing_information":""}
""".strip()


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
) -> list[ChatMessage]:
    """Build chat messages for independent (round 1) or critique (round 2+) opinion generation.

    `context` is the round-robin discussion so far: the previous round's final
    opinions plus any peers who have already spoken in the current round, in
    speaking order. Each entry keeps its agent_id/persona/round so the model
    sees an actual discussion transcript rather than an anonymous opinion dump.
    """
    mode = (task_mode or "clinical").strip().lower()
    is_safety_officer = persona.strip().lower() == "safety_officer"
    stance_directive = (
        "Base your decision strictly on the provided evidence. "
        "If the evidence strongly supports a conclusion, choose 'yes' or 'no'. "
        "If the evidence is genuinely conflicting or insufficient to answer the question, you MUST choose 'maybe'."
    )
    if mode == "pubmedqa":
        if is_safety_officer:
            persona_text = PERSONA_INSTRUCTIONS["safety_officer"]
            schema_block = SAFETY_OPINION_SCHEMA
        else:
            persona_text = PUBMEDQA_PERSONA_INSTRUCTIONS.get(
                persona, PUBMEDQA_PERSONA_INSTRUCTIONS["generalist"]
            )
            persona_text = f"{persona_text}\n{stance_directive}"
            active_label_rule = (
                PUBMEDQA_LABEL_RULE_ADVOCATE
                if persona == "uncertainty_advocate"
                else PUBMEDQA_LABEL_RULE
            )
            schema_block = (
                f"{PUBMEDQA_COMPACT_SCHEMA}\n\n{active_label_rule}"
                if compact
                else f"{CLINICAL_OPINION_SCHEMA}\n\n{active_label_rule}"
            )
    else:
        if is_safety_officer:
            persona_text = PERSONA_INSTRUCTIONS["safety_officer"]
            schema_block = SAFETY_OPINION_SCHEMA
        else:
            persona_text = PERSONA_INSTRUCTIONS.get(persona, PERSONA_INSTRUCTIONS["generalist"])
            persona_text = f"{persona_text}\n{stance_directive}"
            schema_block = CLINICAL_OPINION_SCHEMA

    system = (
        f"You are clinical debate agent `{agent_id}` with persona `{persona}`.\n"
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

    if context:
        current_round = max(entry.round for entry in context)
        serialized = [
            _serialize_context_entry(entry, current_round=current_round, compact=compact, mode=mode)
            for entry in context
        ]
        parts.append(
            "PEER OPINIONS SO FAR (previous round's final opinions, plus anyone who "
            "has already spoken this round, in speaking order; critique weak "
            "arguments, update hypotheses, and resolve contradictions where "
            "possible):\n"
            f"{json.dumps(serialized, ensure_ascii=False)}"
        )
        parts.append(
            "Produce an UPDATED ClinicalOpinion that reflects what you accept, "
            "reject, or still find uncertain after reviewing peers."
        )
    else:
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
) -> dict:
    """Render one peer turn as an identified, ordered discussion entry."""
    opinion = entry.opinion
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
