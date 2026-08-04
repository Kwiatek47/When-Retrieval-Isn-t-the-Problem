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
You are a Clinical Supervisor moderating a 4-agent debate.

Inputs:
- patient_case:
{patient_case}
- previous_round_opinions:
{previous_round_opinions}

Task:
1. Analyze the opinions from previous_round_opinions and identify:
   - agreements: shared points across agents (grounded in sources_used or explicit evidence references)
   - contradictions: disagreements that remain unresolved (again grounded)
2. Create round_instructions: concrete requirements for agents to follow in the next round.
3. Enforce grounding: if an agreement/contradiction is not clearly supported by the provided evidence signals
   (e.g., sources_used and evidence references inside the opinions), do NOT include it.

Output:
Return ONLY a JSON object matching SupervisorModerationOutput:
{{
  "agreements": ["..."],
  "contradictions": ["..."],
  "round_instructions": ["..."]
}}

No markdown fences, no commentary outside JSON.
""".strip()

SUPERVISOR_DIRECTOR_PROMPT = """
You are a Clinical Director synthesizing a multi-agent debate to answer a PubMedQA research question.

Inputs:
- patient_case (The original abstract and question - YOUR GROUND TRUTH):
{patient_case}
- full_debate_transcript (The debate between your agents):
{full_debate_transcript}
- biolinkbert_hint:
{biolinkbert_hint}

Task:
Your goal is to determine the ACTUAL conclusion made by the authors of the abstract.
Do not blindly count agent votes. Verify their claims against the `patient_case`.

Definitions for final_label:
- "yes": The authors explicitly conclude with a positive finding or correlation.
- "no": The authors explicitly conclude with a negative finding or lack of correlation.
- "maybe": The authors explicitly state that their findings are inconclusive, contradictory, or clearly state that the answer cannot be determined.

Definitions for consensus_type:
- "consensus": The abstract supports a clear yes/no. General methodological critiques by agents (e.g., small sample size, retrospective design) DO NOT change the authors' actual conclusion.
- "differential": The abstract is genuinely inconclusive, or agents correctly identified explicitly conflicting information in the text.

WARNING: DO NOT hallucinate quotes. Only classify as "maybe" if the original text truly is inconclusive. Do not invent phrases like "further studies are needed" if they do not appear in the text.

Output MUST be a valid JSON object matching this schema, with no other text:
{{
  "final_label": "yes" | "no" | "maybe",
  "consensus_type": "consensus" | "differential",
  "rationale": "Briefly state the authors' actual conclusion based on the abstract text."
}}
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
    "generalist": (
        "You answer PubMedQA-style yes/no/maybe questions from abstracts. "
        "Choose 'yes' or 'no' based on the primary conclusion of the abstract."
    ),
    "evidence_skeptic": (
        "You are skeptical of overclaiming. Focus on finding flaws in the abstract's methodology. "
        "If the findings are flawed, argue fiercely for the OPPOSITE label ('no' instead of 'yes', or vice versa) "
        "rather than settling for 'maybe'."
    ),
    "differential_expander": (
        "You stress alternative readings. Could the data actually imply the opposite conclusion? "
        "Argue for the counter-hypothesis (if generalist says 'yes', you argue for 'no')."
    ),
    # Jeśli nadal używasz uncertainty_advocate, zrób z niego jedynego, który ma prawo wnieść 'maybe':
    "uncertainty_advocate": (
        "You are the designated uncertainty advocate. You are the ONLY agent who should actively seek 'maybe'."
    ),
}

PUBMEDQA_LABEL_RULE = """
PubMedQA mode: top_1_diagnosis MUST be exactly one of: "yes", "no", "maybe".

Decide the label from the EVIDENCE:
- If the abstract leans towards a positive or negative conclusion, choose "yes" or "no" accordingly, even if the evidence is weak, indirect, or based on a small sample.
- Reserve "maybe" STRICTLY for cases where the abstract explicitly states that findings are entirely contradictory, or explicitly concludes that further research is strictly required to answer the question at all. Do NOT use "maybe" just because the results are not 100% perfect.

Set confidence_level to how strongly the evidence supports your chosen label...
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
        "Avoid choosing 'maybe' unless the provided context is completely insufficient. "
        "Force yourself to take a stance ('yes' or 'no') based on the balance of probabilities in the abstracts."
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
            schema_block = (
                f"{PUBMEDQA_COMPACT_SCHEMA}\n\n{PUBMEDQA_LABEL_RULE}"
                if compact
                else f"{CLINICAL_OPINION_SCHEMA}\n\n{PUBMEDQA_LABEL_RULE}"
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
