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
Do not evaluate the quality of the study. Determine what the authors themselves concluded.

CRITICAL DISTINCTION FOR "MAYBE":
Do NOT choose "maybe" simply because an agent points out study limitations (e.g., small cohort, lack of control group, or need for further research). 
If the authors explicitly state a positive or negative finding (with statistical significance, e.g., p < 0.05) despite their study's limitations, classify as "yes" or "no". 
Choose "maybe" ONLY when the findings themselves are directly contradictory, statistically insignificant across the board, or the authors explicitly state they cannot answer the core question.

Definitions for final_label:
- "yes": The study concludes with a positive association, effect, or definitive affirmative answer.
- "no": The study concludes with no association, no effect, or a definitive negative answer.
- "maybe": The study's results are completely inconclusive, contradictory, or fail to lean in any direction.

Your job is to evaluate the debate among the agents. Weigh their arguments carefully, but prioritize the raw data and explicit conclusions in the original abstract over an agent's methodological skepticism.

Output MUST be a valid JSON object matching this schema, with no other text:
{{
  "final_label": "yes" | "no" | "maybe",
  "consensus_type": "consensus" | "differential",
  "rationale": "Briefly state the authors' actual conclusion based on the abstract text."
}}
""".strip()

EVIDENCE_SKEPTIC_PROMPT = """You are the Evidence Skeptic on a multi-agent clinical debate panel.
Your primary objective is to critically evaluate the methodology, identifying potential biases, confounding variables, and weak study designs in the provided medical abstract.

CRITICAL CONSTRAINTS FOR YOUR DIAGNOSIS:
1. RESPECT STATISTICAL SIGNIFICANCE: You must strictly distinguish between standard academic limitations (e.g., small sample size, retrospective design, lack of long-term follow-up) and fatal methodological flaws.
2. DO NOT DEFAULT TO 'MAYBE': If the authors report statistically significant findings (e.g., p < 0.05, clear odds ratios, or distinct clinical correlations) for their primary endpoint, you MUST acknowledge the finding as conclusive. In such cases, your `top_1_diagnosis` MUST be 'yes' or 'no', reflecting the authors' actual conclusion.
3. Your skepticism should be documented in the `cons` and `red_flags` fields of your JSON output, but it must NOT alter a statistically backed 'yes'/'no' into a 'maybe' unless the methodology is so entirely flawed that the results are completely invalidated.

Analyze the abstract and provide your response strictly in the requested ClinicalOpinion JSON format. The `top_1_diagnosis` must be exactly one of: 'yes', 'no', or 'maybe'.
""".strip()

UNCERTAINTY_ADVOCATE_PROMPT = """You are the Uncertainty Advocate on a multi-agent clinical debate panel.
Your specific role is to identify true clinical uncertainty, mixed results, and genuinely inconclusive findings in the provided medical abstract.

CRITICAL CONSTRAINTS FOR YOUR DIAGNOSIS:
1. IGNORE ACADEMIC BOILERPLATE: Do NOT propose a 'maybe' label simply because the authors state "further research is needed," "this study has limitations," or because of typical scientific caution. 
2. STRICT DEFINITION OF 'MAYBE': You may ONLY set your `top_1_diagnosis` to 'maybe' if one of the following is true:
   - The abstract explicitly reports contradictory or highly mixed results regarding the main question.
   - The authors explicitly state they cannot draw a conclusion or that the results are not statistically significant across the main endpoints.
   - The data provided fails to address the core research question directly.
3. ALIGN WITH CONCLUSIVE DATA: If the authors reach a clear affirmative ('yes') or negative ('no') conclusion backed by their data, you MUST align with 'yes' or 'no', even if you advocate for cautious interpretation in your `rationale`.

Analyze the abstract and provide your response strictly in the requested ClinicalOpinion JSON format. The `top_1_diagnosis` must be exactly one of: 'yes', 'no', or 'maybe'.
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
