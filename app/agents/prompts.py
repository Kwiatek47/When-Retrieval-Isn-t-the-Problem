"""Prompt builders for multi-agent clinical debate (inject point for LLM / BioLinkBERT hint)."""

from __future__ import annotations

import json
from typing import Any, Literal

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
You are a Clinical Supervisor moderating a multi-agent debate.

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
4. Record the primary endpoint/result, the authors' current yes/no/maybe conclusion,
   and any residual uncertainty that the next round must resolve.

Output:
Return ONLY a JSON object matching SupervisorModerationOutput:
{{
  "agreements": ["..."],
  "contradictions": ["..."],
  "round_instructions": ["..."],
  "primary_endpoint_result": "...",
  "author_conclusion": "yes" | "no" | "maybe" | "unclear",
  "residual_uncertainty": ["..."]
}}

No markdown fences, no commentary outside JSON.
""".strip()

SUPERVISOR_DIRECTOR_PROMPT = """
You are a Clinical Director synthesizing a multi-agent debate to answer a PubMedQA research question.

Inputs:
- patient_case (original abstract and question — YOUR GROUND TRUTH):
{patient_case}
- full_debate_transcript (agent opinions across rounds):
{full_debate_transcript}

Task:
Determine the ACTUAL conclusion made by the authors of the abstract.
Do not grade study quality. Ask: what did the authors conclude about the research question?

CRITICAL RULES FOR CHOOSING THE LABEL:
1. DISTINGUISHING "yes" AND "no":
   - Choose "yes" if the authors conclude a positive association, effect, or affirmative answer.
   - Choose "no" if the authors conclude NO association, NO effect, or a definitive negative answer. A definitive finding that something DOES NOT work is a "no", not a "yes".
2. THE "BOILERPLATE" BAN:
   - Do NOT choose "maybe" only because an agent cites boilerplate limitations (small sample, retrospective design). If the authors report a clear primary finding, prioritize the authors' explicit conclusion.
3. TRUE UNCERTAINTY ("maybe") & COVERAGE:
   - You MUST set `question_coverage` to "partial" and `final_label` to "maybe" if the primary findings are genuinely mixed/contradictory.
   - SPECULATIVE UTILITY: You MUST choose "maybe" if the question asks about a clinical/diagnostic role, and the authors only prove a correlation, concluding that the intervention "may", "could", or "has potential to" have a role in the future. Suggesting a hypothesis is not a definitive "yes".

Discount opinions whose sources_used include "fallback". Weigh agent arguments carefully, but prioritize the abstract text. 

Output ONLY a valid JSON object (no markdown, no commentary):
{{
  "question_coverage": "full" | "partial" | "none",
  "findings_decisive_for_question": true | false,
  "final_label": "yes" | "no" | "maybe",
  "consensus_type": "consensus" | "differential" | "escalation",
  "rationale": "Briefly state the authors' conclusion grounded in the abstract."
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


# Legacy single-role prompt (kept for backward-compatible tests / references).
UNCERTAINTY_ADVOCATE_PROMPT = """You are the Uncertainty Advocate on a PubMedQA debate panel.
Your ONLY job is to identify fundamental gaps that prevent a definitive 'yes' or 'no' conclusion.

You MUST champion the 'maybe' label if you detect:
1. PARTIAL COVERAGE: The study investigates a related metric but doesn't fully answer the core question (e.g., using a surrogate endpoint).
2. INTERNAL CONTRADICTIONS: Primary and secondary endpoints point in opposite directions.
3. SPECULATIVE CLINICAL UTILITY: If the question asks whether X has a diagnostic/therapeutic role, and the authors only prove that X *correlates* with a disease, concluding that it "may/might" have a role. Proposing a future clinical application based on a correlation is a 'maybe', NOT a 'yes'.
4. INSIGNIFICANT DATA: The main claim relies on statistically insignificant results (p > 0.05).

CRITICAL CONFIDENCE RULE: When you choose 'maybe', you must set your `confidence_level` HIGH (e.g., 0.90 - 1.0). Do not use a low confidence score to reflect the paper's uncertainty; you must be highly confident IN your detection of that uncertainty.

Respond with ClinicalOpinion JSON only. top_1_diagnosis must be exactly 'yes', 'no', or 'maybe'.
""".strip()

PUBMEDQA_UNCERTAINTY_PERSONAS = frozenset(
    {"uncertainty_advocate"}
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
    "uncertainty_advocate": (
        UNCERTAINTY_ADVOCATE_PROMPT +
        "\n\nCRITICAL INSTRUCTION FOR DEBATE ROUNDS: You are the sole auditor of uncertainty. "
        "Do NOT easily yield to the Generalist. "
        "HOWEVER, if peer arguments logically resolve the apparent contradictions or prove the abstract fully addresses the question (e.g., via a valid surrogate endpoint), "
        "you MUST update your diagnosis to 'yes' or 'no'. Only maintain 'maybe' if the gaps are genuine and unresolved."
    ),
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
- Choose "maybe" if the findings are mixed, contradictory, express genuine uncertainty, or only partially answer the posed question (surrogate/subgroup).

Set confidence_level to how strongly the text supports your chosen label.
""".strip()


PUBMEDQA_COMPACT_SCHEMA = """
Return ONLY one compact JSON object (no markdown) with exactly:
{"top_1_diagnosis":"yes|no|maybe","evidence_conclusiveness":"conclusive|inconclusive","top_3_differential_diagnoses":["yes","no","maybe"],"pros":["one short reason"],"cons":["one short caveat"],"required_further_tests":[],"confidence_level":0.0,"sources_used":["abstract"],"red_flags":[],"missing_information":""}
""".strip()

PeerContextMode = Literal["nl", "compact-json", "full-json"]


def _clip_text(value: Any, *, max_chars: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def format_opinion_nl(
    entry: AgentRoundOpinion,
    *,
    current_round: int | None = None,
) -> str:
    """Render one peer opinion as a short natural-language line block."""
    opinion = entry.opinion
    spoken_mark = ""
    if current_round is not None and entry.round == current_round:
        spoken_mark = "*"
    conf = float(opinion.confidence_level)
    conclusiveness = (opinion.evidence_conclusiveness or "").strip() or "unspecified"
    header = (
        f"[R{entry.round}{spoken_mark}] {entry.persona} "
        f"(conf={conf:.2f}, {conclusiveness}): {opinion.top_1_diagnosis}"
    )
    lines = [header]
    pros = [p for p in (opinion.pros or []) if str(p).strip()]
    cons = [c for c in (opinion.cons or []) if str(c).strip()]
    if pros:
        lines.append(f"  Pro: {_clip_text(pros[0])}")
    if cons:
        lines.append(f"  Con: {_clip_text(cons[0])}")
    safety = getattr(opinion, "safety_opinion", None)
    if safety is not None:
        lines.append(
            "  Safety: "
            f"passed={bool(safety.safety_passed)}, "
            f"immediate={bool(safety.immediate_intervention_required)}, "
            f"flags={_clip_text(', '.join(safety.red_flags_detected or []) or 'none', max_chars=120)}"
        )
    return "\n".join(lines)


def format_peer_context(
    entries: list[AgentRoundOpinion],
    *,
    peer_context: PeerContextMode = "nl",
    compact: bool = False,
    mode: str = "clinical",
) -> str:
    """Serialize peer opinions for prompt injection (NL or JSON)."""
    if not entries:
        return ""
    style = (peer_context or "nl").strip().lower()
    current_round = max(entry.round for entry in entries)

    if style == "nl":
        blocks = [
            format_opinion_nl(entry, current_round=current_round) for entry in entries
        ]
        return "\n".join(blocks)

    use_compact = style == "compact-json" or compact or (mode or "").strip().lower() == "pubmedqa"
    if style == "full-json":
        use_compact = False
    serialized = [
        _serialize_context_entry(
            entry,
            current_round=current_round,
            compact=use_compact,
            mode=mode,
        )
        for entry in entries
    ]
    return json.dumps(serialized, ensure_ascii=False)


def format_moderation_nl(moderation_output: Any) -> str:
    """Render supervisor moderation as concise natural language."""
    if hasattr(moderation_output, "model_dump"):
        data = moderation_output.model_dump()
    elif isinstance(moderation_output, dict):
        data = moderation_output
    else:
        data = {"raw": str(moderation_output)}

    lines = [
        "Supervisor moderation from the previous round:",
        f"- author_conclusion: {data.get('author_conclusion', 'unclear')}",
        f"- primary_endpoint_result: {_clip_text(data.get('primary_endpoint_result', ''), max_chars=220)}",
    ]
    for key, label in (
        ("agreements", "Agreements"),
        ("contradictions", "Contradictions"),
        ("residual_uncertainty", "Residual uncertainty"),
        ("round_instructions", "Round instructions"),
    ):
        items = [str(item).strip() for item in (data.get(key) or []) if str(item).strip()]
        if not items:
            lines.append(f"- {label}: (none)")
            continue
        lines.append(f"- {label}:")
        for item in items[:6]:
            lines.append(f"  • {_clip_text(item, max_chars=200)}")
    return "\n".join(lines)


def format_opinions_for_supervisor(
    opinions: dict[str, Any] | list[AgentRoundOpinion],
    *,
    peer_context: PeerContextMode = "nl",
) -> str:
    """Format previous-round opinions for the supervisor moderator prompt."""
    style = (peer_context or "nl").strip().lower()
    if isinstance(opinions, list):
        entries = opinions
        if style == "nl":
            return format_peer_context(entries, peer_context="nl")
        if style == "full-json":
            payload = {
                entry.agent_id: json.loads(entry.opinion.model_dump_json())
                for entry in entries
            }
            return json.dumps(payload, ensure_ascii=False)
        # compact-json
        payload = {
            entry.agent_id: {
                "label": entry.opinion.top_1_diagnosis,
                "evidence_conclusiveness": entry.opinion.evidence_conclusiveness,
                "confidence": entry.opinion.confidence_level,
                "pros": (entry.opinion.pros or [])[:1],
                "cons": (entry.opinion.cons or [])[:1],
            }
            for entry in entries
        }
        return json.dumps(payload, ensure_ascii=False)

    # Legacy dict[str, opinion_dump]
    if style == "nl":
        blocks: list[str] = []
        for agent_id, raw in opinions.items():
            data = raw if isinstance(raw, dict) else {}
            label = data.get("top_1_diagnosis", "unknown")
            conf = float(data.get("confidence_level", 0.0) or 0.0)
            conclusiveness = data.get("evidence_conclusiveness") or "unspecified"
            pros = data.get("pros") or []
            cons = data.get("cons") or []
            block = [
                f"[agent] {agent_id} (conf={conf:.2f}, {conclusiveness}): {label}"
            ]
            if pros:
                block.append(f"  Pro: {_clip_text(pros[0])}")
            if cons:
                block.append(f"  Con: {_clip_text(cons[0])}")
            blocks.append("\n".join(block))
        return "\n".join(blocks)
    return json.dumps(opinions, ensure_ascii=False)


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
    peer_context: PeerContextMode = "nl",
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
                if persona in PUBMEDQA_UNCERTAINTY_PERSONAS
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
        rendered = format_peer_context(
            context,
            peer_context=peer_context,
            compact=compact,
            mode=mode,
        )
        parts.append(
            "PEER OPINIONS SO FAR (previous round's final opinions, plus anyone who "
            "has already spoken this round, in speaking order; critique weak "
            "arguments, update hypotheses, and resolve contradictions where "
            "possible):\n"
            f"{rendered}"
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
