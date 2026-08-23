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
- biolinkbert_hint (an external classifier's prediction, if any; a signal to weigh critically, never ground truth on its own):
{biolinkbert_hint}

Task:
Determine the ACTUAL conclusion made by the authors of the abstract.
Do not grade study quality. Ask: what did the authors conclude about the research question?

CAUTION: Some agents may be instructed to keep defending their Round 1 label in later rounds (a role-play device to surface counter-arguments, not a structural guarantee — they remain free to change their label). A prolonged, aggressive argument does NOT mean the medical abstract is inconclusive. Judge the abstract's actual conclusion independently of how hard any agent argued for it.

CRITICAL RULES FOR CHOOSING THE LABEL:
1. DISTINGUISHING "yes" AND "no":
   - Choose "yes" if the authors conclude a positive association, effect, or affirmative answer.
   - Choose "no" if the authors conclude NO association, NO effect, or a definitive negative answer. A definitive finding that something DOES NOT work is a "no", not a "yes".
2. THE "BOILERPLATE" BAN:
   - Do NOT choose "maybe" only because an agent cites boilerplate limitations (small sample, retrospective design). If the authors report a clear primary finding, prioritize the authors' explicit conclusion.
3. TRUE UNCERTAINTY ("maybe") & COVERAGE:
   - You MUST set `question_coverage` to "partial" and `final_label` to "maybe" if the primary findings are genuinely mixed/contradictory.
   - SPECULATIVE UTILITY: Choose "maybe" only when there is a STRUCTURAL gap between what was measured and what was asked — e.g. the authors used a surrogate/proxy endpoint instead of the outcome the question asks about, or the primary and secondary endpoints point in different directions. Hedging verbs alone ("may", "could", "has potential to") are NOT sufficient grounds for "maybe" on their own — check whether the measured endpoint actually differs from the question before applying this rule.
4. DO NOT TALLY VOTES: Agents may be under role-play instructions to argue adversarially. A majority of agents voting "yes" or "no" means NOTHING. Do not count their votes. You must base your final_label SOLELY on the logic you write in your rationale.

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

Set `confidence_level` to how strongly the abstract text supports your chosen label — including when that label is 'maybe'. A genuinely ambiguous abstract warrants a moderate confidence score, not an artificially high one.

Respond with ClinicalOpinion JSON only. top_1_diagnosis must be exactly 'yes', 'no', or 'maybe'.
""".strip()

PUBMEDQA_UNCERTAINTY_PERSONAS = frozenset(
    {"uncertainty_advocate"}
)

ROUND2_STRUCTURED_CRITICISM_RULE = (
    "In your 'cons' or 'pros', you MUST explicitly name an agent you disagree with "
    "using a JSON-safe tag like [uncertainty_advocate] or [evidence_skeptic], and refute "
    "their specific argument. Do not just restate your previous opinion."
)

ROUND2_NO_VERBATIM_QUOTE_RULE = (
    "DO NOT copy or quote other agents' text verbatim in your pros/cons. "
    "Synthesize your own counter-arguments."
)

ROUND2_JSON_SAFETY_RULE = (
    "Your output MUST be valid JSON. Use double-quoted strings in pros/cons. "
    "Do NOT use @mentions or possessive apostrophes (write [evidence_skeptic] argument, "
    "never evidence_skeptic's). Limit pros and cons to up to 3 short sentences each "
    "(max 40 words per sentence). Escape any internal double quotes as \\\"."
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

# Round 1: no peer context exists yet, so the task must be answerable standalone.
PUBMEDQA_DIFFERENTIAL_EXPANDER_R1 = (
    "You ground the differential in what was literally measured. Read the RESULTS "
    "section itself, not just the CONCLUSIONS paraphrase, and extract the primary "
    "numeric/statistical finding. Then check whether the question asks exactly what "
    "RESULTS measured, or whether there is a gap (surrogate endpoint, subgroup-only "
    "result, mismatched outcome). State any such gap explicitly."
)

# Round 2+: peer context exists, so contrarian stress-testing becomes meaningful.
PUBMEDQA_DIFFERENTIAL_EXPANDER_R2PLUS = (
    "You stress-test the panel's reading. Could the data actually imply the opposite "
    "conclusion? Argue for the counter-hypothesis to whatever your peers converged on, "
    "grounded in the literal RESULTS wording rather than the CONCLUSIONS paraphrase."
)

PUBMEDQA_PERSONA_INSTRUCTIONS: dict[str, str] = {
    "generalist": (
        "You answer PubMedQA-style yes/no/maybe questions from abstracts. "
        "Choose 'yes' or 'no' based on the primary conclusion of the abstract."
    ),
    "evidence_skeptic": EVIDENCE_SKEPTIC_PROMPT,
    "differential_expander": PUBMEDQA_DIFFERENTIAL_EXPANDER_R1,
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
Return ONLY one MINIFIED single-line JSON object (no markdown, no indentation, no line breaks) with exactly:
{"top_1_diagnosis":"yes|no|maybe","evidence_conclusiveness":"conclusive|inconclusive","top_3_differential_diagnoses":["yes","no","maybe"],"pros":["one short reason max 20 words"],"cons":["one short caveat max 20 words"],"required_further_tests":[],"confidence_level":0.0,"sources_used":["abstract"],"red_flags":[],"missing_information":""}
""".strip()

# Round 2+: agents are asked to name and refute a specific peer, which needs more
# room than round 1's single-sentence justification allows.
PUBMEDQA_COMPACT_SCHEMA_R2PLUS = """
Return ONLY one MINIFIED single-line JSON object (no markdown, no indentation, no line breaks) with exactly:
{"top_1_diagnosis":"yes|no|maybe","evidence_conclusiveness":"conclusive|inconclusive","top_3_differential_diagnoses":["yes","no","maybe"],"pros":["up to 3 reasons, each max 40 words"],"cons":["up to 3 caveats, each max 40 words"],"required_further_tests":[],"confidence_level":0.0,"sources_used":["abstract"],"red_flags":[],"missing_information":""}
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


def format_moderator_instruction_block(
    moderation_output: Any,
    *,
    peer_context: PeerContextMode = "nl",
) -> str:
    """Render supervisor moderation as a dedicated moderator block for agent prompts."""
    style = (peer_context or "nl").strip().lower()
    if style == "nl":
        body = format_moderation_nl(moderation_output) + "\nRespond to these instructions."
    else:
        moderation_payload = (
            moderation_output.model_dump()
            if hasattr(moderation_output, "model_dump")
            else moderation_output
        )
        body = (
            "Supervisor moderation from the previous round:\n"
            + json.dumps(moderation_payload, ensure_ascii=False)
            + "\nRespond to these instructions."
        )
    return f"[SYSTEM INSTRUCTION FROM MODERATOR]\n{body}"


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
    frozen_label: str | None = None,
    moderator_instruction: str | None = None,
    round_number: int | None = None,
) -> list[ChatMessage]:
    """Build chat messages for independent (round 1) or critique (round 2+) opinion generation.

    `context` is the previous round's final peer opinions (excluding self).
    Agents revise in isolation: they never see same-round peer drafts.
    Each entry keeps its agent_id/persona/round so the model sees an identified
    discussion transcript rather than an anonymous opinion dump.
    """
    mode = (task_mode or "clinical").strip().lower()
    is_safety_officer = persona.strip().lower() == "safety_officer"
    peer_revision = bool(context) or bool(moderator_instruction)
    use_compact_schema = compact or mode == "pubmedqa"
    resolved_round = (
        round_number
        if round_number is not None
        else (max((entry.round for entry in context), default=1) if context else 1)
    )
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
            if persona == "differential_expander":
                persona_text = (
                    PUBMEDQA_DIFFERENTIAL_EXPANDER_R2PLUS
                    if resolved_round > 1
                    else PUBMEDQA_DIFFERENTIAL_EXPANDER_R1
                )
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
            compact_schema = (
                PUBMEDQA_COMPACT_SCHEMA_R2PLUS if resolved_round > 1 else PUBMEDQA_COMPACT_SCHEMA
            )
            schema_block = (
                f"{compact_schema}\n\n{active_label_rule}"
                if use_compact_schema
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
    adversarial_directive = ""
    if frozen_label and resolved_round > 1:
        adversarial_directive = (
            f"\n\n[SYSTEM ARCHITECTURE OVERRIDE]\n"
            f"In Round 1, you diagnosed the answer as '{frozen_label}'. "
            f"The system has FROZEN your stance. You are now the defense attorney for the '{frozen_label}' label.\n"
            f"1. Your top_1_diagnosis MUST remain '{frozen_label}'.\n"
            f"2. ALIGNMENT RULE: Your 'pros' MUST logically support '{frozen_label}'. If your label is 'no' or 'maybe', your pros MUST explain what is wrong with the study or why it fails. NEVER use arguments that support the opposite label.\n"
            f"3. ANTI-LAZINESS RULE: You MUST explicitly attack the PEERS who voted differently. Do not use generic phrases. Quote specific data points from the abstract to prove your peers are wrong.\n"
        )

    system = (
        f"You are clinical debate agent `{agent_id}` with persona `{persona}`.\n"
        f"agent_id={agent_id}\n"
        f"task_mode={mode}\n"
        f"{persona_text}\n\n"
        f"{schema_block}"
        f"{adversarial_directive}"
    )
    if repair:
        system += (
            "\n\nYour previous reply was invalid JSON. "
            "Respond again with ONLY a valid JSON object matching the schema. "
            f"{ROUND2_JSON_SAFETY_RULE}"
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

    if peer_revision:
        peer_block_parts: list[str] = []
        if moderator_instruction and moderator_instruction.strip():
            peer_block_parts.append(moderator_instruction.strip())
        if context:
            rendered = format_peer_context(
                context,
                peer_context=peer_context,
                compact=use_compact_schema,
                mode=mode,
            )
            if rendered.strip():
                peer_block_parts.append(rendered)
        parts.append(
            "PEER OPINIONS FROM THE PREVIOUS ROUND (final opinions only; peers in "
            "this round write in isolation — you do not see their current drafts. "
            "Critique weak arguments, update hypotheses, and resolve contradictions "
            "where possible):\n"
            + "\n\n".join(peer_block_parts)
        )
        parts.append(
            "Produce an UPDATED ClinicalOpinion that reflects what you accept, "
            "reject, or still find uncertain after reviewing peers."
        )
        parts.append(ROUND2_STRUCTURED_CRITICISM_RULE)
        parts.append(ROUND2_NO_VERBATIM_QUOTE_RULE)
        parts.append(ROUND2_JSON_SAFETY_RULE)
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
