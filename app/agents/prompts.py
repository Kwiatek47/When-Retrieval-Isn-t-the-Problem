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

DEFENSE_OPINION_SCHEMA = """
Return ONLY a single JSON object with exactly these fields:
- internal_monologue (string, briefly think step-by-step: what is my frozen label, who opposed me, and how can I logically defend my label using the abstract text?)
- top_1_diagnosis (string, MUST BE EXACTLY '{frozen_label}')
- evidence_conclusiveness (string; one of "conclusive", "inconclusive")
- top_3_differential_diagnoses (array of strings)
- best_evidence_supporting_my_label (array of strings, quote the abstract to prove why '{frozen_label}' is logically correct)
- explicit_attack_on_opposing_peers (array of strings, name specific peers by their persona and destroy their arguments using data)
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

5. ADJUDICATE THE DISSENT. This is your most important job. When agents split, a
   majority vote would silently discard the minority and the panel would learn
   nothing. Instead:
   - Identify who is in the minority and what label they hold.
   - State their single strongest claim, grounded in the abstract — the best version
     of their argument, not a weak paraphrase.
   - Judge honestly whether the majority ACTUALLY ANSWERED that claim. Restating
     their own position, asserting the minority is wrong, or citing generic study
     limitations is NOT an answer. Set majority_has_addressed_it=false in that case.
   - Write one directed question the MAJORITY must answer next round, and one the
     MINORITY must answer. Make them specific and answerable from the abstract.
   If the panel is unanimous, leave the dissent fields empty and set
   majority_has_addressed_it to true.

Output:
Return ONLY a JSON object matching SupervisorModerationOutput:
{{
  "agreements": ["..."],
  "contradictions": ["..."],
  "round_instructions": ["..."],
  "primary_endpoint_result": "...",
  "author_conclusion": "yes" | "no" | "maybe" | "unclear",
  "residual_uncertainty": ["..."],
  "dissent": {{
    "minority_agents": ["..."],
    "minority_label": "yes" | "no" | "maybe" | "",
    "minority_core_claim": "...",
    "majority_has_addressed_it": true | false,
    "directed_challenge_to_majority": "...",
    "directed_challenge_to_minority": "..."
  }}
}}

No markdown fences, no commentary outside JSON.
""".strip()

SUPERVISOR_DIRECTOR_PROMPT = """
You are a Clinical Director synthesizing a multi-agent debate to answer a PubMedQA research question.

Inputs:
- patient_case (original abstract and question — YOUR GROUND TRUTH):
{patient_case}
- panel_vote_summary (who voted what, computed mechanically — not agent claims):
{panel_vote_summary}
- full_debate_transcript (agent opinions across rounds):
{full_debate_transcript}

Task:
Determine the ACTUAL conclusion made by the authors of the abstract.
Do not grade study quality. Ask: what did the authors conclude about the research question?

CRITICAL NOTE: The agents in the transcript are structurally forced by the system to stubbornly defend their Round 1 labels (playing Devil's Advocate). A prolonged, aggressive argument does NOT mean the medical abstract is inconclusive. You must cut through their forced stubbornness and independently judge the abstract's actual conclusion.

CRITICAL RULES FOR CHOOSING THE LABEL:
1. DISTINGUISHING "yes" AND "no":
   - Choose "yes" if the authors conclude a positive association, effect, or affirmative answer.
   - Choose "no" if the authors conclude NO association, NO effect, or a definitive negative answer. A definitive finding that something DOES NOT work is a "no", not a "yes".
2. THE "BOILERPLATE" BAN:
   - Do NOT choose "maybe" only because an agent cites boilerplate limitations (small sample, retrospective design). If the authors report a clear primary finding, prioritize the authors' explicit conclusion.
3. TRUE UNCERTAINTY ("maybe") & COVERAGE:
   - You MUST set `question_coverage` to "partial" and `final_label` to "maybe" if the primary findings are genuinely mixed/contradictory.
   - SPECULATIVE UTILITY: You MUST choose "maybe" if the question asks about a clinical/diagnostic role, and the authors only prove a correlation, concluding that the intervention "may", "could", or "has potential to" have a role in the future. Suggesting a hypothesis is not a definitive "yes".
5. VOTES ARE CALIBRATED EVIDENCE, NOT A VERDICT: Use `panel_vote_summary` as evidence about the abstract, weighted by how much each vote is worth:
   - ROUND 1 IS THE HONEST SIGNAL: round-1 labels were formed independently, before any peer contamination or forced defense. Later-round labels are FROZEN by the system and carry no new information — a label repeated in round 3 is not a second vote.
   - LONE DISSENT DOES NOT WIN BY DEFAULT: when one agent dissents against an otherwise agreeing panel, you may only follow the dissenter if you can point to the specific sentence or number in the abstract that the majority misread. Name that sentence in your rationale. If you cannot, go with the majority reading.
   - DISCOUNT THE STRUCTURAL MAYBE-HUNTER: [uncertainty_advocate] is instructed by the system to search for reasons to answer "maybe", and its stated confidence is not a reliable measure of how real the gap is. Treat its "maybe" as a hypothesis to verify against the abstract, never as evidence on its own.
   - You may still overrule the entire panel — but only on the strength of the abstract text, which you must quote in your rationale.

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
UNCERTAINTY_ADVOCATE_PROMPT = """You are the Coverage Auditor on a PubMedQA debate panel.
You answer ONE question: does this abstract actually settle the question that was asked?

You are a DETECTOR, not an advocate. A detector that fires on everything is worthless —
it carries no information and the panel learns to ignore it. Measured on this benchmark,
answering 'maybe' indiscriminately scores no better than guessing. Your value comes
entirely from separating the abstracts that leave the question open from those that close it.

DEFAULT: most abstracts DO answer their own question. Start from 'yes' or 'no' following the
authors' primary finding, and move to 'maybe' only when you can name a specific gap below and
quote the text that shows it.

Answer 'maybe' ONLY when one of these is concretely present, and say which one:
1. PARTIAL COVERAGE: the question asks about X, the study measured a surrogate or proxy for X.
2. SUBGROUP-ONLY: the effect holds in a subgroup but not in the population the question asks about.
3. INTERNAL CONTRADICTION: primary and secondary endpoints point in opposite directions.
4. SPECULATIVE UTILITY: the question asks whether X has a clinical role, the authors prove only
   correlation and propose the role as future work ("may", "could", "has potential").
5. INSIGNIFICANT DATA: the main claim rests on results that failed significance (p > 0.05).

These do NOT justify 'maybe': small sample, retrospective design, short follow-up, single centre,
a call for further research, or cautious academic phrasing around a clear primary finding.

CONFIDENCE CALIBRATION RULE: `confidence_level` measures YOUR detection of the gap, not the paper's own uncertainty. Report it honestly — downstream consensus logic reads this number and is misled by inflated values:
- 0.85-1.0: the gap is explicit in the text (authors state the question stays unresolved, the endpoint is openly a surrogate, primary and secondary results contradict each other).
- 0.5-0.8: you are inferring the gap from what the abstract does not say.
- below 0.5: you suspect a gap but the text mostly supports a definitive answer.
Never inflate this score to make your 'maybe' harder to overrule.

Respond with ClinicalOpinion JSON only. top_1_diagnosis must be exactly 'yes', 'no', or 'maybe'.
""".strip()

PUBMEDQA_UNCERTAINTY_PERSONAS = frozenset(
    {"uncertainty_advocate"}
)

ENGAGEMENT_OPINION_SCHEMA = """
Return ONLY a single JSON object with exactly these fields:
- strongest_opposing_argument (string; state the BEST argument against your current
  label, in its strongest form. If a peer disagrees with you, this is their argument,
  put as well as they could put it. Never write "none" while a peer disagrees.)
- my_answer_to_it (string; answer that argument using the abstract. "I disagree" or
  restating your own position is not an answer.)
- position_changed (boolean; true if you are changing your label this round)
- what_changed_my_mind (string; if position_changed is true, name the specific
  argument or sentence that moved you. "The majority disagreed with me" is NOT a
  valid reason — changing because you are outnumbered is the one thing you must
  never do. If position_changed is false, leave this empty.)
- top_1_diagnosis (string; "yes", "no" or "maybe")
- evidence_conclusiveness (string; one of "conclusive", "inconclusive")
- top_3_differential_diagnoses (array of strings)
- pros (array of strings)
- cons (array of strings)
- required_further_tests (array of strings)
- confidence_level (number between 0.0 and 1.0)
- sources_used (array of strings)
- red_flags (array of strings)
- missing_information (string)

No markdown fences, no commentary outside JSON.
""".strip()

DISSENT_ENGAGEMENT_RULE = (
    "HOW A REAL PANEL ARGUES — follow this before you write your label:\n"
    "1. A disagreement is information. If a peer reached a different conclusion from "
    "the same abstract, they saw something you did not, or you saw something they did "
    "not. Find out which before you decide.\n"
    "2. You MUST fill strongest_opposing_argument with the best case against your own "
    "label, and answer it in my_answer_to_it using the abstract text.\n"
    "3. Being outnumbered is NOT evidence. Do not move to the majority label because it "
    "is the majority. Move only if a specific argument defeats your reading, and then "
    "name that argument in what_changed_my_mind.\n"
    "4. Equally, do not dig in out of stubbornness. If the objection is answered, say so "
    "and update.\n"
    "5. If you hold your position, your answer must explain why the opposing argument "
    "fails — not merely that you still believe your own."
)

ROUND2_STRUCTURED_CRITICISM_RULE = (
    "In your attack/critique fields, you MUST explicitly name an agent you disagree with "
    "using a JSON-safe tag like [uncertainty_advocate] or [evidence_skeptic], and refute "
    "their specific argument. Do not just restate your previous opinion."
)

ROUND2_NO_VERBATIM_QUOTE_RULE = (
    "DO NOT copy or quote other agents' text verbatim in your arguments. "
    "Synthesize your own counter-arguments."
)

ROUND2_JSON_SAFETY_RULE = (
    "Your output MUST be valid JSON. Use double-quoted strings for all argument arrays. "
    "Do NOT use @mentions or possessive apostrophes (write [evidence_skeptic] argument, "
    "never evidence_skeptic's). Limit your evidence and attacks to exactly one short sentence each "
    "(max 25 words). Escape any internal double quotes as \\\"."
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
        "\n\nCRITICAL INSTRUCTION FOR DEBATE ROUNDS: You are the sole auditor of coverage, so "
        "hold a gap you can still point to in the text — but holding 'maybe' on every case "
        "makes your vote uninformative and it will be discounted. "
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

    dissent = data.get("dissent") or {}
    if isinstance(dissent, dict) and dissent.get("minority_agents"):
        addressed = bool(dissent.get("majority_has_addressed_it"))
        lines.append("- OPEN DISAGREEMENT ON THE PANEL:")
        lines.append(
            f"  • minority: {', '.join(str(a) for a in dissent['minority_agents'])} "
            f"holding '{dissent.get('minority_label', '')}'"
        )
        lines.append(
            f"  • their strongest claim: {_clip_text(dissent.get('minority_core_claim', ''), max_chars=260)}"
        )
        lines.append(
            "  • has the majority answered it? "
            + ("yes" if addressed else "NO — it still stands unanswered")
        )
        maj = str(dissent.get("directed_challenge_to_majority") or "").strip()
        mino = str(dissent.get("directed_challenge_to_minority") or "").strip()
        if maj:
            lines.append(f"  • MAJORITY must answer: {_clip_text(maj, max_chars=240)}")
        if mino:
            lines.append(f"  • MINORITY must answer: {_clip_text(mino, max_chars=240)}")
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
    dissent_protocol: bool = False,
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

    resolved_round = (
        round_number
        if round_number is not None
        else (max((entry.round for entry in context), default=1) if context else 1)
    )
    adversarial_directive = ""
    if dissent_protocol and resolved_round > 1 and not frozen_label and not is_safety_officer:
        # Engagement schema: the agent must answer the opposing case before voting.
        # safety_officer keeps its own schema — it audits, it does not take a side.
        schema_block = (
            f"{ENGAGEMENT_OPINION_SCHEMA}\n\n{active_label_rule}"
            if mode == "pubmedqa"
            else ENGAGEMENT_OPINION_SCHEMA
        )
    if frozen_label and resolved_round > 1:
        # Semantic steering: defense-round JSON keys replace generic pros/cons.
        schema_block = DEFENSE_OPINION_SCHEMA.format(frozen_label=frozen_label)
        adversarial_directive = (
            f"\n\n[SYSTEM ARCHITECTURE OVERRIDE]\n"
            f"You are the defense attorney for the '{frozen_label}' label. "
            f"Populate the JSON fields carefully to defend '{frozen_label}' and attack anyone who disagreed with it."
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
        if dissent_protocol and not frozen_label:
            parts.append(DISSENT_ENGAGEMENT_RULE)
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
