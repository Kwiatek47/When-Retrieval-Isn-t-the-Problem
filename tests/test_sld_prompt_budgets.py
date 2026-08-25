"""Regression tests for the token-budget bug found on the first real dev-90
run: PromptBudgetExceeded on build_director_prompt because real R2 rationale/
complement/self_audit text is much longer than short test fixtures, and the
prompt re-rendered the full abstract on top of the ledger + opinions for no
reason (same bug as build_moderator_prompt, fixed earlier but missed here).

These use deliberately verbose (unbounded, repeated) text for every field —
tighter than any real model is likely to produce — across a sample of real
PQA-L abstracts, so a future edit that removes the truncation/redundancy
fixes fails loudly here instead of on a multi-hour cluster run.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.agents.sld import prompts
from app.agents.sld.ledger import (
    Claim,
    ConclusionReconstructorContribution,
    EvidenceLedger,
    FindingsAuditorContribution,
    Gap,
    GapAuditorContribution,
    LedgerConflict,
    QuestionFramerContribution,
    RoundTwoOpinion,
)
from app.agents.sld.segmentation import extract_abstract_text, split_sentences

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = (
    PROJECT_ROOT / "data" / "benchmarks" / "pubmedqa" / "official_pqal_test" / "corpus.json"
)
SAMPLE_SIZE = 40

_VERBOSE = (
    "This is a realistically verbose model-generated rationale sentence that restates the "
    "finding at some length because 7B instruction models routinely ignore be-concise "
    "instructions. "
) * 2


def _sample_sentence_maps() -> list[dict[str, str]]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    maps = []
    for doc in corpus[:SAMPLE_SIZE]:
        abstract = extract_abstract_text(doc["content"])
        sentences = dict(split_sentences(abstract))
        if len(sentences) >= 3:
            maps.append(sentences)
    return maps


class DirectorPromptBudgetTests(unittest.TestCase):
    def test_verbose_ledger_and_four_verbose_opinions_stay_within_budget(self) -> None:
        for sentences in _sample_sentence_maps():
            ids = list(sentences)[:3]
            ledger = EvidenceLedger(
                target_population=Claim(text=_VERBOSE, sentence_ids=[ids[0]]),
                target_exposure=Claim(text=_VERBOSE, sentence_ids=[ids[0]]),
                target_outcome=Claim(text=_VERBOSE, sentence_ids=[ids[0]]),
                question_type="utility",
                primary_endpoint=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
                direction="positive",
                significance=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
                effect_magnitude=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
                reconstructed_conclusion=Claim(text=_VERBOSE, sentence_ids=[ids[2]]),
                conclusion_direction="positive",
                conclusion_strength="qualified",
                gaps=[Gap(gap_type="underpowered", description=_VERBOSE, sentence_ids=[ids[0]])],
                conflicts=[
                    LedgerConflict(
                        description=_VERBOSE,
                        agent_ids=["findings_auditor", "conclusion_reconstructor"],
                        sentence_ids=[ids[0], ids[1]],
                    )
                ],
                open_questions=[_VERBOSE],
                round_instructions=[_VERBOSE, _VERBOSE],
            )
            opinions = [
                RoundTwoOpinion(
                    agent_id=persona,
                    label="yes",
                    rationale=_VERBOSE,
                    citations=ids,
                    complement=_VERBOSE,
                    self_audit=_VERBOSE,
                )
                for persona in (
                    "question_framer",
                    "findings_auditor",
                    "gap_auditor",
                    "conclusion_reconstructor",
                )
            ]
            # Must not raise PromptBudgetExceeded.
            prompts.build_director_prompt(
                "Is X valuable in Y?", prompts.render_ledger(ledger), opinions
            )


class RoundTwoPromptBudgetTests(unittest.TestCase):
    """This is the exact case that broke on the first real cluster run of
    dev-90: a real R2 call hit PromptBudgetExceeded at ~1610 tokens because
    the "YOUR OWN ROUND-1 NOTE" render (a single contribution, not an
    aggregate of several) was the one render_contribution call site left
    uncapped — every *aggregating* call site had already been capped by the
    Director/Moderator fix, but a single-item render looked "obviously safe"
    and wasn't. It wasn't: a fully-populated ledger + own note + 3 round
    instructions + the longest real abstract can still exceed even the
    tightened budget, which is why R2 gets its own ROUND_TWO_MAX_PROMPT_TOKENS
    instead of reusing the R1 constant."""

    def _full_ledger_and_own_note(self, ids: list[str]) -> tuple[EvidenceLedger, FindingsAuditorContribution]:
        ledger = EvidenceLedger(
            target_population=Claim(text=_VERBOSE, sentence_ids=[ids[0]]),
            target_exposure=Claim(text=_VERBOSE, sentence_ids=[ids[0]]),
            target_outcome=Claim(text=_VERBOSE, sentence_ids=[ids[0]]),
            question_type="utility",
            primary_endpoint=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
            direction="positive",
            significance=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
            effect_magnitude=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
            reconstructed_conclusion=Claim(text=_VERBOSE, sentence_ids=[ids[2]]),
            conclusion_direction="positive",
            conclusion_strength="qualified",
            gaps=[Gap(gap_type="underpowered", description=_VERBOSE, sentence_ids=[ids[0]])],
            conflicts=[
                LedgerConflict(
                    description=_VERBOSE, agent_ids=["a", "b"], sentence_ids=[ids[0], ids[1]]
                )
            ],
            open_questions=[_VERBOSE],
            round_instructions=[_VERBOSE, _VERBOSE, _VERBOSE],
        )
        own_note = FindingsAuditorContribution(
            agent_id="findings_auditor",
            primary_endpoint=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
            direction="positive",
            significance=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
            effect_magnitude=Claim(text=_VERBOSE, sentence_ids=[ids[1]]),
        )
        return ledger, own_note

    def test_full_ledger_and_own_note_stay_within_budget(self) -> None:
        for sentences in _sample_sentence_maps():
            ids = list(sentences)[:3]
            ledger, own_note = self._full_ledger_and_own_note(ids)
            # Must not raise PromptBudgetExceeded.
            prompts.build_round_two_prompt(
                "Is X valuable in Y?",
                sentences,
                prompts.render_ledger(ledger),
                own_note,
                ledger.round_instructions,
            )

    def test_peer_notes_variant_stays_within_budget(self) -> None:
        for sentences in _sample_sentence_maps():
            ids = list(sentences)[:3]
            ledger, own_note = self._full_ledger_and_own_note(ids)
            peer_notes = "\n\n".join(
                prompts.render_contribution(own_note, max_field_chars=180) for _ in range(3)
            )
            # Must not raise PromptBudgetExceeded.
            prompts.build_round_two_peer_prompt(
                "Is X valuable in Y?",
                sentences,
                peer_notes,
                own_note,
                ledger.round_instructions,
            )


class ModeratorPromptBudgetTests(unittest.TestCase):
    def test_four_verbose_r1_contributions_stay_within_budget(self) -> None:
        for sentences in _sample_sentence_maps():
            ids = list(sentences)[:1]
            contributions = [
                QuestionFramerContribution(
                    agent_id="question_framer",
                    target_population=Claim(text=_VERBOSE, sentence_ids=ids),
                    target_exposure=Claim(text=_VERBOSE, sentence_ids=ids),
                    target_outcome=Claim(text=_VERBOSE, sentence_ids=ids),
                    question_type="utility",
                    yes_requires=_VERBOSE,
                    no_requires=_VERBOSE,
                ),
                FindingsAuditorContribution(
                    agent_id="findings_auditor",
                    primary_endpoint=Claim(text=_VERBOSE, sentence_ids=ids),
                    direction="positive",
                    significance=Claim(text=_VERBOSE, sentence_ids=ids),
                    effect_magnitude=Claim(text=_VERBOSE, sentence_ids=ids),
                ),
                GapAuditorContribution(
                    agent_id="gap_auditor",
                    gaps=[Gap(gap_type="underpowered", description=_VERBOSE, sentence_ids=ids)],
                ),
                ConclusionReconstructorContribution(
                    agent_id="conclusion_reconstructor",
                    reconstructed_conclusion=Claim(text=_VERBOSE, sentence_ids=ids),
                    direction="positive",
                    strength="qualified",
                ),
            ]
            # Must not raise PromptBudgetExceeded.
            prompts.build_moderator_prompt("Is X valuable in Y?", contributions)


class NeutralPromptBudgetTests(unittest.TestCase):
    def test_full_abstract_stays_within_budget(self) -> None:
        """build_neutral_prompt does all four personas' extraction in one
        call (ablation c), so it legitimately needs AGGREGATE_MAX_PROMPT_TOKENS
        rather than the standard per-persona budget — this locks that in."""
        from app.agents.sld.segmentation import extract_stats_profile, tag_sections

        with open(CORPUS_PATH, encoding="utf-8") as f:
            corpus = json.load(f)
        for doc in corpus[:SAMPLE_SIZE]:
            abstract = extract_abstract_text(doc["content"])
            sentence_list = split_sentences(abstract)
            sentences = dict(sentence_list)
            if len(sentences) < 3:
                continue
            tags = tag_sections(sentence_list)
            profile = extract_stats_profile(sentence_list)
            # Must not raise PromptBudgetExceeded.
            prompts.build_neutral_prompt(doc["title"], sentences, tags, profile)


if __name__ == "__main__":
    unittest.main()
