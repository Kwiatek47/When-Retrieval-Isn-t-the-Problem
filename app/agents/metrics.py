"""Metrics that compare debate architectures rather than score a single answer.

Accuracy alone cannot say *why* one architecture beats another. These measure the
failure modes the supervisor architectures are meant to remove:

* `subversion_rate` - agents who were right alone and wrong after the discussion.
* `rescue_rate` - the reverse. Reporting subversion without rescue is how a paper
  accidentally claims discussion is harmful when it is merely churning; the net
  of the two is what a claim can rest on.
* `adoption_rate` - how often an agent simply moves to the majority position it
  was shown, whether or not that position is correct.
* `identity_bias_coefficient` - how much adoption drops when the same panel can no
  longer see who said what. This one is computed across two runs, not within one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.agents.aggregation import opinion_label
from app.agents.models import AgentRoundOpinion


@dataclass(frozen=True)
class FlipCounts:
    """Per-case movement of individual agents between the first and final round."""

    subverted: int
    """Right in round 1, wrong at the end."""
    rescued: int
    """Wrong in round 1, right at the end."""
    stable_correct: int
    stable_wrong: int

    @property
    def evaluated(self) -> int:
        return self.subverted + self.rescued + self.stable_correct + self.stable_wrong

    def as_dict(self) -> dict[str, int]:
        return {
            "subverted": self.subverted,
            "rescued": self.rescued,
            "stable_correct": self.stable_correct,
            "stable_wrong": self.stable_wrong,
        }


def flip_counts(rounds: list[list[AgentRoundOpinion]], gold_label: str) -> FlipCounts:
    """Classify each agent's round-1 -> final movement against the gold label.

    Agents are matched by id across rounds; an agent missing from either end is
    skipped rather than counted as a flip.
    """
    if len(rounds) < 2:
        return FlipCounts(0, 0, 0, 0)

    first = {entry.agent_id: opinion_label(entry.opinion) for entry in rounds[0]}
    last = {entry.agent_id: opinion_label(entry.opinion) for entry in rounds[-1]}

    subverted = rescued = stable_correct = stable_wrong = 0
    for agent_id, start_label in first.items():
        end_label = last.get(agent_id)
        if start_label is None or end_label is None:
            continue
        was_right = start_label == gold_label
        is_right = end_label == gold_label
        if was_right and not is_right:
            subverted += 1
        elif not was_right and is_right:
            rescued += 1
        elif was_right:
            stable_correct += 1
        else:
            stable_wrong += 1
    return FlipCounts(subverted, rescued, stable_correct, stable_wrong)


def subversion_rate(counts: list[FlipCounts]) -> float:
    """Share of agent-turns that started correct and ended wrong.

    Denominator is all evaluated agent-turns, so this is comparable across
    architectures with different panel sizes or round counts.
    """
    total = sum(item.evaluated for item in counts)
    if not total:
        return 0.0
    return sum(item.subverted for item in counts) / total


def rescue_rate(counts: list[FlipCounts]) -> float:
    """Share of agent-turns that started wrong and ended correct."""
    total = sum(item.evaluated for item in counts)
    if not total:
        return 0.0
    return sum(item.rescued for item in counts) / total


def adoption_rate(rounds: list[list[AgentRoundOpinion]]) -> float:
    """How often an agent moved onto the majority label of the peers it had seen.

    Gold-label-free by design: this measures conformity, which is a behaviour
    worth counting whether the majority happened to be right or wrong. Turns
    where the agent already held the peer-majority label are not counted as
    adoption, since nothing moved, and turns where the peers were tied are
    skipped entirely - there is no majority to conform to, and breaking the tie
    arbitrarily would make the metric depend on the order agents were listed in.
    """
    if len(rounds) < 2:
        return 0.0

    moves = 0
    adoptions = 0
    for index in range(1, len(rounds)):
        previous = {entry.agent_id: opinion_label(entry.opinion) for entry in rounds[index - 1]}
        for entry in rounds[index]:
            before = previous.get(entry.agent_id)
            after = opinion_label(entry.opinion)
            if before is None or after is None:
                continue
            peers = [
                label
                for agent_id, label in previous.items()
                if agent_id != entry.agent_id and label is not None
            ]
            peer_majority = _strict_majority(peers)
            if peer_majority is None or before == peer_majority:
                continue
            moves += 1
            if after == peer_majority:
                adoptions += 1
    if not moves:
        return 0.0
    return adoptions / moves


def _strict_majority(labels: list[str]) -> str | None:
    """The single most common label, or None when the top two are tied."""
    if not labels:
        return None
    ranked = Counter(labels).most_common(2)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def identity_bias_coefficient(
    identified_adoption: float,
    anonymized_adoption: float,
) -> float:
    """How much conformity the identity labels alone were buying.

    Positive means agents conformed more when they could see who was speaking,
    which is the effect anonymization is supposed to remove. Feed it the
    `adoption_rate` of two runs of the *same* panel on the *same* cases,
    differing only in whether the transcript was anonymized - otherwise the
    difference is confounded and the number means nothing.
    """
    return identified_adoption - anonymized_adoption
