"""Context partitioning: give each agent a different slice of the same evidence.

This is what makes agents differ once their personas are gone. Instead of four
identical analysts all reading the whole abstract - and therefore all making the
same mistake for the same reason - each one sees a contiguous block of sentences
and has to ask peers about the rest.

The split is deterministic and model-free: identical input always yields the same
partitions, and nothing here consumes tokens, so cost comparisons between the
shared-context and partitioned architectures are not polluted by the split
itself.

Two invariants matter:

* **The question is in every partition.** An agent that cannot see what is being
  asked cannot produce a usable answer or a meaningful information request.
* **Partitions round-trip through `parse_pubmedqa_patient_case`.** Each agent's
  view is re-emitted in the exact `RESEARCH QUESTION:` / `EVIDENCE:` / `SOURCE`
  layout the rest of the package parses, so nothing downstream needs to know
  whether it is looking at a full case or a slice of one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.backends import parse_pubmedqa_patient_case

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[0-9])")


@dataclass(frozen=True)
class ContextPartition:
    """One agent's view of the case: the shared question plus its evidence slice."""

    segment_id: str
    index: int
    patient_case: str
    """Full case text as this agent sees it, ready to hand to `ClinicalAgent`."""
    evidence_text: str
    """Just the evidence sentences, used by the routing eligibility gate."""
    sentence_span: tuple[int, int]
    """Half-open [start, end) range of sentence indices, including any overlap."""


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences on terminal punctuation followed by a capital."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def partition_patient_case(
    patient_case: str,
    *,
    k: int,
    overlap: int = 0,
) -> list[ContextPartition]:
    """Split a case into `k` overlapping evidence segments, question kept in each.

    `overlap` is the number of sentences each segment borrows from its neighbours
    on either side, which keeps a claim that straddles a boundary readable to at
    least one agent. Returns fewer than `k` partitions when the evidence has fewer
    than `k` sentences - inventing empty segments would give some agent nothing to
    reason from.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")

    question, documents = parse_pubmedqa_patient_case(patient_case)
    header = _header(patient_case)

    sentences: list[str] = []
    owners: list[int] = []
    for doc_index, document in enumerate(documents):
        for sentence in split_sentences(document.content):
            sentences.append(sentence)
            owners.append(doc_index)

    if not sentences:
        # Nothing to split: every agent sees the case unchanged rather than nothing.
        return [
            ContextPartition(
                segment_id="s1",
                index=0,
                patient_case=patient_case,
                evidence_text="",
                sentence_span=(0, 0),
            )
        ]

    partitions: list[ContextPartition] = []
    for index, (start, end) in enumerate(_core_spans(len(sentences), k)):
        window_start = max(start - overlap, 0)
        window_end = min(end + overlap, len(sentences))
        slice_sentences = sentences[window_start:window_end]
        slice_owners = owners[window_start:window_end]
        partitions.append(
            ContextPartition(
                segment_id=f"s{index + 1}",
                index=index,
                patient_case=_render_case(
                    header=header,
                    question=question,
                    documents=documents,
                    sentences=slice_sentences,
                    owners=slice_owners,
                ),
                evidence_text=" ".join(slice_sentences),
                sentence_span=(window_start, window_end),
            )
        )
    return partitions


def _core_spans(total: int, k: int) -> list[tuple[int, int]]:
    """Contiguous, disjoint, exhaustive spans; the first spans absorb the remainder."""
    parts = min(k, total)
    base, extra = divmod(total, parts)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        spans.append((cursor, cursor + size))
        cursor += size
    return spans


def _header(patient_case: str) -> str:
    """The instruction line preceding `RESEARCH QUESTION:`, if the case has one."""
    prefix = patient_case.split("RESEARCH QUESTION:", 1)[0].strip()
    return prefix


def _render_case(
    *,
    header: str,
    question: str,
    documents: list,
    sentences: list[str],
    owners: list[int],
) -> str:
    """Re-emit an agent's slice in the canonical case format."""
    by_document: dict[int, list[str]] = {}
    for sentence, owner in zip(sentences, owners):
        by_document.setdefault(owner, []).append(sentence)

    blocks: list[str] = []
    for doc_index, doc_sentences in sorted(by_document.items()):
        document = documents[doc_index]
        block = f"SOURCE {document.id}"
        if document.title:
            block += f"\nTitle: {document.title}"
        block += "\n" + " ".join(doc_sentences)
        blocks.append(block)

    parts: list[str] = []
    if header:
        parts.append(header)
    parts.append(f"RESEARCH QUESTION:\n{question}")
    parts.append("EVIDENCE:\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)
