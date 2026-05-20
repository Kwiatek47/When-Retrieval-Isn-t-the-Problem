import re

from app.schemas import Citation, CitationValidation


_CITATION_BLOCK_PATTERN = re.compile(r"\[([^\]]*S[^\]]*)\]|\(([^\)]*S[^\)]*)\)", re.IGNORECASE)
_CITATION_ID_PATTERN = re.compile(r"\bS([1-9][0-9]*)\b", re.IGNORECASE)
_CITATION_RANGE_PATTERN = re.compile(r"\bS([1-9][0-9]*)\s*[-–]\s*S?([1-9][0-9]*)\b", re.IGNORECASE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_MAX_CITATIONS_PER_SENTENCE = 3
_NON_CLAIM_PATTERN = re.compile(
    r"(cannot provide|can'?t provide|not enough evidence|knowledge base did not|not medical advice|"
    r"consult(?:ing)? (?:a |an |your )?(?:qualified |licensed )?"
    r"(?:health\s*care provider|health\s*care professional|clinician|doctor|physician|medical professional)|"
    r"personalized (?:medical )?advice|personalised (?:medical )?advice|"
    r"baza wiedzy|nie mogę udzielić|nie moge udzielic|niewystarczająco|niewystarczajaco|"
    r"to nie jest porada medyczna|nie stanowi porady medycznej|skonsultuj|"
    r"indywidualn(?:a|ej) porad(?:a|y)|porad(?:a|y) medyczn(?:a|ej)|lekarz(?:em|a)?)",
    re.IGNORECASE,
)


def validate_citations(content: str, available_citations: list[Citation]) -> CitationValidation:
    """Validate that an LLM response only cites retrieved source labels."""

    allowed_ids = {citation.id for citation in available_citations}
    cited_ids = extract_citation_ids(content)
    unique_cited_ids = sorted(set(cited_ids), key=_citation_sort_key)
    claim_stats = _claim_level_stats(content, has_available_citations=bool(allowed_ids))

    missing_citation_ids = sorted(set(unique_cited_ids) - allowed_ids, key=_citation_sort_key)
    unused_citation_ids = sorted(allowed_ids - set(unique_cited_ids), key=_citation_sort_key)
    has_required_citation = (
        not allowed_ids
        or claim_stats["claim_count"] == 0
        or claim_stats["cited_claims_count"] == claim_stats["claim_count"]
    )
    citation_precision = _citation_precision(cited_ids, allowed_ids)

    issues = []
    if allowed_ids and claim_stats["claim_count"] and not unique_cited_ids:
        issues.append("response_missing_inline_citations")
    if missing_citation_ids:
        issues.append("response_contains_unknown_citations")
    if _has_noncanonical_citations(content):
        issues.append("response_contains_noncanonical_citation_format")
    issues.extend(claim_stats["issues"])

    return CitationValidation(
        passed=not issues,
        cited_ids=unique_cited_ids,
        missing_citation_ids=missing_citation_ids,
        unused_citation_ids=unused_citation_ids,
        has_required_citation=has_required_citation,
        claim_count=claim_stats["claim_count"],
        cited_claims_count=claim_stats["cited_claims_count"],
        citation_recall=claim_stats["citation_recall"],
        citation_precision=citation_precision,
        uncited_claims=claim_stats["uncited_claims"],
        shotgun_citation_claims=claim_stats["shotgun_citation_claims"],
        orphan_citations=claim_stats["orphan_citations"],
        issues=_unique_preserving_order(issues),
    )


def extract_citation_ids(content: str) -> list[str]:
    """Extract source labels from inline citations.

    Supports both separate citations (`[S1] [S3]`) and grouped citations
    (`[S1, S3]`, `[S1-S3]`).
    """

    cited_ids: list[str] = []
    for block in _citation_blocks(content):
        cited_ids.extend(_citation_ids_from_block(block))
    return cited_ids


def strip_citation_tags(content: str) -> str:
    def replace_citation(match: re.Match[str]) -> str:
        block = _block_content(match)
        if _citation_ids_from_block(block):
            return ""
        return match.group(0)

    return _CITATION_BLOCK_PATTERN.sub(replace_citation, content).strip()


def normalize_citation_format(content: str) -> str:
    """Normalize tolerated citation variants to the strict public format.

    Converts grouped and parenthetical source labels such as `[S1, S3]`,
    `[S1-S3]`, and `(S2)` into separate bracketed citations: `[S1] [S3]`.
    Non-citation bracket/parenthesis content is left unchanged.
    """

    def replace_citation(match: re.Match[str]) -> str:
        block = _block_content(match)
        citation_ids = _citation_ids_from_block(block)
        if not citation_ids:
            return match.group(0)
        return " ".join(f"[{citation_id}]" for citation_id in citation_ids)

    normalized = _CITATION_BLOCK_PATTERN.sub(replace_citation, content)
    return re.sub(r"\s+([.,;:])", r"\1", normalized).strip()


def requires_citation(sentence: str) -> bool:
    """Return whether a sentence is a substantive claim requiring source support."""

    return _requires_citation(sentence)


def _claim_level_stats(content: str, *, has_available_citations: bool) -> dict[str, object]:
    empty_stats = {
        "claim_count": 0,
        "cited_claims_count": 0,
        "citation_recall": None,
        "uncited_claims": [],
        "shotgun_citation_claims": [],
        "orphan_citations": [],
        "issues": [],
    }
    if not has_available_citations:
        return empty_stats

    issues = []
    claim_count = 0
    cited_claims_count = 0
    uncited_claims = []
    shotgun_citation_claims = []
    orphan_citations = []

    for sentence in _sentences(content):
        cited_ids = extract_citation_ids(sentence)
        if _is_orphan_citation(sentence):
            orphan_citations.append(sentence)
            issues.append("response_contains_orphan_citation")
            continue

        if not _requires_citation(sentence):
            continue

        claim_count += 1
        if cited_ids:
            cited_claims_count += 1
        else:
            uncited_claims.append(sentence)
            issues.append("claim_missing_citation")

        if len(set(cited_ids)) > _MAX_CITATIONS_PER_SENTENCE:
            shotgun_citation_claims.append(sentence)
            issues.append("response_contains_shotgun_citation")

    citation_recall = cited_claims_count / claim_count if claim_count else None
    return {
        "claim_count": claim_count,
        "cited_claims_count": cited_claims_count,
        "citation_recall": citation_recall,
        "uncited_claims": uncited_claims,
        "shotgun_citation_claims": shotgun_citation_claims,
        "orphan_citations": orphan_citations,
        "issues": _unique_preserving_order(issues),
    }


def _citation_precision(cited_ids: list[str], allowed_ids: set[str]) -> float | None:
    if not cited_ids:
        return None
    valid_count = sum(1 for citation_id in cited_ids if citation_id in allowed_ids)
    return valid_count / len(cited_ids)


def _expand_citation_ranges(content: str) -> list[str]:
    citation_ids: list[str] = []
    for start_raw, end_raw in _CITATION_RANGE_PATTERN.findall(content):
        start = int(start_raw)
        end = int(end_raw)
        if start <= end and end - start <= 20:
            citation_ids.extend(f"S{index}" for index in range(start, end + 1))
    return citation_ids


def _sentences(content: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", content).strip()
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(normalized) if sentence.strip()]


def _is_orphan_citation(sentence: str) -> bool:
    return not strip_citation_tags(sentence).strip(" \t\r\n.,;:")


def _requires_citation(sentence: str) -> bool:
    if _NON_CLAIM_PATTERN.search(sentence):
        return False

    statement = strip_citation_tags(sentence).strip()
    if not statement:
        return False
    if _is_short_heading(statement):
        return False
    return True


def _is_short_heading(statement: str) -> bool:
    if not statement.endswith(":"):
        return False
    return len(statement.split()) <= 8


def _unique_preserving_order(values: list[str]) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _citation_blocks(content: str) -> list[str]:
    return [_block_content(match) for match in _CITATION_BLOCK_PATTERN.finditer(content)]


def _block_content(match: re.Match[str]) -> str:
    return next(group for group in match.groups() if group is not None)


def _citation_ids_from_block(content: str) -> list[str]:
    citation_ids = [
        *_expand_citation_ranges(content),
        *(f"S{match}" for match in _CITATION_ID_PATTERN.findall(content)),
    ]
    unique_ids = []
    seen = set()
    for citation_id in citation_ids:
        normalized = citation_id.upper()
        if normalized not in seen:
            unique_ids.append(normalized)
            seen.add(normalized)
    return unique_ids


def _has_noncanonical_citations(content: str) -> bool:
    for match in _CITATION_BLOCK_PATTERN.finditer(content):
        block = _block_content(match)
        citation_ids = _citation_ids_from_block(block)
        if not citation_ids:
            continue
        canonical = " ".join(f"[{citation_id}]" for citation_id in citation_ids)
        if match.group(0) != canonical:
            return True
    return False


def _citation_sort_key(citation_id: str) -> int:
    try:
        return int(citation_id.removeprefix("S"))
    except ValueError:
        return 0
