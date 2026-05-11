import re

from app.schemas import Citation, CitationValidation


_CITATION_PATTERN = re.compile(r"\[S([1-9][0-9]*)\]")


def validate_citations(content: str, available_citations: list[Citation]) -> CitationValidation:
    """Validate that an LLM response only cites retrieved source labels."""

    allowed_ids = {citation.id for citation in available_citations}
    cited_ids = [f"S{match}" for match in _CITATION_PATTERN.findall(content)]
    unique_cited_ids = sorted(set(cited_ids), key=_citation_sort_key)

    missing_citation_ids = sorted(set(unique_cited_ids) - allowed_ids, key=_citation_sort_key)
    unused_citation_ids = sorted(allowed_ids - set(unique_cited_ids), key=_citation_sort_key)
    has_required_citation = bool(unique_cited_ids) if allowed_ids else True

    issues = []
    if allowed_ids and not unique_cited_ids:
        issues.append("response_missing_inline_citations")
    if missing_citation_ids:
        issues.append("response_contains_unknown_citations")

    return CitationValidation(
        passed=not issues,
        cited_ids=unique_cited_ids,
        missing_citation_ids=missing_citation_ids,
        unused_citation_ids=unused_citation_ids,
        has_required_citation=has_required_citation,
        issues=issues,
    )


def _citation_sort_key(citation_id: str) -> int:
    try:
        return int(citation_id.removeprefix("S"))
    except ValueError:
        return 0
