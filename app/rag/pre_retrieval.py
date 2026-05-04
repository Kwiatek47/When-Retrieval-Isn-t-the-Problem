import re

from app.rag.models import PreRetrievalResult
from app.schemas import ChatMessage


class PreRetriever:
    """Prepare a clinical user query for retrieval."""

    _NON_RETRIEVAL_PATTERNS = (
        re.compile(r"^\s*(hi|hello|hey|czesc|cześć|dzień dobry|dzien dobry)\s*[!.]?\s*$", re.IGNORECASE),
        re.compile(r"^\s*(thanks|thank you|dzieki|dzięki|ok|okay)\s*[!.]?\s*$", re.IGNORECASE),
    )
    _ABBREVIATIONS = {
        "bp": "blood pressure",
        "hr": "heart rate",
        "mi": "myocardial infarction",
        "pad": "peripheral artery disease",
        "pe": "pulmonary embolism",
        "dm": "diabetes mellitus",
        "t2dm": "type 2 diabetes mellitus",
        "ckd": "chronic kidney disease",
        "copd": "chronic obstructive pulmonary disease",
        "nsaid": "nonsteroidal anti-inflammatory drug",
        "nsaids": "nonsteroidal anti-inflammatory drugs",
    }

    def prepare(self, messages: list[ChatMessage]) -> PreRetrievalResult:
        query = self._latest_user_message(messages)
        normalized_query = self._normalize(query)
        expanded_query = self._expand_abbreviations(normalized_query)
        requires_retrieval = self._requires_retrieval(normalized_query)

        search_queries = [normalized_query]
        if expanded_query != normalized_query:
            search_queries.append(expanded_query)

        filters = self._extract_filters(normalized_query)
        notes = []
        if expanded_query != normalized_query:
            notes.append("Medical abbreviations expanded for retrieval.")
        if not requires_retrieval:
            notes.append("Retrieval skipped for simple conversational input.")

        return PreRetrievalResult(
            original_query=query,
            normalized_query=normalized_query,
            search_queries=search_queries,
            requires_retrieval=requires_retrieval,
            filters=filters,
            notes=notes,
        )

    def _latest_user_message(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return messages[-1].content

    def _normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    def _expand_abbreviations(self, query: str) -> str:
        expanded = query
        for abbreviation, full_name in self._ABBREVIATIONS.items():
            pattern = re.compile(rf"\b{re.escape(abbreviation)}\b", re.IGNORECASE)
            expanded = pattern.sub(f"{abbreviation.upper()} ({full_name})", expanded)
        return expanded

    def _requires_retrieval(self, query: str) -> bool:
        if not query:
            return False
        return not any(pattern.match(query) for pattern in self._NON_RETRIEVAL_PATTERNS)

    def _extract_filters(self, query: str) -> dict[str, str]:
        filters = {}
        icd_codes = re.findall(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b", query.upper())
        if icd_codes:
            filters["icd_code"] = ",".join(sorted(set(icd_codes)))
        return filters

