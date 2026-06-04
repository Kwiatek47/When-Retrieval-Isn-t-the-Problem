from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "nice" / "chunks_clinical.parquet"
DEFAULT_RETRIEVAL_OUT = PROJECT_ROOT / "data" / "benchmarks" / "nice" / "eval_nice_guidelines_retrieval_500.json"
DEFAULT_RAG_OUT = PROJECT_ROOT / "data" / "benchmarks" / "nice" / "eval_nice_guidelines_rag_100.json"

STOPWORDS = {
    "a",
    "about",
    "after",
    "against",
    "all",
    "also",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "during",
    "each",
    "for",
    "from",
    "guideline",
    "guidelines",
    "have",
    "having",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "more",
    "most",
    "must",
    "need",
    "of",
    "on",
    "or",
    "our",
    "overview",
    "people",
    "person",
    "recommendation",
    "recommendations",
    "refer",
    "referral",
    "referrals",
    "section",
    "sections",
    "should",
    "the",
    "their",
    "this",
    "those",
    "through",
    "to",
    "treatment",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
    "would",
}

GENERIC_SECTION_MARKERS = {
    "about this guideline",
    "assessment",
    "context",
    "general recommendations",
    "how has it been developed?",
    "how does it relate to statutory and non-statutory guidance?",
    "introduction",
    "overview",
    "recommendations",
    "who is it for?",
    "who should use this guidance?",
    "using this guideline",
    "why is it needed?",
    "why the committee made the recommendations",
}

SECTION_HINT_MARKERS = {
    "assessment",
    "diagnos",
    "management",
    "referral",
    "treatment",
    "suspect",
    "suspected",
    "when to suspect",
    "could this be sepsis",
    "initial",
    "risk",
    "support",
    "care",
    "monitor",
    "prevention",
    "prescrib",
    "recommend",
    "information and support",
    "individualised",
    "choice",
    "first-line",
    "first line",
}

QUESTION_TEMPLATES = (
    "What does NICE recommend for {focus} in {title}?",
    "According to NICE, how should {focus} be managed in {title}?",
    "What guidance does NICE give for {focus} in {title}?",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build larger NICE benchmark datasets from NICE chunks.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--retrieval-out", type=Path, default=DEFAULT_RETRIEVAL_OUT)
    parser.add_argument("--rag-out", type=Path, default=DEFAULT_RAG_OUT)
    parser.add_argument("--retrieval-size", type=int, default=500)
    parser.add_argument("--rag-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _load_rows(args.chunks)
    docs = _group_by_document(rows)
    if not docs:
        raise RuntimeError(f"No NICE documents found in {args.chunks}.")

    doc_order = _document_order(docs)
    retrieval_cases = _build_cases(
        docs=docs,
        doc_order=doc_order,
        target_size=args.retrieval_size,
        mode="retrieval",
        seed=args.seed,
    )
    rag_cases = _build_cases(
        docs=docs,
        doc_order=doc_order,
        target_size=args.rag_size,
        mode="rag",
        seed=args.seed,
    )

    _write_json(args.retrieval_out, retrieval_cases)
    _write_json(args.rag_out, rag_cases)

    print(
        "Wrote NICE benchmark datasets "
        f"retrieval={args.retrieval_out} cases={len(retrieval_cases)} "
        f"rag={args.rag_out} cases={len(rag_cases)} "
        f"docs={len(docs)} chunks={len(rows)}"
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"NICE chunks parquet not found: {path}")
    table = pq.read_table(path, columns=["chunk_id", "document_id", "external_id", "title", "section", "header_path", "text"])
    return table.to_pylist()


def _group_by_document(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        document_id = str(row.get("document_id") or "").strip()
        if not document_id:
            continue
        docs[document_id].append(row)
    for document_id in docs:
        docs[document_id].sort(key=_chunk_sort_key)
    return dict(docs)


def _document_order(docs: dict[str, list[dict[str, Any]]]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for document_id, rows in docs.items():
        external_id = _external_id(rows[0])
        grouped[_external_prefix(external_id)].append(document_id)

    prefix_order = ["AMR", "CG", "NG", "PH", "TA", "HTG", "HST", "CSG", "SC", "MIB", "ES", "QS"]
    ordered_prefixes = sorted(grouped, key=lambda value: (prefix_order.index(value) if value in prefix_order else len(prefix_order), value))

    ordered_docs: list[str] = []
    for prefix in ordered_prefixes:
        ordered_docs.extend(sorted(grouped[prefix], key=lambda document_id: (_external_id(docs[document_id][0]), document_id)))
    return ordered_docs


def _build_cases(
    *,
    docs: dict[str, list[dict[str, Any]]],
    doc_order: list[str],
    target_size: int,
    mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    selected_chunk_ids: set[str] = set()
    max_depth = max((len(rows) for rows in docs.values()), default=0)

    if mode == "rag":
        depth_limit = 1
    else:
        depth_limit = max_depth

    for depth in range(depth_limit):
        for index, document_id in enumerate(doc_order):
            rows = docs[document_id]
            if depth >= len(rows):
                continue
            row = rows[depth]
            chunk_id = str(row.get("chunk_id") or "")
            if not chunk_id or chunk_id in selected_chunk_ids:
                continue
            if mode == "rag" and depth > 0:
                continue
            case = _build_case(row, mode=mode, index=len(cases), seed=seed)
            if case is None:
                continue
            selected_chunk_ids.add(chunk_id)
            cases.append(case)
            if len(cases) >= target_size:
                return cases

    if len(cases) < target_size:
        for depth in range(depth_limit, max_depth):
            for document_id in doc_order:
                rows = docs[document_id]
                if depth >= len(rows):
                    continue
                row = rows[depth]
                chunk_id = str(row.get("chunk_id") or "")
                if not chunk_id or chunk_id in selected_chunk_ids:
                    continue
                case = _build_case(row, mode=mode, index=len(cases), seed=seed)
                if case is None:
                    continue
                selected_chunk_ids.add(chunk_id)
                cases.append(case)
                if len(cases) >= target_size:
                    return cases

    if len(cases) < target_size:
        raise RuntimeError(f"Only built {len(cases)} {mode} cases; need {target_size}.")

    return cases


def _build_case(row: dict[str, Any], *, mode: str, index: int, seed: int) -> dict[str, Any] | None:
    title = str(row.get("title") or "").strip()
    external_id = _external_id(row)
    document_id = str(row.get("document_id") or "").strip()
    section = str(row.get("section") or "").strip()
    header_path = str(row.get("header_path") or "").strip()
    text = str(row.get("text") or "").strip()
    body = _body_from_chunk(text)
    focus = _focus_phrase(title=title, section=section, header_path=header_path, body=body)
    if not focus:
        return None

    answer = _answer_snippet(body)
    terms = _expected_terms(focus=focus, answer=answer)
    if mode == "rag" and len(terms) < 3:
        return None

    template_index = (index + seed) % len(QUESTION_TEMPLATES)
    question = QUESTION_TEMPLATES[template_index].format(focus=focus, title=_short_title(title))

    if mode == "retrieval":
        return {
            "id": f"{document_id}:{index:04d}",
            "question": question,
            "intent": _intent_from_text(section, header_path, text),
            "relevant_document_ids": [document_id],
            "relevant_pmids": [],
            "acceptable_publication_types": ["Practice Guideline", "Guideline", "Clinical Guideline"],
            "must_not_answer_without_sources": True,
            "source_urls": [f"https://www.nice.org.uk/guidance/{external_id.lower()}" if external_id else ""],
            "benchmark_focus": focus,
            "benchmark_chunk_id": row.get("chunk_id"),
            "benchmark_section": section,
        }

    return {
        "id": f"{document_id}:{index:04d}",
        "question": question,
        "intent": _intent_from_text(section, header_path, text),
        "expected_status": "grounded",
        "relevant_document_ids": [document_id],
        "relevant_pmids": [],
        "expected_answer_terms": terms,
        "forbidden_answer_terms": ["PubMed"],
        "source_urls": [f"https://www.nice.org.uk/guidance/{external_id.lower()}" if external_id else ""],
        "answer": answer,
        "benchmark_focus": focus,
        "benchmark_chunk_id": row.get("chunk_id"),
        "benchmark_section": section,
    }


def _body_from_chunk(text: str) -> str:
    parts = text.split("\n\n", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    return text.strip()


def _focus_phrase(*, title: str, section: str, header_path: str, body: str) -> str:
    candidates = [
        _clean_focus(section),
        _clean_focus(_tail_from_header_path(header_path)),
        _clean_focus(title),
        _clean_focus(body[:240]),
    ]
    for candidate in candidates:
        if candidate and len(candidate.split()) >= 3:
            return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _clean_focus(text: str) -> str:
    tokens = _content_tokens(text)
    if not tokens:
        return ""
    return " ".join(tokens[:8])


def _tail_from_header_path(header_path: str) -> str:
    if "->" not in header_path:
        return header_path
    return header_path.split("->")[-1].strip()


def _answer_snippet(body: str, *, max_words: int = 42) -> str:
    cleaned = re.sub(r"\s+", " ", body).strip()
    if not cleaned:
        return ""
    bullets = re.findall(r"(?:^|[.!?]\s+)([^.?!]{40,240})", cleaned)
    if bullets:
        cleaned = bullets[0].strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chosen: list[str] = []
    words = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        chosen.append(sentence)
        words += len(_tokenize(sentence))
        if words >= max_words or len(chosen) >= 2:
            break
    snippet = " ".join(chosen).strip()
    return snippet[:320].rstrip()


def _expected_terms(*, focus: str, answer: str) -> list[str]:
    terms: list[str] = []
    for source in (focus, answer):
        for term in _content_tokens(source):
            if term not in terms:
                terms.append(term)
            if len(terms) >= 4:
                return terms
    return terms


def _content_tokens(text: str) -> list[str]:
    tokens = []
    for raw in _tokenize(text):
        normalized = raw.strip("-'").lower()
        if not normalized or normalized in STOPWORDS:
            continue
        if len(normalized) < 4 and not raw.isupper():
            continue
        if normalized.isdigit():
            continue
        if normalized in GENERIC_SECTION_MARKERS:
            continue
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-']*", text)


def _intent_from_text(section: str, header_path: str, text: str) -> str:
    haystack = " ".join([section, header_path, text]).lower()
    if "treat" in haystack or "prescrib" in haystack or "medication" in haystack:
        return "treatment"
    if "diagnos" in haystack or "suspect" in haystack or "recognition" in haystack or "referral" in haystack:
        return "diagnosis"
    if "prevention" in haystack or "risk" in haystack:
        return "prevention"
    return "general"


def _external_id(row: dict[str, Any]) -> str:
    value = str(row.get("external_id") or "").strip()
    return value.upper()


def _external_prefix(external_id: str) -> str:
    match = re.match(r"([A-Z]+)", external_id)
    if match:
        return match.group(1)
    return "ZZ"


def _short_title(title: str) -> str:
    cleaned = title.strip()
    if ":" in cleaned:
        return cleaned.split(":", 1)[0].strip()
    return cleaned


def _chunk_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    section = str(row.get("section") or "")
    header_path = str(row.get("header_path") or "")
    text = str(row.get("text") or "")
    score = _chunk_score(section=section, header_path=header_path, text=text)
    chunk_index = _safe_int(row.get("chunk_index"))
    word_count = _safe_int(row.get("word_count"))
    chunk_id = str(row.get("chunk_id") or "")
    return (-score, chunk_index, -word_count, chunk_id)


def _chunk_score(*, section: str, header_path: str, text: str) -> float:
    score = 0.0
    lower_section = section.lower().strip()
    lower_header = header_path.lower()
    lower_text = text.lower()

    if lower_section in GENERIC_SECTION_MARKERS:
        score -= 4.0
    if any(marker in lower_section for marker in SECTION_HINT_MARKERS):
        score += 4.0
    if any(marker in lower_header for marker in SECTION_HINT_MARKERS):
        score += 2.0
    if any(marker in lower_text for marker in ("- 1.", " refer ", " offer ", " consider ", " should ", " do not ")):
        score += 1.5
    if _safe_word_count(text) >= 35:
        score += 1.0
    if _safe_word_count(text) >= 120:
        score += 0.5
    if len(_content_tokens(section)) >= 3:
        score += 0.5
    return score


def _safe_word_count(text: str) -> int:
    return len(_tokenize(text))


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
