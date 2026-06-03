# Roadmap korpusów MedChat

Ocena scenariusza rozszerzenia RAG (maj 2026) i plan wdrożenia w repozytorium.

## Werdykt architektoniczny

Kierunek jest **poprawny**: jedna kolekcja Qdrant, rozróżnienie źródeł przez `corpusVersion` + filtry `publicationTypes` / `year`, bez mieszania embeddingów o różnych wymiarach.

| Priorytet | Korpus | Wpływ | Status w repo |
| --- | --- | --- | --- |
| P0 | StatPearls | case-based, MCQ, ścieżki decyzyjne | **pilot** — `scripts/data/statpearls/` |
| P0 | PubMed reviews / RCT | evidence, PICO | częściowo — pipeline PubMed + sample ingest |
| P1 | DailyMed (SPL) | dawkowanie, CI, interakcje | planowany |
| P1 | NICE / wytyczne | postępowanie krok po kroku | planowany |
| P2 | UMLS (mapowania) | query expansion, nie pełny indeks | planowany |

**Unikać na start:** surowy PubMed bez filtrów, fora, pełne triple UMLS, przestarzałe wytyczne bez daty.

## Etap A — StatPearls (wdrożony)

1. `discover_chapters.py` — E-utilities `books`, zapytanie `statpearls[book] AND chapter[bookpart]`.
2. `build_chunks.py` — HTML `?report=printable`, sekcje H2/H3 → chunki MedCPT.
3. Indeks: `01_build_index.py --corpus-version statpearls_v1`.
4. Runtime: `RAG_CORPUS_VERSION=statpearls_v1`.

Rejestr wersji: `scripts/data/corpora/registry.json`.

## Etap B — merge multi-corpus

```bash
python3 scripts/data/corpora/merge_parquet.py \
  --inputs data/processed/chunks.parquet data/processed/statpearls/chunks.parquet \
  --output data/processed/chunks_merged.parquet
```

Po merge przebuduj BM25 (`data/bm25_stats.json`) i Qdrant z jednym `corpusVersion` tylko jeśli chcesz jedną wersję logiczną; w przeciwnym razie zostaw osobne `corpusVersion` w tej samej kolekcji.

## Metryki po każdym korpusie

```bash
make eval-retrieval
make eval-pubmedqa
# case-based: data/benchmarks/rag/
```

Oczekiwany efekt StatPearls: mniej `low_evidence` na pytaniach klinicznych / MedQA-style przy `RAG_CORPUS_VERSION=statpearls_v1` lub po merge.

## Następne kroki (P1)

- **DailyMed:** parser SPL → `publication_types: ["Drug Label"]`, rozszerzyć `_PUBLICATION_TYPE_POLICY` w `pre_retrieval.py`.
- **NICE:** HTML/PDF → `Practice Guideline`, filtr `year`.
- **PubMed RCT:** rozszerzyć `scripts/data/pubmed/pipeline` o zapytanie RCT + `publication_types` zawiera `Randomized Controlled Trial`.
