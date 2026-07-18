# Roadmap korpusów MedChat

Plan rozszerzenia korpusu wiedzy chatbota medycznego i status kolejnych
korpusów w repozytorium.

## Werdykt architektoniczny

Jedna kolekcja Qdrant, jedno `data/processed/chunks.parquet` scalane przez
`scripts/data/corpora/build_processed_chunks.py` z canonical schema
(`chunks_schema_v2`) i per-corpus adapterów. Rozróżnianie źródeł na etapie
retrieval przez pola payload `sourceName`, `sourceType`, `publicationTypes` i
`corpusVersion` (do filtrów A/B per korpus w jednej kolekcji), bez mieszania
embeddingów o różnych wymiarach.

Rejestr wersji: `scripts/data/corpora/registry.json`.

| Priorytet | Korpus | Wpływ | Status |
| --- | --- | --- | --- |
| P0 | PubMed reviews / systematic reviews / meta-analyses | evidence, PICO | active |
| P0 | NICE guidelines | guideline coverage, treatment/diagnosis pathways | pilot (chunks locally, index tested) |
| P0 | StatPearls | case-based, exam-style, decision paths | pilot (pipeline + adapter merged; run local) |
| P1 | DailyMed (SPL) | dawkowanie, przeciwwskazania, interakcje | planowany |
| P1 | PubMed RCT rozszerzenie | primary evidence dla intent=treatment | planowany |
| P2 | UMLS mapowania | query expansion, nie pełny indeks | planowany |

**Odrzucone:**

- **S2ORC (Semantic Scholar Open Research Corpus)**: Semantic Scholar
  indeksuje głównie MEDLINE/PubMed jako źródło biomedyczne, więc podzbiór
  filtrowany po `s2FieldsOfStudy=Medicine` w >90% pokrywa się z PubMed
  abstract corpus, który już mamy. Dodałby noise i storage (~10-30 GB) bez
  istotnego information gain. Preprinty (bioRxiv/medRxiv) to marginalna
  część i można je dodać osobnym dedykowanym pipeline, nie masową syndykacją.

## Ukończone etapy

### PubMed reviews (P0, active)

- Pipeline: `scripts/data/pubmed/pipeline/` (01_esearch → 02_fetch → 04_clean_dedupe_chunk → 05_quality_report → 06_write_manifest).
- Output: `data/interim/pubmed/chunks.parquet` (pełny corpus, ~kilkaset tysięcy chunków).
- Adapter: `scripts/data/corpora/adapters/pubmed.py`.
- Corpus version: `pubmed-reviews-v1`.

### NICE guidelines (P0, pilot)

- Pipeline: `scripts/data/nice/pipeline/` (00_fetch_catalog → 00_download_pdfs → 01_pdf_to_markdown → 01b_clean_markdown → 02_build_documents → 03_write_manifest) + `scripts/embeddings/nice_markdown_to_parquet.py`.
- Output: `data/interim/nice/chunks.parquet` (broad, ~39k chunków) lub `chunks_clinical.parquet` (~17k).
- Adapter: `scripts/data/corpora/adapters/nice.py`.
- Corpus version: `nice-guidelines-v1`.

### StatPearls (P0, pilot)

- Pipeline: `scripts/data/statpearls/` (`discover_chapters.py` → `build_chunks.py`).
- Output: `data/interim/statpearls/chunks.parquet` (~9k rozdziałów, ~200-500k chunków).
- Adapter: `scripts/data/corpora/adapters/statpearls.py`.
- Corpus version: `statpearls-v1`.
- Benchmark: `data/benchmarks/retrieval/eval_statpearls_sample.json` (12 case-based pytań, ground truth po `doc_id`).

### Merger + canonical schema (etap wspólny)

- Schema: `scripts/data/corpora/schema.py` (`chunks_schema_v2`, jednolity PyArrow schema, brak `promote_options="default"`).
- Registry: `scripts/data/corpora/registry.json` (adapter_class, dedupe_priority, default_include, local_chunks_path per corpus).
- Merger: `scripts/data/corpora/build_processed_chunks.py` (cross-corpus dedupe po pmid/doi/opcjonalnie title, walidacja, manifest z dedupe_stats + counts per source/version).
- Ablation runner: `scripts/data/corpora/run_ablation.py`.

## Następne kroki

- **DailyMed (SPL)**: parser SPL → `publication_types: ["Drug Label"]`, adapter `DailyMedAdapter`, rozszerzyć `_PUBLICATION_TYPE_POLICY` w `pre_retrieval.py` o `Drug Label` dla intent=treatment/adverse_effects.
- **PubMed RCT rozszerzenie**: dodać drugi query set (RCT + PICO-oriented queries) w `scripts/data/pubmed/pipeline/01_esearch_pmids.py`, drugi `corpus_version` (`pubmed-rct-v1`), oba wpisy w registry.
- **UMLS mapowania**: nie pełny indeks - lightweight lookup do query expansion w pre-retrieval.

## Metryki i weryfikacja

Po każdym nowym korpusie:

```bash
make eval-retrieval           # PubMedQA retrieval (baseline musi się nie pogorszyć)
make eval-nice-retrieval      # NICE guideline sample
make eval-statpearls-retrieval  # StatPearls case-based sample
make corpus-ablation          # pełne ablation study (pubmed_only vs +nice vs +statpearls vs all)
```

Raport: `reports/corpus_ablation.md` + `reports/corpus_ablation_summary.json`.
