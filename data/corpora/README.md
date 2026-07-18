# Korpusy wiedzy (preprocessing)

Struktura przed scaleniem do `data/processed/chunks.parquet`. Każdy korpus ma własny katalog w `data/raw/` i `data/interim/`.

End-to-end instrukcja dla NICE + PubMed, transferu lokalnych artefaktów, merge,
indeksowania Qdrant i testu cytowań jest w
[`docs/data/rag-corpus-runbook.md`](../../docs/data/rag-corpus-runbook.md).

Repozytorium wersji korpusów i benchmarków jest w
[`scripts/data/corpora/registry.json`](../../scripts/data/corpora/registry.json).

## Układ

```text
data/raw/{corpus}/              # pliki źródłowe (PDF, JSON, XML) — lokalne, gitignore
data/interim/{corpus}/          # wynik preprocessingu — lokalne, gitignore
  markdown/                     # surowy wynik parsera PDF
  markdown_clean/               # po usunięciu boilerplate (do chunkingu)
  documents.parquet             # rejestr dokumentów (poziom dokumentu, nie chunk)
  chunks.parquet                # domyślny wariant chunków danego korpusu
  chunks_*.parquet              # opcjonalne warianty, np. clinical/broad
  manifest.json                 # manifest datasetu tego korpusu

data/processed/                 # PÓŹNIEJ: jeden zbiór po merge wszystkich korpusów
  documents.parquet
  chunks.parquet
  manifest.json
```

## Korpusy

| Korpus | `source_type` | `source_name` | Skrypty |
|--------|---------------|---------------|---------|
| PubMed | `literature` | `PubMed` | `scripts/data/pubmed/pipeline/` |
| NICE | `guideline` | `NICE` | `scripts/data/nice/pipeline/` |
| StatPearls | `clinical_overview` | `StatPearls` | `scripts/data/statpearls/` |
| OpenFDA | `drug_label` | `OpenFDA` | (planowane) |

Aktualne wersje logiczne:

- `pubmed-reviews-v1` - PubMed reviews / systematic reviews.
- `nice-guidelines-v1` - NICE guideline corpus, wariant broad albo clinical
  zależnie od wybranego parquetu przy indeksowaniu.
- `statpearls-v1` - StatPearls chapters (NCBI Bookshelf), rozbite na
  sekcje/paragrafy.

## Pola wspólne (dokument)

Zgodnie z notatką o `chunks.parquet` — na etapie preprocessingu w `documents.parquet` trzymamy metadane dokumentu; chunki (`chunk_id`, `header_path`, `text` fragmentów) powstaną dopiero przy merge + adaptacyjnym chunkingu.

- `document_id`, `source_type`, `source_name`
- `title`, `external_id` (np. TA1158), `source_url`, `published_at`
- `markdown_path` / ścieżki do artefaktów
- `metadata` (JSON: typ wytycznej, warunki, leki itd.)

## Pola wspólne (chunk)

Canonical schema (`chunks_schema_v2`, patrz
[`scripts/data/corpora/schema.py`](../../scripts/data/corpora/schema.py))
używa jednolitych, stałych typów PyArrow. Wymagane kolumny (non-null):

- `chunk_id`, `doc_id`, `source`, `source_type`, `source_name`
- `title`, `text`, `text_hash`
- `word_count`, `chunk_index`, `corpus_version`

Opcjonalne, nullable:

- `url`, `section`, `header_path`, `parent_chunk_id`, `parent_word_count`
- `pmid`, `doi`, `journal`, `year`, `publication_date`, `publication_types`,
  `is_review`, `is_systematic_review`
- `external_id`, `guidance_type`
- `metadata` (JSON string dowolnych dodatkowych pól per corpus)

Które opcjonalne pola są wypełniane, zależy od adaptera:

- PubMed: `pmid`, `doi`, `journal`, `year`, `publication_types`, `is_review`,
  `is_systematic_review`, `url` (pubmed.ncbi.nlm.nih.gov/{pmid}/)
- NICE: `external_id`, `guidance_type`, `header_path`, `publication_date`,
  `publication_types` = ["Practice Guideline", "Guideline"]
- StatPearls: `external_id` (NBK id), `journal` = "StatPearls",
  `publication_types` = ["Clinical Overview"], `year`, `is_review` = true

## Merge lokalnych artefaktów

Wygenerowane parquet nie są trzymane w git. Zakładamy, że duże pliki są pobierane lokalnie
z dysku/Google Drive/handoffu do katalogów `data/interim/{corpus}/` albo innej lokalnej ścieżki.

Merge korzysta z registry (`scripts/data/corpora/registry.json`) - jedno miejsce
z definicjami wersji, dedupe_priority, ścieżek per corpus i adapterów:

```bash
# domyślnie: wszystkie korpusy z default_include=true, których local_chunks_path istnieje
.venv/bin/python scripts/data/corpora/build_processed_chunks.py

# jawnie wybrane korpusy
.venv/bin/python scripts/data/corpora/build_processed_chunks.py \
  --corpora pubmed_reviews_v1 nice_guidelines_v1 statpearls_v1 \
  --out-chunks data/processed/chunks.parquet \
  --manifest-out data/processed/manifest.json

# clinical wariant NICE zamiast broad
.venv/bin/python scripts/data/corpora/build_processed_chunks.py \
  --corpora pubmed_reviews_v1 nice_guidelines_clinical_v1
```

Skrypt uruchamia per-corpus adapter (canonical schema mapping), robi
cross-corpus dedupe po `pmid` → `doi` → opcjonalnie `title_hash`
(`--dedupe-titles`) z priorytetem z `dedupe_priority`, waliduje unikalność
`chunk_id`, puste teksty i deterministyczny schemat PyArrow, potem zapisuje
`data/processed/chunks.parquet` i `manifest.json` z licznikami per source /
per corpus_version i statystyką dedupe.

## Benchmark NICE

NICE nie ma PMID, więc benchmarki PubMedQA i PubMed retrieval nie są dobrą
miarą dla indeksu NICE-only. Do szybkiego testu korpusu NICE używaj:

```bash
make eval-nice-retrieval
make eval-nice-rag
```

Te sample używają ground truth po `documentId`, np. `nice-amr1`,
`nice-ng127`, `nice-cg150`, `nice-ng28` i `nice-ng253`.
