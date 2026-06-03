# Korpusy wiedzy (preprocessing)

Struktura przed scaleniem do `data/processed/chunks.parquet`. Każdy korpus ma własny katalog w `data/raw/` i `data/interim/`.

End-to-end instrukcja dla NICE + PubMed, transferu lokalnych artefaktów, merge,
indeksowania Qdrant i testu cytowań jest w
[`docs/data/rag-corpus-runbook.md`](../../docs/data/rag-corpus-runbook.md).

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
| OpenFDA | `drug_label` | `OpenFDA` | (planowane) |

## Pola wspólne (dokument)

Zgodnie z notatką o `chunks.parquet` — na etapie preprocessingu w `documents.parquet` trzymamy metadane dokumentu; chunki (`chunk_id`, `header_path`, `text` fragmentów) powstaną dopiero przy merge + adaptacyjnym chunkingu.

- `document_id`, `source_type`, `source_name`
- `title`, `external_id` (np. TA1158), `source_url`, `published_at`
- `markdown_path` / ścieżki do artefaktów
- `metadata` (JSON: typ wytycznej, warunki, leki itd.)

## Pola wspólne (chunk)

Każdy korpus może mieć własne kolumny dodatkowe, ale merge do `data/processed/chunks.parquet`
opiera się na wspólnym minimum:

- `chunk_id`, `doc_id`, `text`, `title`
- `source`, np. `pubmed`, `nice`
- `url` / `source_url`
- `section`, `word_count`, `chunk_index`, `text_hash`

Kolumny specyficzne dla korpusu zostają jako opcjonalne:

- PubMed: `pmid`, `doi`, `journal`, `year`, `publication_types`, `is_review`
- NICE: `external_id`, `guidance_type`, `header_path`, `source_type`, `source_name`

## Merge lokalnych artefaktów

Wygenerowane parquet nie są trzymane w git. Zakładamy, że duże pliki są pobierane lokalnie
z dysku/S3/handoffu do katalogów `data/interim/{corpus}/` albo innej lokalnej ścieżki.

Przykład scalenia po pobraniu PubMed i wygenerowaniu NICE:

```bash
.venv/bin/python scripts/data/merge_corpora.py \
  --chunks pubmed=/path/to/pubmed/chunks.parquet nice=data/interim/nice/chunks.parquet \
  --out-chunks data/processed/chunks.parquet \
  --manifest-out data/processed/manifest.json
```

Dla ostrożniejszego indeksu NICE można podmienić ścieżkę na
`data/interim/nice/chunks_clinical.parquet`.

Skrypt waliduje wymagane kolumny, puste teksty i unikalność `chunk_id`, a brakujące kolumny
uzupełnia jako `null` przez wspólny schema union.
