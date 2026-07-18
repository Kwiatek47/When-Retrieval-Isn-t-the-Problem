# NICE guidelines — preprocessing

Zgodnie z notatką o architekturze RAG: najpierw **layout-aware parsing → Markdown + metadane dokumentu**, potem (na końcu) adaptacyjny chunking i jeden `chunks.parquet` łączący PubMed, wytyczne i etykiety.

Praktyczny przepływ end-to-end od lokalnych parquetów do indeksu Qdrant i testu
cytowań w chatbocie jest w [`rag-corpus-runbook.md`](rag-corpus-runbook.md).

## Pobieranie wszystkich wytycznych (bez ręcznych ID)

Katalog (~2561 opublikowanych pozycji) jest pobierany ze strony NICE (`__NEXT_DATA__`, paginacja `?pa=1&ps=500`):

```bash
.venv/bin/python scripts/data/nice/pipeline/00_fetch_published_catalog.py
.venv/bin/python scripts/data/nice/pipeline/00_download_pdfs.py --all-published --skip-existing --delay 0.5
```

Nie każda pozycja ma pełny PDF (np. część HTG/MIB) — takie wpisy trafiają do `pdf_download_report.json` jako `no_pdf`. Resumowanie: ponowne uruchomienie z `--skip-existing`.

Pełny preprocessing (bez `chunks.parquet`):

```bash
bash scripts/data/nice/run_bulk_preprocessing.sh
```

## Co jest teraz

| Etap | Artefakt | Status |
|------|----------|--------|
| PDF źródłowy | `data/raw/nice/pdf/` | `00_download_pdfs.py` lub lokalnie |
| Markdown surowy | `data/interim/nice/markdown/` | po parserze |
| Markdown oczyszczony | `data/interim/nice/markdown_clean/` | wejście do chunkingu NICE |
| Rejestr dokumentów | `data/interim/nice/documents.parquet` | `source_type=guideline`, `source_name=NICE` |
| Chunki NICE broad-controlled | `data/interim/nice/chunks.parquet` | domyślny artefakt korpusu NICE do pełnego RAG |
| Chunki NICE clinical | `data/interim/nice/chunks_clinical.parquet` | konserwatywny guideline-first wariant |
| Manifest korpusu | `data/interim/nice/manifest.json` | opis datasetu |
| Chunki globalne | `data/processed/chunks.parquet` | **jeszcze nie** |

## Schemat `documents.parquet`

Pola z `scripts/data/nice/configs/documents_schema_v1.json`. To odpowiednik „poziomu dokumentu” przed polami chunków (`header_path`, `chunk_id`, wektory).

## Łączenie korpusów (później)

```text
data/interim/pubmed/     -> documents + chunks (osobny pipeline)
data/interim/nice/       -> documents + markdown
data/interim/openfda/    -> (planowane)

merge ->
  data/processed/documents.parquet
  data/processed/chunks.parquet
  data/processed/manifest.json
```

W `chunks.parquet` każdy wiersz dostanie wspólne kolumny z notatki: `source_type`, `source_name`, `document_id`, `text`, `header_path`, `section_name`, `metadata`, później `dense_vector` / `sparse_vector`.

## Zalecana selekcja pod RAG

Nie indeksuj surowych `markdown/`. Najpierw uruchom czyszczenie:

```bash
.venv/bin/python scripts/data/nice/pipeline/01b_clean_markdown.py
.venv/bin/python scripts/data/nice/pipeline/02_build_documents.py
```

Do konserwatywnego wariantu klinicznego użyj dokumentów stricte guideline/prescribing:

```bash
.venv/bin/python scripts/embeddings/nice_markdown_to_parquet.py \
  --input_dir data/interim/nice/markdown_clean \
  --documents data/interim/nice/documents.parquet \
  --output_file data/interim/nice/chunks_clinical.parquet \
  --section-policy clinical-core \
  --include-prefix ng cg amr mpg ph csg sg sc
```

Domyślny wariant do pełnego RAG to broad-controlled: dodaje technology appraisals,
healthtech i highly specialised technologies, ale nadal filtruje sekcje przez
`clinical-core`, żeby nie ładować kosztowo-organizacyjnego boilerplate.

```bash
.venv/bin/python scripts/embeddings/nice_markdown_to_parquet.py \
  --input_dir data/interim/nice/markdown_clean \
  --documents data/interim/nice/documents.parquet \
  --output_file data/interim/nice/chunks.parquet \
  --section-policy clinical-core \
  --include-prefix ng cg amr mpg ph csg sg sc ta htg hst
```

Na start dalej wykluczaj `qs mib es esnm esuom esmpb nr`, bo to częściej standardy
jakości, briefingi albo materiały wtórne, a nie rdzeń wiedzy klinicznej.

## Przenoszenie i merge z PubMed

`data/interim/nice/chunks.parquet` jest odpowiednikiem korpusowego artefaktu PubMed:
można go spakować, wrzucić na dysk/S3 i pobrać lokalnie przed merge. Nie trzeba
commitować generated parquet do repo. Jeśli chcesz ostrożniejszy indeks, użyj
`chunks_clinical.parquet` zamiast `chunks.parquet`.

Po pobraniu PubMed `chunks.parquet` i wygenerowaniu/pobraniu NICE:

```bash
.venv/bin/python scripts/data/merge_corpora.py \
  --chunks pubmed=/path/to/pubmed/chunks.parquet nice=data/interim/nice/chunks.parquet \
  --out-chunks data/processed/chunks.parquet \
  --manifest-out data/processed/manifest.json
```

Wspólne kolumny pod RAG to `chunk_id`, `doc_id`, `source`, `title`, `text`, `url`,
`section`, `word_count`, `text_hash` i `chunk_index`. Kolumny specyficzne dla
NICE (`external_id`, `guidance_type`, `header_path`) oraz PubMed (`pmid`, `doi`,
`journal`, `year`, `publication_types`) zostają opcjonalne i są uzupełniane jako
`null`, gdy dany korpus ich nie ma.
