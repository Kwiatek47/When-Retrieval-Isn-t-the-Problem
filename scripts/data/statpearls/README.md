# StatPearls corpus (Etap A)

Pipeline buduje `chunks.parquet` z rozdziałów [StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK430685/) w NCBI Bookshelf.

## Wymagania

```bash
export NCBI_EMAIL="twoj.email@example.com"
# opcjonalnie przy większym wolumenie:
export NCBI_API_KEY="..."
```

Zależności: `httpx`, `tenacity`, `pyarrow` (już w głównym `requirements.txt` / venv projektu).

## Kroki

### 1. Manifest rozdziałów (~8967 chapter w StatPearls)

```bash
python3 scripts/data/statpearls/discover_chapters.py --limit 200
```

Wyjście: `data/raw/statpearls/chapter_manifest.jsonl`

### 2. Chunki + parquet

```bash
python3 scripts/data/statpearls/build_chunks.py --limit 50
```

Wyjście: `data/processed/statpearls/chunks.parquet`

### 3. Indeks Qdrant (jedna kolekcja, filtr `corpusVersion`)

```bash
export CORPUS_VERSION=statpearls_v1
python3 scripts/rag/01_build_index.py \
  --chunks data/processed/statpearls/chunks.parquet \
  --corpus-version statpearls_v1
```

W aplikacji:

```bash
export RAG_CORPUS_VERSION=statpearls_v1
```

### 4. Scalenie z PubMed (opcjonalnie)

```bash
python3 scripts/data/corpora/merge_parquet.py \
  --inputs data/processed/chunks.parquet data/processed/statpearls/chunks.parquet \
  --output data/processed/chunks_merged.parquet
```

## Makefile

```bash
make discover-statpearls LIMIT=200
make build-statpearls-chunks LIMIT=50
make index-statpearls
```

## Kontrakt chunków

- `chunk_id`: `statpearls:nbk499858:introduction:0`
- `publication_types`: `["Clinical Overview"]`
- `source`: `statpearls`
- `url`: `https://www.ncbi.nlm.nih.gov/books/NBK499858/`

Zgodne z `scripts/rag/01_build_index.py` i `chunks_schema_v1` (wymagane: `chunk_id`, `text`).
