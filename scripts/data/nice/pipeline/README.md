# NICE — preprocessing (bez chunków)

Etap przygotowania danych pod późniejsze scalenie z PubMed i OpenFDA. **Nie tworzy** `data/processed/chunks.parquet`.

## Przepływ

```text
nice.org.uk/guidance/published  (~2561 pozycji)
        |
        v  00_fetch_published_catalog.py
data/interim/nice/guidance_catalog.jsonl
        |
        v  00_download_pdfs.py --all-published
data/raw/nice/pdf/*.pdf
        |
        v  (opcjonalnie pojedyncze ID zamiast --all-published)
data/raw/nice/pdf/*.pdf
        |
        v  01_pdf_to_markdown.py
data/interim/nice/markdown/*.md
        |
        v  01b_clean_markdown.py
data/interim/nice/markdown_clean/*.md
        |
        v  02_build_documents.py
data/interim/nice/documents.parquet
        |
        v  03_write_manifest.py
data/interim/nice/manifest.json
```

## Uruchomienie

```bash
.venv/bin/pip install -r scripts/data/nice/requirements.txt

# Wszystkie opublikowane wytyczne (bez ręcznego wpisywania ID)
.venv/bin/python scripts/data/nice/pipeline/00_fetch_published_catalog.py
.venv/bin/python scripts/data/nice/pipeline/00_download_pdfs.py --all-published --skip-existing --delay 0.5

# Albo jednym skryptem (kroki 1-5):
bash scripts/data/nice/run_bulk_preprocessing.sh

# Test na 5 dokumentach:
.venv/bin/python scripts/data/nice/pipeline/00_download_pdfs.py --all-published --limit 5 --dry-run

# PDF -> Markdown
.venv/bin/python scripts/data/nice/pipeline/01_pdf_to_markdown.py
.venv/bin/python scripts/data/nice/pipeline/01b_clean_markdown.py

# Rejestr dokumentów
.venv/bin/python scripts/data/nice/pipeline/02_build_documents.py
.venv/bin/python scripts/data/nice/pipeline/03_write_manifest.py
```

## Kolejny etap (osobno)

Adaptacyjny chunking i merge do wspólnego `chunks.parquet`: `scripts/embeddings/nice_markdown_to_parquet.py` — uruchamiaj na `markdown_clean/`, dopiero po ustaleniu selekcji korpusów.

Konserwatywny wariant kliniczny:

```bash
.venv/bin/python scripts/embeddings/nice_markdown_to_parquet.py \
  --input_dir data/interim/nice/markdown_clean \
  --documents data/interim/nice/documents.parquet \
  --output_file data/interim/nice/chunks_clinical.parquet \
  --section-policy clinical-core \
  --include-prefix ng cg amr mpg ph csg sg sc
```

Domyślny broad-controlled wariant do pełnego RAG:

```bash
.venv/bin/python scripts/embeddings/nice_markdown_to_parquet.py \
  --input_dir data/interim/nice/markdown_clean \
  --documents data/interim/nice/documents.parquet \
  --output_file data/interim/nice/chunks.parquet \
  --section-policy clinical-core \
  --include-prefix ng cg amr mpg ph csg sg sc ta htg hst
```
