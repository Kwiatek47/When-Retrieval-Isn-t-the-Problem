# PubMed Data Pipeline

This document describes the active PubMed data pipeline contract used by the RAG index. The old handoff and upload notes were moved to `archive/` because they describe a previous coordination stage, not the current runtime.

## Scope

Dataset version: `pubmed_reviews_v1`

Source: PubMed abstracts from the last 5 years, English review and systematic review records, with abstracts, excluding retracted publications.

Final processed run summary:

```text
input PubMed PMIDs: 990,391
metadata rows fetched: 982,019
final documents: 977,777
final chunks: 977,777
```

The full generated corpus is not stored in git. Runtime and indexing expect local artifacts under `data/processed/`, `data/embeddings/`, and `data/indexes/`.

## Active Files

```text
scripts/data/pubmed/pipeline/      PubMed fetch, clean, chunk, report, manifest scripts
scripts/data/pubmed/configs/       chunk schema and manifest example
scripts/data/pubmed/examples/      small repo-safe sample chunk
scripts/data/pubmed/reports/       repo-safe completed-run summaries
scripts/data/pubmed/requirements.txt
```

## Output Contract

The embedding and indexing pipeline expects:

```text
data/processed/chunks.parquet
data/processed/documents.parquet
data/processed/manifest.json
data/processed/cleaning_stats.json
```

Minimum chunk columns used by RAG indexing:

```text
chunk_id
text
pmid
title
doi
journal
year
publication_types
```

The canonical embedding text column is `text`.

## Runbook

Install the pipeline dependencies when working on data generation:

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/data/pubmed/requirements.txt
```

Typical flow:

```bash
python scripts/data/pubmed/pipeline/01_esearch_pmids.py ...
python scripts/data/pubmed/pipeline/02_fetch_metadata.py ...
python scripts/data/pubmed/pipeline/04_clean_dedupe_chunk.py ...
python scripts/data/pubmed/pipeline/05_quality_report.py ...
python scripts/data/pubmed/pipeline/06_write_manifest.py ...
```

Large outputs, raw PubMed data, logs, credentials, presigned URLs, and Parquet outputs remain local and are ignored by git.

## Storage Notes

The historical completed run used an S3 prefix for team handoff:

```text
s3://medical-rag-pubmed-207909165547-eu-central-1/processed/pubmed_reviews_v1/
```

Do not commit AWS credentials, SSH keys, `.env`, `.pem`, presigned URLs, raw XML, or generated Parquet files.
