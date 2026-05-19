# PubMed Data Pipeline for Medical RAG

Repo-safe package for the data part of a medical chatbot project based on an open-source LLM, RAG, embeddings, and a vector database.

This folder contains documentation, commands, schemas, and lightweight helper scripts. It intentionally does not contain raw PubMed data, processed Parquet files, AWS credentials, SSH keys, `.env` files, or presigned URLs.

## Scope

Dataset version: `pubmed_reviews_v1`

Data source: PubMed abstracts from the last 5 years, filtered to English review and systematic review records with abstracts and excluding retracted publications.

Final storage target:

```text
s3://medical-rag-pubmed-207909165547-eu-central-1/processed/pubmed_reviews_v1/
```

Final generated files:

```text
chunks.parquet
documents.parquet
pmids.txt
manifest.json
data_quality_report.md
data_quality_report.json
cleaning_stats.json
```

Final processed count from the completed run:

```text
input PubMed PMIDs: 990,391
metadata rows fetched: 982,019
final documents: 977,777
final chunks: 977,777
```

## What To Commit

Safe to commit:

```text
README.md
docs/
configs/
scripts/
reports/
examples/
.gitignore
requirements.txt
```

Do not commit:

```text
*.parquet
*.pem
.env
data/
logs/
presigned URLs
AWS keys
raw PubMed XML dumps
```

## Team Contract

The embedding owner should use:

```text
s3://medical-rag-pubmed-207909165547-eu-central-1/processed/pubmed_reviews_v1/chunks.parquet
```

Embedding column:

```text
text
```

Citation metadata:

```text
pmid, title, doi, journal, year
```

