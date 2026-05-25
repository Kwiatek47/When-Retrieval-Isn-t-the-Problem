# Message To Embedding Owner

Dataset: `pubmed_reviews_v1`

Plik dla embeddingów:

```text
s3://medical-rag-pubmed-207909165547-eu-central-1/processed/pubmed_reviews_v1/chunks.parquet
```

Embeddinguj kolumnę:

```text
text
```

Manifest:

```text
s3://medical-rag-pubmed-207909165547-eu-central-1/processed/pubmed_reviews_v1/manifest.json
```

Raport jakości:

```text
s3://medical-rag-pubmed-207909165547-eu-central-1/processed/pubmed_reviews_v1/data_quality_report.md
```

Do cytowań używaj pól:

```text
pmid, title, doi, journal, year
```

Jeśli dostajesz presigned URL, nie wrzucaj go do repozytorium.

