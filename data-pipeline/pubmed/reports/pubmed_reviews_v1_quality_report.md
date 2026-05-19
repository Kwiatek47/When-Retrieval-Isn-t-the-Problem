# PubMed Reviews V1 Quality Report

This is a repo-safe summary of the completed pipeline run. It does not contain raw data.

## Counts

```text
input PMID count: 990,391
metadata rows fetched: 982,019
cleaned rows: 979,499
final documents: 977,777
final chunks: 977,777
```

## Removed Records

```text
removed by cleaning filters: 2,520
duplicates by PMID: 0
duplicates by DOI: 57
duplicates by title hash: 1,665
duplicates by content hash: 0
```

## Output

```text
documents.parquet: 1.7 GB
chunks.parquet: 936 MB
pmids.txt: 8.6 MB
```

## Notes

The dataset is intended for the embedding and RAG stages. The embedding stage should process the `text` column from `chunks.parquet` and preserve citation metadata.

