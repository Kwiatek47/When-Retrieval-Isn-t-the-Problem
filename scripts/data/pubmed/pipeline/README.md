# Data Scripts

Te skrypty są częścią pipeline:

```text
01_esearch_pmids.py       -> pobranie listy PMID z PubMed
02_fetch_metadata.py      -> pobranie metadanych i abstraktów przez EFetch
04_clean_dedupe_chunk.py  -> czyszczenie, deduplikacja, chunking
05_quality_report.py      -> raport jakości
06_write_manifest.py      -> manifest datasetu
```

W repozytorium trzymamy skrypty i instrukcje, ale nie trzymamy wygenerowanych danych.

