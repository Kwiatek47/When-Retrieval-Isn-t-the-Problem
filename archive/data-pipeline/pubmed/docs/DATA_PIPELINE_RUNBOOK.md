# Data Pipeline Runbook

Ten dokument opisuje kroki uruchomienia pipeline na EC2 i zapisania wyników do S3.

## 1. Setup EC2

Przykład dla Amazon Linux 2023:

```bash
sudo dnf update -y
sudo dnf install -y python3 python3-pip git unzip awscli
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Struktura katalogów

```bash
mkdir -p data/interim/pmids
mkdir -p data/interim/metadata
mkdir -p data/processed/pubmed_reviews_v1
mkdir -p data/tmp/duckdb
```

## 3. Query PubMed

Finalne query:

```text
((review[pt] OR systematic[sb])
AND hasabstract
AND english[la]
AND ("2021/05/10"[dp] : "2026/05/10"[dp])
NOT (retracted publication[pt] OR retraction notice[pt]))
```

## 4. Pobranie PMID

```bash
python scripts/data/01_esearch_pmids.py \
  --date-from 2021-05-10 \
  --date-to 2026-05-10 \
  --out data/interim/pmids/pmids.txt \
  --out-dir data/interim/pmids \
  --history-out-dir data/interim/pmids \
  --max-window-count 9999 \
  --max-uid 60000000
```

W zakończonym przebiegu wynik był:

```text
review PMIDs: 975,220
systematic PMIDs: 209,393
final unique PMIDs: 990,391
```

## 5. Pobranie metadanych i abstraktów

```bash
nohup python scripts/data/02_fetch_metadata.py \
  --pmids data/interim/pmids/pmids.txt \
  --review-pmids data/interim/pmids/review_pmids.txt \
  --systematic-pmids data/interim/pmids/systematic_pmids.txt \
  --out data/interim/metadata/pubmed_metadata.parquet \
  --batch-size 200 \
  > metadata_fetch.log 2>&1 &
```

Monitoring:

```bash
tail -f metadata_fetch.log
ls -lh data/interim/metadata/pubmed_metadata.parquet
```

## 6. Cleaning, deduplication, chunking

Na większym zbiorze używamy DuckDB, bo zwykły Python/Pandas może zużyć za dużo RAM.

```bash
nohup python scripts/data/04_clean_dedupe_chunk.py \
  --input data/interim/metadata/pubmed_metadata.parquet \
  --out-documents data/processed/pubmed_reviews_v1/documents.parquet \
  --out-chunks data/processed/pubmed_reviews_v1/chunks.parquet \
  --stats-out data/processed/pubmed_reviews_v1/cleaning_stats.json \
  --duckdb-memory-limit 5GB \
  --duckdb-temp-dir data/tmp/duckdb \
  > clean_dedupe.log 2>&1 &
```

## 7. Quality report

```bash
cp data/interim/pmids/pmids.txt data/processed/pubmed_reviews_v1/pmids.txt

python scripts/data/05_quality_report.py \
  --documents data/processed/pubmed_reviews_v1/documents.parquet \
  --chunks data/processed/pubmed_reviews_v1/chunks.parquet \
  --pmids data/processed/pubmed_reviews_v1/pmids.txt \
  --out data/processed/pubmed_reviews_v1/data_quality_report.md
```

## 8. Upload do S3

```bash
export AWS_REGION=eu-central-1
export BUCKET=medical-rag-pubmed-207909165547-eu-central-1

aws s3 sync \
  data/processed/pubmed_reviews_v1/ \
  s3://$BUCKET/processed/pubmed_reviews_v1/
```

Sprawdzenie:

```bash
aws s3 ls s3://$BUCKET/processed/pubmed_reviews_v1/
```

Oczekiwane pliki:

```text
chunks.parquet
documents.parquet
pmids.txt
manifest.json
data_quality_report.md
data_quality_report.json
cleaning_stats.json
```

