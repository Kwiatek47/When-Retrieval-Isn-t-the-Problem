# GitHub Upload

Ten folder można bezpiecznie wrzucić do repozytorium jako część odpowiedzialna za dane PubMed.

## Proponowana lokalizacja w repo

```text
data-pipeline/pubmed/
```

## Kopiowanie do repo

Przykład:

```bash
cp -R pubmed-data-pipeline-repo-safe /path/to/repo/data-pipeline/pubmed
cd /path/to/repo
```

## Kontrola przed commitem

```bash
find data-pipeline/pubmed \
  \( -name "*.parquet" -o -name "*.pem" -o -name ".env" -o -name "*.log" -o -name "*.xml" -o -name "*.gz" -o -name "*.zip" \) \
  -print
```

Ta komenda nie powinna nic wypisać.

Sprawdź też:

```bash
git status
git diff --cached
```

## Commit

```bash
git add data-pipeline/pubmed
git commit -m "Add PubMed data pipeline docs and scripts"
git push
```

## Czego nie wrzucać

```text
chunks.parquet
documents.parquet
raw PubMed XML
metadata parquet
AWS credentials
SSH keys
presigned URLs
local logs
```

