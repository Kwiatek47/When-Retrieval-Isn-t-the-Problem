# Repo Safety Checklist

Przed wrzuceniem na GitHub sprawdź:

```bash
git status
git diff --cached
find . -name "*.pem" -o -name ".env" -o -name "*.parquet" -o -name "*.log"
```

Nie powinno być:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
session token
private key
SSH key
presigned URL
chunks.parquet
documents.parquet
raw XML
interim data
```

Można wrzucić:

```text
dokumentację
schematy
małe raporty jakości
statystyki czyszczenia
skrypty pipeline
przykładowy manifest bez sekretów
```

