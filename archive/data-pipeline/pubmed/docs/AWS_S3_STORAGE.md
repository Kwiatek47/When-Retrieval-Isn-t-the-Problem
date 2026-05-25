# AWS S3 Storage

## Bucket

```text
medical-rag-pubmed-207909165547-eu-central-1
```

Region:

```text
eu-central-1
```

Dataset prefix:

```text
processed/pubmed_reviews_v1/
```

## Security

Bucket powinien mieć:

```text
Block Public Access: ON
Versioning: Enabled
Server-side encryption: AES256
No public ACLs
No public bucket policy
```

## Minimalne zasady

```text
Nie commitujemy AWS keys.
Nie commitujemy plików .pem.
Nie commitujemy presigned URLs.
Nie robimy bucketu publicznego.
Do EC2 używamy IAM role zamiast kluczy w plikach.
```

## Udostępnianie zespołowi

Najbezpieczniejsze opcje:

```text
1. IAM user/role z ograniczonym dostępem do jednego prefixu S3.
2. Jednorazowy presigned URL z krótkim czasem życia.
3. Kopia tylko chunks.parquet, jeśli osoba od embeddingów nie potrzebuje documents.parquet.
```

Presigned URL jest wygodny, ale nie powinien trafiać do repozytorium.

