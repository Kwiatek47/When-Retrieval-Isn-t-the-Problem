# Team Handoff

## Dane

Dataset:

```text
pubmed_reviews_v1
```

S3 path:

```text
s3://medical-rag-pubmed-207909165547-eu-central-1/processed/pubmed_reviews_v1/
```

Plik dla embeddingów:

```text
chunks.parquet
```

Kolumna do embeddingowania:

```text
text
```

Pola do cytowań:

```text
pmid
title
doi
journal
year
```

## Osoba od embeddingów

Zadania:

```text
1. Pobrać chunks.parquet z S3.
2. Sprawdzić schemat kolumn.
3. Wygenerować embeddingi dla kolumny text.
4. Zachować chunk_id i metadane cytowania.
5. Przekazać wynik do vector DB.
```

## Osoba od RAG

Zadania:

```text
1. Wybrać vector DB.
2. Zaimplementować retriever.
3. Dodać reranker.
4. Zwracać odpowiedzi z cytowaniami.
5. Testować retrieval na pytaniach medycznych.
```

## Osoba od fine-tuningu

Zadania:

```text
1. Dopasować styl odpowiedzi.
2. Ustalić format cytowań.
3. Dodać reguły bezpieczeństwa.
4. Przetestować LoRA/QLoRA tylko na danych instruktażowych, nie jako zamiennik RAG.
5. Przygotować ewaluację odpowiedzi.
```

## Ważna zasada

Fine-tuning nie zastępuje bazy wiedzy. Wiedza medyczna ma pochodzić z retrievalu, a model ma generować odpowiedź na podstawie znalezionych fragmentów.

