# Audyt Architektury Medical RAG

Data audytu: 2026-05-18  
Projekt: `Architektura-multiagentowego-systemu-diagnostycznego`  
Zakładane modele generujące: open-source LLM 7B oraz 26B/27B  
Zakładany typ systemu: prosty medyczny chatbot RAG oparty o Vector DB, bez rozbudowanej warstwy agentowej

## 1. Executive Summary

Obecna architektura jest sensownym MVP medycznego RAG-a:

- FastAPI + UI czatu,
- lokalny LLM przez Ollama,
- osobny `embedding-service`,
- Qdrant jako Vector DB,
- MedCPT jako biomedical dense embedding,
- BM25 sparse retrieval,
- hybrid retrieval przez RRF,
- opcjonalny cross-encoder reranking,
- walidacja cytowań,
- prosta ocena groundedness.

Największe problemy nie są w samym pomyśle, tylko w kilku miejscach wykonania:

1. Indeksowanie pełnego korpusu jest zbyt pamięciożerne, bo skrypt czyta całe pliki Parquet do pamięci.
2. Chunking jest zbyt prosty: jeden abstrakt PubMed to jeden chunk, a najdłuższe chunki są bardzo długie.
3. Query rewrite zastępuje praktycznie oryginalne pytanie, zamiast robić multi-query retrieval.
4. Metadata filtering jest obecnie opisany w danych, ale nie jest realnie używany jako część retrievalu.
5. `embedding-service` miesza liczenie embeddingów z odpytywaniem Qdranta.
6. Ewaluacja jest za mała, żeby bezpiecznie stroić retrieval i prompt.

Najkrótsza droga do stabilnej wersji:

```text
streaming indexer
+ metadata filtering
+ evidence filtering
+ multi-query retrieval
+ BM25 top50 -> MedCPT/cross-encoder rerank -> top5
+ osobne profile dla 7B i 26B
+ realny eval set minimum 100-300 pytań
```

Nie rekomenduję na tym etapie budowania złożonego systemu multiagentowego. Najpierw trzeba ustabilizować prosty, mierzalny RAG.

## 2. Obecny Stan Systemu

### 2.1. Aplikacja

Aktualny przepływ:

```text
Użytkownik
  -> UI / static
  -> FastAPI /api/chat
  -> PreRetriever
  -> Retriever
     -> embedding-service
     -> Qdrant
  -> PostRetriever
  -> Ollama
  -> odpowiedź z cytowaniami
```

Główne komponenty:

- `app/rag/pre_retrieval.py` - normalizacja pytania, query rewrite, proste filtry,
- `app/rag/retrieval.py` - retrievery do Qdranta/embedding-service,
- `app/rag/post_retrieval.py` - reranking, deduplikacja, budowa kontekstu,
- `app/rag/citation_validation.py` - walidacja cytowań `[S1]`, `[S2]`,
- `app/rag/answer_quality.py` - heurystyczna ocena groundedness,
- `services/embedding-service` - MedCPT query/document encoder i endpoint hybrid query,
- `scripts/rag/01_build_index.py` - budowa indeksu Qdrant.

### 2.2. Dane

Lokalna paczka danych:

```text
pubmed_reviews_v1_local/data/processed/chunks.parquet
pubmed_reviews_v1_local/data/processed/documents.parquet
pubmed_reviews_v1_local/data/embeddings/embeddings_shard_0.parquet
pubmed_reviews_v1_local/data/embeddings/embeddings_shard_1.parquet
```

Jakość danych:

- dokumenty: `977777`,
- chunki: `977777`,
- duplicate `chunk_id`: `0`,
- duplicate PMID: `0`,
- empty text rows: `0`,
- średnia długość chunku: `221.58` słów,
- mediana długości chunku: `214` słów,
- maksimum: `4793` słowa.

Embeddingi:

- model: `medcpt-ncbi-v1`,
- document encoder: `ncbi/MedCPT-Article-Encoder`,
- query encoder: `ncbi/MedCPT-Query-Encoder`,
- wymiar: `768`,
- shard 0: `488889` embeddingów,
- shard 1: `488888` embeddingów,
- łącznie: `977777`,
- status walidacji shardów: `PASS`.

### 2.3. Retrieval

Obecnie są dwie ścieżki:

1. Domyślna:

```text
app -> embedding-service -> Qdrant hybrid query
```

2. Alternatywna:

```text
app -> QdrantHybridKnowledgeRetriever -> Qdrant
```

Domyślna ścieżka robi:

```text
MedCPT query embedding
+ BM25 sparse query vector
+ Qdrant RRF fusion
+ opcjonalny cross-encoder rerank
+ top_k do promptu
```

Domyślne parametry:

```text
RAG_CANDIDATE_K=50
RAG_TOP_K=5
RAG_MAX_CONTEXT_CHARS=8000
CROSS_ENCODER_MODEL=ncbi/MedCPT-Cross-Encoder
```

To jest dobry kierunek. Problemem jest brak wystarczającej kontroli jakości retrieved evidence.

## 3. Najważniejsze Ryzyka

### 3.1. Indeksowanie nie jest gotowe na pełny korpus

`scripts/rag/01_build_index.py` używa:

```python
table = pq.read_table(path)
rows = table.to_pylist()
```

To oznacza, że dla prawie miliona chunków i dużych embeddingów wszystko ląduje w pamięci procesu.

Ryzyko:

- OOM przy pełnym indeksowaniu,
- trudne resume po przerwaniu,
- brak naturalnej obsługi shardów,
- konieczność tworzenia dużego scalonego `embeddings.parquet`,
- słaba powtarzalność buildu.

Rekomendacja:

Przepisać index builder na streaming:

```text
czytaj chunks.parquet batchami
czytaj embeddings_shard_*.parquet batchami
join po chunk_id w kontrolowanym buforze
buduj BM25 globalnie w pass 1
upsertuj Qdrant batchami w pass 2
zapisuj manifest i checkpoint
```

Minimalny wariant:

```text
pass 1: policz BM25 stats po całym corpusie
pass 2: dla każdego shardu embeddingów znajdź metadane chunków i upsertuj do Qdranta
```

### 3.2. Chunking jest zbyt płaski

Manifest mówi, że chunking to:

```text
one_pubmed_abstract_equals_one_chunk
```

To jest OK dla MVP, ale nie jest wystarczające produkcyjnie.

Problem:

- część abstraktów jest bardzo długa,
- MedCPT document encoder ma `MEDCPT_DOCUMENT_MAX_LENGTH=512`,
- długie chunki będą ucinane na etapie embeddingu,
- długi chunk w promptcie może zabrać miejsce kilku lepszym źródłom,
- model 7B jest szczególnie wrażliwy na szum w kontekście.

Rekomendacja:

Zostawić obecny abstrakt jako jednostkę źródłową, ale dodać warstwę chunków potomnych dla outlierów:

```text
document_id = PMID
parent_chunk_id = pubmed:{pmid}:abstract
chunk_id = pubmed:{pmid}:abstract:{n}
section = abstract
chunk_index = n
```

Reguła:

- jeśli abstract <= ok. 350-450 słów, zostaje jednym chunkiem,
- jeśli abstract jest dłuższy, dzielić po zdaniach do limitu tokenów,
- zachować overlap tylko mały, np. 1 zdanie, nie agresywne 200-token overlap.

### 3.3. Query rewrite traci oryginalne pytanie

`PreRetriever` tworzy:

```text
search_queries = [normalized_query, rewritten_query]
```

Ale retriever używa praktycznie tylko ostatniego zapytania:

```python
return search_queries[-1]
```

Problem:

- jeśli rewrite zgubi nazwę leku, skrót, liczbę, populację lub język pacjenta, retrieval się pogorszy,
- dla modeli 7B rewrite może być niestabilny,
- brak merge wyników z oryginalnego i przepisanego pytania.

Rekomendacja:

Nie zastępować query. Robić multi-query retrieval:

```text
query A: oryginalne pytanie
query B: rewritten query
query C: opcjonalnie medyczne synonimy/expanded query

dla każdego query:
  dense retrieval
  sparse retrieval

merge:
  RRF albo weighted RRF
  dedupe po chunkId/documentId
```

Wariant prosty:

```text
original_query weight = 1.0
rewritten_query weight = 0.8
```

### 3.4. Metadata filtering jest niewykorzystany

Dane mają metadane:

```text
pmid
doi
journal
year
publicationTypes
isReview
isSystematicReview
corpusVersion
```

Ale retrieval nie używa ich jako realnych filtrów lub boostów.

W medycznym RAG-u to duża strata. Paper o koreańskim chatbocie medycznym na open-weight LLM-ach pokazał, że zwykły RAG nie zawsze poprawia wyniki, a RAG z metadata filtering dawał istotną poprawę dla większości modeli. Testowane były m.in. modele klasy 7B/8B oraz 20B/27B.

Rekomendacja:

Dodać metadata filtering jako obowiązkową warstwę:

```text
hard filters:
  corpusVersion == current
  source in allowed_sources
  year >= min_year, jeśli pytanie dotyczy aktualnych zaleceń

soft boosts:
  isSystematicReview
  publicationTypes contains "Systematic Review"
  publicationTypes contains "Practice Guideline"
  publicationTypes contains "Guideline"
  publicationTypes contains "Review"
  nowszy rok publikacji
```

Nie należy ślepo boostować samej świeżości. Dla mechanizmów biologicznych starsze źródło może być nadal dobre. Dla leczenia, wytycznych i bezpieczeństwa leków świeżość i typ publikacji są ważniejsze.

### 3.5. `embedding-service` ma za dużo odpowiedzialności

Obecnie `embedding-service`:

- liczy embeddingi dokumentów,
- liczy embeddingi query,
- koduje BM25 query,
- odpytuje Qdrant,
- mapuje wyniki do API.

To działa, ale zaciera granice.

Rekomendacja docelowa:

```text
embedding-service:
  /embed/query
  /embed/documents

retrieval-service albo main API:
  BM25 query vector
  Qdrant query
  metadata filtering
  RRF
  evidence filtering
  reranking
```

Minimalnie:

- zostawić endpoint `/embed/hybrid/query` dla kompatybilności,
- nowy kod rozwijać w `QdrantHybridKnowledgeRetriever`,
- docelowo traktować `embedding-service` jako stateless encoder.

### 3.6. Ewaluacja jest zbyt mała

Obecny eval set ma 5 przypadków.

To jest smoke test, nie benchmark.

Rekomendacja:

Zbudować eval set:

```text
100-300 pytań minimum
podział na kategorie kliniczne
gold PMIDs/chunkIds
ocena ekspercka albo półautomatyczna
osobno pytania proste i złożone
osobno pytania aktualne klinicznie
osobno pytania, gdzie należy odmówić odpowiedzi
```

Metryki:

```text
retrieval:
  Recall@5, Recall@10, Recall@50
  MRR
  nDCG@10
  evidence precision@5
  latency p50/p95

generation:
  citation precision
  citation recall
  unsupported claim rate
  no-answer accuracy
  safety score
```

## 4. Usprawnienia Z Paperów I Praktyki Medical RAG

Poniżej są usprawnienia, które warto dodać do obecnego planu. Celowo pomijam ciężkie i egzotyczne podejścia. To są praktyki pasujące do prostego chatbota RAG + Vector DB.

### 4.1. Metadata filtering przed augmentacją

Wniosek:

```text
Nie każdy retrieved chunk powinien trafić do promptu.
```

W badaniu koreańskiego chatbota medycznego na open-weight LLM-ach porównywano baseline, RAG-only i RAG z metadata filtering. RAG-only potrafił nie dawać istotnej poprawy albo pogarszać wynik, natomiast RAG + metadata filtering poprawiał większość testowanych modeli. To jest bezpośrednio istotne dla naszych modeli 7B i 26B/27B.

Do wdrożenia:

```text
retrieved candidates
  -> metadata filter
  -> evidence filter
  -> reranker
  -> top_k context
```

Przykładowa polityka:

```text
dla pytań o leczenie:
  preferuj systematic reviews, guidelines, reviews
  boostuj nowsze źródła
  odrzuć bardzo stare źródła, jeśli są alternatywy

dla pytań o patofizjologię:
  nie wymuszaj najnowszego roku
  preferuj review/systematic review

dla pytań o bezpieczeństwo leków:
  boostuj adverse events, contraindications, guidelines, systematic reviews
```

### 4.2. Hybrid retrieval zostaje, ale trzeba go mierzyć

Paper o efektywnym i reprodukowalnym biomedical QA wskazuje, że BM25 + MedCPT dobrze balansuje jakość i koszt. Szczególnie praktyczny wariant to pobranie większej puli kandydatów tanim lexical retrieverem, a potem reranking biomedical modelem.

Obecna architektura ma już:

- BM25 sparse,
- MedCPT dense,
- Qdrant RRF,
- MedCPT cross-encoder reranker.

Nie trzeba wymyślać nowej architektury. Trzeba dodać tryby porównawcze:

```text
tryb A: dense MedCPT top50 -> cross-encoder top5
tryb B: BM25 top50 -> cross-encoder/MedCPT top5
tryb C: Qdrant RRF dense+sparse top50 -> cross-encoder top5
tryb D: original query + rewritten query -> RRF merge -> top50 -> cross-encoder top5
```

Decyzja ma być na metrykach, nie na intuicji.

### 4.3. Evidence filtering po retrievalu

Medical RAG często psuje odpowiedź przez słaby retrieval albo zły wybór dowodów. Paper z dużą ekspercką oceną medical RAG pokazuje, że standardowy RAG potrafi obniżać factuality/completeness, jeśli evidence retrieval i evidence selection są słabe. Proste strategie typu evidence filtering i query reformulation istotnie pomagają.

Do wdrożenia:

Każdy kandydat po retrievalu dostaje dodatkową ocenę:

```text
query-term coverage
metadata priority
retrieval score
reranker score
source freshness
publication type
```

Odrzucamy:

```text
score poniżej progu
brak pokrycia kluczowych terminów z pytania
niewłaściwy typ publikacji dla pytania klinicznego
źródło z innego corpusVersion
duplikaty tego samego PMID bez dodatkowej wartości
```

Ważne:

Evidence filtering nie może być zbyt agresywny. Jeśli zostaje mniej niż 2-3 źródła, system powinien oznaczyć `low_evidence` i odpowiedzieć ostrożniej albo odmówić odpowiedzi.

### 4.4. Nie zwiększać kontekstu bez potrzeby

MedRAG/MIRAGE oraz prace o long-context medical RAG pokazują problem `lost-in-the-middle`: ważne źródło może zostać pominięte przez model, jeśli jest wrzucone w środek dużego kontekstu.

Wniosek dla nas:

```text
RAG_TOP_K=5 jest rozsądne.
Nie zwiększać domyślnie do 10-20.
```

Lepsza strategia:

- poprawić ranking,
- skrócić chunki,
- zachować tylko najlepsze fragmenty,
- układać kontekst według siły dowodu,
- w promptcie wymusić cytowania przy każdym claimie.

Dla modelu 7B:

```text
top_k = 4-5
max_context_chars = 6000-8000
krótkie odpowiedzi
zero temperature albo bardzo niskie temperature
```

Dla modelu 26B/27B:

```text
top_k = 5-8
max_context_chars = 8000-12000, jeśli model stabilnie obsługuje dłuższy kontekst
dalej unikać wrzucania szumu
```

### 4.5. Query reformulation, ale bez utraty oryginału

Medical RAG korzysta z query reformulation, ale nie powinien ufać wyłącznie rewrite.

Rekomendowany prosty wariant:

```text
queries = [
  original_user_query,
  normalized_query,
  rewritten_medical_query
]
```

Każde query idzie przez retrieval, potem wyniki są scalane.

Nie robić na start:

- wieloetapowego chain-of-thought w promptach użytkownika,
- skomplikowanej agentowości,
- wielu iteracji dla każdego pytania.

Można dodać później:

- jedną iterację follow-up query tylko dla złożonych pytań,
- HyDE tylko jako fallback, gdy top wyniki są słabe.

### 4.6. Iterative RAG tylko jako tryb dla trudnych pytań

i-MedRAG pokazuje, że follow-up queries pomagają przy złożonych pytaniach, gdzie potrzeba kilku rund szukania. To ma sens, ale nie jako default dla prostego chatbota.

Proponowany gating:

```text
jeśli pytanie proste:
  standard RAG

jeśli pytanie zawiera wiele warunków:
  choroba + lek + populacja + przeciwwskazanie + outcome
  -> pozwól na 1 dodatkowe follow-up query

jeśli retrieval confidence jest niski:
  -> 1 dodatkowe query reformulation

limit:
  max 2 rundy retrievalu
```

Dla 7B:

- iterative RAG raczej sterowany regułami,
- model 7B nie powinien sam swobodnie planować wielu zapytań.

Dla 26B:

- można pozwolić na jedną lepszą reformulację,
- nadal kontrolować liczbę rund i koszt.

### 4.7. Rationale-guided retrieval jako inspiracja, nie pierwszy krok

RAG^2 używa rationale-guided retrieval i filtruje nieinformatywne fragmenty. To jest ciekawy kierunek, ale dla obecnego projektu może być zbyt ciężki.

Prosty odpowiednik bez komplikowania:

```text
1. wygeneruj krótkie "information need" bez chain-of-thought
2. użyj go jako dodatkowego query
3. filtruj retrieved chunks po pokryciu tego information need
```

Przykład:

```text
User query:
"Czy SGLT2 są bezpieczne w CKD i niewydolności serca?"

Information need:
"SGLT2 inhibitors effects on kidney outcomes, heart failure hospitalization, mortality, adverse events in CKD or type 2 diabetes"
```

Nie zapisywać ani nie pokazywać użytkownikowi prywatnego rozumowania.

### 4.8. Źródła: guidelines i reviews są ważniejsze niż pojedyncze abstrakty

Thyro-GenAI zwraca uwagę na różnicę między pojedynczymi artykułami a materiałami syntetycznymi typu guidelines/textbooks. W prostym medycznym chatbocie warto preferować źródła, które są bliższe praktyce klinicznej.

Dla obecnego PubMed corpus:

```text
boost:
  Practice Guideline
  Guideline
  Systematic Review
  Meta-Analysis
  Review

neutral:
  Randomized Controlled Trial
  Clinical Trial

ostrożnie:
  case reports
  letters
  editorials
```

Jeśli w przyszłości dodamy inne korpusy:

```text
guidelines/textbooks:
  wysoki priorytet dla zaleceń klinicznych

PubMed abstracts:
  dobry materiał dowodowy, ale wymaga selekcji

public web:
  tylko allowlist z wersjonowaniem
```

## 5. Rekomendowana Architektura Docelowa

Nie rozbudowywać systemu w agentowy framework. Wystarczy modularny RAG:

```text
offline data pipeline
  -> processed documents/chunks
  -> embeddings
  -> streaming index builder
  -> versioned Qdrant collection

runtime API
  -> query normalization
  -> metadata intent detection
  -> multi-query retrieval
  -> metadata filtering
  -> hybrid ranking/RRF
  -> evidence filtering
  -> cross-encoder rerank
  -> context assembly
  -> answer generation
  -> citation validation
  -> answer/evidence logging
```

### 5.1. Offline Pipeline

```text
PubMed/source data
  -> cleaning/dedupe
  -> chunking
  -> chunk quality validation
  -> document embeddings
  -> embedding validation
  -> BM25 stats
  -> Qdrant index build
  -> index manifest
```

Wymagane manifesty:

```text
dataset version
chunking version
embedding model
embedding dimension
BM25 stats checksum
Qdrant collection name
Qdrant vector names
source file checksums
build timestamp
build command/config
```

### 5.2. Versioned Qdrant Collections

Nie pisać bezpośrednio do jednej stałej kolekcji `MedicalChunk`.

Lepszy wzorzec:

```text
MedicalChunk_pubmed_reviews_v1_medcpt_20260518
MedicalChunk_pubmed_reviews_v2_medcpt_20260601
```

Alias:

```text
MedicalChunk_current -> MedicalChunk_pubmed_reviews_v1_medcpt_20260518
```

Korzyści:

- bezpieczne rebuildy,
- rollback,
- porównanie wersji indeksu,
- brak ryzyka częściowo zbudowanej kolekcji w produkcji.

### 5.3. Runtime Retrieval

Proponowany przepływ:

```text
Input question
  -> normalize
  -> classify intent:
       treatment / diagnosis / adverse effects / prognosis / mechanism / general
  -> build queries:
       original
       rewrite
       optional information_need
  -> retrieve candidates:
       dense MedCPT
       sparse BM25
       optional lexical-only mode
  -> merge by weighted RRF
  -> apply metadata filter/boost
  -> evidence filter
  -> rerank top50
  -> select top_k
```

### 5.4. Context Assembly

Każde źródło w kontekście powinno mieć:

```text
[S1] title
PMID / DOI / URL
year
publication type
why selected / evidence score
short excerpt
```

Nie trzeba dawać modelowi pełnego abstraktu, jeśli wystarczy fragment.

Docelowo:

```text
full chunk in Qdrant payload
short selected excerpt in prompt
full source available in citations metadata
```

## 6. Profile Dla Modeli 7B I 26B

### 6.1. Profil 7B

Założenie:

Model 7B ma mniejszą odporność na szum, słabsze utrzymanie instrukcji i gorszą syntezę sprzecznych źródeł.

Rekomendacja:

```text
temperature = 0.0-0.2
RAG_TOP_K = 4-5
RAG_CANDIDATE_K = 50
max_context_chars = 6000-8000
cross_encoder = on
metadata_filtering = strict
evidence_filtering = strict
answer length = concise
citations = required per medical claim
no_answer_policy = strict
```

Prompt policy:

```text
Answer only from retrieved sources.
If sources are weak or missing, say the knowledge base does not contain enough evidence.
Do not provide diagnosis or treatment instructions without citations.
Every medical claim must cite [Sx].
```

7B nie powinien:

- samodzielnie robić wielu kroków agentowych,
- generować długich analiz klinicznych,
- rozwiązywać konfliktów źródeł bez jasnych instrukcji,
- dostawać 15-20 chunków w kontekście.

### 6.2. Profil 26B/27B

Założenie:

Model 26B/27B będzie lepszy w syntezie, ale nadal może halucynować, jeśli retrieval jest słaby. Większy model nie naprawia złego retrievalu.

Rekomendacja:

```text
temperature = 0.0-0.2
RAG_TOP_K = 5-8
RAG_CANDIDATE_K = 50-100
max_context_chars = 8000-12000
cross_encoder = on
metadata_filtering = on
evidence_filtering = on
optional single follow-up query for complex questions
```

26B może dostać:

- minimalnie więcej źródeł,
- jedną dodatkową reformulację,
- bardziej rozbudowaną syntezę,
- porównanie źródeł, jeśli wykryto konflikt.

26B nadal nie powinien:

- odpowiadać z wiedzy własnej, gdy RAG nie ma źródeł,
- mieszać rekomendacji z różnych populacji pacjentów,
- ignorować słabych cytowań,
- dostawać kontekstu bez selekcji.

## 7. Konkretne Zmiany W Kodzie

### P0 - Stabilizacja indeksowania

1. Dodać obsługę wielu plików embeddingów:

```bash
python3 scripts/rag/01_build_index.py \
  --chunks data/processed/chunks.parquet \
  --embeddings data/embeddings/embeddings_shard_0.parquet \
  --embeddings data/embeddings/embeddings_shard_1.parquet \
  --collection MedicalChunk_pubmed_reviews_v1_medcpt_20260518 \
  --qdrant-url http://localhost:6333 \
  --recreate
```

2. Usunąć pełne `to_pylist()` dla dużych plików.

3. Dodać checkpoint:

```text
indexed shard
last row group
last batch
upserted_count
failed_count
```

4. Zapisywać manifest indeksu.

### P0 - Metadata filtering

Dodać do `PreRetrievalResult`:

```text
intent
preferred_publication_types
min_year
requires_recent_evidence
```

Dodać Qdrant filter:

```text
corpusVersion == active
optional year range
optional publicationTypes
optional isSystematicReview
```

Jeśli Qdrant filtering okaże się za restrykcyjny, używać filtering/boosting po retrievalu.

### P1 - Multi-query retrieval

Zmienić retrieval z:

```text
use search_queries[-1]
```

na:

```text
for query in search_queries:
  retrieve candidates
merge candidates with weighted RRF
dedupe by chunkId/documentId
```

### P1 - Evidence filtering

Dodać etap po retrievalu:

```text
candidate -> evidence_score
```

Składniki:

```text
retrieval_score
reranker_score
query_term_coverage
publication_type_score
recency_score
source_priority_score
```

Prosty próg:

```text
if evidence_score < threshold:
  remove from final context
```

Jeśli zostanie za mało źródeł:

```text
retrieval.status = low_evidence
```

### P1 - Context compression

Dodać prosty excerpt extractor:

```text
weź 1-3 zdania z chunku, które mają największe pokrycie terminów z pytania
```

Nie robić na start abstrakcyjnego summarizera. Extractive excerpt jest tańszy, bardziej kontrolowalny i bezpieczniejszy.

### P1 - Eval

Rozszerzyć `data/eval_retrieval_sample.json` do minimum 100 pytań.

Format:

```json
{
  "id": "sglt2-ckd-heart-failure",
  "question": "Czy inhibitory SGLT2 zmniejszają hospitalizacje z powodu niewydolności serca u pacjentów z CKD?",
  "intent": "treatment",
  "relevant_pmids": ["..."],
  "acceptable_publication_types": ["Meta-Analysis", "Systematic Review", "Randomized Controlled Trial"],
  "must_not_answer_without_sources": true
}
```

Dodać raport porównujący:

```text
dense only
sparse only
hybrid RRF
hybrid + metadata filtering
hybrid + metadata filtering + reranker
multi-query + hybrid + filtering + reranker
```

### P2 - Runtime observability

Logować dla każdego requestu:

```text
query
rewritten_query
intent
retrieval_mode
candidate_count
final_source_count
source_pmids
source_scores
publication_types
reranker_scores
answer citation ids
unsupported claims
latency breakdown
model name
```

Nie przechowywać danych wrażliwych użytkownika bez decyzji produktowo-prawnej. Jeśli to ma być realna aplikacja medyczna, potrzebna jest polityka anonimizacji/logowania.

## 8. Rekomendowane Parametry Startowe

### 8.1. 7B

```bash
RAG_RETRIEVER=qdrant_hybrid
RAG_CANDIDATE_K=50
RAG_TOP_K=5
RAG_MAX_CONTEXT_CHARS=7000
CROSS_ENCODER_MODEL=ncbi/MedCPT-Cross-Encoder
ANSWER_QUALITY_METHOD=token_overlap_with_retrieved_context
QUERY_REWRITE_MODEL=<lekki-model-7b-lub-3b>
QUERY_REWRITE_TIMEOUT=10
```

Generacja:

```text
temperature: 0.0-0.2
max answer length: krótko
style: konkret + cytowania
```

### 8.2. 26B/27B

```bash
RAG_RETRIEVER=qdrant_hybrid
RAG_CANDIDATE_K=75
RAG_TOP_K=6
RAG_MAX_CONTEXT_CHARS=10000
CROSS_ENCODER_MODEL=ncbi/MedCPT-Cross-Encoder
QUERY_REWRITE_TIMEOUT=15
```

Generacja:

```text
temperature: 0.0-0.2
answer length: umiarkowana
allow conflict summary: yes
allow one follow-up retrieval query: only for complex questions
```

## 9. No-Answer I Safety Policy

Medyczny RAG powinien umieć nie odpowiedzieć.

Warunki odmowy lub odpowiedzi ostrożnej:

```text
brak źródeł
mniej niż 2 sensowne źródła przy pytaniu klinicznym
źródła sprzeczne bez jasnego rozstrzygnięcia
źródła nie dotyczą populacji z pytania
źródła zbyt stare dla pytania o aktualne leczenie
retrieval confidence poniżej progu
```

Przykładowa odpowiedź:

```text
Baza wiedzy nie zwróciła wystarczająco mocnych źródeł dla tego pytania, więc nie mogę udzielić odpowiedzi opartej na cytowanych danych. W decyzjach medycznych skonsultuj się z lekarzem.
```

To jest szczególnie ważne dla 7B.

## 10. Priorytetowy Roadmap

### Etap 1 - Żeby pełny corpus działał

- streaming index builder,
- obsługa `embeddings_shard_0.parquet` i `embeddings_shard_1.parquet`,
- manifest indeksu,
- wersjonowana kolekcja Qdrant,
- smoke test `/search`.

### Etap 2 - Żeby retrieval był medycznie sensowny

- metadata filtering,
- publication type boosting,
- original query + rewritten query retrieval,
- evidence filtering,
- top50 -> rerank -> top5.

### Etap 3 - Żeby dało się mierzyć jakość

- eval set 100-300 pytań,
- porównanie trybów retrievalu,
- osobne raporty dla 7B i 26B,
- latency p50/p95,
- citation precision i unsupported claims.

### Etap 4 - Dopiero później dodatki

- HyDE jako fallback,
- jedna iteracja follow-up query dla pytań złożonych,
- context excerpt extraction,
- lepsze wykrywanie konfliktów źródeł.

## 11. Czego Nie Robić Teraz

Nie robić teraz:

- pełnego systemu multiagentowego,
- długich chain-of-thought promptów,
- wielu rund retrievalu dla każdego pytania,
- zwiększania `RAG_TOP_K` bez ewaluacji,
- mieszania różnych modeli embeddingów w jednej kolekcji,
- odpowiadania z wiedzy własnej LLM, gdy RAG nie znalazł źródeł,
- budowania skomplikowanego knowledge graph przed naprawieniem retrievalu.

To są rzeczy, które mogą wyglądać atrakcyjnie, ale najpierw trzeba mieć stabilny i mierzalny prosty RAG.

## 12. Docelowy Minimalny Standard Jakości

Przed uznaniem systemu za gotowy do szerszych testów:

```text
Recall@50 >= 0.85 na eval secie
nDCG@10 rośnie po metadata filtering/rerank
minimum 95% odpowiedzi ma poprawne cytowania
unsupported claim rate mierzalnie spada względem baseline
no-answer działa dla pytań bez źródeł
latency p95 akceptowalna dla 7B i 26B osobno
pełny rebuild indeksu jest powtarzalny
rollback kolekcji Qdrant jest możliwy
```

## 13. Źródła

- [Benchmarking Retrieval-Augmented Generation for Medicine](https://arxiv.org/abs/2402.13178) - MIRAGE/MedRAG, wpływ RAG na medical QA, znaczenie kombinacji corpora/retrieverów, lost-in-the-middle.
- [Efficient and Reproducible Biomedical Question Answering using Retrieval Augmented Generation](https://arxiv.org/abs/2505.07917) - porównanie BM25, BioBERT, MedCPT i hybryd; praktyczny trade-off jakości i latencji.
- [Korean Medical Consultation With Open-Weight Large Language Models: Pilot Comparative Evaluation of RAG With Metadata Filtering](https://pubmed.ncbi.nlm.nih.gov/42060907/) - open-weight LLM, RAG-only vs RAG + metadata filtering, wyniki dla modeli klasy 7B/8B i 20B/27B.
- [Thyro-GenAI: A Chatbot Using Retrieval-Augmented Generative Models for Personalized Thyroid Disease Management](https://pmc.ncbi.nlm.nih.gov/articles/PMC11989359/) - modularny RAG, vector DB, HyDE, BM25, reranking, cytowania i logowanie.
- [Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights](https://arxiv.org/abs/2511.06738) - standardowy RAG może pogarszać wyniki, jeśli retrieval/evidence selection są słabe; evidence filtering i query reformulation pomagają.
- [Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions](https://arxiv.org/abs/2408.00727) - i-MedRAG, follow-up queries dla złożonych pytań medycznych.
- [Rationale-Guided Retrieval Augmented Generation for Medical Question Answering](https://arxiv.org/abs/2411.00300) - filtrowanie nieinformatywnych fragmentów i użycie rationale/query guidance jako inspiracja dla bardziej selektywnego retrievalu.
- [MKRAG: Medical Knowledge Retrieval Augmented Generation for Medical Question Answering](https://arxiv.org/abs/2309.16035) - prosty transparentny RAG poprawiający open-source model klasy 7B bez fine-tuningu.

