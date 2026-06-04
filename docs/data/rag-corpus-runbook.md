# RAG corpus runbook

Ten runbook opisuje praktyczny przeplyw dla korpusu wiedzy chatbota medycznego:
NICE jako guideline corpus, PubMed jako literature corpus, merge do jednego
`chunks.parquet`, indeksowanie w Qdrant i szybki test cytowan w chatbocie.

## Co jest w git, a co jest lokalne

Do git trafia kod pipeline, konfiguracja i dokumentacja. Dane zrodlowe i artefakty
sa lokalne, bo PDF, markdown i parquet moga byc duze albo odtwarzalne:

```text
data/raw/                  # lokalne pliki zrodlowe, gitignore
data/interim/              # lokalne artefakty korpusow, gitignore
data/processed/            # lokalny wynik merge, gitignore
artifacts/*.tar.gz         # paczki do przenoszenia danych, gitignore
```

Aktualny artefakt NICE w tym workspace:

```text
data/interim/nice/chunks.parquet           # broad-controlled NICE, domyslny wariant RAG
data/interim/nice/chunks_clinical.parquet  # bardziej konserwatywny wariant kliniczny
data/interim/nice/documents.parquet        # rejestr dokumentow NICE
data/interim/nice/manifest.json            # manifest lokalnego datasetu
```

W obecnie wygenerowanym artefakcie `chunks.parquet` dla NICE jest ok. 39 tys.
chunkow z ok. 1,9 tys. dokumentow. Wariant `chunks_clinical.parquet` ma ok.
17 tys. chunkow z ok. 337 dokumentow. Do pelnego RAG domyslnie uzywaj
`chunks.parquet`; do ostrozniejszego indeksu tylko guideline-first uzywaj
`chunks_clinical.parquet`.

## 1. Przygotowanie NICE lokalnie

Zainstaluj zaleznosci pipeline NICE:

```bash
.venv/bin/pip install -r scripts/data/nice/requirements.txt
```

Pobierz katalog NICE, PDF-y, zbuduj markdown, oczyszczony markdown, dokumenty i manifest:

```bash
bash scripts/data/nice/run_bulk_preprocessing.sh
```

Ten skrypt jest resumowalny: ponowne uruchomienie pomija juz pobrane PDF-y i
istniejace markdowny.

Jesli chcesz sterowac etapami recznie:

```bash
.venv/bin/python scripts/data/nice/pipeline/00_fetch_published_catalog.py
.venv/bin/python scripts/data/nice/pipeline/00_download_pdfs.py --all-published --skip-existing --delay 0.5
.venv/bin/python scripts/data/nice/pipeline/01_pdf_to_markdown.py --skip-existing
.venv/bin/python scripts/data/nice/pipeline/01b_clean_markdown.py --skip-existing
.venv/bin/python scripts/data/nice/pipeline/02_build_documents.py
.venv/bin/python scripts/data/nice/pipeline/03_write_manifest.py
```

Zbuduj domyslny broad-controlled wariant chunkow NICE:

```bash
.venv/bin/python scripts/embeddings/nice_markdown_to_parquet.py \
  --input_dir data/interim/nice/markdown_clean \
  --documents data/interim/nice/documents.parquet \
  --output_file data/interim/nice/chunks.parquet \
  --section-policy clinical-core \
  --include-prefix ng cg amr mpg ph csg sg sc ta htg hst
```

Opcjonalnie zbuduj bardziej konserwatywny wariant kliniczny:

```bash
.venv/bin/python scripts/embeddings/nice_markdown_to_parquet.py \
  --input_dir data/interim/nice/markdown_clean \
  --documents data/interim/nice/documents.parquet \
  --output_file data/interim/nice/chunks_clinical.parquet \
  --section-policy clinical-core \
  --include-prefix ng cg amr mpg ph csg sg sc
```

## 2. Spakowanie danych do przeniesienia

Nie commituj parquetow. Spakuj tylko artefakty potrzebne na drugiej maszynie:

```bash
mkdir -p artifacts
NICE_ARCHIVE="artifacts/nice_corpus_$(date +%Y%m%d).tar.gz"
tar -czf "$NICE_ARCHIVE" \
  data/interim/nice/chunks.parquet \
  data/interim/nice/chunks_clinical.parquet \
  data/interim/nice/documents.parquet \
  data/interim/nice/manifest.json \
  data/interim/nice/chunks_inspection_report.md \
  data/interim/nice/chunks_clinical_inspection_report.md

sha256sum "$NICE_ARCHIVE"
```

Przeniesienie na maszyne GPU:

```bash
scp "$NICE_ARCHIVE" user@gpu:/path/to/repo/artifacts/
```

Na maszynie GPU:

```bash
cd /path/to/repo
git fetch origin
git checkout ftr/nice-corpus
tar -xzf artifacts/nice_corpus_20260603.tar.gz
sha256sum artifacts/nice_corpus_20260603.tar.gz
```

## 3. Merge NICE z PubMed

PubMed musi byc dostarczony jako osobny lokalny parquet, np.
`/path/to/pubmed/chunks.parquet`. Merge tworzy jeden wspolny artefakt RAG:

```bash
.venv/bin/python scripts/data/merge_corpora.py \
  --chunks pubmed=/path/to/pubmed/chunks.parquet nice=data/interim/nice/chunks.parquet \
  --out-chunks data/processed/chunks.parquet \
  --manifest-out data/processed/manifest.json
```

Test NICE-only, bez PubMed:

```bash
.venv/bin/python scripts/data/merge_corpora.py \
  --chunks nice=data/interim/nice/chunks.parquet \
  --out-chunks data/processed/chunks.parquet \
  --manifest-out data/processed/manifest.json
```

Wynikowy `data/processed/chunks.parquet` ma wspolne pola RAG, m.in.
`chunk_id`, `doc_id`, `source`, `title`, `text`, `url`, `section`,
`word_count`, `text_hash` i `chunk_index`. Pola specyficzne dla NICE, takie jak
`external_id`, `guidance_type`, `source_name` i `header_path`, zostaja zachowane
jako metadane i sa potem uzywane w filtrach, boostingu i cytowaniach.

## 4. Uruchomienie Qdrant i embedding-service na GPU

Najprostszy wariant uzywa domyslnej kolekcji z `docker-compose.yml`:
`MedicalChunk_pubmed_reviews_v1_medcpt_20260518`. Najwazniejsze jest, zeby
embedding-service, build index i chatbot wskazywaly na te sama kolekcje.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build qdrant embedding-service
```

W drugim terminalu zbuduj indeks:

```bash
.venv/bin/python scripts/rag/01_build_index.py \
  --chunks data/processed/chunks.parquet \
  --qdrant-url http://localhost:6333 \
  --embedding-service-url http://localhost:8081 \
  --corpus-version medical-knowledge-v1 \
  --recreate
```

Do szybkiego pilota mozna dodac `--limit 500`, ale pelny test cytowan powinien
uzywac docelowego `data/processed/chunks.parquet`.

Wygodne targety Makefile dla pilota NICE:

```bash
make embed-nice
make index-nice NICE_LIMIT=500
```

Domyslnie targety uzywaja:

```text
NICE_CHUNKS=data/interim/nice/chunks_clinical.parquet
NICE_EMBEDDINGS=data/embeddings/nice_clinical_embeddings.parquet
NICE_COLLECTION=MedicalChunk_nice_pilot_medcpt_20260603
NICE_CORPUS_VERSION=nice-guidelines-v1
```

## 5. Uruchomienie chatbota

Uruchom aplikacje tak, zeby widziala Qdrant i embedding-service:

```bash
RAG_CORPUS_VERSION=medical-knowledge-v1 \
EMBEDDING_SERVICE_URL=http://localhost:8081 \
QDRANT_HOST=localhost \
QDRANT_PORT=6333 \
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Jesli LLM idzie przez Ollama, upewnij sie, ze model jest dostepny lokalnie:

```bash
ollama list
ollama pull qwen2.5:7b
```

## 6. Test cytowania NICE

Najlatwiejszy test to pytanie z jawnym ID NICE, bo pre-retrieval doda wtedy filtr
`externalId`:

```text
NICE AMR1: kiedy ceftazidime-avibactam jest recommended?
```

Oczekiwane zachowanie:

- retrieval znajduje fragmenty z `source=nice` i `external_id=AMR1`;
- odpowiedz cytuje zrodlo jako `NICE AMR1`;
- metadane zrodla wskazuja `https://www.nice.org.uk/guidance/amr1`;
- w kontekscie dla modelu widac `Guidance ID`, `Guidance type` i sciezke sekcji.

Mozna tez testowac bez jawnego ID:

```text
Jakie sa zalecenia NICE dotyczace uzycia ceftazidime-avibactam?
```

Wtedy retrieval powinien nadal preferowac guideline NICE, ale filtr nie bedzie
tak waski jak przy pytaniu z `AMR1`.

## 7. Benchmark NICE

Benchmarki PubMedQA sa nadal przydatne dla indeksu PubMed albo indeksu
mieszanego z PubMed, ale nie sa dobra miara dla NICE-only, bo ground truth jest
oparty o PMID. Dla NICE dodane sa male benchmarki z ground truth po
`documentId`:

```text
data/benchmarks/retrieval/eval_nice_guidelines_sample.json
data/benchmarks/rag/eval_nice_guidelines_sample.json
```

Po uruchomieniu API przeciwko kolekcji NICE:

```bash
make eval-nice-retrieval
make eval-nice-rag
```

Pierwszy benchmark odpytuje `GET /search` i liczy MRR, Recall@k, Precision@k i
nDCG@k. Drugi odpytuje `POST /api/rag/trace` oraz `POST /api/chat`, sprawdza
status `grounded`, `source_hit_at_3`, cytowania, groundedness i wymagane termy
w odpowiedzi.

Jesli chcesz wiekszy zestaw, najpierw wygeneruj go z aktualnych NICE chunks:

```bash
make build-nice-benchmarks
```

Potem uruchom:

```bash
make eval-nice-retrieval-large
make eval-nice-rag-large
```

Duzy retrieval benchmark ma 500 case'ow, a duzy RAG benchmark 100 case'ow.
To nadal benchmark po `documentId`, ale z wiekszym pokryciem i lepsza
stabilnoscia niz 5-case smoke test.

## 8. Szybka diagnostyka

Sprawdzenie, czy serwisy odpowiadaja:

```bash
curl http://localhost:6333/collections
curl http://localhost:8081/health
curl http://localhost:8000/health
```

Najczestsze problemy:

- chatbot nie cytuje NICE, bo indeks byl zbudowany na starej wersji `chunks.parquet`;
- build index, API i embedding-service wskazuja na rozne kolekcje Qdrant;
- `data/processed/chunks.parquet` zawiera tylko PubMed, bo merge nie dostal sciezki NICE;
- dane sa na pierwszej maszynie, ale nie zostaly rozpakowane na maszynie GPU;
- Ollama albo embedding-service nie maja pobranych modeli.

## 9. Kiedy regenerowac dane

Regeneruj NICE, gdy chcesz zaktualizowac katalog NICE, zmienic filtr prefiksow,
zmienic polityke sekcji albo poprawic czyszczenie markdown. Nie trzeba
regenerowac danych tylko po zmianie kodu cytowan lub boostingu RAG, ale wtedy
trzeba przebudowac indeks Qdrant, jesli zmienily sie payload metadata albo
`chunks.parquet`.
