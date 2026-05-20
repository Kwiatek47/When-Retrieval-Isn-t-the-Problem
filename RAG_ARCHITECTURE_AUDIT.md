# Aktualna diagnoza RAG

Data: 2026-05-20
Branch: `rag-improvement-rag-optimization`
Zakres: dane -> embeddingi -> indeks -> retrieval -> evidence judge -> writer -> API -> eval

## 1. Najkrotsza diagnoza

Mamy juz duzo wiecej niz prosty RAG. System ma pipeline danych, embeddingi MedCPT, hybrydowe wyszukiwanie dense + BM25, reranking/scoring, guardraile cytowan, endpoint trace i osobny `EvidenceJudge`, ktory podejmuje decyzje przed writerem.

Najmocniejsza czesc systemu to obecnie retrieval. Na oficjalnym PubMedQA 500 test system znajduje poprawne zrodlo na pierwszej pozycji w `98.2%` przypadkow. To znaczy, ze problem nie jest glownie w tym, ze RAG nie znajduje papieru.

Najslabsza czesc systemu to decyzja z evidence: czy zrodlo oznacza `yes`, `no` czy `maybe`. Na oficjalnym PubMedQA 500 mamy `49.6%` accuracy przy modelu `qwen2.5:7b`. To jest ponizej paperowego majority baseline PubMedQA okolo `55.2%`, mimo ze retrieval trafia bardzo dobrze.

Najprosciej:

```text
retriever dowozi zrodlo
judge nie zawsze dobrze rozumie, co z tego zrodla wynika
writer i cytowania sa juz mocno ograniczone guardrailami
```

Dlatego kolejny realny skok jakosci nie bedzie z samego "lepszego promptu do chatbota". Najwiekszy zysk bedzie z lepszego evidence judge, lepszego datasetu treningowo-walidacyjnego i multi-corpus retrievalu dla produkcyjnych pytan klinicznych.

## 2. Co teraz mamy w repo

Glowne katalogi:

```text
app/
  api/                 endpointy FastAPI
  core/                konfiguracja
  providers/           Ollama provider
  rag/                 caly pipeline RAG
services/
  embedding-service/   MedCPT query/document encoder + hybrid search
scripts/
  embeddings/          embedding pipeline
  rag/                 index, search, eval, PubMedQA benchmark
data-pipeline/
  pubmed/              dokumentacja i kontrakt danych PubMed
data/                  sample corpora i male benchmarki
reports/               raporty ewaluacyjne
tests/                 testy jednostkowe RAG
```

Najwazniejsze pliki runtime:

- `app/rag/pre_retrieval.py` - czyszczenie i planowanie query.
- `app/rag/retrieval.py` - retrievery, weighted RRF, metadata boost.
- `app/rag/post_retrieval.py` - ranking evidence, excerpt extraction, prompt context.
- `app/rag/evidence_judge.py` - osobny judge przed writerem.
- `app/rag/pipeline.py` - orkiestracja RAG.
- `app/api/routes.py` - `/api/chat`, `/api/search`, `/api/rag/trace`.
- `app/core/config.py` - wszystkie flagi i domyslne parametry.
- `services/embedding-service/` - MedCPT + BM25 + Qdrant hybrid query.
- `scripts/rag/01_build_index.py` - streamingowy indexer do Qdranta.

## 3. Dane

### Co mamy

W repo sa male datasety i benchmarki:

- `data/pubmed_sample.json`
- `data/sample/medical_documents.json`
- `data/pubmedqa_strict_corpus_90.json`
- `data/eval_pubmedqa_strict_90.json`
- `data/pubmedqa_benchmark_corpus.json`
- `data/eval_pubmedqa_benchmark.json`
- `data/eval_rag_english_real_sources.json`
- `data/pubmedqa_official_pqal_test/` - official PubMedQA PQA-L 500 eval artifacts

Mamy tez dokumentacje pelnego pipeline PubMed:

- `data-pipeline/pubmed/docs/`
- `data-pipeline/pubmed/configs/chunks_schema_v1.json`
- `data-pipeline/pubmed/reports/pubmed_reviews_v1_quality_report.md`

Wedlug repo-safe raportu jakosci dla `pubmed_reviews_v1` pipeline zbudowal:

```text
input PMID count: 990,391
metadata rows fetched: 982,019
cleaned rows: 979,499
final documents: 977,777
final chunks: 977,777
```

Usuniete rekordy:

```text
removed by cleaning filters: 2,520
duplicates by DOI: 57
duplicates by title hash: 1,665
duplicates by PMID: 0
duplicates by content hash: 0
```

To jest dobry fundament. Dane sa odszumione i maja kontrakt pod RAG.

### Czego brakuje

Pelny produkcyjny korpus PubMed i pelne embeddingi nie siedza w samym repo. Repo ma pipeline, kontrakt, raporty i official PubMedQA 500 benchmark artifacts, ale nie ma gotowego produkcyjnego indeksu jako artefaktu.

Do pelnej powtarzalnosci brakuje jeszcze:

- jednoznacznego manifestu aktualnie uzytego indeksu Qdrant,
- wersji `chunks.parquet`,
- wersji shardow embeddingow,
- wersji BM25 stats dla pelnego korpusu,
- prostego polecenia "od zera zbuduj dokladnie ten sam produkcyjny index".

To nie blokuje developmentu, ale blokuje idealnie powtarzalne porownania miedzy osobami i maszynami.

## 4. Chunking

Obecny model danych zaklada `chunks.parquet`. Dla PubMed zwykle jeden abstrakt to jeden chunk.

Dodany zostal opcjonalny parent-child chunking:

- plik: `scripts/rag/00_build_child_chunks.py`
- wejscie: `chunks.parquet`
- wyjscie: `chunks_child.parquet`
- krotkie chunki zostaja bez zmian,
- dlugie chunki `word_count > 450` sa dzielone po zdaniach,
- child chunk dostaje `parent_chunk_id`, `parent_word_count`, `chunk_index`.

To jest dobra zmiana, bo MedCPT document encoder ma limit dlugosci. Bardzo dlugi abstrakt moze byc uciety przy embedowaniu albo dac za duzo szumu do promptu. Child chunking zmniejsza ten problem.

Slaby punkt: to jest tylko skrypt. Jesli nie uruchomimy go przy pelnym buildzie indeksu, runtime nie skorzysta z child chunkow.

## 5. Embeddingi

Embedding service uzywa MedCPT:

```text
query encoder:    ncbi/MedCPT-Query-Encoder
document encoder: ncbi/MedCPT-Article-Encoder
dimension:        768
```

To jest dobry wybor dla biomedycznego RAG, lepszy niz ogolny embedding model do PubMed.

Serwis ma dwa glowne zadania:

1. liczyc embeddingi dokumentow/chunkow,
2. liczyc embedding query i robic hybrid search z Qdrantem.

Wyszukiwanie jest hybrydowe:

```text
MedCPT dense vector
+ BM25 sparse vector
+ Qdrant RRF fusion
```

Domyslna kolekcja:

```text
MedicalChunk_pubmed_reviews_v1_medcpt_20260518
```

Najwazniejsza uwaga: nie wolno mieszac embeddingow z roznych modeli albo roznych wymiarow w jednej kolekcji. Repo juz ma `embeddingModel` i `corpusVersion` w metadanych, co jest dobrym kierunkiem.

## 6. Indeks Qdrant

Indexer jest teraz sensowniejszy niz pierwotne MVP:

- czyta `chunks.parquet` batchami,
- buduje lokalny SQLite chunk store,
- liczy BM25 stats,
- czyta shardy embeddingow,
- laczy embeddingi z payloadem po `chunk_id`,
- upsertuje batchami do Qdranta,
- zapisuje manifest i checkpoint.

Plik:

```text
scripts/rag/01_build_index.py
```

Payload w Qdrancie obsluguje m.in.:

- `chunkId`
- `documentId`
- `pmid`
- `doi`
- `title`
- `journal`
- `year`
- `publicationTypes`
- `isReview`
- `isSystematicReview`
- `wordCount`
- `parentChunkId`
- `parentWordCount`
- `embeddingModel`
- `corpusVersion`

To jest wazne, bo RAG nie powinien traktowac kazdego zrodla tak samo. Guideline, systematic review i RCT powinny miec inna wage niz case report albo editorial.

## 7. Pre-retrieval

`PreRetriever` robi teraz kilka waznych rzeczy przed szukaniem:

- bierze ostatnie pytanie usera,
- normalizuje query,
- wykrywa, czy retrieval w ogole jest potrzebny,
- klasyfikuje intent, np. `treatment`, `diagnosis`, `adverse_effects`,
- dobiera preferowane typy publikacji,
- ustawia filtr swiezosci dla pytan o aktualne wytyczne/bezpieczenstwo,
- robi deterministyczne rozwijanie skrotow,
- opcjonalnie robi LLM rewrite,
- ogranicza liczbe query wariantow do 4.

Rozwijane skroty:

```text
PAD, T2DM, CKD, AF, DOAC, EGFR, SGLT2, ICS
```

Kolejnosc query:

```text
original -> normalized -> acronym expansion -> LLM rewrite
```

To jest dobre, bo nie tracimy oryginalnego pytania. Wczesniejszy problem typowy dla RAG to zastapienie query rewrite'em, ktory czasem gubi sens pytania. Teraz query warianty sa laczone.

## 8. Retrieval

Mamy dwa retrievery:

1. `EmbeddingServiceHybridRetriever` - domyslnie przez `embedding-service`.
2. `QdrantHybridKnowledgeRetriever` - bezposrednio z aplikacji do Qdranta.

Dodane/obecne mechanizmy:

- wieksza pula kandydatow dla pytan klinicznych,
- weighted RRF dla wielu query,
- zapisywanie `rrfScore`,
- zapisywanie `matchedQueryCount`,
- zapisywanie `rawRetrievalScores`,
- metadata-aware boost,
- neutralny fallback, gdy brakuje metadanych.

Candidate expansion:

```text
expanded_limit = max(limit, min(limit * 2, 100))
```

Metadata boost premiuje:

- `Systematic Review`
- `Guideline`
- `Practice Guideline`
- `Meta-Analysis`
- `Randomized Controlled Trial`
- `Clinical Trial`

Lekko karze:

- `Case Reports`
- `Letter`
- `Editorial`
- `Comment`

To jest dobre i bezpieczne. Nie zmieniamy tresci dokumentu, tylko lepiej ustawiamy kolejnosc zrodel.

## 9. Post-retrieval i pakowanie evidence

`PostRetriever` robi selekcje kontekstu przed writerem.

Najwazniejsze rzeczy:

- deduplikacja po `parentChunkId`, `chunkId`, `documentId`,
- opcjonalny cross-encoder reranking,
- evidence scoring,
- query term coverage z wielu query wariantow,
- `multiQueryMatchScore`,
- priorytet dla mocniejszych typow publikacji,
- limit fragmentu per zrodlo przez `RAG_MAX_EXCERPT_CHARS`,
- extractive excerpt extraction,
- wykrywanie potencjalnych konfliktow,
- budowanie bloku `MEDICAL_KNOWLEDGE_BASE`,
- wymuszanie cytowan `[S1]`, `[S2]`, itd.

Obecny kontekst jest ukladany wedlug `evidenceScore`, a nie tylko wedlug surowego score z retrievera. To jest wazne, bo surowy retrieval score nie zawsze oznacza "najlepsze zrodlo do odpowiedzi".

Slaby punkt: evidence scoring nadal jest heurystyczny. Jest znacznie lepszy niz samo top-k z wektora, ale to nadal nie jest nauczony evidence filter.

## 10. Evidence Judge

To jest najwazniejsza zmiana architektoniczna.

Wczesniej system dzialal mniej wiecej tak:

```text
retriever -> writer
```

Teraz docelowy podzial jest taki:

```text
retriever -> evidence judge -> answer writer
```

Znaczenie:

- retriever znajduje potencjalne zrodla,
- evidence judge ocenia, co te zrodla naprawde mowia,
- writer ma skladac odpowiedz z decyzji i cytacji, a nie samemu "zgadywac".

Plik:

```text
app/rag/evidence_judge.py
```

Dla pytan PubMedQA-style (`yes/no/maybe`) judge moze:

- wezwac tani lokalny LLM jako klasyfikator evidence,
- zwrocic `answer_label`,
- zwrocic `confidence`,
- zwrocic krotkie rationale z cytowaniem,
- ominac glownego writera i dac deterministyczna odpowiedz.

Dzieki temu mala 7B nie musi robic wszystkiego naraz. To jest poprawny kierunek dla malych modeli.

### Evidence Judge v3

V3 dodaje ostrzejsze reguly:

- nie traktuje samego tytulu/pytania/celu badania jako dowodu,
- `yes` tylko gdy wyniki jasno wspieraja teze,
- `no` gdy jest bezposrednie zaprzeczenie albo brak efektu,
- `maybe` gdy evidence jest posrednie, slabe, mieszane, subgroup-only albo niepewne,
- po decyzji LLM robi kalibracje regexami na typowe pomylki.

To poprawilo wynik na malym, kontrolowanym tescie 90 przypadkow.

### Voting

Dodany jest opcjonalny tryb voting:

```text
RAG_EVIDENCE_JUDGE_VOTING_ENABLED=true
RAG_EVIDENCE_JUDGE_VOTES=3
```

W praktyce na obecnym tescie voting nie dal lepszej accuracy, a zwiekszyl latency z okolo `2.76s` do `7.56s`. Dlatego defaultowo powinien zostac wylaczony.

## 11. Answer writer i guardraile

Dla zwyklego chatu writer nadal generuje odpowiedz z kontekstu RAG.

Guardraile:

- wymuszaja canonical citations `[S1]`,
- naprawiaja brakujace cytowania, jesli sie da,
- pilnuja, zeby model nie odpowiadal bez zrodel przy `low_evidence`,
- dla PubMedQA-style wymuszaja format `Answer: yes/no/maybe`,
- maja extractive fallback.

Dla PubMedQA-style obecny system moze ominac glownego writera i zwrocic odpowiedz bezposrednio z `EvidenceJudge`. To zmniejsza halucynacje i poprawia kontrolowalnosc, ale przenosi ciezar accuracy na judge.

## 12. API i debug

Najwazniejsze endpointy:

```text
POST /api/chat
GET  /api/search
POST /api/search
POST /api/rag/trace
```

`/api/rag/trace` jest bardzo wazny diagnostycznie, bo pokazuje etapy RAG bez generowania odpowiedzi przez LLM.

Trace zwraca m.in.:

- `original_query`
- `normalized_query`
- `search_queries`
- `intent`
- `filters`
- `preferred_publication_types`
- kandydatow po RRF,
- kandydatow po metadata boost,
- finalne dokumenty po evidence scoringu,
- `retrieval.status`,
- `evidence_decision`,
- preview kontekstu.

To daje nam narzedzie do debugowania, czy problem jest w:

```text
query -> retrieval -> ranking -> judge -> writer
```

Bez trace bardzo latwo zgadywac w zlym miejscu.

## 13. Aktualne wyniki eval

### Strict PubMedQA 90

Dataset:

```text
data/eval_pubmedqa_strict_90.json
```

Model:

```text
qwen2.5:7b
```

Wyniki:

| Wariant | Accuracy | Case pass | Source@1 | Citation pass | Mean latency |
|---|---:|---:|---:|---:|---:|
| Evidence Judge v3 | 61.1% | 57.8% | 97.8% | 100.0% | 2.76s |
| Evidence Judge v3 + voting | 61.1% | 58.9% | 97.8% | 100.0% | 7.56s |

Wniosek:

```text
v3 pomaga
voting na razie nie pomaga
retrieval prawie zawsze trafia zrodlo
```

### Official PubMedQA PQA-L test 500

Raport:

```text
reports/pubmedqa_official_pqal_test_v3.md
reports/pubmedqa_official_pqal_test_v3.json
reports/pubmedqa_official_pqal_test_v3_predictions.json
```

Model:

```text
qwen2.5:7b
```

Wyniki:

| Metryka | Wynik |
|---|---:|
| Cases | 500 |
| Label accuracy | 49.6% |
| Macro-F1 | 46.9% |
| Case pass rate | 48.2% |
| Source hit@1 | 98.2% |
| Source hit@3 | 98.2% |
| Citation pass rate | 99.8% |
| Grounded status rate | 99.2% |
| Mean hallucination rate | 1.8% |
| Mean latency | 2.64s |

Per label:

| Label | Count | Accuracy |
|---|---:|---:|
| maybe | 55 | 54.5% |
| no | 169 | 52.7% |
| yes | 276 | 46.7% |

Predykcje sa przesuniete w strone `maybe`:

```text
true labels:      yes 276, no 169, maybe 55
predicted labels: maybe 203, yes 164, no 133
```

Najwazniejszy wniosek:

```text
retrieval: bardzo dobry
cytowania: bardzo dobre
grounding: dobry
label reasoning: za slaby
```

System za czesto mowi `maybe`, kiedy powinien powiedziec `yes`, i nadal myli czesc `no`.

## 14. Porownanie do paper results

To nie jest jeszcze paper-level accuracy.

Dla orientacji PubMedQA paper raportowal mniej wiecej:

```text
majority baseline: ok. 55.2%
BioBERT:           ok. 68.1%
human:             ok. 78.0%
```

Nasz official 500 wynik:

```text
49.6%
```

Czyli:

- jestesmy ponizej prostego majority baseline,
- nie jestesmy jeszcze blisko BioBERT paper result,
- ale retrieval nie jest glownym winowajca, bo `source@1` jest `98.2%`.

To jest wazne. Gdyby source@1 bylo np. 60%, najpierw naprawialibysmy retrieval. Tutaj glowna strata jest juz po znalezieniu dobrego zrodla.

## 15. Co faktycznie zrobilismy w tej serii zmian

Zrobione rzeczy:

1. Przeniesiony i rozbudowany pipeline na branchu `rag-improvement-rag-optimization`.
2. Dodany candidate pool expansion dla pytan klinicznych.
3. Dodany weighted RRF dla multi-query retrieval.
4. Dodany metadata-aware boost dla mocniejszych typow publikacji.
5. Dodane deterministic acronym expansion dla medycznych skrotow.
6. Dodane multi-query term coverage i `multiQueryMatchScore`.
7. Dodany limit excerptu per zrodlo przez `RAG_MAX_EXCERPT_CHARS`.
8. Dodany diagnostyczny endpoint `POST /api/rag/trace`.
9. Dodany parent-child chunking script dla dlugich abstraktow.
10. Dodany adaptive retrieval v1 po `low_evidence`.
11. Dodany `EvidenceJudge` jako osobna rola miedzy retrieverem i writerem.
12. Dodany `EvidenceJudge v3` z lepszymi regulami decyzji.
13. Dodany opcjonalny voting za flaga.
14. Dodane raporty eval w `reports/`.
15. Uruchomiony realny official PubMedQA PQA-L test 500.

## 16. Mocne strony obecnej architektury

### 16.1. Dobry retriever biomedyczny

MedCPT + BM25 + RRF to sensowny stack do PubMed. Wynik `98.2% Source@1` na official PubMedQA potwierdza, ze dla tego benchmarku retriever znajduje wlasciwe zrodla.

### 16.2. Query planning jest juz praktyczny

System nie polega na jednym query. Uzywa oryginalu, normalizacji, rozszerzen skrotow i opcjonalnego rewrite. To ogranicza ryzyko, ze jedno zle query rozwali caly retrieval.

### 16.3. Metadata zaczyna miec znaczenie

RAG rozroznia typy publikacji. To jest konieczne w medycynie, bo case report i systematic review nie powinny miec tej samej sily.

### 16.4. Jest trace endpoint

Bez `/api/rag/trace` kazda optymalizacja bylaby zgadywaniem. Teraz mozemy sprawdzic, gdzie wynik sie psuje.

### 16.5. Writer jest ograniczony

Cytowania, guardraile i extractive fallback zmniejszaja ryzyko halucynacji. Na official 500 citation pass to `99.8%`, a mean hallucination rate to `1.8%`.

## 17. Slabe punkty

### 17.1. Evidence Judge nie jest jeszcze wystarczajaco dobry

To jest najwiekszy problem.

Objawy:

- official 500 accuracy tylko `49.6%`,
- zbyt duzo odpowiedzi `maybe`,
- `yes` ma tylko `46.7%` per-label accuracy,
- wynik jest ponizej majority baseline.

To znaczy, ze judge jest zbyt ostrozny albo nie rozpoznaje wystarczajaco dobrze, kiedy evidence jasno wspiera teze.

### 17.2. Brakuje uczonego evidence filtera

Mamy heurystyki i LLM judge. Nie mamy jeszcze malego modelu/fine-tuned classifiera, ktory bylby uczony konkretnie do:

```text
question + evidence -> supported/refuted/uncertain
```

To nie jest "trenowanie pod test". Poprawny cel to nauczenie modelu rozpoznawania relacji miedzy pytaniem a wynikiem badania.

### 17.3. Mamy jeden glowny corpus

Architektura nadal jest glownie PubMed-centric.

Do produkcyjnego chatbota medycznego powinny dojsc osobne korpusy:

- PubMed abstracts/articles,
- clinical guidelines,
- drug labels / SmPC / FDA / EMA,
- local protocols,
- possibly textbooks/knowledge summaries,
- benchmark-specific corpora do ewaluacji.

Wtedy retriever powinien byc balanced, a nie "jedna kolekcja i top-k".

### 17.4. Brakuje produkcyjnego clinical eval

PubMedQA jest dobry do testowania evidence reasoning, ale nie jest pelnym testem chatbota klinicznego.

Brakuje:

- pytan o leczenie,
- pytan o dawki,
- pytan o przeciwwskazania,
- pytan o interakcje,
- pytan z konfliktami guidelines vs starsze badania,
- recenzji eksperta medycznego.

### 17.5. Voting jest drogi i na razie nieoplacalny

Na strict 90 voting nie poprawil accuracy, a prawie potroil latency. To nie powinien byc domyslny tryb.

### 17.6. Cross-encoder jest opcjonalny i kosztowny

Cross-encoder moze poprawic ranking, ale na MacBooku/CPU moze byc wolny. Trzeba go testowac osobno i wlaczac tam, gdzie daje realny zysk.

## 18. Diagnoza warstwa po warstwie

### Dane

Stan: dobry fundament, ale artefakty pelnego korpusu sa poza repo.
Ryzyko: trudniejsza powtarzalnosc pelnego buildu.
Priorytet: manifesty i skrypt "rebuild eval index".

### Chunking

Stan: jest parent-child script.
Ryzyko: nie daje zysku, jesli nie uzyjemy `chunks_child.parquet` przy indeksowaniu.
Priorytet: wlaczyc go w pelnym buildzie i porownac wyniki.

### Embeddingi

Stan: dobry model biomedyczny MedCPT.
Ryzyko: mieszanie wersji embeddingow albo corpusVersion.
Priorytet: twarde manifesty i walidacja kolekcji.

### Retrieval

Stan: mocny.
Ryzyko: dla PubMedQA nie jest bottleneckiem, ale dla produkcyjnych pytan brakuje multi-corpus.
Priorytet: nie przepalac czasu na drobne tuningowanie retrievera pod PubMedQA; lepiej dodac multi-corpus dla kliniki.

### Post-retrieval

Stan: solidny heurystycznie.
Ryzyko: scoring nadal jest reczny.
Priorytet: learned/rationale evidence filter.

### Evidence Judge

Stan: najwazniejszy komponent i najwiekszy bottleneck.
Ryzyko: za duzo `maybe`, za malo pewnego `yes`.
Priorytet: poprawa judge na oficjalnym dev/test flow, najlepiej przez osobny classifier lub silniejszy judge model.

### Writer

Stan: wystarczajaco kontrolowany dla cytowan i groundedness.
Ryzyko: ogolne odpowiedzi kliniczne nie sa jeszcze szeroko ocenione.
Priorytet: clinical answer eval, nie tylko PubMedQA.

## 19. Co to znaczy praktycznie

Nie powinnismy teraz losowo dopisywac kolejnych promptow do glownego chatbota.

Najlepszy kierunek:

```text
1. utrzymac mocny retrieval
2. poprawic evidence judge
3. dodac multi-corpus retrieval
4. zrobic powtarzalny eval
5. dopiero potem fine-tuning/fine-grained judge
```

Jesli chcemy zblizyc sie do paper results na malych modelach, musimy rozdzielic problem:

```text
retrieval accuracy != answer accuracy
```

U nas retrieval accuracy jest juz bardzo wysokie. Answer accuracy spada na etapie interpretacji evidence.

## 20. Najlepsze kolejne kroki

### P0 - zrobic eval w 100% powtarzalny

Dodac do repo jeden skrypt/runbook, ktory odtwarza official PubMedQA 500 eval:

```text
download official labels
build chunks
build BM25 stats
index Qdrant collection
run eval
write report
```

Bez tego latwo porownywac wyniki z roznych indeksow albo roznych filtrow.

### P1 - poprawic Evidence Judge

Cel:

```text
question + top evidence -> yes/no/maybe + rationale
```

Najpierw bez fine-tuningu:

- zrobic error analysis official 500,
- zobaczyc, kiedy `yes` zamienia sie w `maybe`,
- zobaczyc, kiedy `no` myli sie z `maybe/yes`,
- dodac testy regresyjne na konkretne typy bledow.

Potem fine-tuning:

- nie po to, zeby "zapamietac test",
- tylko po to, zeby nauczyc relacji pytanie-evidence-decyzja,
- najlepiej na train/dev split, a official test zostawic do koncowej walidacji.

### P2 - multi-corpus retrieval

Dla prawdziwego chatbota klinicznego PubMed nie wystarczy.

Docelowo:

```text
corpus: pubmed
corpus: guidelines
corpus: drug_labels
corpus: local_protocols
```

Retriever powinien dawac zbalansowane top-k, np.:

```text
2 guidelines
2 reviews/trials
1 drug label
```

Zalezne od intencji pytania.

### P3 - learned/rationale evidence filter

Po retrievalu i przed writerem dodac filter:

```text
czy ten fragment realnie pomaga odpowiedziec na pytanie?
czy wspiera, zaprzecza, czy jest niepewny?
```

To moze byc:

- cross-encoder/NLI model,
- maly lokalny LLM judge,
- fine-tuned classifier.

### P4 - produkcyjny clinical eval

Oprocz PubMedQA potrzebujemy eval setu po angielsku z pytaniami typu:

- treatment recommendation,
- diagnosis,
- contraindications,
- drug interactions,
- safety,
- recent guideline question,
- insufficient evidence question.

PubMedQA sprawdza reasoning z abstraktu. Nie sprawdza calego chatbota klinicznego.

## 21. Czego nie robic teraz

Nie warto teraz:

- tuningowac bez konca glownego promptu writera,
- wlaczac voting defaultowo,
- oceniac systemu tylko po kilku recznych pytaniach,
- fine-tunowac model na official test set,
- mieszac nowych korpusow bez pola `corpus` i bez balanced retrievera,
- zakladac, ze wiekszy LLM sam rozwiaze problem RAG.

Wiekszy model pomoze w evidence judge, ale jesli pipeline nie rozroznia corpusow, evidence i konfliktow, to dalej bedzie trudno kontrolowac odpowiedzi.

## 22. Obecny werdykt

Architektura jest teraz w dobrym miejscu do dalszej optymalizacji.

Najkrotszy werdykt:

```text
RAG technicznie dziala.
Retriever jest mocny.
Cytowania i grounding sa dobre.
Glowne ograniczenie to evidence reasoning.
Do produkcji brakuje multi-corpus, powtarzalnego eval i lepszego judge.
```

Jesli mamy dzisiaj wybrac jeden najwazniejszy kierunek, to nie jest kolejny rewrite retrievera. To jest dopracowanie `EvidenceJudge` i przygotowanie go pod uczony, powtarzalnie oceniany classifier.
