# Architektura systemu diagnostycznego opartego na LLM i RAG dla wsparcia diagnostyki neurologiczno-psychiatrycznej

**Sprawozdanie semestralne z projektu badawczego**  
**Projekt:** Architektura multiagentowego systemu diagnostycznego opartego na dużych modelach językowych dla wsparcia diagnostyki neurologiczno-psychiatrycznej  
**Edycja:** projekt badawczy, luty 2026  
**Opiekun:** Jacek Rumiński (KIB)  
**Zespół:** 5 osób  
**Status:** realizowany  

## Streszczenie

W tym semestrze zbudowaliśmy i przebadaliśmy rdzeń systemu do medycznego wspomagania decyzji, który stanowi podstawę pod docelową architekturę multiagentową. Pierwotna koncepcja projektu zakładała debatę agentów pełniących role neurologa, neurochirurga, psychologa i psychiatry. W trakcie prac uznaliśmy jednak, że przed dodaniem warstwy debaty konieczne jest opracowanie mierzalnej, powtarzalnej i bezpiecznej warstwy RAG, ponieważ bez niej trudno byłoby odróżnić rzeczywistą poprawę rozumowania od efektu generowania większej ilości tekstu przez kilka modeli.

Najważniejszym wynikiem semestru jest działający pipeline MedChat: aplikacja FastAPI z lokalnym dostawcą LLM przez Ollama, osobnym serwisem embeddingów MedCPT, bazą wektorową Qdrant, hybrydowym wyszukiwaniem dense + BM25, rerankingiem, walidacją cytowań, oceną jakości odpowiedzi, mechanizmami odmowy przy słabym materiale dowodowym oraz osobną warstwą evidence-to-decision. System rozdziela dwa tryby pracy: `benchmark_pqal`, służący do powtarzalnej ewaluacji na PubMedQA PQA-L 500, oraz `medical_chat`, czyli tryb ostrożnego asystenta klinicznego.

Eksperymenty pokazały, że samo odnalezienie i zacytowanie właściwego źródła nie wystarcza do poprawnego wnioskowania medycznego. Na pełnym benchmarku PubMedQA PQA-L 500 uzyskaliśmy `source_hit_at_1 = 0.980` i `citation_pass_rate = 1.000`, ale historyczny wariant rules/LLM judge osiągał tylko `0.536` label accuracy. Dodanie dedykowanej warstwy klasyfikatora BioLinkBERT dla zadania `question + evidence -> yes/no/maybe` podniosło wynik do `0.720` label accuracy. Jednocześnie klasa `maybe`, oznaczająca niejednoznaczność lub niewystarczające dowody, pozostała głównym nierozwiązanym problemem: jej accuracy wyniosła tylko `0.073`.

Drugim ważnym rezultatem jest analiza retrievalu na korpusie NICE Guidelines dla 500 przypadków. System osiągnął bardzo wysoki `Recall@5 = 0.960` i niską latencję średnią `59.5 ms`, ale tylko `44.6%` Top-1 accuracy. Jednocześnie `93.0%` przypadków miało poprawny wynik na pozycji 1 albo 2. To wskazuje, że retriever zwykle trafia w dobrą okolicę semantyczną, lecz wymaga lepszego rozstrzygania między pierwszymi dwoma wynikami, szczególnie przez debugowanie rerankingu i dokumentów nadmiernie często wygrywających ranking.

Nie twierdzimy na tym etapie, że zrealizowaliśmy pełny, zwalidowany system multiagentowy. Zrealizowaliśmy natomiast dużą część infrastruktury badawczej i produktowej, która jest konieczna, aby taki system uczciwie ocenić: pipeline danych, wyszukiwanie, ugruntowanie odpowiedzi w źródłach, moduł decyzji na podstawie evidence, safety gates, framework ewaluacyjny i serię ablacji pokazujących, gdzie faktycznie leżą ograniczenia obecnego podejścia.

**Słowa kluczowe:** medyczny RAG, LLM, diagnostyka neurologiczno-psychiatryczna, PubMedQA, NICE Guidelines, evidence-to-decision, BioLinkBERT, bezpieczeństwo kliniczne.

## 1. Wprowadzenie

Celem projektu było opracowanie architektury wspierającej diagnostykę neurologiczno-psychiatryczną z użyciem dużych modeli językowych. Punkt wyjścia stanowiła hipoteza, że role-based multi-agent system, w którym różni agenci analizują przypadek z perspektywy własnej specjalności, może poprawić jakość diagnozy w złożonych przypadkach klinicznych. W docelowej wersji agent neurologiczny, neurochirurgiczny, psychologiczny i psychiatryczny mieliby analizować dane pacjenta, formułować hipotezy, krytykować argumenty innych agentów i doprowadzać do bardziej spójnego rozstrzygnięcia.

W praktyce pierwszym problemem okazała się nie sama debata, ale wiarygodność pojedynczego strumienia wnioskowania. LLM może wygenerować odpowiedź brzmiącą przekonująco, nawet gdy opiera się na słabym materiale, błędnie interpretuje źródło albo nie rozpoznaje, że dowody są niejednoznaczne. Dlatego zdecydowaliśmy, że w tym semestrze najpierw budujemy ugruntowany, mierzalny rdzeń RAG i dopiero na nim będziemy rozwijać architekturę multiagentową.

Ta decyzja zmieniła nacisk projektu z "zbudować od razu kilku agentów" na "zbudować system, w którym każdy przyszły agent będzie musiał pracować na jawnych źródłach, cytować evidence i przechodzić przez te same mechanizmy walidacji". Uważamy, że jest to dojrzalsze podejście badawcze, ponieważ debata agentów bez kontroli źródeł mogłaby zwiększać objętość i pozorną pewność odpowiedzi, ale niekoniecznie jej poprawność.

## 2. Zakres wykonanych prac

W semestrze wykonaliśmy następujące elementy projektu:

| Obszar | Status | Co powstało |
|---|---|---|
| Przegląd literatury i pozycjonowanie problemu | wykonane | Notatki badawcze o luce między retrieval quality, citation faithfulness i evidence-to-decision. |
| Dobór danych i benchmarków | wykonane częściowo | PubMedQA PQA-L 500 jako główny benchmark, clinical safety golden set, NICE Guidelines retrieval 500, planowane HealthSearchQA/BioASQ/MedQA. |
| Pipeline danych | wykonane | Kontrakt PubMed `pubmed_reviews_v1`, pipeline czyszczenia i chunkowania, repo-safe benchmark artifacts. |
| Retrieval i indeksowanie | wykonane | MedCPT embeddings, Qdrant, BM25 sparse vectors, RRF fusion, candidate expansion, reranking i metadata-aware scoring. |
| Aplikacja RAG | wykonane | FastAPI, statyczny UI, endpointy `/api/chat`, `/api/rag/trace`, `/api/search`, tryby `medical_chat` i `benchmark_pqal`. |
| Evidence judge | wykonane | Warstwa rules/LLM/classifier, integracja klasyfikatora BioLinkBERT dla PQA-L. |
| Guardraile i bezpieczeństwo | wykonane częściowo | Red-flag routing, odmowa przy low evidence, walidacja cytowań, answer-quality gate, extractive fallback; safety suite ujawnia nadal istotne braki. |
| Ewaluacja i ablacje | wykonane | Pełny PQA-L 500, quick diagnostics, oracle evidence prompts, LLM judge ablations, classifier ablation, option-ranker diagnostics, NICE retrieval analysis. |
| Pełny MAS z debatą agentów | niezakończone | Przygotowaliśmy architekturę bazową i hipotezy dla przyszłej warstwy agentów, ale nie raportujemy jeszcze zwalidowanej poprawy przez debatę. |

## 3. Architektura systemu

Obecna implementacja nosi nazwę MedChat i jest sekwencyjną aplikacją RAG, a nie pełnym systemem multiagentowym. Składa się z kilku warstw, które docelowo mogą być współdzielone przez agentów specjalistycznych.

Główne komponenty to:

| Komponent | Rola |
|---|---|
| FastAPI app | Obsługuje API, chat, trace, search, konfigurację i zależności runtime. |
| Static UI | Prosty interfejs czatu do testowania systemu lokalnie. |
| Ollama provider | Lokalny dostawca LLM, domyślnie `qwen2.5:7b`. |
| Embedding service | Osobny serwis HTTP dla MedCPT query/document embeddings i hybrydowego wyszukiwania. |
| Qdrant | Baza wektorowa dla dense `medcpt_dense` i sparse `bm25_sparse`. |
| PreRetriever | Normalizacja zapytań, klasyfikacja intencji, rozwijanie skrótów, query rewrite i filtry metadanych. |
| MedicalKnowledgeRetriever | Hybrydowe wyszukiwanie dense + BM25 oraz łączenie wyników przez RRF. |
| PostRetriever | Deduplicacja, cross-encoder reranking, evidence scoring, selekcja źródeł i budowa kontekstu. |
| EvidenceJudge | Decyzja na podstawie źródeł: rules, LLM albo classifier. |
| Guardrails | Red flags, odmowy, walidacja cytowań, naprawa cytowań, answer-quality gate, fallback ekstrakcyjny. |
| Evaluation suite | Powtarzalne benchmarki i bramki regresji. |

Przepływ jednego zapytania w trybie klinicznym wygląda następująco:

1. System sprawdza, czy pytanie zawiera objawy alarmowe wymagające natychmiastowej eskalacji.
2. `PreRetriever` normalizuje pytanie, klasyfikuje intencję, rozwija skróty i przygotowuje warianty zapytań.
3. Retriever wysyła zapytanie do embedding-service, gdzie liczony jest embedding MedCPT Query Encoder i wektor BM25.
4. Qdrant wykonuje wyszukiwanie hybrydowe, a wyniki są łączone przez reciprocal rank fusion.
5. `PostRetriever` deduplikuje dokumenty, opcjonalnie uruchamia `ncbi/MedCPT-Cross-Encoder`, ocenia evidence score i buduje kontekst.
6. `EvidenceJudge` ocenia, czy źródła wspierają odpowiedź, zaprzeczają jej, są niejednoznaczne albo niewystarczające.
7. Writer LLM generuje odpowiedź tylko wtedy, gdy evidence i guardraile na to pozwalają.
8. Odpowiedź przechodzi walidację cytowań, ocenę jakości i ewentualny fallback ekstrakcyjny.

W trybie `benchmark_pqal` część zachowań pacjenckich jest celowo wyłączona. Benchmark PQA-L nie jest poradą dla pacjenta, tylko zadaniem klasyfikacji evidence do etykiet `yes`, `no`, `maybe`. Dlatego tryb benchmarkowy pomija red-flag routing i query rewrite, a po znalezieniu PMID może podmienić wybrany chunk na pełny oficjalny abstrakt z korpusu PQA-L. Gold label nie jest używany ani do wyboru źródła, ani do decyzji.

## 4. Dane i źródła wiedzy

W projekcie rozdzieliliśmy dane produkcyjne, repo-safe benchmarki i dane eksperymentalne.

Pipeline PubMed `pubmed_reviews_v1` obejmuje abstrakty z ostatnich 5 lat, rekordy anglojęzyczne, review/systematic review oraz wykluczenie publikacji wycofanych. Udokumentowany przebieg pipeline'u obejmował:

| Etap | Liczba |
|---|---:|
| Wejściowe PMID | 990 391 |
| Pobrane rekordy metadanych | 982 019 |
| Finalne dokumenty | 977 777 |
| Finalne chunki | 977 777 |

Pełny korpus nie jest commitowany do repozytorium. W repo znajdują się natomiast skrypty, kontrakty danych, sample, benchmarki oraz manifesty pozwalające odtworzyć przebieg eksperymentów.

Główne wykorzystane lub przygotowane zbiory:

| Zbiór | Cel |
|---|---|
| PubMedQA official PQA-L 500 | Główny held-out benchmark evidence-to-decision `yes/no/maybe`. |
| PubMedQA quick balanced90 i first100_yes | Szybka diagnostyka przed pełnym runem. |
| Clinical safety golden set | Mały zestaw przypadków wysokiego ryzyka dla trybu `medical_chat`. |
| NICE Guidelines retrieval 500 | Ewaluacja jakości retrievalu na wytycznych klinicznych. |
| PubMed sample i medical_documents sample | Lokalne smoke testy. |
| MedQA archive | Historyczny kontekst/SFT, obecnie nieaktywny jako główny benchmark. |

## 5. Decyzje projektowe

Najważniejszą decyzją było rozdzielenie benchmark mode i product mode. PubMedQA PQA-L mierzy, czy system potrafi z tekstu artykułu wywnioskować `yes`, `no` albo `maybe`. Tryb pacjencki musi natomiast rozpoznawać objawy alarmowe, odmawiać odpowiedzi przy słabych dowodach, unikać definitywnej diagnozy i kierować do lekarza w sytuacjach ryzykownych. Łączenie tych dwóch celów w jedną metrykę prowadziłoby do błędnych wniosków.

Drugą decyzją było potraktowanie retrievalu, cytowań i decyzji evidence-to-conclusion jako osobnych etapów. Początkowo można było oczekiwać, że jeżeli RAG znajdzie właściwe źródło i poprawnie je zacytuje, to LLM poprawnie odpowie. Eksperymenty pokazały, że to założenie jest fałszywe: `source_hit_at_1 = 0.980` i `citation_pass_rate = 1.000` nadal nie gwarantowały poprawnej decyzji `yes/no/maybe`.

Trzecią decyzją był wybór wyspecjalizowanych modeli biomedycznych do retrievalu. Użyliśmy MedCPT Query Encoder i MedCPT Article Encoder, ponieważ są przeznaczone do par zapytanie biomedyczne - artykuł biomedyczny. Zamiast polegać tylko na dense embeddings, dodaliśmy sparse BM25 i fusion w Qdrant. Pozwala to zachować zarówno dopasowanie semantyczne, jak i leksykalne.

Czwartą decyzją było dodanie dedykowanej warstwy evidence judge. Warianty rules/LLM okazały się niewystarczające, dlatego przygotowaliśmy integrację klasyfikatora BioLinkBERT dla zadania `question + evidence -> yes/no/maybe`. Kodowa nazwa metody w runtime pozostaje generyczna (`deberta_classifier`), ale najlepszy raportowany checkpoint to `pubmedqa_biolinkbert_seed47`.

Piątą decyzją było świadome odsunięcie pełnej debaty agentów na kolejny etap. Uważamy, że MAS ma sens dopiero wtedy, gdy każdy agent jest zmuszony do pracy na tych samych zasadach: jawne źródła, cytowania, evidence decision, uncertainty i safety gate. Bez tego multi-agent debate mogłaby generować większą objętość odpowiedzi bez realnej poprawy jakości.

## 6. Eksperymenty i ablacje

### 6.1. PubMedQA PQA-L 500

Główny benchmark paper-facing to PubMedQA official PQA-L 500. Każdy przypadek wymaga odpowiedzi `yes`, `no` albo `maybe` na podstawie abstraktu naukowego. W trybie RAG system najpierw wyszukuje źródło, a następnie warstwa decyzji interpretuje evidence.

Najważniejsze pełne wyniki:

| Wariant | Cases | Label accuracy | Source hit@1 | Citation pass | Mean latency |
|---|---:|---:|---:|---:|---:|
| Majority baseline (`yes`) | 500 | 0.552 | - | - | - |
| RAG + rules/LLM judge | 500 | 0.536 | 0.980 | 1.000 | 1103.9 ms |
| RAG + BioLinkBERT classifier | 500 | 0.720 | 0.980 | 1.000 | 1367.9 ms |

Wynik BioLinkBERT classifier jest obecnie główną wartością raportowaną dla pełnego PQA-L 500. Poprawa względem rules/LLM judge wyniosła `+18.4 pp` label accuracy, przy takim samym `source_hit_at_1`. To jest najważniejszy argument badawczy: największy zysk nie pochodził z kolejnej sztuczki retrievalowej, tylko z lepszej warstwy evidence-to-decision.

Per-label accuracy najlepszego pełnego runu:

| Label | Liczba | Accuracy | Poprawne |
|---|---:|---:|---:|
| yes | 276 | 0.761 | 210 / 276 |
| no | 169 | 0.864 | 146 / 169 |
| maybe | 55 | 0.073 | 4 / 55 |

Macierz pomyłek:

| True \ Pred | yes | no | maybe |
|---|---:|---:|---:|
| yes | 210 | 53 | 13 |
| no | 17 | 146 | 6 |
| maybe | 26 | 25 | 4 |

Interpretacja jest jednoznaczna: klasyfikator bardzo poprawił `yes` i `no`, ale praktycznie nie rozwiązał klasy `maybe`. System przewidział `maybe` tylko 23 razy na 500 przypadków, podczas gdy w zbiorze było 55 prawdziwych przypadków `maybe`. W medycynie ta klasa jest szczególnie ważna, bo odpowiada sytuacji, w której dowody są niejednoznaczne albo niewystarczające.

### 6.2. Prompt i oracle-evidence ablations

Sprawdziliśmy, czy problem evidence-to-decision da się rozwiązać samym promptowaniem LLM. Na szybkim balanced90 dla qwen2.5:7b uzyskaliśmy:

| Wariant | Cases | Accuracy | Macro F1 | yes recall | no recall | maybe recall |
|---|---:|---:|---:|---:|---:|---:|
| Direct, no evidence | 90 | 0.322 | 0.165 | 0.000 | 0.000 | 0.967 |
| Oracle evidence, compact | 90 | 0.522 | 0.507 | 0.767 | 0.333 | 0.467 |
| Oracle evidence, definitions | 90 | 0.533 | 0.528 | 0.700 | 0.467 | 0.433 |
| Oracle evidence, cite-then-answer | 90 | 0.478 | 0.442 | 0.800 | 0.500 | 0.133 |
| Oracle evidence, sufficiency-first | 90 | 0.456 | 0.435 | 0.667 | 0.200 | 0.500 |

Wniosek: evidence pomaga, ale format instrukcji nie wystarcza. Nawet przy oracle evidence najlepszy wariant osiągnął około `0.533` accuracy. To osłabia hipotezę, że wystarczy dodać lepszy prompt typu "najpierw oceń wystarczalność dowodów".

### 6.3. LLM judge ablation

Porównaliśmy także różne modele jako evidence judge. Wyniki diagnostyczne nie pokazały, żeby większy lub bardziej medycznie nazwany LLM automatycznie rozwiązywał problem.

Balanced diagnostic:

| Model | Cases | Accuracy | Maybe acc/recall | No acc/recall | Hit@1 | Citation | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | 30 | 0.467 | 0.000 | 0.600 | 1.000 | 1.000 | 5093.6 ms |
| BioMistral-7B Q4_K_M | 90 | 0.567 | 0.300 | 0.567 | 0.978 | 0.989 | 1824.1 ms |
| MedGemma-27B Q4_K_M | 90 | 0.544 | 0.333 | 0.533 | 0.978 | 0.989 | 4249.1 ms |

First-yes diagnostic:

| Model | Cases | Accuracy | Hit@1 | Citation | Latency |
|---|---:|---:|---:|---:|---:|
| qwen2.5:7b | 30 | 0.767 | 1.000 | 0.967 | 1578.2 ms |
| BioMistral-7B Q4_K_M | 100 | 0.710 | 0.990 | 1.000 | 1392.5 ms |
| MedGemma-27B Q4_K_M | 100 | 0.650 | 0.990 | 1.000 | 3910.0 ms |

Wniosek: LLM judge jest przydatnym baseline'em, ale nie powinien być traktowany jako domyślnie wiarygodny komponent medycznego rozumowania.

### 6.4. RAG vs classifier na balanced90

Na szybkim balanced90 porównaliśmy wariant RAG z LLM judge i RAG z classifier decision layer:

| Wariant | Cases | Accuracy | Macro F1 | yes recall | no recall | maybe recall | Hit@1 | Citation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG LLM top1 | 90 | 0.533 | 0.509 | 0.767 | 0.600 | 0.233 | 0.978 | 0.989 |
| RAG classifier top1 | 90 | 0.644 | 0.572 | 0.900 | 0.900 | 0.133 | 0.978 | 1.000 |

Ten wynik jest diagnostyczny, nie główny paper-facing. Pokazuje jednak ten sam wzorzec co pełny PQA-L 500: klasyfikator poprawia wynik zagregowany i klasy `yes/no`, ale nadal ma problem z `maybe`.

### 6.5. Option-ranker

Przygotowaliśmy też eksperymentalny option-ranker, który punktuje trzy alternatywy:

```text
question + evidence + yes    -> score
question + evidence + no     -> score
question + evidence + maybe  -> score
```

Pierwsze logi treningowe wyglądały obiecująco pod kątem accuracy, ale ujawniły problem nierównowagi:

| Epoch | Dev accuracy | Macro F1 | Balanced accuracy | Maybe recall | Predicted maybe |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.947 | 0.633 | 0.635 | 0.000 | 0 |
| 2 | 0.952 | 0.636 | 0.638 | 0.000 | 0 |
| 3 | 0.961 | 0.642 | 0.644 | 0.000 | 0 |

Wniosek: option-ranker jest ciekawą architekturą decyzyjną, ale nie można jej uznać za poprawioną, dopóki nie poprawi held-out zachowania dla `maybe`. Sama wysoka accuracy na niezbalansowanym dev secie jest myląca.

### 6.6. NICE Guidelines retrieval 500

Dodatkowo przeanalizowaliśmy retrieval na 500 przypadkach dla korpusu NICE Guidelines. Konfiguracja:

| Parametr | Wartość |
|---|---|
| `RAG_CANDIDATE_K` | `50` |
| `RAG_TOP_K` | `5` |
| `CROSS_ENCODER_MODEL` | `ncbi/MedCPT-Cross-Encoder` |
| `RAG_RETRIEVER` | `embedding_service` |
| `RAG_CORPUS_VERSION` | `nice-guidelines-v1` |

Rozkład intencji:

| Intent | Liczba | Udział |
|---|---:|---:|
| treatment | 264 | 52.8% |
| diagnosis | 104 | 20.8% |
| prevention | 69 | 13.8% |
| general | 63 | 12.6% |

Główne metryki:

| Metryka | Wartość | Interpretacja |
|---|---:|---|
| Mean latency | 59.5 ms | Bardzo dobra latencja retrieval/reranking. |
| MRR | 0.6980 | Dobry wynik, obniżany przez trafienia na pozycji 2. |
| Recall@5 | 0.9600 | Poprawny wynik prawie zawsze jest w Top-5. |
| Recall@10 | 0.9680 | Top-10 daje mały przyrost względem Top-5. |
| Precision@5 | 0.1920 | Blisko maksimum `0.20`, jeśli jest jeden relevant wynik. |
| Precision@10 | 0.0968 | Blisko maksimum `0.10` dla jednego relevant wyniku. |
| nDCG@5 | 0.7654 | Problem głównie w ustawieniu wyniku na pozycji 1. |
| nDCG@10 | 0.7680 | Minimalna poprawa po dodaniu pozycji 6-10. |

Rozkład pierwszego relevant rank:

| Pierwszy relevant rank | Liczba | Udział |
|---:|---:|---:|
| 1 | 223 | 44.6% |
| 2 | 242 | 48.4% |
| 3 | 9 | 1.8% |
| 4 | 4 | 0.8% |
| 5 | 2 | 0.4% |
| 6-8 | 4 | 0.8% |
| brak w Top-10 | 16 | 3.2% |

Najważniejszy wniosek z NICE jest inny niż z PubMedQA. Tutaj retrieval ma bardzo dobry recall, ale słaby Top-1. `Top-1 accuracy = 44.6%`, natomiast `Top-2 accuracy = 93.0%`. Oznacza to, że system zwykle znajduje właściwy dokument, ale zbyt często ustawia go na pozycji 2. Zwiększanie `Top-K` z 5 do 10 nie rozwiąże problemu, ponieważ `Recall@10` rośnie tylko o `0.8 pp`.

Najbardziej podejrzanym wzorcem było błędne promowanie dokumentów-magnesów. Dokument `nice-cg103` ("Delirium: prevention, diagnosis and management in hospital and long-term care") pojawił się jako błędny Top-1 aż 102 razy, czyli w `20.4%` całego zbioru. To sugeruje, że pewne ogólne guideline'y mogą mieć zbyt dużą atrakcyjność rankingową.

Zidentyfikowaliśmy też problem debugowalności score. W danych końcowy `score` wyglądał jak funkcja pozycji w rankingu, a nie jak surowa pewność semantyczna modelu. To utrudnia progowanie jakości, wykrywanie sytuacji "system nie wie" i analizę różnicy między Top-1 a Top-2. Rekomendacja na kolejny etap to logowanie osobno raw retriever score, raw cross-encoder score i final fusion score.

### 6.7. Clinical safety golden set

Tryb `medical_chat` przetestowaliśmy na małym golden secie bezpieczeństwa klinicznego. Wyniki pierwszego realnego end-to-end runu:

| Metryka | Wartość |
|---|---:|
| Cases | 12 |
| Critical cases | 5 |
| Case pass rate | 0.500 |
| Behavior pass rate | 0.500 |
| Safety pass rate | 0.667 |
| Severe harm count | 4 |
| Refusal rate | 1.000 |
| Urgent referral pass rate | 0.200 |
| Forbidden violation rate | 0.000 |
| Mean latency | 12 339.8 ms |

Ten wynik nie jest sukcesem produktowym, ale jest ważnym wynikiem inżynierskim: safety suite działa i ujawnia realne ryzyka, których PQA-L nie mierzy. Największy problem dotyczył emergency triage i negation/contraindication. To znaczy, że system nie powinien być przedstawiany jako gotowy asystent pacjencki, mimo dobrych wyników retrievalu i PQA-L.

## 7. Dyskusja

Najważniejsza obserwacja z semestru brzmi: w medycznym RAG retrieval quality nie jest tym samym co decision quality. Na PubMedQA system znajdował właściwy PMID na pierwszej pozycji w `98.0%` przypadków i przechodził walidację cytowań w `100.0%`, ale LLM/rules judge nadal osiągał tylko `53.6%` accuracy. To oznacza, że wniosek medyczny nie wynika automatycznie z posiadania właściwego źródła w kontekście.

Druga obserwacja dotyczy niejednoznaczności. Klasa `maybe` nie jest pobocznym szczegółem benchmarku. W realnej medycynie zdanie "dowody są niewystarczające" bywa bezpieczniejszą i bardziej poprawną odpowiedzią niż wymuszone `yes` albo `no`. Nasz najlepszy klasyfikator nadal niemal nie rozpoznaje tej klasy. To jest główny kierunek przyszłych badań: nie tylko accuracy, ale sufficiency, uncertainty i calibrated confidence.

Trzecia obserwacja dotyczy multiagentowości. Wyniki nie dowodzą jeszcze, że MAS poprawia diagnozę. Dowodzą natomiast, że przyszły MAS powinien być oceniany etapowo: czy agenci poprawiają retrieval, czy poprawiają interpretację evidence, czy lepiej wykrywają sprzeczności, czy lepiej rozpoznają `maybe`, i czy nie pogarszają safety. Sama debata agentów nie powinna być traktowana jako wartość sama w sobie.

Czwarta obserwacja dotyczy NICE. W tym benchmarku bottleneck jest bardziej rankingowy niż decyzyjny: poprawny dokument jest bardzo często w Top-2, ale zbyt rzadko na Top-1. To prowadzi do praktycznej rekomendacji, aby nie zwiększać mechanicznie Top-K, tylko poprawić rozstrzyganie Top-1/Top-2, debugować dokumenty-magnesy i logować surowe score'y.

## 8. Ograniczenia

Najważniejsze ograniczenia obecnego stanu projektu:

1. Nie mamy jeszcze zwalidowanej warstwy multiagentowej z debatą specjalistów. Obecna implementacja to rdzeń RAG i evidence decision layer.
2. PQA-L 500 mierzy evidence classification, ale nie mierzy pełnej jakości diagnozy neurologiczno-psychiatrycznej.
3. Clinical safety golden set jest mały i pierwszy realny run nie przeszedł bramek bezpieczeństwa.
4. Klasa `maybe` pozostaje bardzo słaba mimo poprawy aggregate accuracy.
5. Pełny produkcyjny korpus PubMed i embeddingi nie są commitowane do repozytorium, więc pełna odtwarzalność wymaga lokalnych artefaktów i manifestów.
6. NICE retrieval 500 pokazał problem Top-1 ranking i dokumentów-magnesów.
7. Obecne metryki answer-quality nie są w pełni dopasowane do krótkich odpowiedzi classifier-only, dlatego w PQA-L raportujemy przede wszystkim label accuracy, retrieval hit i citation pass.
8. Nie przeprowadziliśmy jeszcze walidacji na rzeczywistych przypadkach neurologiczno-psychiatrycznych z obrazowaniem CT/MRI.

## 9. Realizacja celów projektu

| Cel z opisu projektu | Stopień realizacji | Komentarz |
|---|---|---|
| Przegląd LLM/RAG/MAS w diagnostyce medycznej | częściowo wykonane | Powstały notatki badawcze i positioning paper, szczególnie wokół medical RAG i evidence-to-decision. |
| Analiza i dobór danych/benchmarków | wykonane | Wybrano PQA-L 500, clinical safety golden, NICE 500; zaplanowano HealthSearchQA, BioASQ, MedQA. |
| Architektura agentów neurolog/neurochirurg/psycholog/psychiatra | koncepcyjnie | Zdefiniowano potrzebny rdzeń RAG i przyszłą rolę agentów, ale nie zaimplementowano pełnej debaty. |
| Pipeline Query/Retrieval/Augmentation/Generation | wykonane | Działa pełny pipeline RAG z pre/post retrieval, evidence judge i writerem. |
| Integracja walidowanej bazy wiedzy i RAG | wykonane częściowo | PubMed i PQA-L działają; NICE retrieval przeanalizowany; pełny multi-corpus kliniczny jest future work. |
| Procedura ewaluacyjna | wykonane | Framework rozdziela PQA-L, retrieval, cytowania, safety, regression gates i error analysis. |
| Eksperymenty single-agent / ablation | wykonane | Przeprowadzono rules/LLM/classifier, prompt, oracle, LLM judge, option-ranker, NICE retrieval i safety diagnostics. |
| Raport/paper | wykonane wstępnie | Obecny dokument syntetyzuje wyniki semestru i przygotowuje materiał pod artykuł. |

## 10. Plan dalszych prac

W kolejnym etapie chcemy przejść od stabilnego rdzenia RAG do rzeczywistej architektury agentowej. Proponowany plan:

1. Dodać agentów specjalistycznych jako warstwę nad wspólnym evidence store: neurolog, neurochirurg, psychiatra, psycholog/neuropsycholog.
2. Wymusić na każdym agencie format: hipoteza, evidence, przeciwwskazania, uncertainty, pytania do uzupełnienia, ryzyka.
3. Dodać verifier/judge, który nie streszcza debaty, tylko ocenia, czy zmieniła ona decyzję i czy poprawiła evidence sufficiency.
4. Testować MAS przede wszystkim na przypadkach trudnych: `maybe`, niska pewność klasyfikatora, konflikty źródeł, błędny Top-1, high-risk triage.
5. Poprawić NICE Top-1 ranking przez analizę raw score, chunków dokumentów-magnesów i final fusion.
6. Rozwinąć clinical safety golden set o więcej przypadków neurologicznych: stroke, seizure/status epilepticus, CNS infection, raised intracranial pressure, cauda equina, suicidal ideation, acute psychosis.
7. Dodać benchmarki zewnętrzne dopiero po jasnym kontrakcie danych i metryk: HealthSearchQA dla pytań pacjenckich, BioASQ dla retrieval/evidence QA, MedQA dla rozumowania egzaminacyjnego.
8. Poprawić obsługę `maybe` przez balanced training, calibration, sufficiency loss albo osobny model wykrywający niewystarczalność evidence.

## 11. Wnioski

W semestrze zrealizowaliśmy znaczący, mierzalny etap projektu. Nie powstał jeszcze pełny system wielu agentów prowadzących debatę diagnostyczną, ale powstała warstwa, bez której taka debata byłaby trudna do naukowej oceny. Zbudowaliśmy lokalny medyczny RAG, pipeline danych, embedding-service, Qdrant hybrid retrieval, reranking, evidence judge, guardraile, tryby benchmarkowe i kliniczne oraz framework ewaluacyjny.

Najważniejszy wynik badawczy jest następujący: system może bardzo skutecznie znaleźć i zacytować właściwe źródło, a mimo to błędnie wyciągnąć wniosek medyczny. Dedykowany klasyfikator BioLinkBERT poprawił PubMedQA PQA-L 500 z `0.536` do `0.720` label accuracy przy `source_hit_at_1 = 0.980` i `citation_pass_rate = 1.000`, ale `maybe` pozostało głównym ograniczeniem. To daje mocny kierunek paperowy: medyczny RAG wymaga etapowej ewaluacji i osobnej warstwy evidence-to-decision, a nie tylko lepszych cytowań.

Analiza NICE 500 pokazała z kolei, że retrieval guideline'ów jest blisko bardzo dobrego wyniku pod względem coverage (`Recall@5 = 0.960`), ale wymaga poprawy rankingu Top-1. Clinical safety golden set pokazał, że tryb pacjencki nadal nie jest gotowy do użycia bez dalszych prac nad triage i bezpieczeństwem.

Podsumowując, osiągnęliśmy główne cele infrastrukturalne i badawcze semestru: mamy działający system, powtarzalne benchmarki, wyniki ablacyjne, zidentyfikowane bottlenecki i jasny plan przejścia do multiagentowości. Kolejny etap powinien skupić się nie na samej liczbie agentów, ale na tym, czy agenci poprawiają interpretację evidence, rozpoznawanie niepewności i bezpieczeństwo decyzji.

## Bibliografia

[1] Jin, D. et al. What Disease Does This Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams. *Applied Sciences*, 2021.  
[2] Singhal, K. et al. Towards expert-level medical question answering with large language models. arXiv, 2023.  
[3] Tu, T. et al. Towards conversational diagnostic AI. *Nature*, 2024.  
[4] Johnson, A. E. W. et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 2023.  
[5] NEJM Clinicopathological Conferences / Case Records.  
[6] PubMedQA official repository and PQA-L benchmark.  
[7] NICE Guidelines retrieval corpus, internal evaluation report, 500 cases.  
[8] RAGAS and ARES evaluation literature for stage-separated RAG evaluation.  
[9] BioASQ Task B benchmark documentation for biomedical retrieval and QA.  
[10] Project repository documentation: `README.md`, `docs/evaluation/evaluation-framework.md`, `docs/research/pubmedqa-evidence-to-decision-findings.md`, `docs/research/medical-rag-research-gap-and-positioning.md`.
