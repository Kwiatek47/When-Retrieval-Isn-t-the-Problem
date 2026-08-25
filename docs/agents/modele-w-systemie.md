# Modele w systemie — rozpiska

Stan na: 2026-08-25. Zebrane z kodu, `.env` i artefaktów na dysku, nie z założeń projektowych.

---

## 1. Skrót

| model | rola | rozmiar | gdzie działa |
|---|---|---|---|
| **BioLinkBERT-large** (fine-tuned) | klasyfikator yes/no/maybe — baseline i „hint" dla agentów | 1.3 GB, 24L/1024 | lokalnie, PyTorch |
| **qwen2.5:7b** | agenci debaty, przepisywanie zapytań RAG | 4.7 GB Q4_K_M | Ollama |
| **qwen2.5:14b** | supervisor (moderator + Director) | 9.0 GB Q4_K_M | Ollama |
| **qwen2.5:32b** | supervisor w cięższych runach | 19.9 GB Q4_K_M | Ollama |
| **qwen2.5-supervisor-14b-stage1** | SFT supervisora, etap 1 (Director) | 9.0 GB Q4_K_M | Ollama |
| **qwen2.5-supervisor-14b-ft** | SFT supervisora, etap 2 (multitask) | 9.0 GB Q4_K_M | Ollama |
| **MedCPT** (Query + Article Encoder) | embeddingi do wyszukiwania w Qdrant | 768 wymiarów | serwis na porcie 8081 |
| **MedCPT Cross-Encoder** | reranking wyników wyszukiwania | — | `CROSS_ENCODER_MODEL` |
| **nli-deberta-v3-base** | audyt NLI dla klasy `maybe` | — | **moduł nieużywany w pipelinie debaty** |
| **paraphrase-multilingual-MiniLM-L12-v2** | bramka jakości odpowiedzi w czacie RAG | — | `ANSWER_QUALITY_MODEL` |

---

## 2. Ścieżka debaty PubMedQA

To jest ścieżka, na której prowadzone są benchmarki.

### BioLinkBERT-large — klasyfikator
`artifacts/classifier/pubmedqa_biolinkbert_seed47/best`

- `BertForSequenceClassification`, 24 warstwy, hidden 1024, 3 klasy (`yes` / `no` / `maybe`)
- **Dwie role naraz**, co jest istotne przy interpretacji wyników:
  1. **baseline** — punkt odniesienia dla całej architektury (0.726 na `eval.json`, 0.656 na `balanced90`)
  2. **hint** — przy `--hint biolinkbert` jego etykieta trafia do promptów **trzech z czterech agentów**; `uncertainty_advocate` jest ślepy przy `--blind-critic all-rounds`
- Konsekwencja: panel i klasyfikator **nie są niezależni**. Zgodność debaty z BERT sięgała 0.956; dopiero protokół sporu obniżył ją do 0.872.
- Jego `confidence` jest **nieskalibrowana** — 0.97–0.99 również przy błędnych predykcjach. Nadaje się jako sygnał dopiero w skrajnym ogonie (>0.99) i tylko w połączeniu z jednomyślnością panelu.

### qwen2.5:7b — agenci debaty
Ustawiany przez `OLLAMA_MODEL`. Cztery persony (`PUBMEDQA_PERSONAS`):

| persona | zadanie |
|---|---|
| `generalist` | czyta wniosek autorów abstraktu |
| `evidence_skeptic` | krytyka metodologii, bez defaultowania do `maybe` |
| `differential_expander` | alternatywne odczytania danych (od `c06a078` **neutralny kierunkowo**) |
| `uncertainty_advocate` | audyt pokrycia pytania — czy abstrakt w ogóle rozstrzyga |

Wcześniejsze runy v1–v6 używały tu **14B**; zejście na 7B nastąpiło dla przyspieszenia i sprawia, że tamte wyniki nie są bezpośrednio porównywalne.

### qwen2.5:14b / :32b — supervisor
Jedna klasa `SupervisorAgent`, **dwie różne role**:

- **Moderator** (`moderate_round`) — między rundami: streszcza zgodności i sprzeczności, a od `bad573f` **osądza spór** (`DissentAssessment`: kto jest w mniejszości, jaki ma najmocniejszy argument, czy większość na niego odpowiedziała, pytania skierowane do obu stron)
- **Director** (`synthesize_decision`) — przy `--aggregate-mode llm_director` wydaje finalną etykietę

Temperatura 0.2, naprawa JSON przy 0.0.

**32B jest praktycznie porzucony** — jedno wystąpienie w repo. Runy v1–v3 pokazały, że Director jako sędzia przegrywa z prostym głosowaniem większościowym o 5.5–6.7 p.p., więc obecne benchmarki używają `--aggregate-mode majority`, a supervisor pełni już tylko rolę moderatora. Przy 14B pojawia się koszt: ~2.4% moderacji nie parsuje się jako JSON (przy 32B tego nie było); mechanizm `--supervisor-fail peer-round` obsługuje to bez utraty case'a.

### Modele SFT — dotrenowane supervisory
`artifacts/sft/qwen14b-supervisor/` (133 GB artefaktów)

- Baza: `unsloth/Qwen2.5-14B-Instruct-bnb-4bit`, LoRA r=32
- `stage1-director` → eksport `qwen2.5-supervisor-14b-stage1`
- `stage2-multitask` → eksport `qwen2.5-supervisor-14b-ft`
- Dane buduje `scripts/sft/prepare_pubmedqa_supervisor_sft.py` z zapisanych debat
- **Nie były używane w benchmarkach z 23–25 sierpnia** — te szły na bazowym `qwen2.5:14b`

---

## 3. Ścieżka RAG (osobna od debaty)

Pętla debaty **nie woła Qdrant ani retrievalu** — dostaje abstrakt bezpośrednio z korpusu PubMedQA. Poniższe modele obsługują aplikację czatową.

| model | zmienna | rola |
|---|---|---|
| `ncbi/MedCPT-Query-Encoder` | `MEDCPT_QUERY_MODEL` | embedding zapytania |
| `ncbi/MedCPT-Article-Encoder` | `MEDCPT_DOCUMENT_MODEL` | embedding dokumentów |
| `ncbi/MedCPT-Cross-Encoder` | `CROSS_ENCODER_MODEL` | reranking kandydatów |
| `qwen2.5:7b` | `QUERY_REWRITE_MODEL` | przepisywanie zapytań użytkownika |
| `paraphrase-multilingual-MiniLM-L12-v2` | `ANSWER_QUALITY_MODEL` | bramka jakości odpowiedzi |

Kolekcja Qdrant: `MedicalChunk_pubmed_reviews_v1_medcpt_20260518`, wektor `medcpt_dense`, 768 wymiarów. Serwis embeddingowy działa na CPU (`EMBEDDING_DEVICE=cpu`), port 8081.

---

## 4. Model zdefiniowany, ale nieaktywny

**`cross-encoder/nli-deberta-v3-base`** — `app/agents/evidence_audit_nli.py`

Pomysł: zamiast pytać model generatywny „czy to jest niepewne", zapytać dedykowany model NLI, czy abstrakt **pociąga** hipotezę. Docstring modułu podaje uzasadnienie: LLM jako audytor dawał AUROC 0.50–0.56, czyli poziom losowy.

**Nie jest wpięty w `evaluate_debate_pubmedqa.py`.** Zważywszy, że klasa `maybe` pozostaje głównym problemem (0.127 na rozkładzie naturalnym), warto sprawdzić, czy ten moduł działa — to jedyne dostępne podejście do `maybe`, którego jeszcze nie zmierzyliśmy.

Konfiguracja wspomina też ścieżkę `artifacts/classifier/pubmedqa_deberta/best`, ale **tego artefaktu nie ma na dysku** — jest tylko `pubmedqa_biolinkbert_seed47`.

---

## 5. Rozmieszczenie sprzętowe (zmierzone)

| model | VRAM | mieści się na T4 (15.3 GB)? |
|---|---|---|
| qwen2.5:7b | ~6.5 GB | **tak** (~5.4 GB przy 2 slotach) |
| qwen2.5:14b | ~15.2 GB | **nie** — wymaga A40 |
| qwen2.5:32b | ~20 GB | **nie** |
| BioLinkBERT | ~3.0 GB | tak |

**Uwaga o Ollamie w tej instalacji:** używa backendu **Vulkan**, nie CUDA. Zmienna `CUDA_VISIBLE_DEVICES` jest **ignorowana** przy wyborze karty — działa `GGML_VK_VISIBLE_DEVICES`. Dodatkowo każdy `llama-server` trzyma ~4 MB kontekstu graficznego na GPU0 niezależnie od przypisania; to artefakt Vulkana, bez obliczeń.

Weryfikacja rozmieszczenia **musi** iść przez listę procesów (`nvidia-smi --query-compute-apps`), a nie przez `/api/ps` Ollamy — ten drugi raportuje zużycie VRAM, ale nie mówi, na której fizycznej karcie.
