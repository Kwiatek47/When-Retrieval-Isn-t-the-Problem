# RQ2 — jak inne zbiory konstruują klasę „brak wystarczających informacji" (2026-09-26)

> Domyka RQ2 z Fazy 1 `Notes/PLAN-2026-09-26-po-h1.md`. Dla każdego zbioru cztery pytania
> z planu: kto anotował, ilu annotatorów na pozycję, czy publikują surowe etykiety, jaki
> procent stanowi klasa NEI. Wszystko zweryfikowane w tekstach prac (PDF), nie w streszczeniach.
> **Najważniejszy wynik tego przeglądu dotyczy naszego własnego zbioru** — §2.

## 1. Tabela porównawcza

| Zbiór | Kto anotował | Na pozycję | Surowe etykiety per annotator | Klasa NEI |
|---|---|---|---|---|
| **PubMedQA PQA-L** | 2 × kandydat na M.D. | **2** + dyskusja | **TAK** (`ori_pqal.json`) | **11.0%** (110/1000) |
| **SciFact** | 3 ekspertów NLP, 5 studentów i 5 doktorantów nauk o życiu | 1 (232 pary re-anotowane) | Nie podano w pracy | **36.6%** (516/1409) |
| **HealthVer** | 3 autorzy pracy, ekspertyza w biomedical NLP | 1 (603 pary re-anotowane) | Nie podano w pracy | **42.7%** (6117/14330) |
| **ClinDet-Bench** | 1 lekarz board-certified, 10 lat praktyki | **1** | GitHub, MIT | 34.0% (32/94) |
| **MedQAbstain** | brak anotacji niepewności; ryzyko anotowane **LLM-em** | 0 (walidacja na 100) | HF dataset | 100% z konstrukcji |
| **NEI-CAP** | 2 annotatorów, konsensus | 2 | „release package", App. M | audytuje konstrukcję |

Agreement: SciFact κ=0.75 (etykieta) / 0.71 (zdania racjonalizujące); HealthVer κ=0.76;
NEI-CAP κ=0.732 (94.4% zgody) i κ=0.743 na bramce „prawdziwe NEI vs kontaminacja";
ClinDet-Bench — brak, bo jeden annotator i zadanie deterministyczne („human performance
teoretycznie 100%, więc dodatkowa ewaluacja nie była potrzebna").

## 2. Czego nie wiedzieliśmy o PQA-L — Algorithm 1 z pracy Jina

To jest najcenniejsze znalezisko. Procedura zbierania PQA-L (Jin et al. 2019, Alg. 1)
mówi dwie rzeczy, których nie było w naszej analizie:

**(a) Etykieta końcowa w sporach nie pochodzi od trzeciego arbitra — to negocjacja tych samych
dwóch osób.** „Annotator 1 and Annotator 2 discuss for an agreement annotation." Annotator 1
**trzyma w ręku konkluzję autorów** podczas tej dyskusji. Nasza liczba 215/299 (etykieta końcowa
idzie za annotatorem czytającym konkluzję) dostaje więc mechanizm: to nie jest niezależne
rozstrzygnięcie, tylko spór, w którym jedna strona dysponuje uprzywilejowanym dowodem i może go
pokazać. Zdanie „91.6% jest częściowo definicją" przestaje być interpretacją, a staje się opisem
procedury.

**(b) Pozycje, co do których annotatorzy nie doszli do porozumienia, zostały ze zbioru usunięte.**
„if not ∃ l_a then Remove inst and continue to next iteration." PQA-L jest więc **przefiltrowane
z przypadków nieredukowalnie spornych**. Konsekwencja dla naszego sufitu: **0.780 accuracy mierzymy
na zbiorze już oczyszczonym z pozycji, których ludzie nie potrafili uzgodnić.** Prawdziwy sufit na
niefiltrowanej populacji pytań jest niższy, a `maybe` w PQA-L nigdy nie znaczy „nie do rozstrzygnięcia" —
znaczy „obaj się zgodzili, że `maybe`, albo tak ustalili po rozmowie". To osłabia interpretację
`maybe` jako „pytanie jest obiektywnie niepewne" i wzmacnia interpretację proceduralną.

Potwierdzone przy okazji: obaj annotatorzy to **kandydaci na M.D.**; `ReasoningFreeAnnotation ← l1`
gdzie annotator 1 widzi `long answer`, `ReasoningRequiredAnnotation ← l2` gdzie annotator 2 widzi
tylko kontekst. Nasza poprawka ról jest zgodna z algorytmem. Rozkład PQA-L z Table 1: yes 55.2%,
no 33.8%, maybe 11.0% — co do jednego nasze 552/338/110.

## 3. Pojedynczo, co istotne

**SciFact** (Wadden et al. 2020, EMNLP). 1409 twierdzeń, korpus 5183 abstraktów. Podział
Train/Dev/Test: 809/300/300; NOINFO 304/112/100, razem **516**. Twierdzenia pisane z *citance*
(zdania cytującego), więc są naturalne, nie syntetyczne. Weryfikatorzy anotowali pary
twierdzenie–cytowany abstrakt jako SUPPORTS/REFUTES/NOINFO; dowody znaleziono w 63% cytowanych
abstraktów. **Każde twierdzenie ma jedną etykietę**, κ=0.75 na 232 ponownie anotowanych parach.
Istotne dla nas: NOINFO na poziomie twierdzenia jest anotowane przez człowieka, ale w treningu
weryfikatora pary NEI zwykle pochodzą z abstraktów-dystraktorów (N=4259 w korpusie) — i to jest
dokładnie to, co atakuje NEI-CAP.

**HealthVer** (Sarrouti et al. 2021, Findings of EMNLP). 14 330 par dowód–twierdzenie z 1855
twierdzeń i 80 pytań. Rozkład: Supports 4986, Refutes 3227, **Neutral 6117 — klasa największa**.
Anotowali **trzej autorzy pracy** („expertise in biomedical NLP" — nie klinicyści), κ=0.76 na 603
parach. Kluczowa różnica wobec SciFact, podkreślona przez samych autorów: „Compared to the SciFact
and FEVER datasets which do not include evidence for the NOINFO claims, we provide evidence
statements for NEUTRAL/NOINFO examples." Annotatorów proszono, by dla NEUTRAL wybierali zdania
*relevantne, ale niewystarczające*. To najbliższy naszemu `maybe` sposób konstrukcji w tej grupie.
Uboczne, przydatne: model „claim-only" osiąga accuracy **50.00** (Table 6) — bez dowodu nie ma sygnału.

**NEI-CAP** (arXiv 2605.26663). Najmocniejsza metodologicznie i bezpośrednio o naszym problemie.
Osiem rodzin konstrukcji NEI (placeholder, random irrelevant, position-biased, BM25 near-miss,
cited non-rationale, same-document, fixed-claim, missing-hop). Wynik: **kompetencja NEI nie
przenosi się między konstrukcjami**. Model trenowany na placeholderach ma NEI-F1 **1.000** na
dopasowanym teście i **0.000** na BM25 near-miss oraz cited non-rationale — replikowane na trzech
backbone'ach × 5 ziaren. Na ludzko-adjudykowanych trudnych NEI (54 pozycje): trening placeholderowy
recall **0.000**, trening BM25 near-miss **0.691**. Dekoder Qwen2.5-7B SFT na placeholderach:
hard NEI-F1 **0.054**, human recall **0.011**.

**ClinDet-Bench** (arXiv 2602.22771, ACL 2026 Industry). 94 przypadki na 16 skalach klinicznych
(CHADS2, HAS-BLED, qSOFA, CURB-65, Child-Pugh…), zbudowane i zweryfikowane przez **jednego** lekarza.
Complete 32 / Incomplete-Determinable 30 / Incomplete-Undeterminable 32. Wyniki: accuracy 0.98
przy pełnej informacji, **0.57** na undeterminable (przedwczesne wnioski) i **0.88** na determinable
(zbędna abstynencja). Najważniejsze dla nas: **istotna ujemna korelacja między trafnością na tych
dwóch warunkach, Spearman r = −0.45, p = 0.027** — czyli zmierzony kompromis „przedwczesne
zamknięcie ↔ inflacja abstynencji" we wszystkich modelach i strategiach promptowania.

**MedQAbstain** (Cocchieri et al., ACL 2026, „LLMs (Almost) Never Abstain Under Medical Uncertainty").
Przerabia cztery zbiory MCQA: MedQA (1273), MedMCQA (4183), AfriMed-QA (3590), MedXpertQA (2450),
plus multimodalny MedXpertQA-MM (2000). **Konstrukcja: usuwa się złotą odpowiedź i wstawia opcję
„I abstain"** — więc abstynencja jest poprawna z definicji. Etykiety ryzyka (Safe / Life-Threatening)
generuje **LLM**, walidowane przez eksperta na losowej próbce 100 pozycji. Wynik: modele niemal nigdy
nie abstynują, **nawet gdy samo pytanie jest ukryte**, a abstynencja słabo wiąże się z deklarowaną
pewnością.

## 4. Co z tego wynika dla naszego papera

**(1) PQA-L publikuje surowe etykiety obu annotatorów i to wśród tych zbiorów wyjątek.** SciFact
i HealthVer raportują κ na ponownie anotowanym podzbiorze, ale w pracach nie ma informacji
o udostępnianiu etykiet per annotator. Cały nasz audyt — sufit 0.780/0.473, 215/299, rozkład źródeł
`maybe` — jest policzalny **tylko** dlatego, że `ori_pqal.json` wozi `reasoning_required_pred`
i `reasoning_free_pred`. To jest samodzielne zdanie do Contributions: eksploatujemy własność
benchmarku, której porównywalne zbiory nie mają, więc tej analizy nie da się wprost powtórzyć
na SciFact ani HealthVer.

**(2) Nasza klasa NEI jest jedyną rzadką i niekonstruowaną.** 11.0% vs 36.6% (SciFact),
42.7% (HealthVer, największa klasa), 34.0% (ClinDet), 100% (MedQAbstain z konstrukcji). Pozostałe
zbiory *budują* NEI tak, by było liczne; PubMedQA je *zastaje*. To bezpośrednio uzasadnia RQ9a:
niezgodność prioru (trening 0.13% vs test 11%) jest właściwością naszego ustawienia, nie wadą
wykonania, i nikt inny jej nie ma.

**(3) Nasza zapaść `maybe` zgadza się liczbowo z NEI-CAP na niezależnym zbiorze.** Według ich
taksonomii `maybe` w PQA-L to przypadek „hard NEI": ten sam abstrakt, anotowany przez człowieka,
bez konstrukcji przez nieobecność dowodu. Ich dekoder na hard NEI osiąga recall **0.011** i F1
**0.054**; nasz BioLinkBERT ma recall `maybe` **0.073**. Ten sam rząd wielkości, inny zbiór, inna
architektura. To zamienia nasz wynik negatywny z „nasz model jest słaby" w „tak zachowują się
weryfikatory na niekonstruowanym NEI" — i daje najmocniejszy możliwy cytat do Related Work.

**(4) Kompromis, który zmierzyliśmy, jest zmierzony także gdzie indziej.** ClinDet r=−0.45 to
ta sama oś co nasza „inflacja abstynencji vs przedwczesne zamknięcie" i co RQ10. MedQAbstain
dokłada skrajną kontrolę: modele nie abstynują nawet przy ukrytym pytaniu. Warto ją cytować
z zastrzeżeniem — ich niepewność jest *konstruowana przez usunięcie odpowiedzi*, więc krytyka
z NEI-CAP stosuje się do nich tak samo jak do placeholderów.

**(5) Wniosek z §2 zmienia jedno zdanie o naszym sufycie.** 0.780 jest mierzone na zbiorze
przefiltrowanym z przypadków nieuzgadnialnych. Trzeba to napisać wprost przy tabeli sufitu,
bo inaczej recenzent to znajdzie.

## 5. Czego nie zweryfikowałem

- Czy repozytoria SciFact i HealthVer **faktycznie** zawierają etykiety per annotator — sprawdziłem
  teksty prac i stronę GitHuba HealthVer, nie same pliki danych. Przed napisaniem twierdzenia (1)
  w paperze trzeba zajrzeć do plików.
- URL wydania danych NEI-CAP (praca odsyła do Appendix M, w HTML-u nie było linku).
- Per-modelowe tabele abstynencji w MedQAbstain — wziąłem tylko wynik nagłówkowy.
- Trzy pozycje, które plan wymienia w §8 do Related Work, ale nie w RQ2: spin w abstraktach
  (Boutron; Koroleva DeSpin 2020) i Jiang & de Marneffe 2022. Nieobjęte tym przeglądem.

## 6. Źródła

- [PubMedQA (Jin et al. 2019, EMNLP)](https://aclanthology.org/D19-1259.pdf) — Alg. 1, Table 1
- [SciFact (Wadden et al. 2020, EMNLP)](https://aclanthology.org/2020.emnlp-main.609/) — §3.2, §3.3, Table 2
- [HealthVer (Sarrouti et al. 2021, Findings of EMNLP)](https://aclanthology.org/2021.findings-emnlp.297.pdf) — §3.3, §4.1, Table 2
- [NEI-CAP (arXiv 2605.26663)](https://arxiv.org/abs/2605.26663) — Tables 1, 2, 3, 5, 6
- [ClinDet-Bench (arXiv 2602.22771)](https://arxiv.org/html/2602.22771) — także [ACL Anthology](https://aclanthology.org/2026.acl-industry.47/)
- [MedQAbstain (Cocchieri et al., ACL 2026)](https://aclanthology.org/2026.acl-long.1365/) — §3.1, §3.2, §3.3, Table 1
- Powiązane, spoza listy z planu: [Knowing When to Abstain / MedAbstain (EACL 2026)](https://arxiv.org/abs/2601.12471)
