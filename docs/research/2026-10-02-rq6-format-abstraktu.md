# RQ6 — czy `maybe` zależy od formatu abstraktu (2026-10-02)

Skrypt: `scripts/agents/analyze_rq6_format.py` → `reports/debate/analysis/rq6_format.json`;
test: `tests/test_rq6_format.py`. Odtworzenie: `.venv/bin/python scripts/agents/analyze_rq6_format.py`.

**Wynik: nie.** Cechy formatu nie przewidują ani gold `maybe`, ani niezgody annotatorów.

Regresja logistyczna (L2, cechy standaryzowane, bootstrap 2000, seed 47) na 1000 pytaniach PQA-L.
Cechy: log tokenów kontekstu, liczba sekcji, log liczb, obecność p-value, gęstość hedgingu w kontekście,
log tokenów pytania. **Konkluzji nie używamy** — model jej nie widzi.

| Cel | n dodatnich | CV AUROC (wszystkie cechy) |
|---|---:|---:|
| gold `maybe` | 110 | **0.517** |
| annotatorzy niezgodni | 299 | **0.508** |

Iloraz szans na 1 SD, gold `maybe` — **wszystkie przedziały obejmują 1**: długość kontekstu 0.87 [0.69, 1.08],
sekcje 1.17 [0.96, 1.40], liczby 1.12 [0.89, 1.44], p-value 0.86 [0.69, 1.07], hedging w kontekście 1.18 [0.96, 1.42],
długość pytania 1.11 [0.91, 1.34]. Najlepsza pojedyncza cecha: hedging w kontekście, AUROC 0.540.

**Wniosek:** format nie tłumaczy `maybe` ani sporów — spójne z H1/H1b (hedging to styl) i z tezą, że `maybe` to
rozstrzygnięty spór, a nie własność tekstu. Moc: 110 dodatnich wykryłoby efekt rzędu OR ≈ 1.4 na SD; mniejszego nie.

**Nie zrobione:**
- typ pytania (brak pola w tabeli; wymaga tekstu pytania z `ori_pqal.json`);
- ~~drugi punkt RQ6 (cechy a *predykcje* systemów)~~ — zrobione 2026-10-04, niżej;
- wynik jest nierejestrowany z góry; cechy wzięto z planu, bez strojenia.

## Druga część: cechy a predykcje systemów (2026-10-04)

`.venv/bin/python scripts/agents/analyze_rq6_format.py --systems` → `reports/debate/analysis/rq6_format_systems.json`.
Te same sześć cech i ta sama regresja, 500 pytań testowych, cztery systemy z `analyze_h2_human_ceiling.SYSTEMS`.
**Eksploracyjne**: pytania testowe były wcześniej oglądane, 9 celów × 6 cech bez korekty na wielokrotne porównania.

| Cel | n dodatnich z 500 | CV AUROC | OR na 1 SD z przedziałem poza 1 |
|---|---:|---:|---|
| gold `maybe` | 55 | 0.544 | sekcje 1.34 [1.05, 1.69]; hedging 1.33 [1.01, 1.70] |
| BioLinkBERT mówi `maybe` | 24 | **0.651** | długość kontekstu 1.91 [1.31, 3.10]; sekcje 0.49 [0.23, 0.79] |
| debata dissent mówi `maybe` | 44 | **0.652** | długość kontekstu 1.36 [1.01, 1.98]; p-value 0.47 [0.29, 0.68] |
| debata majority mówi `maybe` | 26 | **0.663** | długość kontekstu 1.78 [1.25, 2.77]; sekcje 0.64 [0.32, 0.92]; p-value 0.66 [0.40, 0.99] |
| SC `qwen3:8b` k=4 mówi `maybe` | 80 | 0.490 | — |
| błąd systemu (każdy z czterech) | 129–143 | 0.463–0.489 | — |

- **Systemy oparte na BioLinkBERT mówią `maybe` według formatu**: częściej przy długim kontekście, rzadziej przy
  wielu sekcjach i przy p-value. Etykieta gold reaguje na te cechy słabiej (0.544) i częściowo w przeciwną stronę
  (sekcje: gold OR 1.34, BioLinkBERT 0.49).
- To nie są trzy niezależne obserwacje: obie debaty dostały podpowiedź BioLinkBERT.
- **SC bez podpowiedzi nie reaguje na format** (0.490), mimo że mówi `maybe` najczęściej.
- **Format nie przewiduje błędów** żadnego systemu.
- Dwa przedziały dla gold `maybe` wychodzą poza 1 na 500 testowych, a na 1000 pytaniach żaden — przy tylu
  porównaniach nie traktować tego jako efektu.
- Moc: 24–44 dodatnich; przedziały OR są szerokie.

Odczyt pod tezę: klasyfikator uczony na etykietach końcowych nie odtwarza `maybe`, tylko korelat powierzchniowy.
Zgodne z RQ8 (detektor nie odzyskuje etykiety). Nadaje się na jedno zdanie w appendixie, z oznaczeniem „exploratory”.
