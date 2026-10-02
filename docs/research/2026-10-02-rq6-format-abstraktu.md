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
- drugi punkt RQ6 (cechy a *predykcje* systemów) — raporty 500 pytań (`debate7b_dissent_pqal500_v1.json`)
  nie są na dysku (BRAKI §B2);
- wynik jest nierejestrowany z góry; cechy wzięto z planu, bez strojenia.
