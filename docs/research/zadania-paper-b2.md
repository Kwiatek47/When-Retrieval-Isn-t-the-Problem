# Zadania — Paper B2 (5 osób)

Wpiszcie prawdziwe imię przy „Osoba 5”, jeśli inne niż poniżej.  
Propozycja imion z repo: Antoni, Wiktor, Kamil, Grzegorz — **Osoba 5 = do uzupełnienia**.

| Skrót | Osoba | Rola |
|-------|--------|------|
| **A** | Antoni | Koordynacja, protokół, tekst, profesor |
| **W** | Wiktor | Architektura dyskusji (bez/z supervisor), skrypty B/C |
| **K** | Kamil | Baseline PubMedQA 500, ablacja modeli |
| **G** | Grzegorz | Infra, źródła, lock, powtarzalność |
| **E** | Osoba 5 | Prompty, tabele ewaluacji, logi |

---

## P0
1. Napisać protokół: hipoteza, cel, założenia, co mierzymy — **A**
2. Zaakceptować: PubMedQA 500, zamrożone źródła, bez ról lekarskich — **A** (+ veto zespołu)
3. Ustalić: N niezależnych / # głosów / # rund (koszt ≈ równy) — **A + W + K**
4. Ustalić siatkę modeli do ablacji (producenci, rozmiary) — **K + A**
5. Wpisać: modele, indeks, ziarno → `decyzje-paper-b.md` — **G** (wpis), **A** (akceptacja)
6. Przypisać role w zespole — **A**

## P1
7. Prompty: ten sam prompt startowy (runda 1) — **E**
8. Prompt: runda 2 dyskusji — **E + W**
9. Prompt: supervisor (domknięcie) — **E + W**
10. Skrypt: 1× odpowiedź — **K**
11. Skrypt: N niezależnych + majority — **K**
12. Próba 50 pytań (1× + niezależne) na modelu głównym — **K**
13. Sprawdzić te same źródła — **G**
14. Full 500: 1× + niezależne (model główny) — zamrożony baseline — **K** (+ **G** lock)

## P2
15. Skrypt: dyskusja bez supervisora → majority — **W**
16. Skrypt: dyskusja z supervisorem → final supervisora — **W**
17. Próba 50 (oba warianty dyskusji) — **W**
18. 20 logów — czy dyskutują — **E**
19. Full 500: dyskusja bez supervisora (model główny) — **W**
20. Full 500: dyskusja z supervisorem (model główny) — **W**

## P3
21. Tabela: trafność + koszt (1× / niezależne / dyskusja bez / dyskusja z) — **E**
22. Tabela: tak / nie / może — **E**
23. Tabela: poprawia / psuje vs niezależne — **E**
24. Porównanie: bez supervisora vs z supervisorem — **E + W**
25. Slice „może” — **E**
26. Slice: brak zgody niezależnych — **E**
27. (Opcja) BioLinkBERT na tych samych źródłach — **K**

## P4
28. Ablacja modeli — full 500: 1× + niezależne (każdy model z siatki) — **K**
29. Ablacja modeli — full 500: dyskusja bez / z supervisorem — **W** (run), **K** (kolejka modeli)
30. Tabela: model × wariant — **E + K**

## P5
31. Szkic artykułu — **A**
32. Related work — **A + E**
33. Ograniczenia — **A**
34. Jak powtórzyć — **G**
35. 1 strona dla profesora — **A**
36. Pytania na spotkanie — **A**

## P6 (poza B2)
37. NICE / StatPearls / role lekarskie — nie w tym paperze — **A** (pilnuje)

---

## Widok per osoba

### Antoni (A)
1, 2, 3, 4, 5 (akceptacja), 6, 31, 32, 33, 35, 36, 37

### Wiktor (W)
3, 8, 9, 15, 16, 17, 19, 20, 24, 29

### Kamil (K)
3, 4, 10, 11, 12, 14, 27, 28, 29, 30

### Grzegorz (G)
5, 13, 14 (lock), 34

### Osoba 5 (E)
7, 8, 9, 18, 21, 22, 23, 24, 25, 26, 30, 32
