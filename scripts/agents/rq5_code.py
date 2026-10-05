"""RQ5 — code the blind sheet in the terminal, one question at a time.

Shows the question, the abstract without its conclusion and the authors' conclusion, takes
one code A–F from the codebook (`docs/research/2026-10-04-rq5-codebook-taksonomia-maybe.md`)
and saves after every answer, so the session can be closed and resumed at any point.

The tool reads only the sheet. It never opens the key, the model coder's file or the
sheet's coder columns, and it writes each coder's answers to their own file, so two people
can code independently, also outside the repository: the sheet and this one file (standard
library only, Python 3.8+) are all a coder needs.

Usage:
  python scripts/agents/rq5_code.py --out rq5_codes_wiktor.csv
  python scripts/agents/rq5_code.py --sheet rq5_coding_sheet.csv --out rq5_codes_ania.csv

Commands at the prompt:
  A..F [note]   code the question (an optional short note is kept with the code)
  enter         keep the current code, if the question already has one
  b             back to the previous question
  s             skip for now
  g N           go to question N
  l             list all questions with their codes
  k             show the codebook again
  q             save and quit

Scoring afterwards (in the repository):
  python scripts/agents/rq5_maybe_taxonomy.py score --codes-1 rq5_codes_wiktor.csv --codes-2 rq5_codes_ania.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

DEFAULT_SHEET = Path(__file__).resolve().parents[2] / "reports/debate/analysis/rq5_coding_sheet.csv"
CODES = ("A", "B", "C", "D", "E", "F")

# Verbatim from the codebook; tests/test_rq5_code.py checks that it has not drifted.
CATEGORIES = (
    ("A", "Sprzeczne wyniki", "Wyniki w abstrakcie wskazują w przeciwne strony: jedne punkty końcowe za, inne przeciw."),
    ("B", "Brak istotności albo mocy", "Efekt nieistotny statystycznie, mała próba, autorzy piszą o trendzie lub potrzebie dalszych badań."),
    ("C", "Wynik częściowy albo warunkowy", "Efekt jest, ale tylko w podgrupie, przy pewnym warunku albo dla części tego, o co pyta tytuł."),
    ("D", "Pytanie szersze niż badanie", "Projekt badania nie może rozstrzygnąć pytania: np. pytanie przyczynowe, badanie obserwacyjne; pytanie ogólne, jeden ośrodek."),
    ("E", "Inna populacja albo miara", "Badanie mierzy coś innego niż to, o co pyta tytuł: inna populacja, zastępczy punkt końcowy."),
    ("F", "Brak widocznej przyczyny", "Tekst wygląda na rozstrzygający (yes albo no); kodujący nie widzi powodu dla `maybe`."),
)
RULES = (
    "A wobec C: przeciwne kierunki → A; jeden kierunek z zastrzeżeniem → C.",
    "B wobec D: niepewność z liczb (p, n) → B; niepewność z projektu badania → D.",
    "F tylko wtedy, gdy żadna z A–E nie pasuje. F nie znaczy „nie wiem” — znaczy „moim zdaniem to nie jest `maybe`”.",
)

FIELDS = ("item", "code", "note", "coded_at")


# ---------------------------------------------------------------- data

def load_sheet(path: Path) -> List[Dict[str, str]]:
    """Items in sheet order; only the text columns are kept."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = {"item", "question", "context", "conclusion"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path} is not the RQ5 sheet (missing columns: {', '.join(sorted(missing))}).")
    return [{k: r[k] for k in ("item", "question", "context", "conclusion")} for r in rows]


def load_codes(path: Path) -> Dict[int, Dict[str, str]]:
    """Codes already given, keyed by item number. A missing file means a fresh start."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {int(r["item"]): r for r in csv.DictReader(fh) if r.get("code", "").strip().upper() in CODES}


def save_codes(path: Path, codes: Dict[int, Dict[str, str]]) -> None:
    """Write all codes, sorted by item, through a temporary file so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for item in sorted(codes):
            writer.writerow({f: codes[item].get(f, "") for f in FIELDS})
    os.replace(tmp, path)


def first_uncoded(items: List[int], codes: Dict[int, Dict[str, str]], start: int = 0) -> Optional[int]:
    """Index of the first uncoded item at or after `start`, wrapping around; None when all are coded."""
    n = len(items)
    for step in range(n):
        i = (start + step) % n
        if items[i] not in codes:
            return i
    return None


def parse_answer(text: str) -> Tuple[str, str]:
    """Return (command, argument). Commands: code, keep, back, skip, goto, list, codebook, quit, invalid."""
    text = text.strip()
    if not text:
        return "keep", ""
    head, _, rest = text.partition(" ")
    word = head.lower()
    if len(head) == 1 and head.upper() in CODES:
        return "code", head.upper() + ("|" + rest.strip() if rest.strip() else "")
    if word in ("b", "back"):
        return "back", ""
    if word in ("s", "skip"):
        return "skip", ""
    if word in ("g", "go") and rest.strip().isdigit():
        return "goto", rest.strip()
    if word in ("l", "list"):
        return "list", ""
    if word in ("k", "codebook", "?"):
        return "codebook", ""
    if word in ("q", "quit", "exit"):
        return "quit", ""
    return "invalid", text


# ---------------------------------------------------------------- display

def width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 40)).columns, 110))


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if sys.stdout.isatty() else text


def clear() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def wrap(text: str, indent: str = "") -> str:
    out = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        label, sep, body = para.partition(": ")
        if sep and label.isupper() and len(label) < 40:  # section label of the abstract, e.g. "METHODS"
            para = f"{bold(label)}: {body}"
        out.append(textwrap.fill(para, width() - len(indent), initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out)


def codebook_text(short: bool = False) -> str:
    lines = []
    for code, name, when in CATEGORIES:
        lines.append(f"  {bold(code)}  {name}" if short else textwrap.fill(
            f"{code}  {name} — {when}", width(), initial_indent="  ", subsequent_indent="      "))
    if not short:
        lines.append("")
        lines += [textwrap.fill(r, width(), initial_indent="  • ", subsequent_indent="    ") for r in RULES]
    return "\n".join(lines)


def show(item: Dict[str, str], position: int, total: int, done: int, current: Optional[Dict[str, str]]) -> None:
    clear()
    line = "─" * width()
    print(f"{bold('Pytanie ' + item['item'])}  ({position + 1}/{total}, zakodowane {done}/{total})")
    print(line)
    print(wrap(item["question"]))
    print(line)
    print(bold("KONTEKST (abstrakt bez konkluzji)"))
    print(wrap(item["context"]))
    print(line)
    print(bold("KONKLUZJA AUTORÓW"))
    print(wrap(item["conclusion"]))
    print(line)
    print(codebook_text(short=True))
    if current:
        note = f" — {current['note']}" if current.get("note") else ""
        print(f"\n  Obecny kod: {bold(current['code'])}{note}  (enter = zostaw)")


def show_list(sheet: List[Dict[str, str]], codes: Dict[int, Dict[str, str]]) -> None:
    clear()
    for row in sheet:
        c = codes.get(int(row["item"]))
        mark = c["code"] if c else "·"
        print(f"  {int(row['item']):>3}  {mark}  {textwrap.shorten(row['question'], width() - 12)}")
    input("\nEnter — powrót ")


# ---------------------------------------------------------------- session

def run(sheet_path: Path, out_path: Path) -> None:
    sheet = load_sheet(sheet_path)
    items = [int(r["item"]) for r in sheet]
    codes = load_codes(out_path)
    unknown = sorted(set(codes) - set(items))
    if unknown:
        raise SystemExit(f"{out_path} has codes for items not in the sheet: {unknown} — wrong sheet or wrong file?")

    clear()
    print(bold("RQ5 — kodowanie przyczyn `maybe`") + f"\nArkusz: {sheet_path}\nZapis:  {out_path}\n")
    print("Każde pytanie dostaje jedną literę — główną przyczynę, dla której odpowiedź yes/no nie wynika z tekstu.")
    print("Nie otwieraj klucza (rq5_coding_key.json) ani kodów modelu (rq5_coding_llm.csv) przed końcem kodowania.\n")
    print(codebook_text())
    print(f"\nKomendy: A–F [notatka] · enter = zostaw · b wstecz · s pomiń · g N idź do · l lista · k codebook · q wyjdź")
    input("\nEnter — zaczynamy ")

    pos = first_uncoded(items, codes)
    if pos is None:
        pos = 0
    history: List[int] = []
    message = ""
    while True:
        row = sheet[pos]
        item = items[pos]
        show(row, pos, len(items), len(codes), codes.get(item))
        if message:
            print(f"\n  {message}")
            message = ""
        try:
            command, arg = parse_answer(input("\nKod> "))
        except (EOFError, KeyboardInterrupt):
            command, arg = "quit", ""

        if command == "code":
            code, _, note = arg.partition("|")
            codes[item] = {"item": str(item), "code": code, "note": note,
                           "coded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            save_codes(out_path, codes)
        elif command == "keep" and item not in codes:
            message = "To pytanie nie ma jeszcze kodu — wpisz A–F."
            continue
        elif command == "back":
            if history:
                pos = history.pop()
            else:
                message = "To pierwsze pytanie w tej sesji."
            continue
        elif command == "goto":
            n = int(arg)
            if n in items:
                history.append(pos)
                pos = items.index(n)
            else:
                message = f"Nie ma pytania {n}."
            continue
        elif command == "list":
            show_list(sheet, codes)
            continue
        elif command == "codebook":
            clear()
            print(codebook_text())
            input("\nEnter — powrót ")
            continue
        elif command == "quit":
            break
        elif command == "invalid":
            message = f"Nie rozumiem „{arg}”. Wpisz A–F albo komendę (b, s, g N, l, k, q)."
            continue

        # code / keep / skip: move on to the next uncoded question after this one
        history.append(pos)
        nxt = first_uncoded(items, codes, pos + 1)
        if nxt is None:
            clear()
            print(f"Wszystkie {len(items)} pytania zakodowane. Zapisano: {out_path}")
            print("Możesz jeszcze przejrzeć listę (l) albo poprawić kod (g N). Wyjście: q.")
            try:
                command, arg = parse_answer(input("\n> "))
            except (EOFError, KeyboardInterrupt):
                break
            if command == "goto" and arg.isdigit() and int(arg) in items:
                pos = items.index(int(arg))
                continue
            if command == "list":
                show_list(sheet, codes)
                continue
            break
        pos = nxt

    save_codes(out_path, codes)
    print(f"\nZapisano {len(codes)}/{len(items)} kodów w {out_path}.")
    if len(codes) < len(items):
        print("Wznowienie: uruchom tę samą komendę jeszcze raz.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET, help="blind coding sheet (CSV)")
    parser.add_argument("--out", type=Path, required=True, help="this coder's own file, e.g. rq5_codes_<name>.csv")
    args = parser.parse_args()
    if not args.sheet.exists():
        raise SystemExit(f"no sheet at {args.sheet}; pass --sheet path/to/rq5_coding_sheet.csv")
    if args.out.resolve() == args.sheet.resolve():
        raise SystemExit("--out must be a separate file, not the sheet itself.")
    run(args.sheet, args.out)


if __name__ == "__main__":
    main()
