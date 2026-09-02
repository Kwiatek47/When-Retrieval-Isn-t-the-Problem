# Build the ML4H Findings draft. Prefers latexmk; falls back to tectonic / pdflatex.
MAIN = main

.PHONY: all clean docker tectonic figures

all:
	@if command -v latexmk >/dev/null 2>&1; then \
	  latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
	  tectonic $(MAIN).tex; \
	else \
	  $(MAKE) manual; \
	fi

tectonic:
	tectonic $(MAIN).tex

figures:
	../.venv/bin/python figures/generate_paper_figures.py

# Explicit cycle if latexmk is unavailable.
manual:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

# Build inside the official TeX Live image (no local TeX required).
docker:
	docker run --rm -v "$(CURDIR)":/work -w /work texlive/texlive:latest \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex 2>/dev/null || true
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk sections/*.aux
