# No Universal AI Dominance -- replication package
#
#   make all       analysis + validation + figures + tests
#   make verify    re-run everything into a scratch directory and compare with
#                  the committed outputs (the reproducibility check)
#   make paper     build paper/no-universal-ai-dominance.pdf
#
# Every simulation is seeded, so a fresh run must reproduce results/*.json byte
# for byte on any platform. Rendered PNGs depend on the Matplotlib, Pillow and
# FreeType versions and on the host's fonts, so `verify` compares the figures as
# images; `verify-strict` demands identical bytes, which holds between two runs
# that share a font stack.

PYTHON  ?= python3
ROOT    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SCRATCH := $(ROOT).verify

RESULTS := results/results.json
VALID   := results/validation.json
PDF     := paper/no-universal-ai-dominance.pdf

# passed to code/compare_figures.py; verify-strict overrides it with --strict
FIGCMP  ?=

.PHONY: all analysis validate figures test paper slides verify verify-strict arxiv archive archive-status clean help
.DEFAULT_GOAL := help

all: analysis validate figures test ## run the whole pipeline and the test suite

analysis: ## code/analysis.py -> results/results.json   (~70 s)
	$(PYTHON) code/analysis.py > /dev/null
	@echo "wrote $(RESULTS)"

validate: ## code/validate_stats.py -> results/validation.json   (~4 s)
	$(PYTHON) code/validate_stats.py > /dev/null
	@echo "wrote $(VALID)"

figures: ## code/figures.py -> paper/figs/*.{pdf,png}   (~5 s)
	$(PYTHON) code/figures.py

test: ## pytest tests/ -- every number in the paper against results.json
	$(PYTHON) -m pytest tests/ -q

slides: ## build the conference deck -> paper/no-universal-ai-dominance-slides.pdf
	@cd paper && \
	if command -v tectonic >/dev/null 2>&1; then tectonic -X compile slides.tex; \
	elif command -v latexmk >/dev/null 2>&1; then latexmk -pdf -quiet slides.tex; \
	else echo "no LaTeX engine found" >&2; exit 1; fi && \
	cp paper/slides.pdf paper/no-universal-ai-dominance-slides.pdf 2>/dev/null || \
	cp slides.pdf no-universal-ai-dominance-slides.pdf
	@echo "wrote paper/no-universal-ai-dominance-slides.pdf"

paper: ## build the PDF (latexmk, pdflatex or tectonic, whichever is available)
	@cd paper && \
	if command -v latexmk >/dev/null 2>&1; then \
	    latexmk -pdf -quiet main.tex && cp main.pdf ../$(PDF); \
	elif command -v pdflatex >/dev/null 2>&1; then \
	    pdflatex -interaction=nonstopmode main.tex >/dev/null && \
	    bibtex main >/dev/null && \
	    pdflatex -interaction=nonstopmode main.tex >/dev/null && \
	    pdflatex -interaction=nonstopmode main.tex >/dev/null && \
	    cp main.pdf ../$(PDF); \
	elif command -v tectonic >/dev/null 2>&1; then \
	    tectonic -X compile main.tex --keep-intermediates && cp main.pdf ../$(PDF); \
	else \
	    echo "no LaTeX engine found (install tectonic, latexmk or texlive)" >&2; exit 1; \
	fi
	@echo "wrote $(PDF)"

verify: ## assert a fresh run reproduces results/ and paper/figs/
	@rm -rf $(SCRATCH) && mkdir -p $(SCRATCH)/results $(SCRATCH)/figs
	@echo "re-running the pipeline into $(SCRATCH) ..."
	@NUAD_RESULTS=$(SCRATCH)/results NUAD_FIGS=$(SCRATCH)/figs $(PYTHON) code/analysis.py > /dev/null
	@NUAD_RESULTS=$(SCRATCH)/results NUAD_FIGS=$(SCRATCH)/figs $(PYTHON) code/validate_stats.py > /dev/null
	@NUAD_RESULTS=$(SCRATCH)/results NUAD_FIGS=$(SCRATCH)/figs $(PYTHON) code/figures.py > /dev/null
	@diff $(RESULTS) $(SCRATCH)/results/results.json     && echo "  OK results.json"
	@diff $(VALID)   $(SCRATCH)/results/validation.json  && echo "  OK validation.json"
	@$(PYTHON) code/compare_figures.py paper/figs $(SCRATCH)/figs $(FIGCMP)
	@rm -rf $(SCRATCH)
	@echo "reproduction verified"

verify-strict: FIGCMP = --strict
verify-strict: verify ## same as verify, but the figures must match byte for byte
	@:

arxiv: ## bundle the arXiv submission (manuscript + figures + ancillary files)
	@rm -rf build/arxiv && mkdir -p build/arxiv/figs build/arxiv/anc
	@cp paper/main.tex paper/main.bbl paper/references.bib build/arxiv/
	@cp paper/figs/*.pdf build/arxiv/figs/
	@cp code/analysis.py code/figures.py code/validate_stats.py code/paths.py \
	    code/archive_urls.sh code/compare_figures.py build/arxiv/anc/
	@cp data/sources.csv results/results.json results/validation.json build/arxiv/anc/
	@cd build/arxiv && zip -qr ../no-universal-ai-dominance-arxiv.zip .
	@echo "wrote build/no-universal-ai-dominance-arxiv.zip"

archive: ## request an Internet Archive snapshot of every source URL (needs web.archive.org)
	bash code/archive_urls.sh

archive-status: ## report which source URLs lack a snapshot or have a stale one
	$(PYTHON) code/archive_status.py

clean: ## remove build artefacts (never the committed results or figures)
	rm -rf build $(SCRATCH) .pytest_cache paper/main.aux paper/main.log \
	    paper/main.out paper/main.fls paper/main.fdb_latexmk paper/main.blg
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

help:
	@grep -hE '^[a-z][a-z-]*:.*?## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'
