# Build the arXiv paper.
#
# This version is a single document: sections/appendix and the GMM derivation are inlined
# into Paper_v1.tex, so there is no separate SI target. Compare paper-jcim, where the
# appendix lives in Paper_JCIM_SI.tex and `make` builds the main paper and then the SI.
#
# pdflatex exits 0 even when references or citations stay undefined, so the log is checked
# explicitly afterwards. The .out file is removed first because a stale hyperref outline
# makes the outline unstable across passes. The three passes after bibtex resolve
# citations, cross-references and the outline.

.PHONY: all main clean

LATEX  := pdflatex -halt-on-error -interaction=nonstopmode
BIBTEX := bibtex

define latex_build
	rm -f -- "$(1).out"
	$(LATEX) "$(1).tex"
	$(BIBTEX) "$(1)"
	$(LATEX) "$(1).tex"
	$(LATEX) "$(1).tex"
	$(LATEX) "$(1).tex"
	@if grep -Eq 'There were undefined (references|citations)' "$(1).log"; then \
		echo "Build failed: unresolved references or citations remain in $(1).log" >&2; \
		exit 1; \
	fi
endef

all: main

main:
	$(call latex_build,Paper_v1)

clean:
	rm -f -- \
		Paper_v1.aux Paper_v1.bbl Paper_v1.blg Paper_v1.log \
		Paper_v1.out Paper_v1.pdf
