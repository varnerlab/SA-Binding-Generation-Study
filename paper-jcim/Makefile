# Build the JCIM submission: the main paper and its supporting information.
#
# pdflatex exits 0 even when references or citations stay undefined, so each job's log is
# checked explicitly afterwards. The .out file is removed first because a stale hyperref
# outline makes the outline unstable across passes. The three passes after bibtex resolve
# citations, cross-references and the outline.

.PHONY: all main si clean

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

all: si

main:
	$(call latex_build,Paper_JCIM)

# The appendix and the GMM SI table live in Paper_JCIM_SI.tex, so building only the main
# paper leaves a stale SI PDF behind. si depends on main so that `make` refreshes both.
si: main
	$(call latex_build,Paper_JCIM_SI)

clean:
	rm -f -- \
		Paper_JCIM.aux Paper_JCIM.bbl Paper_JCIM.blg Paper_JCIM.log \
		Paper_JCIM.out Paper_JCIM.pdf acs-Paper_JCIM.bib \
		Paper_JCIM_SI.aux Paper_JCIM_SI.bbl Paper_JCIM_SI.blg \
		Paper_JCIM_SI.log Paper_JCIM_SI.out Paper_JCIM_SI.pdf \
		acs-Paper_JCIM_SI.bib
