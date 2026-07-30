.PHONY: all main clean

# The arXiv version is a single document: sections/appendix and the GMM derivation are
# inlined into Paper_v1.tex, so there is no separate SI target here. Compare paper-jcim,
# where the appendix lives in Paper_JCIM_SI.tex and `make` builds main then SI.
all: main

main:
	./Build.sh Paper_v1

clean:
	rm -f -- \
		Paper_v1.aux Paper_v1.bbl Paper_v1.blg Paper_v1.log \
		Paper_v1.out Paper_v1.pdf
