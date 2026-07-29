.PHONY: all main si clean

all: si

main:
	./Build.sh Paper_JCIM

si: main
	./Build.sh Paper_JCIM_SI

clean:
	rm -f -- \
		Paper_JCIM.aux Paper_JCIM.bbl Paper_JCIM.blg Paper_JCIM.log \
		Paper_JCIM.out Paper_JCIM.pdf acs-Paper_JCIM.bib \
		Paper_JCIM_SI.aux Paper_JCIM_SI.bbl Paper_JCIM_SI.blg \
		Paper_JCIM_SI.log Paper_JCIM_SI.out Paper_JCIM_SI.pdf \
		acs-Paper_JCIM_SI.bib
