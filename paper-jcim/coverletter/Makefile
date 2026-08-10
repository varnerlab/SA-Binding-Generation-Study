# Build the standalone JCIM cover letter.

.PHONY: all clean

TARGET := Cover_Letter_JCIM
LATEX  := pdflatex -halt-on-error -interaction=nonstopmode

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex
	rm -f -- "$(TARGET).out"
	$(LATEX) "$(TARGET).tex"
	$(LATEX) "$(TARGET).tex"
	@if grep -Eq 'There were undefined (references|citations)' "$(TARGET).log"; then \
		echo "Build failed: unresolved references or citations remain in $(TARGET).log" >&2; \
		exit 1; \
	fi

clean:
	rm -f -- \
		$(TARGET).aux $(TARGET).log $(TARGET).out $(TARGET).pdf
