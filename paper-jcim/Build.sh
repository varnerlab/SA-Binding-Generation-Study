#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 JOBNAME" >&2
    exit 2
fi

job=$1

# Clean the outline file that can cause hyperref instability.
rm -f -- "$job.out"

# Build: pdflatex -> bibtex -> pdflatex x3
# The extra passes resolve citations, cross-refs, and hyperref outlines.
pdflatex -halt-on-error -interaction=nonstopmode "$job.tex"
bibtex "$job"
pdflatex -halt-on-error -interaction=nonstopmode "$job.tex"
pdflatex -halt-on-error -interaction=nonstopmode "$job.tex"
pdflatex -halt-on-error -interaction=nonstopmode "$job.tex"

if grep -Eq 'There were undefined (references|citations)' "$job.log"; then
    echo "Build failed: unresolved references or citations remain in $job.log" >&2
    exit 1
fi
