# Review: Curated Conditioning Transfers Conotoxin Sequence Features

Scope: arXiv manuscript only.

## Review outcome

The subsection is not yet computationally publication-ready. The original raw-sequence indexing bug was partly corrected, but the hard-curation results still mix two alignment frames.

### 1. Blocking: generated and input sequences use different designated matrices

The generator independently cleans the designated FASTA (`code/experiments/run_omega_conotoxin_experiment.jl`, beginning around line 53), while the corrected SAR analysis extracts the designated accessions from the canonical full-family alignment (`code/experiments/compute_conotoxin_sar_agreement.jl`, beginning around line 79).

Both matrices are 23 x 26, but 7 of 23 sequences differ. For example:

- Cys15 frequency: canonical input 1.000; actual generation matrix 0.870.
- K/R21 frequency: canonical input 0.478; actual generation matrix 0.565.

Thus, the current table compares canonical inputs with outputs generated from another matrix.

### 2. Affected claims must be recomputed

This affects the designated-generated statistics in `paper-arxiv/sections/results.tex`, beginning around line 181, including:

- Tyr13, basic-residue, KL-divergence, and novelty values;
- the SAR enrichment table;
- designated heatmaps and residuals;
- the `r = 0.745` entropy correlation;
- top-five sequence substitutions;
- designated ESMFold/AlphaFold2 results and the structural superposition.

The full-family generation and canonical multiplicity sweep are unaffected. Importantly, the central Tyr13 conclusion will probably survive: an independent canonical hard-curation calculation gives `0.979 +/- 0.006` across five replicates (`code/experiments/run_canonical_family_sweeps.jl`, beginning around line 84).

### 3. The "five highest-pLDDT" wording overstates the selection pool

Only the first 50 of 1,550 sequences were modeled (`code/experiments/run_conotoxin_structure_validation.jl`, beginning around line 193). Therefore, the manuscript should say "the five highest-pLDDT sequences among the 50 modeled sequences," not imply that these were the top five of the complete library.

### 4. The structural sentence and figure disagree

The Results section says "the top three sequences," but the figure contains one selected conotoxin sequence (`paper-arxiv/Paper_v1.tex`, beginning around line 526).

The figure also reports TM = 0.53 using query-length normalization (`code/experiments/render_fold_superposition_figures.py`, beginning around line 64), whereas the tables and Methods use reference-length normalization. Reference-length normalization gives 0.52342, which rounds to 0.52.

### 5. Minor caption error

The phrase "full-family seeding reduced both" is misleading. Tyr13 increases from 33.8% to 46.9%, and K/R increases from 9.9% to 12.0% (`paper-arxiv/Paper_v1.tex`, beginning around line 306). The caption should instead say that these values were lower than those obtained under designated-subset seeding.

## Recommended correction

Derive the hard-curation matrix directly from the canonical full-family alignment:

```julia
char_designated = char_full[designated, :]
```

Then regenerate the designated library and all dependent analyses. Add a regression test asserting that the generator's designated input matrix exactly equals the canonical subset. The separately aligned designated FASTA should be used only to supply accession IDs, not sequence coordinates.

