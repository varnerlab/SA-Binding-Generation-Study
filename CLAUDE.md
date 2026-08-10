# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase for the paper "Conditioning Protein Generation via Hopfield Pattern Multiplicity". Extends stochastic attention (SA) protein sequence generation with multiplicity-weighted Hopfield energy to bias generation toward a designated functional subset (e.g., binders) without retraining. Companion to the base SA method ([arXiv:2603.14717](https://arxiv.org/abs/2603.14717)).

## Running experiments

All Julia scripts must be run from the `code/` directory. Every script begins with `include("Include.jl")` which activates the project environment and loads all source modules.

```bash
cd code

# Run any experiment (first run downloads/precompiles dependencies automatically)
julia experiments/run_kunitz_binding_experiment.jl
julia experiments/run_multiplicity_conditioning.jl
julia experiments/run_new_family_validation.jl
julia experiments/run_omega_conotoxin_experiment.jl

# Python validation scripts
python3 experiments/score_esm2_perplexity.py      # requires fair-esm, torch
python3 experiments/render_separation_gap_figure_5fam.py  # matplotlib figures

# HMM baseline (requires HMMER3: brew install hmmer)
julia experiments/run_hmm_baseline.jl

# Build the arXiv paper (single document, appendix and SI derivation inlined)
cd ../paper-arxiv && make

# Build the JCIM submission: main paper, then the supporting information. The appendix and
# the GMM SI table live in Paper_JCIM_SI.tex, so building only the main paper would leave a
# stale SI PDF. `make` refreshes both; `make main` builds only the main paper.
cd ../paper-jcim && make
```

`make` is the only supported build entry point in both paper trees. Each Makefile runs
pdflatex, then bibtex, then pdflatex three more times, and then fails the build if the log
still reports undefined references or citations (pdflatex itself exits 0 in that case).
`make clean` removes the build products.

Julia 1.12 required. Package versions pinned in `code/Manifest.toml`; a human-readable snapshot is in `code/dependency_snapshot.toml`.

## Architecture

### Bootstrap pattern: `Include.jl`

`code/Include.jl` is the single entry point for all Julia scripts. It:
1. Sets path constants (`_ROOT`, `_PATH_TO_SRC`, `_PATH_TO_DATA`, `_PATH_TO_FIG`)
2. Activates the project environment from `Project.toml`/`Manifest.toml`
3. Loads all packages and source modules in order: `Data.jl` -> `Compute.jl` -> `Utilities.jl` -> `Protein.jl` -> `Binding.jl`

The core samplers use caller-local random-number generators via `seed=` or `rng=` arguments. The canonical family-sweep producer also uses local RNGs, so it does not mutate Julia's global RNG state.

### Source modules (`code/src/`)

- **Compute.jl**: Core SA sampler. The `sample(X, xi0, T; beta, alpha, seed)` function runs Unadjusted Langevin dynamics on the modern Hopfield energy, returning the full trajectory.
- **Protein.jl**: Data pipeline. Stockholm/FASTA parsing, alignment cleaning, one-hot encoding (`AA_ALPHABET` = 20 standard AAs), PCA with unit-norm projection, entropy-inflection phase-transition detection, inverse PCA with argmax decoding. Key function: `build_memory_matrix`.
- **Binding.jl**: Multiplicity conditioning and all comparison approaches. Key functions: `weighted_sample` (multiplicity-weighted Langevin dynamics), `multiplicity_vector`, `generate_sequences`, `effective_num_patterns`, `find_weighted_entropy_inflection`. Also implements four alternative approaches: curated memory, biased energy, post-hoc filtering, interface-weighted PCA.
- **Utilities.jl**: Diagnostic metrics (cosine similarity, Hopfield energy, attention entropy, novelty, diversity).
- **Data.jl**: Synthetic memory matrix generation for testing.

### Data layout (`code/data/`)

Per-family subdirectories contain Stockholm seed alignments, generated FASTA files, and multiplicity sweep CSVs. Six protein families are studied: WW (PF00397), Forkhead (PF00250), Kunitz (PF00014), SH3 (PF00018), Homeobox (PF00046), and omega-conotoxin (O-superfamily, under `omega_conotoxin/`). Cross-family results live in `code/data/multi_family_comparison_5fam.csv`.

### Papers (`paper-arxiv/` and `paper-jcim/`)

Two sibling trees, not nested. The arXiv version is `paper-arxiv/Paper_v1.tex`, one document with the appendix and GMM derivation inlined. The JCIM submission is `paper-jcim/Paper_JCIM.tex` plus a separate supporting-information document `paper-jcim/Paper_JCIM_SI.tex` that carries the appendix and the GMM SI table. Build either tree with `make`.

Each tree has its own `sections/`, `References*.bib`, and `sections/figs/`. The nine shared
section files, bibliography, labeled display-item contents, and common figure assets are kept
in sync by `code/test/test_manuscript_consistency.jl`, so edit them in both. The test permits
only layout-specific section lines, float-placement options, the two arXiv-only PNG companions,
and the JCIM-only TOC graphic to differ. Journal metadata and float-placement macros remain
specific to their wrapper files.

Tables and prose numbers under `sections/generated/` are produced by `code/experiments/generate_paper_tables.jl` from the canonical CSVs and byte-compared against both trees by `code/test/test_generated_paper_tables.jl`. Never hand-edit them; regenerate instead.

## Terminology

The method is target-agnostic. Use "designated" and "background" (not "binder" and "non-binder") when discussing sequence subsets, since the labeling comes from external domain knowledge. Legacy code uses "binder" in function names for historical reasons.

## Writing conventions for paper text

- No subsection headings within Introduction, Results, or Discussion; use flowing prose.
- No em dashes or en dashes; use periods or commas instead.
