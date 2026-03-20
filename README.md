# Conditioning Protein Generation via Hopfield Pattern Multiplicity

This repository contains the code, data, and manuscript source for the paper *Conditioning Protein Generation via Hopfield Pattern Multiplicity*. The method extends stochastic attention (SA) protein sequence generation by assigning multiplicity weights to stored patterns in the modern Hopfield energy, tilting the Boltzmann distribution toward a user-specified functional subset without retraining. The only modification to the SA sampler is a log-multiplicity bias added to the softmax logits. A single scalar parameter, the multiplicity ratio, continuously interpolates between unconditioned generation and hard subset curation.

The companion paper describing the base SA method is: *Training-Free Generation of Protein Sequences from Small Family Alignments via Stochastic Attention* ([arXiv:2603.14717](https://arxiv.org/abs/2603.14717)), with code at [github.com/varnerlab/SA-Protein-Modeling-Study](https://github.com/varnerlab/SA-Protein-Modeling-Study).

## Repository organization

The repository has two top-level directories: `code/` for all computational work (Julia and Python) and `paper/` for the LaTeX manuscript.

### Source library (`code/src/`)

The core library extends the base SA codebase with five source files:

- `Compute.jl` implements the stochastic attention sampler: the `sample` function takes a memory matrix, an initial state, and a number of iterations, and returns the full Langevin trajectory.
- `Protein.jl` provides the protein data pipeline (Stockholm/FASTA parsing, alignment cleaning, one-hot encoding, PCA dimensionality reduction with unit-norm projection, phase-transition detection via entropy inflection, and inverse PCA with argmax decoding).
- `Binding.jl` implements the multiplicity-weighted Hopfield energy and its Langevin dynamics (`weighted_sample`), the multiplicity vector construction (`multiplicity_vector`), effective pattern count (`effective_num_patterns`), weighted entropy inflection detection (`find_weighted_entropy_inflection`), and the `generate_sequences` pipeline. It also contains four alternative conditioning approaches (curated memory, biased energy, post-hoc filtering, interface-weighted PCA) used for comparison.
- `Utilities.jl` provides diagnostic functions (cosine similarity, Hopfield energy, attention entropy, novelty, diversity).
- `Data.jl` generates synthetic memory matrices for testing.

Every Julia script begins by calling `Include.jl`, which activates the project environment from `Project.toml` and loads all source modules. Exact package versions used for the paper results are recorded in `code/dependency_snapshot.toml`.

### Data (`code/data/`)

Pre-cached Pfam seed alignments and experiment outputs are organized by family:

| Family | Pfam ID | K | L | d | Designated subset | Marker |
|---|---|---|---|---|---|---|
| WW | PF00397 | 420 | 31 | 186 | 69 | Specificity loop |
| Forkhead | PF00250 | 246 | 87 | 172 | 122 | H/N at H3 helix |
| Kunitz | PF00014 | 99 | 53 | 80 | 32 | K/R at P1 |
| SH3 | PF00018 | 55 | 48 | 46 | 33 | Conserved Trp |
| Homeobox | PF00046 | 136 | 57 | 101 | 102 | Gln at position 50 |

The omega-conotoxin O-superfamily (K=74, L=26, 23 designated Cav2.2 binders) is stored under `code/data/omega_conotoxin/` with MAFFT-aligned FASTA files compiled from SwissProt.

Each family subdirectory contains the Stockholm-format seed alignment, generated sequence FASTA files, and per-family multiplicity sweep results (`multiplicity_sweep.csv`). Cross-family comparison tables are at `code/data/multi_family_comparison_5fam.csv`.

### Experiments (`code/experiments/`)

The main experiment scripts, each self-contained and runnable from the command line:

**Core experiments:**
- `run_kunitz_binding_experiment.jl` — Hard curation on Kunitz domains: phenotype transfer, scaling study (3 to 32 input binders), sequence-level analysis.
- `run_multiplicity_conditioning.jl` — Multiplicity-weighted generation on Kunitz: rho sweep, calibration gap decomposition, phase transition characterization, entropy curves.
- `run_second_family_validation.jl` — Cross-family validation on Kunitz, SH3, and WW domains: Fisher separation index, calibration gap regression.
- `run_new_family_validation.jl` — Extended cross-family validation adding Homeobox and Forkhead (5 Pfam families total).
- `run_omega_conotoxin_experiment.jl` — Omega-conotoxin Cav2.2 binder generation: full-family vs. curated seeding, pharmacophore analysis, SAR agreement.
- `run_conotoxin_multiplicity_sweep.jl` — Multiplicity sweep on the conotoxin family.
- `run_calibration_diagnostics.jl` — Detailed calibration gap decomposition (attention, PCA, argmax layers).

**Validation:**
- `run_structure_validation.jl` — ESMFold structure prediction and TM-score computation against experimental references.
- `run_conotoxin_structure_validation.jl` — Structure validation for conotoxin sequences.
- `run_hmm_baseline.jl` — HMMER3 profile HMM baseline.
- `score_esm2_perplexity.py` — ESM2-650M pseudo-perplexity scoring.
- `compute_conotoxin_sar_agreement.jl` — SAR agreement table computation.
- `prepare_af2_input.py` — Prepare FASTA input for ColabFold AlphaFold2 cross-validation.
- `run_conotoxin_docking_validation.py` — AlphaFold2-multimer complex prediction analysis.

**Figure rendering:**
- `render_separation_gap_figure_5fam.py` — Publication figure: calibration trajectories and S-vs-gap regression (matplotlib).
- `render_entropy_curves.py` — Publication figure: normalized entropy curves (matplotlib).
- `render_sequence_analysis_figure.jl` — Kunitz sequence-level analysis figure.
- `render_conotoxin_sequence_analysis.jl` — Conotoxin sequence-level analysis figure.
- `render_fold_superposition_figures.py` — Structure superposition figure (requires PyMOL).

**Replicate variants** (`*_with_replicates.jl`) run multiple random seeds for uncertainty quantification.

### Paper (`paper/`)

The main file `Paper_v1.tex` inputs section files from `paper/sections/` (abstract, introduction, results, discussion, theory, methods, appendix), and references are in `References_v1.bib`. Figures are in `paper/sections/figs/` and `paper/figs/`. Running `./Build.sh` from the `paper/` directory executes the standard four-pass LaTeX compilation cycle and produces `Paper_v1.pdf`.

## Getting started

The Julia experiments were developed and tested with [Julia](https://julialang.org/downloads/) 1.12. The simplest installation route on all platforms is [juliaup](https://github.com/JuliaLang/juliaup) (`curl -fsSL https://install.julialang.org | sh` on macOS/Linux, or `winget install Julia` on Windows). On the first run, `Include.jl` will download and precompile all Julia dependencies automatically.

Several validation and figure scripts are written in Python (3.9+) and depend on matplotlib, numpy, and BioPython. Structure rendering requires [PyMOL](https://pymol.org/). ESM2 perplexity scoring requires `fair-esm` and `torch`. AlphaFold2 cross-validation runs on [Google Colab](https://github.com/sokrypton/ColabFold) via ColabFold and requires a GPU runtime. [HMMER3](http://hmmer.org/) is required for the HMM baseline (`brew install hmmer` on macOS). Building the paper PDF requires a LaTeX distribution with `pdflatex` and `bibtex` ([TeX Live](https://tug.org/texlive/) or [MiKTeX](https://miktex.org/)).

To reproduce the core experiments:

```bash
git clone https://github.com/varnerlab/SA-Binding-Generation-Study.git
cd SA-Binding-Generation-Study/code

# Kunitz hard curation and scaling study
julia experiments/run_kunitz_binding_experiment.jl

# Multiplicity conditioning and calibration gap (Kunitz)
julia experiments/run_multiplicity_conditioning.jl

# Cross-family validation (5 Pfam families)
julia experiments/run_new_family_validation.jl

# Omega-conotoxin Cav2.2 binder generation
julia experiments/run_omega_conotoxin_experiment.jl

# Structure validation (requires internet for ESMFold API and RCSB PDB)
julia experiments/run_structure_validation.jl

# HMM baseline (requires HMMER3)
julia experiments/run_hmm_baseline.jl

# ESM2 perplexity (requires fair-esm and torch)
python experiments/score_esm2_perplexity.py

# Regenerate publication figures
python experiments/render_separation_gap_figure_5fam.py
python experiments/render_entropy_curves.py

# Paper
cd ../paper && ./Build.sh
```

The experiments can be run independently in any order, with the exception that structure validation and ESM2 scoring require the SA-generated sequence FASTA files produced by the core experiments. Each script writes its output (CSVs, figures, FASTA files) to `code/data/` or `code/figs/`. All random seeds are passed explicitly to sampling functions, so running the scripts from scratch will produce identical results regardless of platform.

## License

This project is released under the [MIT License](LICENSE).
