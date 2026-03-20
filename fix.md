# Path Fixes for `code/experiments`

Applied fixes for path regressions introduced after moving files into `code/experiments` (notably affecting AlphaFold docking/validation scripts).

## Issues fixed

1. Wrong Julia include/data root resolution
- Scripts were resolving `Include.jl` via a non-existent `.../code/experiments/code` path.
- Fixed by setting code-root with `dirname(@__DIR__)`.

Files updated:
- [code/experiments/run_hmm_baseline.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_hmm_baseline.jl)
- [code/experiments/run_all_approaches_with_replicates.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_all_approaches_with_replicates.jl)
- [code/experiments/run_augmented_memory_deepdive_with_replicates.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_augmented_memory_deepdive_with_replicates.jl)
- [code/experiments/run_second_family_validation_with_replicates.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_second_family_validation_with_replicates.jl)
- [code/experiments/run_structure_validation.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_structure_validation.jl)
- [code/experiments/run_kunitz_binding_experiment_with_replicates.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_kunitz_binding_experiment_with_replicates.jl)
- [code/experiments/plot_esm2_results.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/plot_esm2_results.jl)
- [code/experiments/run_combined_multiplicity_iface_pca_with_replicates.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_combined_multiplicity_iface_pca_with_replicates.jl)
- [code/experiments/run_conotoxin_structure_validation.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_conotoxin_structure_validation.jl)
- [code/experiments/run_multiplicity_conditioning_with_replicates.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_multiplicity_conditioning_with_replicates.jl)

2. Hardcoded absolute paths replaced
- Removed user-specific absolute paths that pointed to `/Users/jeffreyvarner/...`.

Files updated:
- [code/experiments/render_fold_superposition_figures.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/render_fold_superposition_figures.py)
- [code/experiments/prepare_af2_input.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/prepare_af2_input.py)

3. CWD-dependent or brittle relative paths fixed
- Switched from `Path("code/..." )` to script-relative roots.

Files updated:
- [code/experiments/extract_scores.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/extract_scores.py)
- [code/experiments/extract_scores_simple.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/extract_scores_simple.py)
- [code/experiments/score_esm2_perplexity.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/score_esm2_perplexity.py)
- [code/experiments/render_separation_gap_figure.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/render_separation_gap_figure.py)

4. Robust repository discovery in separation-gap figure script
- Added parent-directory probing for a true repo root (instead of hardcoded `parents[2]`).
- File updated: [code/experiments/render_separation_gap_figure.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/render_separation_gap_figure.py)

## 5. Docking pipeline run-format fix (ColabFold multimer submission)
- The previous `run_conotoxin_docking_validation.py` wrote two FASTA records (Cav2.2 + conotoxin), which ColabFold treated as two independent monomer queries.
- Updated complex FASTA generation to use multimer syntax in a single FASTA entry with `:` chain separator.
- Added multimer validation in scoring to skip outputs that do not expose `iptm` (or are tagged as single-chain via logs/PDB chain count).
- Added the same skip behavior to extraction scripts so historical monomer-only score directories are excluded from docking summaries.

Files updated:
- [code/experiments/run_conotoxin_docking_validation.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_conotoxin_docking_validation.py)
- [code/experiments/extract_scores.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/extract_scores.py)
- [code/experiments/extract_scores_simple.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/extract_scores_simple.py)

## 6. Validation script for multimer integrity
- Added a validator to audit existing docking run directories for multimer validity.
- The script checks FASTA formatting (`:` multimer separator), output JSON (`iptm` presence), logs (monomer-warning hints), and rank-001 PDB chain count.
- Added `--sample-fasta` mode to write and validate a single multimer-format FASTA example (`header:seq1:seq2`) before rerunning prediction batches.

Files added:
- [code/experiments/validate_docking_multimer.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/validate_docking_multimer.py)

## Notes
- No runtime tests were executed.
