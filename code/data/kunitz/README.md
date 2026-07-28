# Kunitz derived-data provenance

## Current manuscript artifacts

Commit `094a66e` regenerated these gap-sensitive outputs after
`sequence_identity` was corrected to exclude all alignment gap symbols (`.`, `-`,
and `~`):

- `mask_experiment.csv`
- `mask_betasweep.csv`

These are the Kunitz nearest-sequence novelty sources used by the current
manuscript. `mask_residuals.csv` contains no novelty or nearest-sequence identity
column and was unaffected.

## Legacy pre-gap-fix artifacts

The 23 files below are retained historical outputs from the March 2026 experiment
state. They are not direct manuscript inputs. Their noted columns were calculated
before commit `094a66e`, when Pfam `.` gaps were incorrectly counted as residue
mismatches by `sequence_identity`. Re-running their producers with current code
will therefore change those columns even if the generated sequences do not change.

| Files | Count | Gap-sensitive columns | Original artifact commit |
|---|---:|---|---|
| `approach_comparison_{raw_replicates,aggregated}.csv` | 2 | `mean_novelty`; aggregated `novelty_mean`, `novelty_std` | `ebe5449` |
| `binding_experiment_{raw_replicates,aggregated}.csv` | 2 | `mean_seqid`; aggregated `seqid_mean`, `seqid_std` | `ebe5449` |
| `baseline_comparison.csv` | 1 | `seqid_mean`, `seqid_std` | `4b0064e` |
| `deepdive_{scaling,weighted,interpolation,mixed,consensus}.csv` | 5 | `mean_novelty`, `mean_seqid` | `ebe5449` |
| `deepdive_{scaling,weighted,interpolation,mixed,consensus}_raw_replicates.csv` | 5 | `mean_novelty` | `ebe5449` |
| `deepdive_{scaling,weighted,interpolation,mixed,consensus}_aggregated.csv` | 5 | `novelty_mean`, `novelty_std` | `ebe5449` |
| `multiplicity_generation.csv` | 1 | `mean_novelty`, `mean_seqid` | `ebe5449` |
| `multiplicity_generation_raw_replicates.csv` | 1 | `mean_novelty`, `mean_seqid` | `ebe5449` |
| `multiplicity_generation_aggregated.csv` | 1 | `novelty_mean`, `novelty_std` | `ebe5449` |

Important distinctions:

- In the binding and HMM-baseline pipelines, `novelty` is PCA cosine distance
  (`sample_novelty`) and is not gap-sensitive. Only their sequence-identity
  columns are stale.
- `approach_comparison.csv` has no nearest-sequence metric and is unaffected.
- There is no tracked `binding_experiment.csv`; only its raw-replicate and
  aggregated variants exist.

## Regeneration policy

Do not mix corrected metric columns into these historical CSVs by hand. If one of
these experiment families becomes an active result again:

1. Re-run its producer from a recorded code commit and environment.
2. Regenerate raw and aggregated outputs together.
3. Verify that non-gap-sensitive columns remain scientifically consistent, and
   explain any broader changes caused by code drift since March 2026.
4. Record the producing commit and command in this file.

The current producer scripts are:

- `experiments/run_all_approaches_with_replicates.jl`
- `experiments/run_kunitz_binding_experiment_with_replicates.jl`
- `experiments/run_hmm_baseline.jl`
- `experiments/run_augmented_memory_deepdive.jl`
- `experiments/run_augmented_memory_deepdive_with_replicates.jl`
- `experiments/run_multiplicity_conditioning.jl`
- `experiments/run_multiplicity_conditioning_with_replicates.jl`
