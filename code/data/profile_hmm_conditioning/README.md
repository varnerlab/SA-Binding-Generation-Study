# Matched profile-HMM conditioning benchmark

This directory contains the canonical outputs of
`code/experiments/run_profile_hmm_conditioning_benchmark.jl` and
`code/experiments/score_profile_hmm_baseline.py`.

- `raw_replicates.csv`: five HMMER3 emission replicates for each of six families, eight
  multiplicity ratios, and the designated-subset endpoint.
- `aggregated.csv`: across-replicate means and standard deviations.
- `execution.csv`: benchmark grid, sample counts, HMMER options, and seed origin.
- `kunitz_rho500_rep1.fasta`: weighted profile-HMM Kunitz replicate used for the matched ESM2
  comparison.
- `kunitz_sa_rho500_rep1.fasta`: exactly reproduced canonical SA Kunitz replicate used for the
  matched ESM2 comparison.
- `kunitz_rho500_matched_esm2_raw.csv` and
  `kunitz_rho500_matched_esm2_summary.csv`: local ESM2-650M masked-marginal scores for the first
  50 sequences from each matched replicate.

The profile HMM receives the same cleaned alignment, designation labels, and relative
multiplicity ratio as SA. `hmmbuild` is run with
`--amino --symfrac 0.0 --wgiven --enone`, and `hmmemit -a` output is reduced to RF-annotated
match columns before metrics are calculated.
