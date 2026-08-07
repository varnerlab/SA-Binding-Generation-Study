#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$paper_dir/.." && pwd)"
output_dir="${1:-$repo_dir/output/arxiv}"
archive="$output_dir/SA-Binding-Generation-Study-arxiv-v2.tar.gz"

files=(
  Paper_v1.tex
  Paper_v1.bbl
  References_v1.bib
  neurips_2026.sty
  sections/figs/fold_superposition_combined.pdf
  sections/abstract.tex
  sections/appendix.tex
  sections/discussion.tex
  sections/introduction.tex
  sections/methods.tex
  sections/results.tex
  sections/si_gmm_derivation.tex
  sections/theory.tex
  sections/figs/fig2_separation_vs_gap.pdf
  sections/figs/fig5_entropy_curves.pdf
  sections/figs/fig_mask_betasweep.pdf
  sections/figs/loop_heatmap_with_residuals.pdf
  sections/figs/plddt_vs_tmscore_scatter.pdf
  sections/figs/sequence_analysis_conotoxin.pdf
  sections/figs/sequence_analysis_kunitz.pdf
  sections/figs/study1_binder_scaling.pdf
  sections/generated/numbers.tex
  sections/generated/tab_beta_sweep.tex
  sections/generated/tab_cross_family.tex
  sections/generated/tab_forkhead_rho.tex
  sections/generated/tab_homeobox_rho.tex
  sections/generated/tab_kunitz_rho.tex
  sections/generated/tab_omega_conotoxin_rho.tex
  sections/generated/tab_per_family_rho.tex
  sections/generated/tab_profile_hmm_benchmark.tex
  sections/generated/tab_profile_hmm_rho.tex
  sections/generated/tab_sar_agreement.tex
  sections/generated/tab_sh3_rho.tex
  sections/generated/tab_ww_rho.tex
)

for file in "${files[@]}"; do
  if [[ ! -f "$paper_dir/$file" ]]; then
    printf 'Missing arXiv source dependency: %s\n' "$file" >&2
    exit 1
  fi
done

mkdir -p "$output_dir"
tar -czf "$archive" -C "$paper_dir" "${files[@]}"
printf '%s\n' "$archive"
