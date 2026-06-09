# Design: Correct the Kunitz hard-mask sub-study

Date: 2026-06-09
Branch: `kunitz-attention-mask`
Status: approved (design), pending spec review

## Motivation

An audit of the committed Kunitz hard-mask study (commits `ace21d0`..`cf01504`)
found the **code correct** but the **headline scientific conclusion overstated
relative to the study's own data**, plus an uncontrolled confound. Specifically:

1. **The data contradicts "the gap is purely beta-bound."** Interpolating the
   mask beta-sweep envelope at each multiplicity point's own `beta_w`, the
   finite-rho points sit *systematically below* the envelope, monotone in rho:
   residuals approx -0.117 (f=0.5), -0.079 (f=0.7), -0.020 (f=0.9), -0.015
   (f=0.95), +0.001 (f=0.99); unconditional approx -0.149. They converge onto
   the envelope only as rho -> inf. This is the signature of background
   attention mass suppressing recovery at finite rho, which contradicts the
   Results sentence "the finite-rho conditions fall on the same
   recovery-versus-beta curve traced by the mask, so multiplicity strength acts
   on the decoded phenotype chiefly through the beta* it induces."

2. **Warm-start confound.** `generate_masked_sequences` warm-starts every chain
   from an in-set (designated) pattern, while the conditions it is compared to
   (`generate_sequences`, `generate_weighted_sequences`) warm-start via
   `mod1(chain, K)` over all columns (~32% in-set). This inflates the same
   residuals; the background-mass effect and the warm-start head start cannot be
   separated with the current design.

3. **Confirmation-bias diagnostic.** The script's logged "corrected analysis"
   computes only the single f=0.99 matched-beta point (`argmax(mult.f_target)`),
   the one row that has converged onto the envelope, so the contradicting
   low-rho residuals never surfaced.

4. **No uncertainty.** Chains are autocorrelated; effective n is ~ n_chains, so
   chain-level SE on P1 K/R is ~0.11, not the ~0.02 a naive binomial on n=620
   implies. n_chains is unequal across conditions (30 vs 20). No error bars are
   reported anywhere.

The fix corrects the confound, quantifies uncertainty, measures the gap
decomposition honestly, and reframes the Results to match. The final wording
strength is **data-contingent** (see Decision Rule).

## Goals

- Equalize the warm-start policy and n_chains so the mask vs multiplicity vs
  unconditional comparison is fair.
- Report chain-level mean +/- SE for every reported quantity and every residual.
- Measure the matched-beta residual (empirical `Delta_attn`) for **all** five
  multiplicity conditions, at exactly matched beta (no interpolation, no grid
  quantization).
- Reframe the Results paragraph as the gap-decomposition reading: the mask is
  the `Delta_attn = 0` envelope (decode-limited ceiling, closes with beta);
  finite-rho points sit below it by the empirical `Delta_attn`, converging as
  rho -> inf. Drop the "single curve / rho acts only through beta*" claim.
- Regenerate data, figures, the `tab:mask-recovery` table, and rebuild the PDF.

## Non-goals / out of scope

- **No change** to the sampler core (`sample`, `logit_bias_sample`,
  `masked_sample`, `weighted_sample`, `mask_vector`) — these are correct and
  tested; the masked == SA-on-reduced-matrix identity holds.
- **No change** to `generate_sequences` or `generate_weighted_sequences` — they
  are shared across all six families in the paper; changing their warm-start
  would invalidate unrelated committed results. Only the mask pipeline is
  brought into line with them.
- No re-run of the other families or other experiments.
- No multi-seed sweep (chain-level error bars were chosen as sufficient).

## Changes

### C1. `code/src/Binding.jl` — `generate_masked_sequences` warm-start

Change the per-chain initialization from in-set-only to the neutral baseline
policy, matching `generate_sequences`/`generate_weighted_sequences`:

```julia
# before
k = keep_indices[mod1(chain, length(keep_indices))]   # warm-start in-set
# after
k = mod1(chain, K)                                     # neutral warm-start (matches baselines)
```

Update the docstring to state the warm-start now matches the baseline pipelines
(remove the "warm-start from kept (in-set) patterns ... masked-conditional
protocol" sentence). Masking still confines the chain to the designated subset
after burn-in; only the start distribution changes. No other change to this
function.

### C2. `code/experiments/run_kunitz_mask_experiment.jl`

- **Global determinism:** add `Random.seed!(42)` near the top (after
  `include`), matching the base seed already passed to every `generate_*` call.
  This makes the whole script reproducible and removes the chain-1 warm-start
  dependence on uncontrolled process RNG. (Consistent with "each experiment
  manages its own seeds.")
- **Equalize n_chains = 30** for every `generate_*` call (multiplicity sweep and
  beta-sweep were 20).
- **`eval_condition` returns per-chain P1 K/R.** No generator signature change
  is needed: the flat `seqs` array is ordered chain-by-chain with a fixed
  `samples_per_chain = length(burnin:thin:T)`, so `eval_condition` takes
  `n_chains` and slices chain `c` as `(c-1)*spc+1 : c*spc`. Compute:
  - `p1_kr` = overall fraction (unchanged headline number),
  - `p1_kr_chain` = vector of length n_chains of per-chain fractions,
  - `p1_kr_se` = `std(p1_kr_chain) / sqrt(n_chains)`.
  Apply the same per-chain treatment to novelty (per-chain mean nearest-identity)
  so novelty also gets an SE. Add an `n >= 2` guard at the top of `eval_condition`
  (return NaNs with a warning rather than dividing by zero) — latent robustness.
- **Exact matched-beta mask runs (sub-choice A, accepted):** after the
  multiplicity sweep, for each multiplicity condition `i` with inflection
  `beta_w_i`, run `generate_masked_sequences(X_hat_all, pca_all, L,
  designated_idx; beta = beta_w_i, n_chains = 30, ...)` and record the mask P1
  K/R (with chain-level SE) at exactly `beta_w_i`. The matched-beta residual is
  `residual_i = mult_p1kr_i - mask_p1kr_at_beta_w_i`, with
  `se_i = sqrt(mult_se_i^2 + mask_se_i^2)`. This is the empirical `Delta_attn`
  at condition `i`.
- **Rewrite the "corrected analysis" block** to log the full residual table
  (all five conditions: f_target, beta_w, mult P1 K/R +/- SE, mask P1 K/R at
  beta_w +/- SE, residual +/- SE), plus the monotonicity check and a one-line
  summary of whether residuals are within error bars. Remove the single-point
  `argmax(mult.f_target)` analysis.
- Keep the existing fixed beta-sweep `[2,4,8,...,512]` (now 30 chains, with
  per-point SE) as the mask envelope for the figure.
- Write the residual table to a new CSV `code/data/kunitz/mask_residuals.csv`.
  The existing `mask_experiment.csv` and `mask_betasweep.csv` gain SE columns.

### C3. Figures (sub-choice B, accepted: single unifying figure + table residuals)

- **`fig_mask_betasweep`** (the unifying figure): mask beta-envelope with
  per-point error bars; multiplicity points overlaid at `(beta_w, p1_kr)` with
  x/y error bars; unconditional and curation references with error bars. The
  visual story is the multiplicity points lying below the envelope and
  converging onto it as rho grows. Add the exact-beta_w mask points (faint) so
  the vertical residual is directly readable.
- Regenerate `fig_mask_calibration` and `fig_mask_novelty_tradeoff` with error
  bars for consistency.
- No second decomposition panel (rejected per sub-choice B).

### C4. Paper

- **`paper/sections/results.tex`** — rewrite the "hard mask as the rho -> inf
  endpoint" paragraph:
  - Frame the mask as the `Delta_attn = 0` envelope / decode-limited ceiling
    that closes with beta (this part stands).
  - State that finite-rho conditions sit below the envelope by the empirical
    `Delta_attn`, which shrinks monotonically to 0 as rho -> inf — i.e. recovery
    depends on both retrieval sharpness (beta) and residual background mass (rho).
  - **Delete** the sentence "the finite-rho conditions fall on the same
    recovery-versus-beta curve traced by the mask, so multiplicity strength acts
    on the decoded phenotype chiefly through the beta* it induces."
  - Report the matched-beta residuals with CIs; wording strength per Decision Rule.
- **`paper/sections/methods.tex`** — one sentence: the mask, multiplicity, and
  unconditional conditions use an identical neutral warm-start (`mod1(chain,K)`)
  and 30 chains; recovery and novelty are reported as chain-level mean +/- SE.
- **`paper/Paper_v1.tex`** — update `tab:mask-recovery` values to the re-run
  numbers, add chain-level CIs, and (if it reads cleanly) a matched-beta residual
  column. Update the `fig:mask-betasweep` caption to the corrected claim.
- **Rebuild** `Paper_v1.pdf` via `paper/Build.sh Paper_v1`.

### C5. Tests

- Update `code/test/test_mask.jl`: the
  `generate_masked_sequences shape + validity` test currently passes
  `keep_idx = [1..6]` and only checks shape; keep that, and add an assertion
  that with the neutral warm-start the function still returns finite PCA vectors
  and correct counts. No test asserts the old in-set warm-start, so nothing
  breaks. Verify the full suite still passes.

## Statistics (chain-level error bars)

Each Langevin chain is an independent seeded replicate. Treat the per-chain P1
K/R fraction (and per-chain mean novelty) as the unit of replication:
`mean +/- std/sqrt(n_chains)` over n_chains = 30. Residual SEs combine in
quadrature. This requires no extra compute beyond the single corrected re-run
plus the five exact-beta_w mask runs.

## Decision rule for final wording

After the corrected re-run, inspect the residual table:
- If residuals are **large and monotone** (well outside the combined SE for the
  low-f conditions): state the two-factor result strongly — recovery is governed
  by both beta and background mass; the mask isolates the decode-limited ceiling.
- If residuals **collapse to within error bars** (the confound was inflating
  them): narrow the claim to the endpoint (with background eliminated, beta
  alone drives recovery to 1.0) and explicitly note the finite-rho points are
  consistent with the envelope within error. Either way the "single curve / rho
  only through beta*" sentence stays deleted.

The corrected numbers will be shown to the user at the checkpoint before the
prose is finalized.

## Sequencing

1. C1 + C2 + C5 code changes.
2. Smoke run (small n_chains / short T) to validate the script runs and the
   per-chain SE + residual plumbing works; measure full-run wall-clock.
3. Full re-run (30 chains, T=5000) -> regenerate CSVs + figures.
4. **Checkpoint: show the user the corrected residual table and figure.**
5. Write Results/methods/table prose to match (Decision Rule); rebuild PDF.
6. Update memory (`project_kunitz_mask_extension.md`) to the resolved finding.
7. Run the test suite; commit.

## Acceptance criteria

- `generate_masked_sequences` uses `mod1(chain, K)`; no change to shared pipelines.
- Experiment is deterministic from a single top-level seed; all conditions use 30 chains.
- `mask_experiment.csv`, `mask_betasweep.csv` carry SE columns; `mask_residuals.csv`
  exists with per-condition matched-beta residuals + SE.
- `fig_mask_betasweep` shows error bars and the finite-rho points relative to the envelope.
- Results paragraph no longer contains the "single curve / rho only through beta*"
  claim; the gap-decomposition framing is present; `tab:mask-recovery` matches the
  re-run; PDF rebuilds without error.
- `code/test/test_mask.jl` passes.
