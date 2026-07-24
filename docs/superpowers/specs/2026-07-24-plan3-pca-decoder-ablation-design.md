# Plan 3 Design: W3 — PCA Decoder-Fidelity Ablation (gated)

Date: 2026-07-24
Status: design, approved by author; awaiting spec review before writing-plans
Parent spec: `docs/superpowers/specs/2026-07-22-arxiv-revision-design.md` (workstream W3, opt-in add-ons)
Predecessors: Plan 1 (math foundations) merged at 9c974fd; Plan 2 (canonical CSV reconcile)
committed at 1b04df4 / 0dd17e9 / 92c9095.

## 1. Context and locked decisions

The audit's "PCA representation and decoding" section shows the decoder is not the inverse of the
encoder. `build_memory_matrix` (`code/src/Protein.jl`) one-hot encodes the alignment, fits PCA,
transforms to scores `Z` (d_pca x K) whose column norms are approximately 5.0 to 6.5, then
normalizes each column to unit norm to form the memory matrix `X_hat`. The sampler runs on
`X_hat` (unit-norm memories). `decode_sample` reconstructs a generated PCA-space vector via
`MultivariateStats.reconstruct(pca_model, xi)`, which is affine (`P xi + mu`), then argmax-decodes
each position. Because a generated `xi` sits near a unit-norm memory, the data-dependent term
`P xi` is roughly six times too small relative to the mean `mu`, so argmax leans on the mean
composition. A diagnostic measured reconstruction identity of about 0.833 for unit-norm decoding
versus about 0.996 when decoding from the original-radius scores.

The clean Gaussian-mixture identity that anchors the arXiv-v2 reframe (Plan 1) requires unit-norm
memories. Therefore the encoder and sampler stay unchanged; the ablation varies only the decoder.
Of the audit's three variants, only a radius-restoring decoder both preserves the theory and can
change generated sequences, so the empirical question is: does restoring the radius at decode time
move the marker-positive fraction of generated sequences?

Decisions locked with the author:

- **Scope: gated diagnostic.** Build the decode variants, measure static reconstruction fidelity
  and the generation-time marker fraction, and decide whether a radius-restoring decoder changes
  generation enough to require regenerating canonical outputs. Do not pre-commit to a full re-run.
- **Escalation trigger: beyond replicate noise.** On Kunitz, escalate if the restored-radius
  marker-positive fraction differs from the current (normalized) decoder by more than the replicate
  standard deviation already recorded in the canonical aggregate, at any rho. Otherwise document the
  loss as a disclosed decoder limitation.
- **Restored radius = per-sample nearest-memory norm.** For a generated `xi`, scale by the original
  PCA score-norm `r_{k*}` of its nearest memory (`k* = argmax_k X_hat[:,k]^T xi`) before
  reconstruct. This is the principled inverse of the encoder. A radius sweep provides the SI
  sensitivity curve.
- **Static fidelity on all six families; generation gate on Kunitz only.** Kunitz is the diagnostic
  family (parent spec). Static reconstruction fidelity is cheap (no sampling) so it runs for all six
  to generalize the finding and to fix the stale appendix descriptors. WW is not added to the gate.
- **The gate regenerates the Kunitz sweep once.** Plan 2 did not persist PCA-space `xi`
  trajectories, so the gate re-runs the canonical Kunitz generation with the Plan 2 seeds and
  decodes the same samples two ways. The normalized decode must reproduce the committed aggregate
  (0.581 plus or minus 0.006 at rho=500); this doubles as a consistency check.
- **Escalation produces a decision, not the re-run.** If the gate trips, the six-family
  regeneration under restored-radius decoding is a separate follow-up plan. Weighted MALA (also a W3
  opt-in) stays deferred. The sampler and GMM theory are untouched.
- **Anti-drift continues.** New paper numbers come from committed CSVs via generated table bodies
  and `numbers.tex` macros, per the Plan 2 discipline.

## 2. Decode primitives (`code/src/Protein.jl`)

All additions are backward compatible; `build_memory_matrix` keeps its return signature.

- `memory_radii(char_mat, pca_model) -> Vector{Float64}`: recompute `Z = transform(pca_model,
  onehot_encode(char_mat))` and return its per-column norms `r_k`. Kept separate from
  `build_memory_matrix` so existing callers are unaffected.
- `decode_sample(xi_pca, pca_model, L; scale::Float64=1.0) -> String`: reconstruct `scale .* xi_pca`
  then argmax-decode. `scale=1.0` is byte-identical to the current behavior.
- `nearest_memory_radius(xi, X_hat, r) -> Float64`: `k* = argmax_k X_hat[:,k]^T xi`, return `r[k*]`.

Restored-radius decode of a generated sample is
`decode_sample(xi, pca_model, L; scale = nearest_memory_radius(xi, X_hat, r))`.

Note: for a *stored* memory the exact `r_k` is known, so restored-radius decoding of stored
memories is identical to decoding from the original-radius scores (the audit's "unnormalized"
case). The static table therefore has two meaningful columns, normalized and restored.

## 3. Experiment: `code/experiments/run_pca_decoder_ablation.jl`

Reuses `canonical_family_registry.jl` (family list, data loading, split and marker predicates) so
family definitions are not duplicated. Three parts:

- **(A) Static reconstruction fidelity, all six families.** For each family, encode the cleaned
  alignment, and for each stored memory decode under normalized (`scale=1`) and restored
  (`scale=r_k`). Reconstruction identity is `sequence_identity(decoded, original)` (gaps excluded),
  averaged over memories. Output `code/data/pca_decoder_fidelity.csv` with schema
  `family,K,d_full,d_pca,d_full_over_K,identity_normalized_mean,identity_normalized_std,`
  `identity_restored_mean,identity_restored_std`. This reproduces Kunitz about 0.83 versus about
  1.00 and supplies the corrected "six families" and per-family `d_full/K` descriptors.
- **(B) Generation gate, Kunitz only.** Regenerate the canonical Kunitz multiplicity sweep once
  using the Plan 2 seeds and parameters (rho grid {1,2,5,10,20,50,100,500}, 5 replicates, 20 chains,
  T=5000, burn_in=2000, thin=100; seeds per `data/canonical_sweep_provenance.csv`), capturing the
  PCA-space samples `xi`. Decode the *same* samples under normalized and restored-radius decoders,
  compute the Kunitz marker-positive fraction (marker P1 K/R) per replicate under each, using the
  same pooling as the canonical driver (fraction of marker-positive sequences across the replicate's
  chains and retained samples). Output
  `code/data/kunitz/pca_decoder_gate.csv` with schema
  `rho,f_obs_norm_mean,f_obs_norm_std,f_obs_restored_mean,f_obs_restored_std,`
  `delta_mean,delta_std,replicate_std,escalate`, where `delta` is the paired per-replicate
  (restored minus normalized), `replicate_std` is the canonical `f_obs_std` at that rho, and
  `escalate = abs(delta_mean) > replicate_std`. A row-level `escalate` true at any rho sets the
  run-level decision.
- **(C) Radius sweep, Kunitz (SI).** Decode the same generated samples at a grid of absolute scales
  from 1.0 (current decoder) up to the maximum memory radius `max_k r_k` (about eight points), per
  rho. Output `code/data/kunitz/pca_decoder_radius_sweep.csv` with schema
  `rho,scale,f_obs_mean,f_obs_std`, plus a figure written to the `paper-arxiv` figs path.

Reproducibility constraint: the gate must reuse the canonical driver's generation exactly, not
reimplement it. If `run_canonical_family_sweeps.jl` currently decodes inline, factor its inner
per-(rho, replicate, chain) generation into a function that returns `xi` samples, and have both the
driver and the ablation call it. The normalized-decode marker fraction reproducing the committed
aggregate is the acceptance check that the seed semantics were preserved.

Validation before writing outputs: fail on schema mismatch, non-finite values, fractions outside
`[0,1]`, an incomplete rho grid, unexpected replicate counts, or a normalized-decode rho=500 mean
that does not round to the committed 0.581.

## 4. Anti-drift tooling and paper edits

- Extend `code/experiments/generate_paper_tables.jl` to read `pca_decoder_fidelity.csv` and
  `pca_decoder_gate.csv` and emit into `paper-arxiv/sections/generated/`:
  - `tab_decoder_fidelity.tex`: the fidelity tabular body (per-family normalized vs restored
    identity, `d_full/K`).
  - New macros appended to `numbers.tex`: `\KunitzIdentityNormalized`, `\KunitzIdentityRestored`,
    `\DecoderGateDeltaMax` (max abs paired delta over rho), and `\DecoderGateEscalate` (a word,
    "does" or "does not", for the prose sentence on whether decoding changes generation).
- New SI subsection `paper-arxiv/sections/si_pca_decoder.tex` (`\label{app:decoder}`), `\input` into
  `appendix.tex`, reporting: the fidelity table, the gate outcome (delta versus replicate noise and
  the escalation decision), and the radius-sweep figure. Follows the Plan 1 `si_gmm_derivation.tex`
  pattern.
- Fix the stale appendix descriptors the audit flagged: "four families" becomes six; the
  `d_full/K` range is corrected to include WW (about 620/420 = 1.48). Prefer sourcing these from the
  generated fidelity table so they cannot drift again.
- The regression test regenerates the generated directory into a temporary location and byte-compares
  every file, as in Plan 2.

## 5. Tests (added to `code/test/runtests.jl`)

- Restored-radius identity is greater than or equal to normalized identity on stored memories, per
  family (the fix must not reduce fidelity).
- `decode_sample(xi, pca_model, L; scale=1.0)` equals the current `decode_sample(xi, pca_model, L)`
  output on fixed vectors (backward compatibility).
- Affine sanity: `reconstruct(pca_model, scale .* xi)` equals
  `scale .* (reconstruct(pca_model, xi) .- mu) .+ mu` to tolerance.
- Gate consistency: the gate's normalized `f_obs` at rho=500 equals the committed
  `data/kunitz/multiplicity_sweep_aggregated.csv` value (ties the ablation to the canonical CSV).
- `nearest_memory_radius` returns exactly `r_k` when given a stored unit-norm memory column.

## 6. Verification

- Run the ablation, regenerate tables and macros, build `Paper_v1`, and confirm a clean PDF with
  resolved refs and both `app:gmm` and `app:decoder` labels present.
- `runtests.jl` passes, including the new decoder tests and the byte-for-byte generated-file
  regeneration.
- Report the gate decision explicitly: the maximum absolute paired delta over rho, the replicate
  noise it is compared against, and whether the run escalates. If it escalates, surface it as a real
  result change and stop for author direction before any six-family regeneration.

## 7. Sequencing and dependencies

1. Add decode primitives to `Protein.jl` (Section 2) with unit tests.
2. If needed, refactor the canonical driver's inner generation to return `xi`; verify the driver
   still reproduces the committed Kunitz aggregate.
3. Build `run_pca_decoder_ablation.jl` parts A, B, C and emit the three CSVs plus the sweep figure.
4. Extend the generator; add the SI subsection and appendix descriptor fixes; wire macros and the
   fidelity table.
5. Add the regression tests; build the paper; report the gate decision.

Plan 3 touches `appendix.tex` and the generated directory, both settled by Plan 2; it adds rather
than rewrites, so there is no conflict with Plan 5 prose (which comes later). Plan 3 must run before
the narrative reframe (Plan 5 / W0) because a tripped gate would change numbers the prose depends on.

## 8. Success criteria

- Decode primitives exist, are backward compatible, and are unit tested.
- `pca_decoder_fidelity.csv` exists for all six families and reproduces the Kunitz normalized-versus-
  restored identity gap; the appendix "six families" and `d_full/K` descriptors are corrected and
  generated.
- `pca_decoder_gate.csv` exists for Kunitz across the full rho grid; its normalized column
  reproduces 0.581 plus or minus 0.006 at rho=500; the escalation decision is computed against
  replicate noise and reported.
- `pca_decoder_radius_sweep.csv` and its figure exist; the SI subsection presents the fidelity
  table, gate outcome, and sweep.
- New decoder numbers in the manuscript are `\input` table cells or `numbers.tex` macros.
- `Paper_v1.pdf` builds clean with `app:gmm` and `app:decoder` resolved; `runtests.jl` passes.

## 9. Risks and open questions

- **The gate trips.** If restored-radius decoding moves the Kunitz marker fraction beyond replicate
  noise, Plan 2's canonical numbers are provisional and a six-family regeneration follows. This is
  the intended outcome of a gated experiment, surfaced not absorbed.
- **Seed reproduction.** The gate's value depends on reproducing the canonical generation exactly.
  The reproduce-0.581 check guards this; if the driver cannot be refactored without changing seed
  semantics, treat that as a blocker and resolve before trusting the gate.
- **Nearest-memory assignment noise.** For a mixture draw between components, the nearest memory can
  be ambiguous; the radius sweep contextualizes how sensitive the marker fraction is to the exact
  scale, so the gate does not hinge on a single knife-edge assignment.
- **RNG purity.** As in Plan 2, generation reuses the global-seed pipeline; the local-RNG refactor
  remains a W5 item. If W5 changes seeds later, these CSVs regenerate.
