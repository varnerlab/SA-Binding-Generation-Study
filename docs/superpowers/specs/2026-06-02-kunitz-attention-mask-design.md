# Design: Attention-Masking Conditioning on Kunitz (P1 K/R)

- Date: 2026-06-02
- Author: Jeffrey Varner (with Claude)
- Status: Draft for review
- Topic: Add hard attention masking as the `b = -∞` endpoint of the multiplicity-conditioning axis, and validate it on the Kunitz family as a focus/enrich experiment.

## Motivation

The companion stochastic-attention (SA) paper introduces conditional sampling by masking
the attention softmax: with a bias vector `b ∈ (ℝ ∪ {-∞})^K` added to the logits,
`softmax(β Xᵀξ + b)`, setting `b_k = -∞` zeroes the softmax weight on out-of-set
memories. On Olivetti faces this lifted subject-recovery from 10.4% (≈ random) to 96.0%
by suppressing the inter-basin drift that otherwise carries a chain out of the target
subject (`method.tex:119-128`, `appendix.tex:566-588`, Table `tab:olivetti-recovery`).

That paper explicitly frames *our* multiplicity conditioning as the finite-bias
generalization of its hard mask: "A finite-bias generalization, `b_k ∈ ℝ`, recovers the
Hopfield-multiplicity conditioning of Varner" (`method.tex:128`). So both methods are one
primitive at two ends of a single axis:

| Bias vector `b` | Behavior | Where studied |
|---|---|---|
| `b = 0` everywhere | unconditional SA | base method |
| `b = log r` (finite) | multiplicity conditioning | this project |
| `b = -∞` on background | hard mask = exact curation on reduced matrix | companion paper (faces) |

This project's central open problem is the **calibration gap**: the observed designated
fraction in generated sequences lags the requested target (`regenerate_combined_figures.jl:42-50`
plots "Observed P1 K/R fraction" vs "Target effective binder fraction" against the ideal
`y = x`). The mask is the exact `ρ → ∞` endpoint of that curve, so adding it is not just a
demonstration — it directly probes whether, and how much of, the calibration gap is closable.

## Core idea

**Why the mask is the multiplicity endpoint (softmax shift-invariance).** Multiplicity
conditioning is the `b = log r` case: with `r_designated = ρ`, `r_background = 1`, the bias is
`b = log r = [log ρ on designated, 0 on background]`. Because softmax is invariant to a
constant shift of all logits, subtracting `log ρ` gives the identical dynamics with
`b = [0 on designated, -log ρ on background]`. So multiplicity conditioning is exactly a
**soft background mask** with finite negative bias `-log ρ`, and the **hard mask
(`b = -∞` on background) is its `log ρ → ∞` limit**. The three regimes are one axis through
one core:

| Regime | Bias `b` | Realized by |
|---|---|---|
| Unconditional | `0` | `sample` (≡ `logit_bias_sample(zeros(K))`) |
| Multiplicity (`b = log r`) | `log ρ` on designated, `0` on background (≡ `-log ρ` on background) | `weighted_sample(multiplicity_vector(K, designated; ρ))` |
| Hard mask | `-∞` on background | `masked_sample(keep = designated)` |

The mask is the genuine `ρ → ∞` limit, and it is distinct from the two conditions the
multiplicity experiment already runs:

- `ρ = 500` multiplicity (full basis): background still carries weight 1, so a small but
  nonzero residual mass remains on background patterns (`run_multiplicity_conditioning.jl:210-212`).
- Hard curation (subset basis): rebuilds the memory matrix *and PCA basis* from the
  designated subset only, discarding the background scaffold and failing when the subset is
  too small to fit a PCA (`run_kunitz_binding_experiment.jl:104-108`).
- **Hard mask (full basis, background weight exactly 0):** keeps the full-family PCA
  geometry and only restricts attention. This is new, and it isolates two effects:
  1. **Scaffold preservation** vs curation: same geometry as unconditional, so we can show
     the mask enriches the P1 feature without distorting the rest of the Kunitz scaffold.
  2. **No small-subset failure:** the mask always uses the full-family basis, so it has no
     minimum-subset requirement.

**Calibration-gap decomposition (the analysis payoff).** If `ρ = 500` saturates below 1.0
but the mask reaches ≈ 1.0, the residual gap at finite ρ is residual background attention
mass, removable only by `b = -∞`. If a gap remains even under the mask, that residual is
intrinsic to the decode/geometry (inverse-PCA + per-position argmax) and is irreducible by
conditioning alone. Either outcome is informative and publishable.

## Code design

All sampler code lives in `code/src/`. `softmax` is `NNlib.softmax` (numerically stable;
`-∞` logits map to weight 0 with no NaN, provided ≥ 1 finite entry), so the mask is safe.

All three regimes from the Core idea flow through the one core `logit_bias_sample(X, ξ₀, T, b)`:

- **Unconditional** (`b = 0`): `logit_bias_sample(X, ξ₀, T, zeros(K))`, equivalent to the
  existing `sample` (verified by a unit test below).
- **Multiplicity** (`b = log r`): `weighted_sample(X, ξ₀, T, r)` with
  `r = multiplicity_vector(K, designated; ρ)`. After the refactor `weighted_sample` is itself
  a wrapper that passes `b = log.(r)` to the core, so the existing multiplicity path is
  unchanged in behavior and now shares the core. No new multiplicity function is added.
- **Hard mask** (`b = -∞` on background): `masked_sample(X, ξ₀, T, keep = designated)`.

### 1. New unified core: `logit_bias_sample`

Add to `Binding.jl`. (Name avoids collision with the existing `biased_sample` at
`Binding.jl:205`, which is Approach 2's energy-gradient bias — a different mechanism.)

```julia
logit_bias_sample(X, ξ₀, T, b; β=1.0, α=0.1, seed=nothing)
```

- `b::Vector{Float64}` is the additive logit bias, length `K`; entries may be `-Inf`.
- Update: `logits = β .* (X' * ξ) .+ b; w = softmax(logits)`, then the standard Langevin
  step `ξ .= (1-α)ξ .+ α (X*w) .+ sqrt(2α/β) ε`.
- Validation: `length(b) == K`; `any(isfinite, b)` (at least one reachable memory);
  standard `T>0`, `β>0`, `0<α<1` checks.
- Returns `(t, Ξ)`, identical shape to `weighted_sample`.

### 2. `weighted_sample` becomes a thin wrapper

Refactor `weighted_sample` (`Binding.jl:546`) to delegate: `logit_bias_sample(X, ξ₀, T, log.(w); ...)`.
Relax the positivity guard from `all(w -> w > 0)` (line 559) to `all(w -> w >= 0)` **and**
`any(w -> w > 0)`. `log(0.0) = -Inf` is handled by the core. Existing callers pass strictly
positive weights, so behavior is unchanged (backward compatible).

### 3. New `masked_sample` wrapper

```julia
masked_sample(X, ξ₀, T, keep::AbstractVector{Bool}; β, α, seed)   # length-K keep mask
masked_sample(X, ξ₀, T, keep_indices::Vector{Int}; ...)            # convenience
```

Builds `b = [k ? 0.0 : -Inf for k in keep]` and calls `logit_bias_sample`. This is the
paper-notation API.

### 4. Generation wrapper: `generate_masked_sequences`

Mirror `generate_weighted_sequences` (`Binding.jl:897`). Thin wrapper that builds the
keep mask / bias and calls `masked_sample` per chain, with the same chain/burn-in/thin/decode
loop and the same `decode_sample` inverse-PCA path used everywhere else. (Equivalent to
calling `generate_weighted_sequences` with a 0/1 weight vector now that zeros are allowed;
the dedicated wrapper just reads cleanly in the experiment script and matches paper notation.)

Add helper `mask_vector(K, keep_indices) -> Vector{Float64}` (1 on keep, 0 elsewhere) for
the weight-vector route and for symmetry with `multiplicity_vector` (`Binding.jl:632`).

## Experiment design

Add a new sibling script `experiments/run_kunitz_mask_experiment.jl` that reuses the P1 split
logic from `run_kunitz_binding_experiment.jl` (keeping the mask result modular rather than
bloating the existing experiment). Designated subset = K/R at the P1
position, found automatically as today (`dump_entropy_curves.jl:17-22`,
`run_kunitz_binding_experiment.jl:54-89`). All conditions share the full-family basis
`(X̂_all, pca_all)` except curation.

| # | Condition | Basis | Background weight | Status |
|---|---|---|---|---|
| 1 | Unconditional (`b=0`) | full | 1 | exists |
| 2 | Multiplicity sweep (finite ρ) | full | 1 | exists |
| 3 | `ρ = 500` multiplicity | full | 1 (small residual mass) | exists |
| 4 | **Hard mask (`b=-∞` on background)** | full | **0** | **new** |
| 5 | Hard curation | subset | n/a (background removed) | exists |

- **Primary metric:** fraction K/R at P1 (`p1_phenotype_analysis`), the recovery-rate
  analog. Place condition 4 on the existing observed-vs-target calibration curve as the
  `ρ → ∞` endpoint.
- **Hypothesis:** finite ρ (incl. ρ=500) saturates below 1.0; the mask drives it to ≈ 1.0.
  Report the gap decomposition (residual-mass vs decode/geometry) per the Core idea.
- **Quality guards** (focus must not collapse to memorization):
  - Novelty must stay > 0 (`Utilities.jl` novelty).
  - Diversity (mean pairwise distance).
  - ESM2 pseudo-perplexity for plausibility (`score_esm2_perplexity.py`).
  - Hopfield energy (`weighted_hopfield_energy` / `Utilities.jl`).
  - Scaffold preservation: AA-composition KL and/or non-P1 sequence-identity to the family,
    comparing mask (full basis) vs curation (subset basis).
- **Optional:** fixed-mask β-sweep mirroring the companion paper's `tab:olivetti-betasweep`
  to show the within-designated Hopfield → Hebbian transition holds inside the masked regime
  (reuse `find_weighted_entropy_inflection`, `Binding.jl:715`).

## Deliverables

- A protein analog of `tab:olivetti-recovery`: unconditional vs mask P1 K/R recovery (+ ρ=500
  and curation for context).
- The existing calibration figure with the mask endpoint marked, plus the gap decomposition.
- Paper text: extend methods/results with the mask as the `b=-∞` endpoint, citing the
  companion masking section and its "finite-bias generalization" sentence. Follow paper style
  (no subsection headings in Results, flowing prose, no em/en dashes, "designated/background"
  terminology).

## File-level change list

- `code/src/Binding.jl`: add `logit_bias_sample`, `masked_sample` (2 methods), `mask_vector`,
  `generate_masked_sequences`; refactor `weighted_sample` to delegate and relax its guard.
- `code/experiments/run_kunitz_mask_experiment.jl` (new): reuse the P1 split from
  `run_kunitz_binding_experiment.jl`, add condition 4 (the mask) plus the recovery /
  calibration / guard reporting; write a CSV alongside the existing sweep CSVs.
- `code/test/` (if a test dir exists; otherwise a `experiments/test_mask_equivalence.jl`):
  the equivalence and safety unit tests below.
- `paper/sections/`: methods + results text and the new table/figure (separate pass after the
  experiment lands).

## Testing plan

- **Reduced-matrix equivalence (exact).** The paper's claim is that masked SA on full `X`
  equals unmasked SA on the reduced matrix `X_S`. With the same seed,
  `masked_sample(X, ξ₀, T, keep)` must reproduce `sample(X[:, keep_cols], ξ₀, T)` to machine
  precision (same `d`, identical `randn(d)` draws, `X w` collapses to `X_S w_S`). This both
  validates correctness and verifies the paper's identity.
- **`b = 0` reduces to unconditional.** `logit_bias_sample(X, ξ₀, T, zeros(K))` matches
  `sample(X, ξ₀, T)` at the same seed.
- **`-∞` safety.** No NaN/Inf in the trajectory when `b` has `-∞` entries, as long as ≥ 1
  finite entry; error cleanly when all entries are `-∞`.
- **Backward compatibility.** `weighted_sample` with strictly positive weights produces the
  same output as before the refactor (golden trajectory at a fixed seed).

## Risks and open questions

- **Mask vs ρ=500 may look similar** if the residual background mass at ρ=500 is already
  negligible. Mitigation: that *is* a finding (the gap is decode/geometry-bound, not
  mass-bound); report it as the decomposition result rather than a null.
- **Decode/geometry floor.** The inverse-PCA + argmax decode may impose a recovery ceiling
  below 1.0 even under the mask. This is the irreducible component we want to quantify.
- **β choice under the mask.** Use the generation regime (≈ 2β*) as elsewhere; optionally
  re-derive β* on the masked landscape via `find_weighted_entropy_inflection`.

## Out of scope

- DARPins / any new (7th) protein family. Parked as a separate sub-project (data sourcing
  and variable-repeat alignment are non-trivial).
- Rolling the mask across all 5-6 families (cross-family table). Possible follow-up once the
  Kunitz result is solid.
- Soft/partial masks for per-example ablation or data cleaning. The primitive supports them
  (any finite or `-∞` `b`), but the experiments here use the full-background mask only.
