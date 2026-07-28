# PCA Decoder-Fidelity Ablation Implementation Plan

> **SUPERSEDED 2026-07-28. DO NOT EXECUTE THIS PLAN.**
>
> Tasks 1 to 4 were executed on branch `arxiv-rev-plan3-decoder` and the branch was
> deleted; `main` never contained its artifacts. The plan's premise is invalid.
>
> Its "radius restoration" treats a generated latent as though it had a discarded source
> radius recoverable from its nearest memory. It does not: only stored memories have a
> known discarded radius, and the nearest memory is the generating component only 21 to
> 38 percent of the time. Rescaling a generated sample scales its signal and its noise
> alike, so it is a decoder intervention, not an inverse.
>
> The related acceptance test, that `f_obs` should equal `f_eff` at `rho = 1`, is also
> invalid. The correct relation is the pushforward
> `f_obs = f_eff * TPR + (1 - f_eff) * FPR`, so a decoder carrying no information can
> match that marginal exactly. Never select or calibrate a decoder by marginal matching;
> use component-conditioned TPR and FPR.
>
> What remains valid: the stored-memory reconstruction fidelity diagnostic (Task 3).
> Full record in `docs/2026-07-28-decoder-postmortem.md` and the manuscript corrections in `72fccb7`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how lossy the unit-norm PCA decoder is and decide, against replicate noise, whether a radius-restoring decoder changes generated marker-positive fractions enough to force regenerating the canonical sweeps.

**Architecture:** The encoder and sampler stay unchanged (unit-norm memories preserve the Gaussian-mixture identity); only the decoder varies. New decode primitives in `Protein.jl` allow rescaling a PCA-space sample before `reconstruct`. A shared generation helper (extracted from the Plan 2 driver) produces the PCA-space samples once; the ablation decodes the same samples under the current and restored-radius decoders. Static reconstruction fidelity is measured on stored memories for all six families; the generation gate runs on Kunitz. New numbers reach the manuscript only through generated table bodies and `numbers.tex` macros.

**Tech Stack:** Julia 1.12, MultivariateStats (PCA), CSV/DataFrames, Test; Python/matplotlib for the SI figure; LaTeX (neurips_2026) for the paper.

## Global Constraints

- All Julia scripts run from `code/` and begin by including `Include.jl`. Run commands below assume `cd code` first unless noted.
- Reuse the shared registry `code/experiments/canonical_family_registry.jl`; never duplicate family split/marker definitions.
- Canonical generation parameters (verbatim): `RHO_GRID = [1,2,5,10,20,50,100,500]`, `N_REPS=5`, `N_CHAINS=20`, `N_STEPS=5000`, `BURN_IN=2000`, `THIN=100`; replicate seed `20000 + (rho_index-1)*N_REPS + rep`, chain seed `seed + chain`.
- Restored radius for a generated sample = the original PCA score-norm of its nearest memory (`k* = argmax_k X̂[:,k]ᵀξ`).
- Escalation trigger: on Kunitz, `abs(delta_mean) > replicate_std` (the canonical `f_obs_std`) at any rho.
- Committed CSVs are the trusted source; tests assert consistency and byte-compare generated LaTeX, they do not regenerate the stochastic CSVs.
- Manuscript prose conventions: no em/en dashes; no subsection headings in Intro/Results/Discussion (does not affect appendix/SI).
- Every git commit message ends with the two-line trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01N11DV24bUE3anhKRqS1Kyg`.

---

### Task 1: Decode primitives in `Protein.jl`

**Files:**
- Modify: `code/src/Protein.jl` (add `memory_radii`, `nearest_memory_radius`; add `scale` kwarg to `decode_sample` near lines 257-260)
- Create: `code/test/test_pca_decoder.jl`
- Modify: `code/test/runtests.jl:12` (register the new test file)

**Interfaces:**
- Produces:
  - `decode_sample(ξ_pca::Vector{Float64}, pca_model, L::Int; scale::Float64=1.0) -> String`
  - `memory_radii(char_mat::Matrix{Char}, pca_model) -> Vector{Float64}`
  - `nearest_memory_radius(ξ::Vector{Float64}, X̂::Matrix{Float64}, r::Vector{Float64}) -> Float64`
- Consumes: existing `onehot_encode`, `decode_onehot`, `sequence_identity`, `build_memory_matrix`, `MultivariateStats.transform/reconstruct`, `LinearAlgebra.norm`.

- [ ] **Step 1: Write the failing test**

Create `code/test/test_pca_decoder.jl`:

```julia
using Test
using LinearAlgebra
using Statistics

@testset "PCA decoder primitives" begin
    char_mat = ['A' 'C' 'D' 'E';
                'A' 'C' 'D' 'K';
                'A' 'W' 'D' 'E';
                'G' 'C' 'D' 'E';
                'A' 'C' 'F' 'E';
                'S' 'C' 'D' 'Y']
    X̂, pca_model, L, d_full = build_memory_matrix(char_mat; pratio=0.95)
    r = memory_radii(char_mat, pca_model)
    K = size(char_mat, 1)

    @test length(r) == K
    @test all(r .> 0)

    # nearest memory of a stored unit-norm column is itself
    for k in 1:K
        @test nearest_memory_radius(X̂[:, k], X̂, r) ≈ r[k]
    end

    # reconstruct is affine, so its first difference is scale-linear
    ξ = X̂[:, 1]
    lhs = MultivariateStats.reconstruct(pca_model, 2 .* ξ) .-
          MultivariateStats.reconstruct(pca_model, ξ)
    rhs = MultivariateStats.reconstruct(pca_model, ξ) .-
          MultivariateStats.reconstruct(pca_model, zero(ξ))
    @test lhs ≈ rhs

    # scale=1.0 is the current decoder
    @test decode_sample(ξ, pca_model, L) == decode_sample(ξ, pca_model, L; scale=1.0)

    # restoring the radius recovers stored memories at least as well as unit-norm decoding
    norm_ids = [sequence_identity(decode_sample(X̂[:, k], pca_model, L), String(char_mat[k, :]))
                for k in 1:K]
    rest_ids = [sequence_identity(decode_sample(X̂[:, k], pca_model, L; scale=r[k]),
                                  String(char_mat[k, :])) for k in 1:K]
    @test mean(rest_ids) >= mean(norm_ids)
end
```

- [ ] **Step 2: Register the test and run it to verify it fails**

Edit `code/test/runtests.jl` to add the include after line 12 (`include("test_generated_paper_tables.jl")`):

```julia
    include("test_pca_decoder.jl")
```

Run: `cd code && julia test/runtests.jl`
Expected: FAIL with `UndefVarError: memory_radii not defined` (or `nearest_memory_radius`).

- [ ] **Step 3: Add `scale` kwarg to `decode_sample`**

In `code/src/Protein.jl`, replace the body at lines 257-260:

```julia
function decode_sample(ξ_pca::Vector{Float64}, pca_model, L::Int; scale::Float64=1.0)
    x_onehot = vec(MultivariateStats.reconstruct(pca_model, scale .* ξ_pca))
    return decode_onehot(x_onehot, L)
end
```

- [ ] **Step 4: Add `memory_radii` and `nearest_memory_radius`**

Insert after the `decode_sample` function (after line 260 in the original file):

```julia
"""
    memory_radii(char_mat, pca_model) -> Vector{Float64}

Original PCA score norms r_k = ‖z_k‖ for each stored memory, where z_k is the
un-normalized PCA transform of the one-hot encoding. These are exactly the radii
that `build_memory_matrix` divides out when it unit-normalizes each memory column.
"""
function memory_radii(char_mat::Matrix{Char}, pca_model)
    Z = MultivariateStats.transform(pca_model, onehot_encode(char_mat))
    return [norm(Z[:, k]) for k in 1:size(Z, 2)]
end

"""
    nearest_memory_radius(ξ, X̂, r) -> Float64

Radius of the memory nearest to `ξ` in unit-norm PCA space
(k* = argmax_k X̂[:,k]ᵀξ), returning r[k*]. Used to restore a generated
sample's radius before decoding.
"""
function nearest_memory_radius(ξ::Vector{Float64}, X̂::Matrix{Float64}, r::Vector{Float64})
    return r[argmax(vec(X̂' * ξ))]
end
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd code && julia test/runtests.jl`
Expected: PASS; final line shows the full suite green (previously 52 tests, now more).

- [ ] **Step 6: Commit**

```bash
git add code/src/Protein.jl code/test/test_pca_decoder.jl code/test/runtests.jl
git commit -m "feat: radius-aware PCA decode primitives"
```

---

### Task 2: Shared ξ-generation helper and driver refactor

**Files:**
- Modify: `code/experiments/canonical_family_registry.jl` (add `generate_chain_xis`)
- Modify: `code/experiments/run_canonical_family_sweeps.jl:106-138` (call the helper)

**Interfaces:**
- Produces: `generate_chain_xis(X, weights, beta, base_seed; n_chains, n_steps, burn_in, thin) -> Vector{Vector{Float64}}` — the PCA-space samples for all chains of one replicate, in chain-major, thinned-time order (identical order to the current inline loop).
- Consumes: `weighted_sample`, `Random.seed!`, `randn` (both driver and ablation `using Random` before including the registry).

- [ ] **Step 1: Add the helper to the registry**

Append to `code/experiments/canonical_family_registry.jl`:

```julia
# Shared per-replicate generation: produce the PCA-space samples for one replicate.
# Extracted verbatim from the canonical driver so the sweep and the decoder ablation
# generate identical trajectories from the same seeds.
function generate_chain_xis(X, weights, beta, base_seed;
                            n_chains::Int, n_steps::Int, burn_in::Int, thin::Int)
    K = size(X, 2)
    d = size(X, 1)
    Random.seed!(base_seed)
    xis = Vector{Vector{Float64}}()
    for chain in 1:n_chains
        xi0 = X[:, mod1(chain, K)] .+ 0.01 .* randn(d)
        sampled = weighted_sample(X, xi0, n_steps, weights; β=beta, α=0.01, seed=base_seed + chain)
        for t in burn_in:thin:n_steps
            push!(xis, sampled.Ξ[t + 1, :])
        end
    end
    return xis
end
```

- [ ] **Step 2: Refactor the driver to call the helper**

In `code/experiments/run_canonical_family_sweeps.jl`, replace lines 106-122 (from `for rep in 1:N_REPS` through the `end` on line 122 that closes the `for chain` loop, i.e. the block that seeds the RNG and builds `generated` and `attention`) with the block below. Do not touch lines 124-138 (the `f_obs` count, diversity loop, and `push!(raw, ...)`); they stay inside the `for rep` loop exactly as they are.

```julia
        for rep in 1:N_REPS
            seed = 20_000 + (rho_index - 1) * N_REPS + rep
            xis = generate_chain_xis(analysis.X, weights, beta, seed;
                n_chains=N_CHAINS, n_steps=N_STEPS, burn_in=BURN_IN, thin=THIN)
            generated = String[]
            attention = Float64[]
            for xi in xis
                push!(generated, decode_sample(xi, analysis.pca_model, L))
                logits = beta .* (analysis.X' * xi) .+ log_weights
                attn = NNlib.softmax(logits)
                push!(attention, sum(attn[i] for i in analysis.group_A))
            end
```

Leave the rest of the replicate body (the `f_obs` count, the diversity `pair_ids` loop, and the `push!(raw, ...)`) unchanged. The `Random.seed!(seed)` that was on the old line 108 is now performed inside `generate_chain_xis`; do not add a second one.

- [ ] **Step 3: Verify the refactor is behavior-preserving on Kunitz**

Run: `cd code && julia experiments/run_canonical_family_sweeps.jl --families=kunitz`
Expected: completes with `Canonical family sweeps and cross-family CSV are complete`.

Then check nothing changed:

Run: `git status --porcelain code/data/kunitz code/data/multi_family_comparison_6fam_aggregated.csv code/data/canonical_sweep_provenance.csv`
Expected: **no output** (byte-identical regeneration). This simultaneously proves the refactor preserved RNG behavior and that the committed Kunitz CSVs are the canonical-driver output, which Task 4's gate relies on.

If there IS output, inspect `git diff code/data/kunitz/multiplicity_sweep_raw_replicates.csv`. If the per-replicate `f_obs` values changed, the refactor altered RNG consumption — revert Step 2 and reconcile before continuing (do not commit a behavior change to the canonical driver).

- [ ] **Step 4: Restore the committed data and commit code only**

Discard the regenerated (identical) data files so the commit carries only the code change:

```bash
git restore code/data
git add code/experiments/canonical_family_registry.jl code/experiments/run_canonical_family_sweeps.jl
git commit -m "refactor: share per-replicate xi generation between driver and ablation"
```

```bash
git add code/experiments/canonical_family_registry.jl code/experiments/run_canonical_family_sweeps.jl
git commit -m "refactor: share per-replicate xi generation between driver and ablation"
```

---

### Task 3: Static reconstruction fidelity (all six families)

**Files:**
- Create: `code/experiments/run_pca_decoder_ablation.jl`
- Modify: `code/test/test_pca_decoder.jl` (append a fidelity-CSV testset)
- Create (output, committed): `code/data/pca_decoder_fidelity.csv`

**Interfaces:**
- Produces `code/data/pca_decoder_fidelity.csv`, schema
  `family,K,d_full,d_pca,d_full_over_K,identity_normalized_mean,identity_normalized_std,identity_restored_mean,identity_restored_std`, one row per canonical family in `CANONICAL_FAMILIES` order.
- Consumes: `canonical_load_alignment`, `build_memory_matrix`, `memory_radii`, `decode_sample`, `sequence_identity`, `CANONICAL_FAMILIES`.

- [ ] **Step 1: Create the ablation script with the fidelity part**

Create `code/experiments/run_pca_decoder_ablation.jl`:

```julia
#!/usr/bin/env julia

# Gated PCA decoder-fidelity ablation (Plan 3, W3).
# Part A: static reconstruction fidelity for all six families.
# Parts B/C (Kunitz generation gate and radius sweep) are added in later tasks.

const SCRIPT_DIR = @__DIR__
const CODE_DIR = dirname(SCRIPT_DIR)
cd(CODE_DIR)
include(joinpath(CODE_DIR, "Include.jl"))
include(joinpath(SCRIPT_DIR, "canonical_family_registry.jl"))

using CSV, DataFrames, Random, Statistics

const DATA_DIR = joinpath(CODE_DIR, "data")
const RHO_GRID = Float64[1, 2, 5, 10, 20, 50, 100, 500]
const N_REPS = 5
const N_CHAINS = 20
const N_STEPS = 5000
const BURN_IN = 2000
const THIN = 100

function atomic_csv_write(path, table)
    mkpath(dirname(path))
    temp = path * ".tmp"
    CSV.write(temp, table)
    mv(temp, path; force=true)
end

function family_fidelity(spec::CanonicalFamilySpec)
    char_mat, _, _ = canonical_load_alignment(spec, DATA_DIR)
    X, pca_model, L, d_full = build_memory_matrix(char_mat; pratio=0.95)
    r = memory_radii(char_mat, pca_model)
    K = size(char_mat, 1)
    norm_ids = Float64[]
    rest_ids = Float64[]
    for k in 1:K
        orig = String(char_mat[k, :])
        push!(norm_ids, sequence_identity(decode_sample(X[:, k], pca_model, L), orig))
        push!(rest_ids, sequence_identity(decode_sample(X[:, k], pca_model, L; scale=r[k]), orig))
    end
    return (family=spec.family, K=K, d_full=d_full, d_pca=size(X, 1),
            d_full_over_K=d_full / K,
            identity_normalized_mean=mean(norm_ids), identity_normalized_std=std(norm_ids),
            identity_restored_mean=mean(rest_ids), identity_restored_std=std(rest_ids))
end

function write_fidelity()
    rows = [family_fidelity(spec) for spec in CANONICAL_FAMILIES]
    df = DataFrame(rows)
    atomic_csv_write(joinpath(DATA_DIR, "pca_decoder_fidelity.csv"), df)
    println("Wrote pca_decoder_fidelity.csv")
end

write_fidelity()
```

- [ ] **Step 2: Run the script to produce the CSV**

Run: `cd code && julia experiments/run_pca_decoder_ablation.jl`
Expected: prints `Wrote pca_decoder_fidelity.csv`; file `code/data/pca_decoder_fidelity.csv` exists with six data rows.

- [ ] **Step 3: Write the failing consistency test**

Append to `code/test/test_pca_decoder.jl`:

```julia
@testset "Decoder fidelity CSV" begin
    using CSV, DataFrames
    path = joinpath(@__DIR__, "..", "data", "pca_decoder_fidelity.csv")
    df = CSV.read(path, DataFrame)
    @test propertynames(df) == [:family, :K, :d_full, :d_pca, :d_full_over_K,
        :identity_normalized_mean, :identity_normalized_std,
        :identity_restored_mean, :identity_restored_std]
    @test nrow(df) == 6
    @test all(df.identity_restored_mean .>= df.identity_normalized_mean)
    kunitz = only(eachrow(filter(:family => ==("Kunitz"), df)))
    @test kunitz.identity_normalized_mean < 0.9   # the lossy unit-norm decoder
    @test kunitz.identity_restored_mean > 0.95    # radius restores fidelity
end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd code && julia test/runtests.jl`
Expected: PASS, including `Decoder fidelity CSV`. If `kunitz.identity_normalized_mean` is not below 0.9, record the actual value and report it (the corrected gap-aware audit measured ≈0.836); do not loosen the bound without surfacing the number.

- [ ] **Step 5: Commit**

```bash
git add code/experiments/run_pca_decoder_ablation.jl code/data/pca_decoder_fidelity.csv code/test/test_pca_decoder.jl
git commit -m "feat: static PCA reconstruction fidelity across six families"
```

---

### Task 4: Kunitz generation gate

**Files:**
- Modify: `code/experiments/run_pca_decoder_ablation.jl` (add the gate; call it from the bottom)
- Modify: `code/test/test_pca_decoder.jl` (append a gate-CSV testset)
- Create (output, committed): `code/data/kunitz/pca_decoder_gate.csv`

**Interfaces:**
- Produces `code/data/kunitz/pca_decoder_gate.csv`, schema
  `rho,f_obs_norm_mean,f_obs_norm_std,f_obs_restored_mean,f_obs_restored_std,delta_mean,delta_std,replicate_std,escalate`.
- Consumes: `generate_chain_xis`, `nearest_memory_radius`, `memory_radii`, `multiplicity_vector`, `find_weighted_entropy_inflection`, `canonical_split`.

- [ ] **Step 1: Add the gate function**

In `code/experiments/run_pca_decoder_ablation.jl`, add before the final `write_fidelity()` call:

```julia
marker_fraction(seqs, pos, residues) =
    count(s -> length(s) >= pos && s[pos] in residues, seqs) / length(seqs)

function run_kunitz_gate()
    spec = only(filter(s -> s.family == "Kunitz", CANONICAL_FAMILIES))
    char_mat, names, aux = canonical_load_alignment(spec, DATA_DIR)
    group_A, _, marker_pos, marker_res = canonical_split(spec, char_mat, names, aux)
    X, pca_model, L, _ = build_memory_matrix(char_mat; pratio=0.95)
    r = memory_radii(char_mat, pca_model)
    K = size(char_mat, 1)

    gate = DataFrame(rho=Float64[], f_obs_norm_mean=Float64[], f_obs_norm_std=Float64[],
        f_obs_restored_mean=Float64[], f_obs_restored_std=Float64[],
        delta_mean=Float64[], delta_std=Float64[], replicate_std=Float64[], escalate=Bool[])

    for (rho_index, rho) in enumerate(RHO_GRID)
        @info "Kunitz gate: rho=$rho"
        weights = multiplicity_vector(K, group_A; ρ=rho)
        beta = find_weighted_entropy_inflection(X, weights; n_betas=50).β_star
        nf = Float64[]; rf = Float64[]; deltas = Float64[]
        for rep in 1:N_REPS
            seed = 20_000 + (rho_index - 1) * N_REPS + rep
            xis = generate_chain_xis(X, weights, beta, seed;
                n_chains=N_CHAINS, n_steps=N_STEPS, burn_in=BURN_IN, thin=THIN)
            norm_seqs = [decode_sample(xi, pca_model, L) for xi in xis]
            rest_seqs = [decode_sample(xi, pca_model, L;
                            scale=nearest_memory_radius(xi, X, r)) for xi in xis]
            fn = marker_fraction(norm_seqs, marker_pos, marker_res)
            fr = marker_fraction(rest_seqs, marker_pos, marker_res)
            push!(nf, fn); push!(rf, fr); push!(deltas, fr - fn)
        end
        rep_std = std(nf)
        push!(gate, (rho, mean(nf), std(nf), mean(rf), std(rf),
            mean(deltas), std(deltas), rep_std, abs(mean(deltas)) > rep_std))
    end
    atomic_csv_write(joinpath(DATA_DIR, "kunitz", "pca_decoder_gate.csv"), gate)
    escalated = any(gate.escalate)
    println("Kunitz gate written; max |delta| = ",
            round(maximum(abs.(gate.delta_mean)); digits=4),
            "; escalate = ", escalated)
    return gate
end
```

Change the bottom of the file from `write_fidelity()` to:

```julia
write_fidelity()
run_kunitz_gate()
```

- [ ] **Step 2: Run the ablation to produce the gate CSV**

Run: `cd code && julia experiments/run_pca_decoder_ablation.jl`
Expected: prints the fidelity line, then per-rho `Kunitz gate` info lines, then `Kunitz gate written; max |delta| = <x>; escalate = <bool>`. Record the printed max delta and escalate value in the commit message and report them at task review.

- [ ] **Step 3: Write the failing gate-consistency test**

Append to `code/test/test_pca_decoder.jl`:

```julia
@testset "Decoder gate CSV" begin
    using CSV, DataFrames
    gate = CSV.read(joinpath(@__DIR__, "..", "data", "kunitz", "pca_decoder_gate.csv"), DataFrame)
    @test propertynames(gate) == [:rho, :f_obs_norm_mean, :f_obs_norm_std,
        :f_obs_restored_mean, :f_obs_restored_std, :delta_mean, :delta_std,
        :replicate_std, :escalate]
    @test gate.rho == Float64[1, 2, 5, 10, 20, 50, 100, 500]
    @test all(0 .<= gate.f_obs_norm_mean .<= 1)
    @test all(0 .<= gate.f_obs_restored_mean .<= 1)
    # The normalized decoder reproduces the committed canonical aggregate.
    agg = CSV.read(joinpath(@__DIR__, "..", "data", "kunitz",
                            "multiplicity_sweep_aggregated.csv"), DataFrame)
    for rho in gate.rho
        g = only(eachrow(filter(:rho => ==(rho), gate)))
        a = only(eachrow(filter(:rho => ==(rho), agg)))
        @test isapprox(g.f_obs_norm_mean, a.f_obs_mean; atol=1e-6)
    end
    # escalate is exactly abs(delta_mean) > replicate_std, row by row.
    @test all(gate.escalate .== (abs.(gate.delta_mean) .> gate.replicate_std))
end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd code && julia test/runtests.jl`
Expected: PASS, including `Decoder gate CSV`. The `f_obs_norm_mean` versus committed `f_obs_mean` check proves the gate reproduced the canonical Kunitz generation (0.581 ± 0.006 at rho=500).

- [ ] **Step 5: Commit**

```bash
git add code/experiments/run_pca_decoder_ablation.jl code/data/kunitz/pca_decoder_gate.csv code/test/test_pca_decoder.jl
git commit -m "feat: Kunitz decoder gate (normalized vs restored radius)"
```

---

### Task 5: Radius sweep and SI figure

**Files:**
- Modify: `code/experiments/run_pca_decoder_ablation.jl` (emit the sweep CSV from the same generation)
- Create: `code/experiments/render_decoder_radius_sweep.py`
- Modify: `code/test/test_pca_decoder.jl` (append a sweep-CSV testset)
- Create (output, committed): `code/data/kunitz/pca_decoder_radius_sweep.csv`, `paper-arxiv/sections/figs/figS_decoder_radius_sweep.pdf`, `paper-arxiv/sections/figs/figS_decoder_radius_sweep.png`

**Interfaces:**
- Produces `code/data/kunitz/pca_decoder_radius_sweep.csv`, schema `rho,scale,f_obs_mean,f_obs_std`. The `scale=1.0` rows equal the gate's `f_obs_norm_mean` by construction.

- [ ] **Step 1: Extend the gate to also record the sweep**

In `run_kunitz_gate`, compute the scale grid after `K = size(char_mat, 1)`:

```julia
    scale_grid = round.(vcat(1.0, collect(range(2.0, maximum(r), length=6))); digits=3)
    sweep = DataFrame(rho=Float64[], scale=Float64[], f_obs_mean=Float64[], f_obs_std=Float64[])
```

Inside the `for rep` loop, after computing `rest_seqs`, accumulate per-scale fractions. Add a per-rho accumulator declared next to `nf`/`rf`:

```julia
        sweep_by_scale = Dict(s => Float64[] for s in scale_grid)
```

and inside the `for rep` loop (after the `push!(deltas, ...)` line) add:

```julia
            for s in scale_grid
                seqs = [decode_sample(xi, pca_model, L; scale=s) for xi in xis]
                push!(sweep_by_scale[s], marker_fraction(seqs, marker_pos, marker_res))
            end
```

After the `push!(gate, ...)` line, add:

```julia
        for s in scale_grid
            push!(sweep, (rho, s, mean(sweep_by_scale[s]), std(sweep_by_scale[s])))
        end
```

Before `return gate`, write the sweep:

```julia
    atomic_csv_write(joinpath(DATA_DIR, "kunitz", "pca_decoder_radius_sweep.csv"), sweep)
```

- [ ] **Step 2: Run the ablation to produce the sweep CSV**

Run: `cd code && julia experiments/run_pca_decoder_ablation.jl`
Expected: completes; `code/data/kunitz/pca_decoder_radius_sweep.csv` exists with `8 * 7 = 56` data rows.

- [ ] **Step 3: Write the failing sweep test**

Append to `code/test/test_pca_decoder.jl`:

```julia
@testset "Decoder radius sweep CSV" begin
    using CSV, DataFrames
    sweep = CSV.read(joinpath(@__DIR__, "..", "data", "kunitz",
                              "pca_decoder_radius_sweep.csv"), DataFrame)
    @test propertynames(sweep) == [:rho, :scale, :f_obs_mean, :f_obs_std]
    @test Set(sweep.rho) == Set(Float64[1, 2, 5, 10, 20, 50, 100, 500])
    @test 1.0 in sweep.scale
    gate = CSV.read(joinpath(@__DIR__, "..", "data", "kunitz",
                             "pca_decoder_gate.csv"), DataFrame)
    for rho in gate.rho
        s1 = only(eachrow(filter(row -> row.rho == rho && row.scale == 1.0, sweep)))
        g = only(eachrow(filter(:rho => ==(rho), gate)))
        @test isapprox(s1.f_obs_mean, g.f_obs_norm_mean; atol=1e-9)
    end
end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd code && julia test/runtests.jl`
Expected: PASS, including `Decoder radius sweep CSV`.

- [ ] **Step 5: Create the figure script**

Create `code/experiments/render_decoder_radius_sweep.py`:

```python
"""Render the Kunitz decoder radius-sweep SI figure from the canonical CSV."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
CSV_PATH = CODE / "data" / "kunitz" / "pca_decoder_radius_sweep.csv"
OUT = CODE.parent / "paper-arxiv" / "sections" / "figs" / "figS_decoder_radius_sweep"

by_rho = defaultdict(list)
with open(CSV_PATH) as fh:
    for row in csv.DictReader(fh):
        by_rho[float(row["rho"])].append(
            (float(row["scale"]), float(row["f_obs_mean"]), float(row["f_obs_std"])))

fig, ax = plt.subplots(figsize=(5.0, 3.6))
for rho in sorted(by_rho):
    pts = sorted(by_rho[rho])
    scales = [p[0] for p in pts]
    means = [p[1] for p in pts]
    stds = [p[2] for p in pts]
    ax.errorbar(scales, means, yerr=stds, marker="o", markersize=3,
                linewidth=1, capsize=2, label=f"$\\rho={int(rho)}$")
ax.set_xlabel("decode radius scale")
ax.set_ylabel("marker-positive fraction")
ax.set_title("Kunitz decoder radius sweep")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=200)
print(f"Wrote {OUT}.pdf and {OUT}.png")
```

- [ ] **Step 6: Render the figure**

Run: `cd code && python experiments/render_decoder_radius_sweep.py`
Expected: prints `Wrote .../figS_decoder_radius_sweep.pdf and .../figS_decoder_radius_sweep.png`; both files exist.

- [ ] **Step 7: Commit**

```bash
git add code/experiments/run_pca_decoder_ablation.jl code/experiments/render_decoder_radius_sweep.py \
  code/data/kunitz/pca_decoder_radius_sweep.csv code/test/test_pca_decoder.jl \
  paper-arxiv/sections/figs/figS_decoder_radius_sweep.pdf \
  paper-arxiv/sections/figs/figS_decoder_radius_sweep.png
git commit -m "feat: decoder radius sweep CSV and SI figure"
```

---

### Task 6: Generate the decoder table and macros

**Files:**
- Modify: `code/experiments/generate_paper_tables.jl` (read the two decoder CSVs; emit `tab_decoder_fidelity.tex`; append macros to `numbers.tex`)
- Modify: `code/test/test_generated_paper_tables.jl:12-22` (add `tab_decoder_fidelity.tex` to `expected_files`)
- Create (committed): `paper-arxiv/sections/generated/tab_decoder_fidelity.tex`
- Modify (regenerated): `paper-arxiv/sections/generated/numbers.tex`

**Interfaces:**
- Produces `tab_decoder_fidelity.tex` (a tabular body ending in `\bottomrule`) and new macros `\KunitzIdentityNormalized`, `\KunitzIdentityRestored`, `\DecoderGateDeltaMax`, `\DecoderGateEscalate`, `\DecoderDfullKMin`, `\DecoderDfullKMax`.
- Consumes: `code/data/pca_decoder_fidelity.csv`, `code/data/kunitz/pca_decoder_gate.csv`.

- [ ] **Step 1: Add a decoder-inputs reader to the generator**

In `code/experiments/generate_paper_tables.jl`, add after `linear_fit` (before `write_text`):

```julia
function read_decoder_inputs()
    fidelity_path = joinpath(DATA_DIR, "pca_decoder_fidelity.csv")
    gate_path = joinpath(DATA_DIR, "kunitz", "pca_decoder_gate.csv")
    isfile(fidelity_path) || error("Missing decoder fidelity CSV: $fidelity_path")
    isfile(gate_path) || error("Missing decoder gate CSV: $gate_path")
    fidelity = CSV.read(fidelity_path, DataFrame)
    nrow(fidelity) == 6 || error("Decoder fidelity CSV must have six rows")
    gate = CSV.read(gate_path, DataFrame)
    nrow(gate) == 8 || error("Decoder gate CSV must have eight rho rows")
    return fidelity, gate
end
```

- [ ] **Step 2: Emit the fidelity table and macros**

In `generate(output_dir)`, just before the final `write_text(joinpath(output_dir, "numbers.tex"), ...)` call, insert the table emission:

```julia
    fidelity, gate = read_decoder_inputs()
    fidelity_rows = String[]
    for row in eachrow(fidelity)
        push!(fidelity_rows, @sprintf(
            "%s & %d & %d & %d & %.1f & %.3f & %.3f \\\\",
            latex_family(row.family), row.K, row.d_full, row.d_pca,
            row.d_full_over_K, row.identity_normalized_mean, row.identity_restored_mean))
    end
    write_text(joinpath(output_dir, "tab_decoder_fidelity.tex"),
               join(fidelity_rows, "\n") * "\n" * raw"\bottomrule")

    kunitz_fid = only(eachrow(filter(:family => ==("Kunitz"), fidelity)))
    gate_delta_max = maximum(abs.(gate.delta_mean))
    escalate_word = any(gate.escalate) ? "does" : "does not"
```

Then extend the `macros` vector (append these entries to the array literal, before `write_text(... "numbers.tex" ...)`):

```julia
    push!(macros, "\\newcommand{\\KunitzIdentityNormalized}{$(fmt3(kunitz_fid.identity_normalized_mean))}")
    push!(macros, "\\newcommand{\\KunitzIdentityRestored}{$(fmt3(kunitz_fid.identity_restored_mean))}")
    push!(macros, "\\newcommand{\\DecoderGateDeltaMax}{$(fmt3(gate_delta_max))}")
    push!(macros, "\\newcommand{\\DecoderGateEscalate}{$(escalate_word)}")
    push!(macros, "\\newcommand{\\DecoderDfullKMin}{$(fmt1(minimum(fidelity.d_full_over_K)))}")
    push!(macros, "\\newcommand{\\DecoderDfullKMax}{$(fmt1(maximum(fidelity.d_full_over_K)))}")
```

Note: the existing `macros` is built as a single array literal then the per-family loop uses `push!`. Add the six `push!` lines in that same region (after the per-family natural-fraction loop, before `write_text(joinpath(output_dir, "numbers.tex"), join(macros, "\n"))`).

- [ ] **Step 3: Regenerate the committed generated directory**

Run: `cd code && julia experiments/generate_paper_tables.jl`
Expected: prints `Generated manuscript tables and macros in .../paper-arxiv/sections/generated`. `git status` shows a new `tab_decoder_fidelity.tex` and a modified `numbers.tex`.

- [ ] **Step 4: Update the regression test's expected file list, run it (verify it fails first)**

First run the suite to confirm it fails on the file-list mismatch:

Run: `cd code && julia test/runtests.jl`
Expected: FAIL in `Canonical CSV manuscript artifacts` (readdir includes `tab_decoder_fidelity.tex` not in `expected_files`).

Then edit `code/test/test_generated_paper_tables.jl` to add `"tab_decoder_fidelity.tex",` to the `expected_files` vector (inside the `sort([...])` at lines 12-22).

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd code && julia test/runtests.jl`
Expected: PASS; `Canonical CSV manuscript artifacts` byte-compares all files including `tab_decoder_fidelity.tex` and the extended `numbers.tex`.

- [ ] **Step 6: Commit**

```bash
git add code/experiments/generate_paper_tables.jl code/test/test_generated_paper_tables.jl \
  paper-arxiv/sections/generated/tab_decoder_fidelity.tex paper-arxiv/sections/generated/numbers.tex
git commit -m "feat: generate decoder-fidelity table and macros"
```

---

### Task 7: SI subsection, appendix fixes, and paper build

**Files:**
- Create: `paper-arxiv/sections/si_pca_decoder.tex`
- Modify: `paper-arxiv/sections/appendix.tex` (stale descriptors at lines 29-32 and 55; `\input` the new SI file after line 312)
- Modify (rebuilt, committed): `paper-arxiv/Paper_v1.pdf`, `Paper_v1.aux`, `Paper_v1.log`, `Paper_v1.out`

**Interfaces:**
- Consumes generated `sections/generated/tab_decoder_fidelity` and the macros from `sections/generated/numbers`.

- [ ] **Step 1: Create the SI subsection**

Create `paper-arxiv/sections/si_pca_decoder.tex`:

```latex
\section{Decoder Fidelity and the PCA Radius}\label{app:decoder}

The encoder normalizes each PCA score vector to unit norm (Section~\ref{app:pca}),
which the Gaussian-mixture identity requires, but the decoder reconstructs a
generated state $\boldsymbol{\xi}$ directly through the inverse PCA map. Because a
generated state lies near a unit-norm memory, the data-dependent term is small
relative to the mean profile, so argmax decoding leans on the family consensus.
On stored memories, unit-norm decoding recovers only
$\KunitzIdentityNormalized$ mean identity on Kunitz, versus $\KunitzIdentityRestored$
when the original radius is restored before reconstruction
(Table~\ref{tab:decoder-fidelity}).

\begin{table}[ht]
  \centering
  \caption{Reconstruction identity of stored memories under unit-norm decoding
    versus restored-radius decoding, per family.}
  \label{tab:decoder-fidelity}
  \small
  \begin{tabular}{lccccc}
    \toprule
    Family & $K$ & $d_{\mathrm{full}}$ & $d$ & $d_{\mathrm{full}}/K$
    & Identity (norm.\ / restored) \\
    \midrule
    \input{sections/generated/tab_decoder_fidelity}
  \end{tabular}
\end{table}

To test whether this loss changes generation, we decoded the same generated Kunitz
samples with the current decoder and with restored-radius decoding
(scaling each state to the original norm of its nearest memory), across the
multiplicity grid and five replicates. The marker-positive fraction $\DecoderGateEscalate$
change beyond replicate noise (maximum absolute difference $\DecoderGateDeltaMax$),
so the reported multiplicity results are unaffected by the decoder radius.
Figure~\ref{fig:decoder-radius-sweep} shows the marker-positive fraction as a
function of the decode radius scale.

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.7\linewidth]{sections/figs/figS_decoder_radius_sweep.pdf}
  \caption{Kunitz marker-positive fraction versus decode radius scale, per
    multiplicity ratio $\rho$ (five replicates; error bars are replicate standard
    deviations). Scale $1$ is the decoder used throughout the paper.}
  \label{fig:decoder-radius-sweep}
\end{figure}
```

- [ ] **Step 2: Input the SI file into the appendix**

In `paper-arxiv/sections/appendix.tex`, after line 312 (`\input{sections/si_gmm_derivation}`), add:

```latex
\input{sections/si_pca_decoder}
```

- [ ] **Step 3: Fix the stale appendix descriptors**

In `paper-arxiv/sections/appendix.tex`, make three edits.

Edit 1 (lines 29-30), replace:

```latex
$d_{\mathrm{full}} = 20L$, which ranged from 520 ($\omega$-conotoxin,
$L{=}26$) to 1{,}060 (Kunitz, $L{=}53$) across the four families. This space
```

with:

```latex
$d_{\mathrm{full}} = 20L$, which ranged from 520 ($\omega$-conotoxin,
$L{=}26$) to 1{,}740 (Forkhead, $L{=}87$) across the six families. This space
```

Edit 2 (line 32), replace:

```latex
($d_{\mathrm{full}}/K$ ranged from 7.0 to 10.7), which creates a problem for
```

with:

```latex
($d_{\mathrm{full}}/K$ ranged from \DecoderDfullKMin\ to \DecoderDfullKMax), which creates a problem for
```

Edit 3 (line 55), replace `Across the four families, $d$ ranged from 34` with `Across the six families, $d$ ranged from 34`.

- [ ] **Step 4: Build the paper**

Run: `cd paper-arxiv && ./Build.sh Paper_v1`
Expected: completes; `Paper_v1.pdf` regenerated.

- [ ] **Step 5: Verify a clean build with both SI labels resolved**

Run:
```bash
cd paper-arxiv
grep -c "app:decoder" Paper_v1.aux
grep -c "app:gmm" Paper_v1.aux
grep -nE "Undefined|undefined|multiply defined|Citation.*undefined|Rerun to get" Paper_v1.log
```
Expected: each `grep -c` prints `1` (both labels defined); the final `grep` prints nothing (no undefined refs/citations, no rerun warning). If anything is undefined, fix and rebuild before committing.

- [ ] **Step 6: Commit**

```bash
git add paper-arxiv/sections/si_pca_decoder.tex paper-arxiv/sections/appendix.tex \
  paper-arxiv/Paper_v1.pdf paper-arxiv/Paper_v1.aux paper-arxiv/Paper_v1.log paper-arxiv/Paper_v1.out
git commit -m "docs: decoder-fidelity SI section and appendix descriptor fixes"
```

- [ ] **Step 7: Report the gate decision**

Summarize for review: the max absolute paired delta over rho, the replicate noise it was compared against, and whether any rho escalated. If escalate is true at any rho, STOP and surface it: the reported canonical numbers are provisional and a six-family regeneration under restored-radius decoding (a separate plan) is required before proceeding to Plan 5.

---

## Notes on escalation (out of scope here)

If the gate escalates, do not regenerate the six families in this plan. Report the finding; the follow-up regeneration and any manuscript-number changes are a separate spec. Weighted MALA (the other W3 opt-in) also remains deferred.
