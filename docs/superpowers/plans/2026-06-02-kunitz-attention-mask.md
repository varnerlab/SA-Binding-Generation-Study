# Kunitz Attention-Masking Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hard attention masking (`b = -∞` on background memories) as the `ρ → ∞` endpoint of the multiplicity-conditioning axis, unified through one sampler core, and validate it on the Kunitz family by measuring P1 K/R recovery and decomposing the calibration gap.

**Architecture:** Introduce one core sampler `logit_bias_sample(X, ξ₀, T, b)` that adds an additive logit bias inside the attention softmax. `weighted_sample` (multiplicity, `b = log r`) and a new `masked_sample` (`b = -∞` on background) become thin wrappers over it. A new Kunitz experiment script adds the mask, hard-curation, and unconditional points to the existing calibration curve and reports the gap decomposition.

**Tech Stack:** Julia 1.12; NNlib (`softmax`, `-∞`-safe); MultivariateStats (PCA); DataFrames + CSV; Plots; `Test` stdlib. All source in `code/src/`, experiments in `code/experiments/`, run from `code/`.

**Spec:** `docs/superpowers/specs/2026-06-02-kunitz-attention-mask-design.md`

---

## File Structure

- `code/src/Binding.jl` — add `logit_bias_sample`, refactor `weighted_sample`, add `masked_sample` (2 methods), `mask_vector`, `generate_masked_sequences`.
- `code/test/test_mask.jl` — new test file (project has no test harness yet); one `@testset` per sampler task.
- `code/Project.toml` — add `Test` (stdlib) to `[deps]`.
- `code/experiments/run_kunitz_mask_experiment.jl` — new experiment: conditions, metrics, calibration figure, gap decomposition, CSV.
- `paper/sections/results.tex`, `paper/sections/methods.tex` — recovery table + prose (final task, after the run produces numbers).

Conventions confirmed from the codebase:
- `sample(X, ξ₀, T; β, α, seed)` returns `(t, Ξ)` where `Ξ` is `(T+1)×d`; uses `softmax` = `NNlib.softmax` (`Compute.jl`).
- `weighted_sample(X, ξ₀, T, pattern_weights; β, α, seed)` currently validates `all(w -> w > 0)` and computes `logits = β .* (X' * ξ) .+ log.(w)` (`Binding.jl:546-583`).
- `decode_sample(ξ_pca, pca_model, L)` (`Protein.jl:257`); `build_memory_matrix(char_mat; pratio)` returns `(X̂, pca, L, d_full)`.
- `multiplicity_vector(K, idx; ρ)`, `effective_binder_fraction(r, idx)`, `find_entropy_inflection(X̂).β_star`, `build_multiplicity_conditioned_memory(char_mat, idx; f_target)` returning `.X̂ .pca_model .r .ρ .f_eff .K_eff` (`Binding.jl`, `run_multiplicity_conditioning.jl:173`).
- Metric helpers: `sequence_identity`, `nearest_sequence_identity`, `aa_composition_kl`, `valid_residue_fraction`, `aa_freq_matrix` (`Protein.jl`).

Tests run from `code/` with: `julia --project=. test/test_mask.jl`. The test file includes the bootstrap via `include(joinpath(@__DIR__, "..", "Include.jl"))` so cwd does not matter for paths.

---

### Task 1: Add `Test` dep + core `logit_bias_sample` (b=0 ≡ sample)

**Files:**
- Modify: `code/Project.toml` (add `Test`)
- Create: `code/test/test_mask.jl`
- Modify: `code/src/Binding.jl` (add `logit_bias_sample` above the `weighted_sample` docstring, currently `Binding.jl:520`)

- [ ] **Step 1: Add the `Test` stdlib to the project**

Run (from `code/`):
```bash
julia --project=. -e 'using Pkg; Pkg.add("Test")'
```
Expected: `Test` added to `[deps]` in `Project.toml` and recorded in `Manifest.toml` (stdlib, no version churn).

- [ ] **Step 2: Write the failing test**

Create `code/test/test_mask.jl`:
```julia
# Tests for the unified logit-bias sampler and its mask/weight wrappers.
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test

@testset "logit_bias_sample: b=0 reduces to unconditional sample" begin
    Random.seed!(11)
    d, K, T = 4, 6, 30
    X = randn(d, K); ξ0 = randn(d)
    Ξb = logit_bias_sample(X, ξ0, T, zeros(K); β=1.5, α=0.1, seed=5).Ξ
    Ξs = sample(X, ξ0, T; β=1.5, α=0.1, seed=5).Ξ
    @test Ξb ≈ Ξs
end
```

- [ ] **Step 3: Run the test to verify it fails**

Run (from `code/`):
```bash
julia --project=. test/test_mask.jl
```
Expected: FAIL — `UndefVarError: logit_bias_sample not defined`.

- [ ] **Step 4: Implement `logit_bias_sample`**

Insert into `code/src/Binding.jl` immediately above the `weighted_sample` docstring (`Binding.jl:520`):
```julia
"""
    logit_bias_sample(X, ξ₀, T, b; β=1.0, α=0.1, seed=nothing)

Unified Langevin sampler with an additive bias `b` on the attention logits:

    a_t = softmax(β Xᵀ ξ_t + b)

`b` is a length-K vector and may contain `-Inf` entries (those memories get
softmax weight exactly 0). This is the one primitive behind all conditioning
regimes: `b = 0` is unconditional SA, `b = log r` is multiplicity conditioning
(`weighted_sample`), and `b = -∞` on a subset is hard masking (`masked_sample`).

Not to be confused with `biased_sample` (Approach 2), which adds an energy
gradient to the update rather than a bias to the softmax logits.
"""
function logit_bias_sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int,
                            b::Vector{Float64};
                            β::Float64=1.0, α::Float64=0.1,
                            seed::Union{Int,Nothing}=nothing)
    d, K = size(X)
    length(ξ₀) == d || throw(DimensionMismatch(
        "Initial state has length $(length(ξ₀)) but X has $d rows"))
    length(b) == K || throw(DimensionMismatch(
        "b has length $(length(b)) but X has $K columns"))
    any(isfinite, b) || throw(ArgumentError("b must have at least one finite entry"))
    T > 0   || throw(ArgumentError("T must be positive"))
    β > 0   || throw(ArgumentError("β must be positive"))
    0 < α < 1 || throw(ArgumentError("α must be in (0,1)"))

    if seed !== nothing
        Random.seed!(seed)
    end

    Ξ = Matrix{Float64}(undef, T + 1, d)
    Ξ[1, :] .= ξ₀
    noise_scale = sqrt(2.0 * α / β)

    ξ = copy(ξ₀)
    for t in 1:T
        logits = β .* (X' * ξ) .+ b
        w = softmax(logits)
        ε = randn(d)
        ξ .= (1.0 - α) .* ξ .+ α .* (X * w) .+ noise_scale .* ε
        Ξ[t + 1, :] .= ξ
    end
    return (t = collect(0:T), Ξ = Ξ)
end
```

- [ ] **Step 5: Run the test to verify it passes**

Run (from `code/`):
```bash
julia --project=. test/test_mask.jl
```
Expected: PASS — `Test Summary: ... | Pass  1`.

- [ ] **Step 6: Commit**

```bash
git add code/Project.toml code/Manifest.toml code/test/test_mask.jl code/src/Binding.jl
git commit -m "feat: add logit_bias_sample unified sampler core" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Refactor `weighted_sample` to delegate and allow zero weights

**Files:**
- Modify: `code/src/Binding.jl` (`weighted_sample`, currently `Binding.jl:546-583`)
- Modify: `code/test/test_mask.jl` (append a testset)

- [ ] **Step 1: Write the failing tests**

Append to `code/test/test_mask.jl`:
```julia
@testset "weighted_sample delegates to logit_bias_sample" begin
    Random.seed!(3)
    d, K, T = 4, 6, 30
    X = randn(d, K); ξ0 = randn(d)
    # uniform positive weights == unconditional
    @test weighted_sample(X, ξ0, T, ones(K); β=1.2, seed=8).Ξ ≈ sample(X, ξ0, T; β=1.2, seed=8).Ξ
    # weighted == logit-bias of log-weights
    w = abs.(randn(K)) .+ 0.1
    @test weighted_sample(X, ξ0, T, w; β=1.2, seed=8).Ξ ≈
          logit_bias_sample(X, ξ0, T, log.(w); β=1.2, seed=8).Ξ
    # a single zero weight is now allowed and produces a finite trajectory
    w0 = copy(w); w0[2] = 0.0
    @test all(isfinite, weighted_sample(X, ξ0, T, w0; β=1.2, seed=8).Ξ)
    # all-zero weights throw (no reachable memory)
    @test_throws ArgumentError weighted_sample(X, ξ0, T, zeros(K); β=1.2, seed=8)
end
```

- [ ] **Step 2: Run to verify the new testset fails**

Run (from `code/`):
```bash
julia --project=. test/test_mask.jl
```
Expected: FAIL — the zero-weight cases error (`weighted_sample` currently throws on any `w <= 0`).

- [ ] **Step 3: Replace the `weighted_sample` body**

Replace the entire `weighted_sample` function (`Binding.jl:546-583`) with:
```julia
function weighted_sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int,
                          pattern_weights::Vector{Float64};
                          β::Float64=1.0, α::Float64=0.1,
                          seed::Union{Int, Nothing}=nothing)
    K = size(X, 2)
    length(pattern_weights) == K || throw(DimensionMismatch(
        "pattern_weights has length $(length(pattern_weights)) but X has $K columns"))
    all(w -> w >= 0, pattern_weights) || throw(ArgumentError("All weights must be nonnegative"))
    any(w -> w > 0, pattern_weights) || throw(ArgumentError("At least one weight must be positive"))

    # b = log(w); log(0) = -Inf is handled by the core (softmax weight 0).
    return logit_bias_sample(X, ξ₀, T, log.(pattern_weights); β=β, α=α, seed=seed)
end
```
(Remaining input validation — `ξ₀` length, `T`, `β`, `α` — now happens inside `logit_bias_sample`, so it is not duplicated here.)

- [ ] **Step 4: Run to verify all tests pass**

Run (from `code/`):
```bash
julia --project=. test/test_mask.jl
```
Expected: PASS — both testsets green.

- [ ] **Step 5: Commit**

```bash
git add code/src/Binding.jl code/test/test_mask.jl
git commit -m "refactor: weighted_sample delegates to core, allows zero weights" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `masked_sample` + `mask_vector` with reduced-matrix equivalence

**Files:**
- Modify: `code/src/Binding.jl` (add after `weighted_sample`)
- Modify: `code/test/test_mask.jl` (append a testset)

- [ ] **Step 1: Write the failing tests**

Append to `code/test/test_mask.jl`:
```julia
@testset "masked_sample equals unmasked SA on the reduced matrix" begin
    Random.seed!(7)
    d, K, T = 5, 8, 40
    X = randn(d, K); ξ0 = randn(d)
    keep = Bool[1,1,1,0,0,1,0,0]
    keepcols = findall(keep)
    # The paper identity: masking full X == running SA on X_S.
    Ξm = masked_sample(X, ξ0, T, keep; β=2.0, α=0.05, seed=99).Ξ
    Ξr = sample(X[:, keepcols], ξ0, T; β=2.0, α=0.05, seed=99).Ξ
    @test Ξm ≈ Ξr
    # -∞ safety: no NaN/Inf in the trajectory
    @test all(isfinite, Ξm)
    # index method matches boolean method
    @test masked_sample(X, ξ0, T, keepcols; β=2.0, seed=2).Ξ ≈
          masked_sample(X, ξ0, T, keep; β=2.0, seed=2).Ξ
    # empty keep-set throws (no finite logit)
    @test_throws ArgumentError masked_sample(X, ξ0, T, falses(K); β=2.0, seed=1)
    # mask_vector builds 0/1 weights
    @test mask_vector(5, [1, 3]) == [1.0, 0.0, 1.0, 0.0, 0.0]
end
```

- [ ] **Step 2: Run to verify it fails**

Run (from `code/`):
```bash
julia --project=. test/test_mask.jl
```
Expected: FAIL — `UndefVarError: masked_sample not defined`.

- [ ] **Step 3: Implement `masked_sample` and `mask_vector`**

Insert into `code/src/Binding.jl` immediately after the `weighted_sample` function:
```julia
"""
    masked_sample(X, ξ₀, T, keep; β=1.0, α=0.1, seed=nothing)

Hard attention masking. `keep` is either a length-K Boolean vector or a vector
of kept column indices. Memories outside the keep-set get logit bias `-∞`
(softmax weight 0), so the chain only attends to the kept subset. Equivalent to
running unmasked SA on the reduced memory matrix `X[:, keep]` while retaining the
full-family coordinate basis. This is the `ρ → ∞` endpoint of multiplicity
conditioning.
"""
function masked_sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int,
                        keep::AbstractVector{Bool};
                        β::Float64=1.0, α::Float64=0.1,
                        seed::Union{Int, Nothing}=nothing)
    K = size(X, 2)
    length(keep) == K || throw(DimensionMismatch(
        "keep has length $(length(keep)) but X has $K columns"))
    b = [keep[k] ? 0.0 : -Inf for k in 1:K]
    return logit_bias_sample(X, ξ₀, T, b; β=β, α=α, seed=seed)
end

function masked_sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int,
                        keep_indices::Vector{Int};
                        β::Float64=1.0, α::Float64=0.1,
                        seed::Union{Int, Nothing}=nothing)
    K = size(X, 2)
    keep = falses(K)
    keep[keep_indices] .= true
    return masked_sample(X, ξ₀, T, keep; β=β, α=α, seed=seed)
end

"""
    mask_vector(K, keep_indices) -> Vector{Float64}

Length-K weight vector with 1.0 on kept indices and 0.0 elsewhere. Symmetric
with `multiplicity_vector`; passing it to `weighted_sample`/`generate_weighted_sequences`
reproduces the hard mask via the `log(0) = -∞` route.
"""
function mask_vector(K::Int, keep_indices::Vector{Int})
    v = zeros(K)
    v[keep_indices] .= 1.0
    return v
end
```

- [ ] **Step 4: Run to verify it passes**

Run (from `code/`):
```bash
julia --project=. test/test_mask.jl
```
Expected: PASS — all three testsets green.

- [ ] **Step 5: Commit**

```bash
git add code/src/Binding.jl code/test/test_mask.jl
git commit -m "feat: add masked_sample (b=-inf) and mask_vector" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add `generate_masked_sequences` generation pipeline

**Files:**
- Modify: `code/src/Binding.jl` (add after `generate_weighted_sequences`, currently ends near `Binding.jl:919`)
- Modify: `code/test/test_mask.jl` (append a smoke testset)

- [ ] **Step 1: Write the failing smoke test**

Append to `code/test/test_mask.jl`:
```julia
@testset "generate_masked_sequences shape + validity (small synthetic family)" begin
    Random.seed!(123)
    aas = collect("ACDEFGHIKLMNPQRSTVWY")
    char_mat = [rand(aas) for _ in 1:12, _ in 1:16]   # 12 sequences × 16 positions
    X̂, pca, Lout, _ = build_memory_matrix(char_mat; pratio=0.95)
    keep_idx = [1, 2, 3, 4, 5, 6]
    seqs, pcas = generate_masked_sequences(X̂, pca, Lout, keep_idx;
        β=5.0, n_chains=2, T=50, burnin=10, thin=10, seed=1)
    @test length(seqs) == 2 * length(10:10:50)   # n_chains × samples-per-chain
    @test all(s -> length(s) == Lout, seqs)
    @test length(pcas) == length(seqs)
end
```

- [ ] **Step 2: Run to verify it fails**

Run (from `code/`):
```bash
julia --project=. test/test_mask.jl
```
Expected: FAIL — `UndefVarError: generate_masked_sequences not defined`.

- [ ] **Step 3: Implement `generate_masked_sequences`**

Insert into `code/src/Binding.jl` immediately after `generate_weighted_sequences`:
```julia
"""
    generate_masked_sequences(X̂, pca_model, L, keep_indices; β, n_chains=30,
                              T=5000, α=0.01, burnin=2000, thin=100, seed=42)

Hard-masked generation pipeline. Same interface as `generate_weighted_sequences`
but restricts attention to `keep_indices` via `masked_sample`. Chains warm-start
from kept (in-set) patterns so they begin inside the designated subset, matching
the masked-conditional protocol. Decoding uses the supplied (full-family) PCA basis.
"""
function generate_masked_sequences(X̂::Matrix{Float64}, pca_model, L::Int,
                                    keep_indices::Vector{Int};
                                    β::Float64, n_chains::Int=30, T::Int=5000,
                                    α::Float64=0.01, burnin::Int=2000, thin::Int=100,
                                    seed::Int=42)
    d, K = size(X̂)
    keep = falses(K)
    keep[keep_indices] .= true
    gen_seqs = String[]
    gen_pca = Vector{Float64}[]

    @info "Generating masked sequences: $n_chains chains × $T steps (β=$β, kept=$(length(keep_indices))/$K)"
    for chain in 1:n_chains
        k = keep_indices[mod1(chain, length(keep_indices))]   # warm-start in-set
        ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)
        result = masked_sample(X̂, ξ₀, T, keep; β=β, α=α, seed=seed + chain)
        for t in burnin:thin:T
            ξ = result.Ξ[t + 1, :]
            push!(gen_seqs, decode_sample(ξ, pca_model, L))
            push!(gen_pca, ξ)
        end
    end

    @info "  Generated $(length(gen_seqs)) masked sequences from $n_chains chains"
    return gen_seqs, gen_pca
end
```

- [ ] **Step 4: Run to verify it passes**

Run (from `code/`):
```bash
julia --project=. test/test_mask.jl
```
Expected: PASS — all four testsets green.

- [ ] **Step 5: Commit**

```bash
git add code/src/Binding.jl code/test/test_mask.jl
git commit -m "feat: add generate_masked_sequences pipeline" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Kunitz mask experiment script

**Files:**
- Create: `code/experiments/run_kunitz_mask_experiment.jl`
- Output (created by the run): `code/data/kunitz/mask_experiment.csv`, `code/figs/kunitz/fig_mask_calibration.{png,pdf}`

- [ ] **Step 1: Write the experiment script**

Create `code/experiments/run_kunitz_mask_experiment.jl`:
```julia
# ──────────────────────────────────────────────────────────────────────────────
# run_kunitz_mask_experiment.jl
#
# Adds the hard attention mask (b = -∞ on background) as the ρ → ∞ endpoint of
# the multiplicity calibration curve on the Kunitz family (P1 K/R designated
# subset), and decomposes the calibration gap into residual-background-mass
# (closed by the mask) vs decode/geometry floor (irreducible).
#
# Conditions (all on the full-family PCA basis except hard curation):
#   1. Unconditional  (b = 0)
#   2. Multiplicity calibration sweep  (finite ρ via build_multiplicity_conditioned_memory)
#   3. Hard mask  (b = -∞ on background, full basis)
#   4. Hard curation  (subset basis)
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const PFAM_ID = "PF00014"
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "kunitz")
mkpath(CACHE_DIR); mkpath(FIG_DIR)

# --- Load + clean alignment ---
@info "Loading Kunitz alignment (PF00014)"
sto_file = download_pfam_seed(PFAM_ID; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)
stored_seqs = [String(char_mat[i, :]) for i in 1:K_total]
@info "  $K_total sequences × $L positions"

# --- P1 split (designated = K/R at P1), identical logic to run_kunitz_binding_experiment.jl ---
lys_fracs = zeros(L)
for j in 1:L
    n_lys = count(i -> char_mat[i, j] == 'K', 1:K_total)
    n_valid = count(i -> char_mat[i, j] != '-' && char_mat[i, j] != '.', 1:K_total)
    lys_fracs[j] = n_valid > 0 ? n_lys / n_valid : 0.0
end
p1_candidates = findall(f -> f > 0.2, lys_fracs)
p1_pos = isempty(p1_candidates) ? argmax(lys_fracs) : p1_candidates[argmax(lys_fracs[p1_candidates])]
designated_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
@info "  P1 column $p1_pos; designated (K/R at P1) = $(length(designated_idx))/$K_total"
length(designated_idx) >= 5 || error("Too few designated sequences ($(length(designated_idx))) for this experiment")

# --- Full-family basis (shared by unconditional, multiplicity, mask) ---
X̂_all, pca_all, L_all, _ = build_memory_matrix(char_mat; pratio=0.95)
β_all = find_entropy_inflection(X̂_all).β_star
@info "  Full-family β* = $(round(β_all, digits=3))"

# --- Metric helper: P1 K/R fraction + sequence-level novelty/diversity ---
function eval_condition(seqs::Vector{String}, label::String)
    n = length(seqs)
    p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n
    np = min(500, n * (n - 1) ÷ 2)
    pair_ids = Float64[]
    for _ in 1:np
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
    end
    diversity = 1.0 - mean(pair_ids)
    novelty = 1.0 - mean(nearest_sequence_identity(s, stored_seqs) for s in seqs)
    kl = aa_composition_kl(seqs, stored_seqs)
    @info "  [$label] P1 K/R = $(round(p1_kr,digits=3)), novelty = $(round(novelty,digits=3)), " *
          "diversity = $(round(diversity,digits=3)), KL = $(round(kl,digits=3))"
    return (label=label, p1_kr=p1_kr, novelty=novelty, diversity=diversity, kl=kl, n=n)
end

results = DataFrame(condition=String[], f_target=Float64[], p1_kr=Float64[],
                    novelty=Float64[], diversity=Float64[], kl=Float64[], n=Int[])

# --- Condition 1: Unconditional (b = 0) ---
@info "Condition 1: Unconditional"
uncond_seqs, _ = generate_sequences(X̂_all, pca_all, L; β=β_all, n_chains=30, T=5000, seed=42)
m = eval_condition(uncond_seqs, "unconditional")
push!(results, ("unconditional", effective_binder_fraction(ones(K_total), designated_idx),
                m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Condition 2: Multiplicity calibration sweep (finite ρ) ---
@info "Condition 2: Multiplicity calibration sweep"
for ft in [0.5, 0.7, 0.9, 0.95, 0.99]
    res = build_multiplicity_conditioned_memory(char_mat, designated_idx; f_target=ft)
    βw = find_weighted_entropy_inflection(res.X̂, res.r; n_betas=50).β_star
    seqs, _ = generate_weighted_sequences(res.X̂, res.pca_model, L, res.r;
        β=βw, n_chains=20, T=5000, seed=42)
    m = eval_condition(seqs, "multiplicity(f=$ft)")
    push!(results, ("multiplicity", res.f_eff, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))
end

# --- Condition 3: Hard mask (b = -∞ on background, full basis) ---
@info "Condition 3: Hard mask"
β_mask = find_entropy_inflection(X̂_all[:, designated_idx]).β_star
mask_seqs, _ = generate_masked_sequences(X̂_all, pca_all, L, designated_idx;
    β=β_mask, n_chains=30, T=5000, seed=42)
m = eval_condition(mask_seqs, "mask")
push!(results, ("mask", 1.0, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Condition 4: Hard curation (subset basis) ---
@info "Condition 4: Hard curation (subset basis)"
X̂_cur, pca_cur, _, _ = build_memory_matrix(char_mat[designated_idx, :]; pratio=0.95)
β_cur = find_entropy_inflection(X̂_cur).β_star
cur_seqs, _ = generate_sequences(X̂_cur, pca_cur, L; β=β_cur, n_chains=30, T=5000, seed=42)
m = eval_condition(cur_seqs, "curation")
push!(results, ("curation", 1.0, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Write CSV ---
csv_path = joinpath(CACHE_DIR, "mask_experiment.csv")
CSV.write(csv_path, results)
@info "Wrote $csv_path"

# --- Calibration figure: multiplicity curve + mask/curation/unconditional overlays ---
mult = results[results.condition .== "multiplicity", :]
uncond_row = results[results.condition .== "unconditional", :]
mask_row = results[results.condition .== "mask", :]
cur_row  = results[results.condition .== "curation", :]

p = plot(size=(640, 520), margin=10Plots.mm, legend=:bottomright,
    xlabel="Target effective designated fraction (f_target)",
    ylabel="Observed P1 K/R fraction",
    title="Kunitz: multiplicity calibration with mask endpoint")
plot!(p, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)")
plot!(p, mult.f_target, mult.p1_kr, marker=:circle, color=:steelblue, lw=2, label="multiplicity (finite ρ)")
scatter!(p, uncond_row.f_target, uncond_row.p1_kr, marker=:diamond, ms=8, color=:gray, label="unconditional")
scatter!(p, mask_row.f_target, mask_row.p1_kr, marker=:star5, ms=12, color=:crimson, label="hard mask (b=-∞)")
scatter!(p, cur_row.f_target, cur_row.p1_kr, marker=:utriangle, ms=9, color=:forestgreen, label="hard curation")
savefig(p, joinpath(FIG_DIR, "fig_mask_calibration.png"))
savefig(p, joinpath(FIG_DIR, "fig_mask_calibration.pdf"))
@info "Wrote fig_mask_calibration"

# --- Calibration-gap decomposition ---
mult_top = mult[argmax(mult.f_target), :]
gap_finite = mult_top.f_target - mult_top.p1_kr      # residual at the strongest finite ρ
gap_mask   = 1.0 - mask_row.p1_kr[1]                 # remaining under the mask = decode/geometry floor
@info "=== Calibration-gap decomposition ==="
@info "  Strongest finite ρ (f_target=$(mult_top.f_target)): observed P1 K/R = $(round(mult_top.p1_kr,digits=3)), gap = $(round(gap_finite,digits=3))"
@info "  Hard mask: observed P1 K/R = $(round(mask_row.p1_kr[1],digits=3)), residual (decode/geometry) = $(round(gap_mask,digits=3))"
@info "  Residual background mass closed by the mask ≈ $(round(gap_finite - gap_mask, digits=3))"
@info "  Scaffold/quality: mask novelty=$(round(mask_row.novelty[1],digits=3)) vs curation novelty=$(round(cur_row.novelty[1],digits=3)); " *
      "mask KL=$(round(mask_row.kl[1],digits=3)) vs curation KL=$(round(cur_row.kl[1],digits=3))"
```

- [ ] **Step 2: Run the experiment**

Run (from `code/`):
```bash
julia experiments/run_kunitz_mask_experiment.jl
```
Expected: completes without error (first run precompiles deps and downloads the PF00014 seed alignment; allow several minutes). Logs the four conditions, the gap decomposition, and writes the CSV + figure.

- [ ] **Step 3: Verify the outputs exist and are sane**

Run (from `code/`):
```bash
julia --project=. -e 'using CSV, DataFrames; df = CSV.read("data/kunitz/mask_experiment.csv", DataFrame); show(df, allrows=true); println(); @assert "mask" in df.condition; @assert isfile("figs/kunitz/fig_mask_calibration.png")'
```
Expected: prints the results table including a `mask` row; no assertion error.

- [ ] **Step 4: Commit**

```bash
git add code/experiments/run_kunitz_mask_experiment.jl code/data/kunitz/mask_experiment.csv code/figs/kunitz/fig_mask_calibration.png code/figs/kunitz/fig_mask_calibration.pdf
git commit -m "feat: Kunitz mask experiment with calibration-gap decomposition" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6 (optional): Fixed-mask β-sweep (Olivetti analog)

Mirrors the companion paper's `tab:olivetti-betasweep`: hold the mask fixed and sweep β to show the within-designated Hopfield → Hebbian transition holds inside the masked regime. Skip if the calibration result in Task 5 already tells the story you want.

**Files:**
- Modify: `code/experiments/run_kunitz_mask_experiment.jl` (append)
- Output: `code/data/kunitz/mask_betasweep.csv`

- [ ] **Step 1: Append the β-sweep block**

Append to `code/experiments/run_kunitz_mask_experiment.jl`:
```julia
# --- Fixed-mask β-sweep: novelty vs recall inside the masked regime ---
@info "Optional: fixed-mask β-sweep"
betasweep = DataFrame(β=Float64[], p1_kr=Float64[], novelty=Float64[])
for βb in [2.0, 8.0, 32.0, 128.0, 512.0]
    seqs, _ = generate_masked_sequences(X̂_all, pca_all, L, designated_idx;
        β=βb, n_chains=20, T=5000, seed=42)
    p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / length(seqs)
    nov = 1.0 - mean(nearest_sequence_identity(s, stored_seqs) for s in seqs)
    push!(betasweep, (βb, p1_kr, nov))
    @info "  β=$βb: P1 K/R=$(round(p1_kr,digits=3)), novelty=$(round(nov,digits=3))"
end
CSV.write(joinpath(CACHE_DIR, "mask_betasweep.csv"), betasweep)
@info "Wrote mask_betasweep.csv"
```

- [ ] **Step 2: Run and verify**

Run (from `code/`):
```bash
julia experiments/run_kunitz_mask_experiment.jl
```
Expected: additionally writes `data/kunitz/mask_betasweep.csv`; β-sweep rows logged (novelty decreasing as β grows, P1 K/R staying high because the off-subset is unreachable).

- [ ] **Step 3: Commit**

```bash
git add code/experiments/run_kunitz_mask_experiment.jl code/data/kunitz/mask_betasweep.csv
git commit -m "feat: optional fixed-mask beta-sweep for Kunitz" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Paper text (recovery table + prose)

Do this after Task 5 produces numbers. Read the actual values from `code/data/kunitz/mask_experiment.csv` and the gap-decomposition log lines; populate the table from the named CSV columns.

**Files:**
- Modify: `paper/sections/results.tex` (add table + paragraph)
- Modify: `paper/sections/methods.tex` (one paragraph: the bias axis and the mask as `b=-∞`)

- [ ] **Step 1: Add the recovery table to `results.tex`**

Insert (fill the four numeric cells from the CSV `p1_kr` column for the matching `condition` rows; keep paper style — flowing prose, no subsection headings, no em/en dashes):
```latex
\begin{table}[!t]
\centering
\caption{Kunitz P1 K/R recovery across the conditioning axis (PF00014, designated subset = K/R at P1). The hard mask is the $b=-\infty$ endpoint of the multiplicity axis on the full-family basis; hard curation rebuilds the basis from the designated subset.}
\label{tab:kunitz-mask-recovery}
\begin{tabular}{@{}lc@{}}
\toprule
Condition & Observed P1 K/R fraction $\uparrow$ \\
\midrule
Unconditional ($b=0$)            & $<<unconditional.p1_kr>>$ \\
Multiplicity (strongest finite $\rho$) & $<<multiplicity max f_target row p1_kr>>$ \\
\textbf{Hard mask ($b=-\infty$)} & $\mathbf{<<mask.p1_kr>>}$ \\
Hard curation (subset basis)     & $<<curation.p1_kr>>$ \\
\bottomrule
\end{tabular}
\end{table}
```
Replace each `<<...>>` with the rounded value from the CSV before saving.

- [ ] **Step 2: Add a results paragraph**

Add a flowing-prose paragraph after the table stating: the unconditional baseline P1 K/R fraction, that finite multiplicity saturates below 1.0 (the calibration gap), that the mask drives recovery to its endpoint value, the gap decomposition (residual background mass closed by the mask vs the decode/geometry floor that remains), and that the mask preserves novelty/scaffold relative to curation (cite `tab:kunitz-mask-recovery`). Use the logged decomposition numbers. No section self-references.

- [ ] **Step 3: Add the methods paragraph**

In `methods.tex`, add one paragraph: the conditioning bias `b` on the attention logits, the three regimes (`b=0`, `b=\log r`, `b=-\infty`), the softmax shift-invariance identity that makes multiplicity a soft background mask and the hard mask its `\log\rho\to\infty` limit, and a citation to the companion paper's masking section as the `b=-\infty` case. Use "designated/background" terminology.

- [ ] **Step 4: Style check**

Run the `style-check` skill on the edited sections (em/en dashes, subsection headings, line numbers). Fix any violations.

- [ ] **Step 5: Commit**

```bash
git add paper/sections/results.tex paper/sections/methods.tex
git commit -m "docs: add Kunitz mask recovery table and prose" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** unified core (Task 1), `b=log r` multiplicity via wrapper (Task 2), mask + reduced-matrix equivalence (Task 3), generation (Task 4), experiment with all conditions + calibration figure + gap decomposition + quality guards (Task 5), optional β-sweep (Task 6), deliverables table/prose (Task 7). DARPins and all-family rollout remain out of scope per spec.
- **Naming:** `logit_bias_sample` (core, distinct from existing energy `biased_sample`), `masked_sample`, `mask_vector`, `generate_masked_sequences` are used consistently across tasks. `designated_idx` is the experiment's name for the K/R-at-P1 subset.
- **Quality guards** use the existing `sequence_identity` / `nearest_sequence_identity` / `aa_composition_kl` helpers, matching `run_multiplicity_conditioning.jl`.
