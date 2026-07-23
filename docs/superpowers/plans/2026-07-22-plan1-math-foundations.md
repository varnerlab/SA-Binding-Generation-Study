# Plan 1: Mathematical Foundations & Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and test the exact Gaussian-mixture sampler, the corrected score/gradient, the corrected entropy identities, and an honestly-named transition statistic, then ship a real test suite, one exact-vs-ULA baseline table, and the SI derivation, all with zero re-running of family experiments.

**Architecture:** New pure functions land in the existing source modules (`Binding.jl` for multiplicity/GMM, `Protein.jl` for the transition statistic). Each is locked by a focused TDD test under `code/test/`, aggregated by a new `runtests.jl`. One experiment script productionizes the exact-vs-ULA comparison into a CSV. The SI derivation is a new `.tex` fragment `\input` into the appendix.

**Tech Stack:** Julia 1.12, `Test` stdlib, `NNlib` (softmax), `MultivariateStats` (PCA), `Random`, `LinearAlgebra`, `Statistics`, `DataFrames`, `CSV`. LaTeX (amsmath/amssymb) for the SI.

## Global Constraints

- Julia 1.12. Package versions pinned in `code/Manifest.toml`; do not change deps.
- All Julia scripts run from `code/`; every script begins with `include("Include.jl")`.
- Terminology: use "designated" and "background", not "binder"/"non-binder", in new prose and docstrings. Legacy function names keep "binder".
- No em dashes or en dashes in any paper prose (SI included); use periods or commas.
- Memory columns are unit-norm (`build_memory_matrix`); the GMM identity depends on this.
- The SA sampler is UNCONSTRAINED in R^d (no sphere projection in `sample`); the SI must say so.
- Commits stage ONLY the files named in each task via explicit `git add <paths>`. Never `git add -A` or `git add .`: the working tree has an uncommitted `paper/` -> `paper-arxiv/` rename that must not be swept into these commits.
- New randomized code takes an explicit `rng`/`seed`; do not rely on global RNG state.

---

### Task 0: Branch and confirm the existing suite runs

**Files:**
- None created; git branch only.

**Interfaces:**
- Produces: a working branch `arxiv-rev-plan1-math` off `main`.

- [ ] **Step 1: Create the working branch**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git checkout -b arxiv-rev-plan1-math
```

- [ ] **Step 2: Confirm the current mask tests pass (baseline green)**

Run:
```bash
cd code && julia test/test_mask.jl
```
Expected: test summary with all tests passed, no failures/errors (14 assertions across 4 testsets).

---

### Task 1: Exact Gaussian-mixture sampler

**Files:**
- Modify: `code/src/Binding.jl` (add after `weighted_attention_entropy`, around line 777)
- Test: `code/test/test_gmm_identity.jl`

**Interfaces:**
- Consumes: `Random`, `LinearAlgebra` (loaded by `Include.jl`).
- Produces: `exact_gmm_sample(X::Matrix{Float64}, r::Vector{Float64}, β::Float64, n::Int; rng::AbstractRNG=Random.default_rng()) -> (Ξ::Matrix{Float64}, components::Vector{Int})` where `Ξ` is `d × n`.

- [ ] **Step 1: Write the failing test**

Create `code/test/test_gmm_identity.jl`:
```julia
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random, LinearAlgebra, Statistics

unitcols(d, K, seed) = (X = randn(MersenneTwister(seed), d, K);
                        for k in 1:K; X[:, k] ./= norm(X[:, k]); end; X)

@testset "exact_gmm_sample: shape, components, determinism" begin
    X = unitcols(8, 5, 10)
    r = [1.0, 2.0, 3.0, 4.0, 5.0]
    out  = exact_gmm_sample(X, r, 4.0, 100; rng=MersenneTwister(1))
    out2 = exact_gmm_sample(X, r, 4.0, 100; rng=MersenneTwister(1))
    @test size(out.Ξ) == (8, 100)
    @test all(c -> 1 <= c <= 5, out.components)
    @test out.Ξ == out2.Ξ                      # determinism under same seed
    @test out.components == out2.components
end

@testset "exact_gmm_sample: empirical mean matches Σ w_k m_k" begin
    X = unitcols(10, 6, 20)
    r = abs.(randn(MersenneTwister(21), 6)) .+ 0.5
    w = r ./ sum(r)
    out = exact_gmm_sample(X, r, 6.0, 200_000; rng=MersenneTwister(7))
    @test norm(vec(mean(out.Ξ; dims=2)) - X * w) < 0.05
end

@testset "exact_gmm_sample: designated mass matches f_eff" begin
    X = unitcols(10, 8, 30)
    designated = [2, 4, 6]
    r = ones(8); r[designated] .= 20.0
    f_eff = sum(r[designated]) / sum(r)
    out = exact_gmm_sample(X, r, 5.0, 200_000; rng=MersenneTwister(3))
    @test abs(mean(c -> c in designated, out.components) - f_eff) < 0.01
end

@testset "exact_gmm_sample: input validation" begin
    X = unitcols(4, 3, 1)
    @test_throws DimensionMismatch exact_gmm_sample(X, ones(2), 1.0, 5)
    @test_throws ArgumentError exact_gmm_sample(X, zeros(3), 1.0, 5)
    @test_throws ArgumentError exact_gmm_sample(X, ones(3), -1.0, 5)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd code && julia test/test_gmm_identity.jl
```
Expected: FAIL with `UndefVarError: exact_gmm_sample not defined`.

- [ ] **Step 3: Write minimal implementation**

In `code/src/Binding.jl`, after the `weighted_attention_entropy` function (around line 777), add:
```julia
"""
    exact_gmm_sample(X, r, β, n; rng=Random.default_rng()) -> (Ξ, components)

Draw `n` INDEPENDENT samples from the exact stationary distribution of the
multiplicity-weighted Hopfield energy. For unit-norm memories the target is the
Gaussian mixture (see SI):

    p_β(ξ) = Σ_k w_k N(ξ; m_k, β⁻¹ I),   w_k = r_k / Σ_j r_j.

Procedure: draw component k ~ Categorical(w), then ξ = m_k + β^{-1/2} z with
z ~ N(0, I). No MCMC. Returns latents `Ξ` (d × n) and each draw's component index.
This is the exact-sampler baseline for the ULA generator `weighted_sample`.
"""
function exact_gmm_sample(X::Matrix{Float64}, r::Vector{Float64}, β::Float64, n::Int;
                          rng::AbstractRNG=Random.default_rng())
    d, K = size(X)
    length(r) == K || throw(DimensionMismatch(
        "r has length $(length(r)) but X has $K columns"))
    all(≥(0.0), r) || throw(ArgumentError("All multiplicities must be nonnegative"))
    any(>(0.0), r) || throw(ArgumentError("At least one multiplicity must be positive"))
    β > 0 || throw(ArgumentError("β must be positive, got β = $β"))
    n > 0 || throw(ArgumentError("n must be positive, got n = $n"))

    w  = r ./ sum(r)
    cw = cumsum(w); cw[end] = 1.0            # guard roundoff so u∈[0,1) always lands
    s  = 1.0 / sqrt(β)                       # per-coordinate std dev of each component
    Ξ  = Matrix{Float64}(undef, d, n)
    components = Vector{Int}(undef, n)
    for i in 1:n
        k = min(searchsortedfirst(cw, rand(rng)), K)
        components[i] = k
        @views Ξ[:, i] .= X[:, k] .+ s .* randn(rng, d)
    end
    return (Ξ = Ξ, components = components)
end
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd code && julia test/test_gmm_identity.jl
```
Expected: PASS, all 4 testsets, no failures.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/src/Binding.jl code/test/test_gmm_identity.jl
git commit -m "feat: exact Gaussian-mixture sampler for the SA equilibrium"
```

---

### Task 2: Corrected score and gradient (the missing beta)

**Files:**
- Modify: `code/src/Binding.jl` (add after `exact_gmm_sample`)
- Test: `code/test/test_score_gradient.jl`

**Interfaces:**
- Consumes: `weighted_hopfield_energy(ξ, X, β, r)` (existing, `Binding.jl:751`), `NNlib.softmax`.
- Produces:
  - `weighted_hopfield_gradient(ξ::Vector{Float64}, X::Matrix{Float64}, β::Float64, r::Vector{Float64}) -> Vector{Float64}` computing `ξ - X softmax(β Xᵀξ + log r)`.
  - `weighted_score(ξ, X, β, r) -> Vector{Float64}` computing `-β ∇E_r = β[X softmax(...) - ξ]`.

- [ ] **Step 1: Write the failing test**

Create `code/test/test_score_gradient.jl`:
```julia
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random, LinearAlgebra

unitcols(d, K, seed) = (X = randn(MersenneTwister(seed), d, K);
                        for k in 1:K; X[:, k] ./= norm(X[:, k]); end; X)

@testset "weighted_hopfield_gradient matches central finite differences" begin
    X = unitcols(6, 7, 42)
    r = abs.(randn(MersenneTwister(43), 7)) .+ 0.3
    β = 3.0
    ξ = 0.5 .* randn(MersenneTwister(44), 6)
    g = weighted_hopfield_gradient(ξ, X, β, r)
    h = 1e-6
    gfd = similar(g)
    for i in eachindex(ξ)
        ξp = copy(ξ); ξp[i] += h
        ξm = copy(ξ); ξm[i] -= h
        gfd[i] = (weighted_hopfield_energy(ξp, X, β, r) -
                  weighted_hopfield_energy(ξm, X, β, r)) / (2h)
    end
    @test norm(g - gfd) < 1e-6
end

@testset "weighted_score = -β ∇E_r, and r≡1 reduces to unweighted gradient" begin
    X = unitcols(5, 4, 1)
    β = 2.0
    ξ = randn(MersenneTwister(2), 5)
    r = abs.(randn(MersenneTwister(3), 4)) .+ 0.5
    @test weighted_score(ξ, X, β, r) ≈ -β .* weighted_hopfield_gradient(ξ, X, β, r)
    a = NNlib.softmax(β .* (X' * ξ))
    @test weighted_hopfield_gradient(ξ, X, β, ones(4)) ≈ ξ .- X * a
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd code && julia test/test_score_gradient.jl
```
Expected: FAIL with `UndefVarError: weighted_hopfield_gradient not defined`.

- [ ] **Step 3: Write minimal implementation**

In `code/src/Binding.jl`, after `exact_gmm_sample`, add:
```julia
"""
    weighted_hopfield_gradient(ξ, X, β, r) -> Vector{Float64}

Gradient of the multiplicity-weighted Hopfield energy `E_r`
(see `weighted_hopfield_energy`):

    ∇E_r(ξ) = ξ - X softmax(β Xᵀξ + log r).
"""
function weighted_hopfield_gradient(ξ::Vector{Float64}, X::Matrix{Float64},
                                     β::Float64, r::Vector{Float64})::Vector{Float64}
    a = NNlib.softmax(β .* (X' * ξ) .+ log.(r))
    return ξ .- X * a
end

"""
    weighted_score(ξ, X, β, r) -> Vector{Float64}

Score of the multiplicity-weighted stationary target. Carries the factor β that
the v1 proposition dropped:

    ∇_ξ log p_β(ξ) = -β ∇E_r(ξ) = β[X softmax(β Xᵀξ + log r) - ξ].
"""
weighted_score(ξ::Vector{Float64}, X::Matrix{Float64}, β::Float64, r::Vector{Float64}) =
    -β .* weighted_hopfield_gradient(ξ, X, β, r)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd code && julia test/test_score_gradient.jl
```
Expected: PASS, both testsets.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/src/Binding.jl code/test/test_score_gradient.jl
git commit -m "feat: multiplicity-weighted Hopfield gradient and score (restores beta)"
```

---

### Task 3: Entropy identities (Shannon vs Renyi-2)

**Files:**
- Modify: `code/src/Binding.jl` (add after `weighted_score`)
- Test: `code/test/test_entropy_identities.jl`

**Interfaces:**
- Consumes: `effective_num_patterns(r)` (existing, `Binding.jl:739`), `weighted_attention_entropy(ξ, X, β, r)` (existing, `Binding.jl:768`).
- Produces:
  - `shannon_entropy(p::Vector{Float64}) -> Float64` (Shannon entropy of `p` normalized).
  - `log_effective_num_patterns(r::Vector{Float64}) -> Float64` = `log(effective_num_patterns(r))` (Renyi-2 entropy).

- [ ] **Step 1: Write the failing test**

Create `code/test/test_entropy_identities.jl`:
```julia
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random, LinearAlgebra

unitcols(d, K, seed) = (X = randn(MersenneTwister(seed), d, K);
                        for k in 1:K; X[:, k] ./= norm(X[:, k]); end; X)

@testset "β→0 attention entropy equals Shannon entropy of normalized weights" begin
    X = unitcols(6, 10, 5)
    r = abs.(randn(MersenneTwister(6), 10)) .+ 0.2
    # at β≈0 the weighted attention weights are w = r/Σr, independent of ξ
    Hβ0 = weighted_attention_entropy(zeros(6), X, 1e-9, r)
    @test isapprox(Hβ0, shannon_entropy(r); atol=1e-6)
end

@testset "Shannon (Renyi-1) and log K_eff (Renyi-2) coincide only for equal weights" begin
    r_eq = ones(8)
    @test isapprox(shannon_entropy(r_eq), log_effective_num_patterns(r_eq); atol=1e-10)
    @test isapprox(shannon_entropy(r_eq), log(8); atol=1e-10)
    # non-uniform: Renyi-1 > Renyi-2 strictly, by a material margin
    r = [500.0; ones(31)]
    Hs = shannon_entropy(r)
    Hr = log_effective_num_patterns(r)
    @test Hs > Hr
    @test Hs - Hr > 0.1
end

@testset "shannon_entropy validation" begin
    @test_throws ArgumentError shannon_entropy(zeros(4))
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd code && julia test/test_entropy_identities.jl
```
Expected: FAIL with `UndefVarError: shannon_entropy not defined`.

- [ ] **Step 3: Write minimal implementation**

In `code/src/Binding.jl`, after `weighted_score`, add:
```julia
"""
    shannon_entropy(p) -> Float64

Shannon entropy (nats) of a nonnegative weight vector normalized to sum 1:
`H = -Σ_k w_k log w_k`, `w = p / Σ p`. This is the β→0 limit of the weighted
attention entropy and is NOT equal to `log K_eff` (the Renyi-2 entropy) except
when all weights are equal. See `log_effective_num_patterns`.
"""
function shannon_entropy(p::Vector{Float64})::Float64
    s = sum(p)
    s > 0 || throw(ArgumentError("weights must sum to a positive value"))
    H = 0.0
    for pk in p
        w = pk / s
        w > 0 && (H -= w * log(w))
    end
    return H
end

"""
    log_effective_num_patterns(r) -> Float64

`log K_eff(r) = log((Σ r_k)² / Σ r_k²) = -log Σ w_k²`, the Renyi-2 entropy of the
normalized weights. Equals `shannon_entropy(r)` only when all weights are equal.
"""
log_effective_num_patterns(r::Vector{Float64}) = log(effective_num_patterns(r))
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd code && julia test/test_entropy_identities.jl
```
Expected: PASS, all 3 testsets.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/src/Binding.jl code/test/test_entropy_identities.jl
git commit -m "feat: Shannon vs Renyi-2 entropy helpers (separates H from log K_eff)"
```

---

### Task 4: Honestly-named transition statistic

**Files:**
- Modify: `code/src/Protein.jl` (add after `find_entropy_inflection`, around line 411)
- Test: `code/test/test_transition_statistic.jl`

**Interfaces:**
- Consumes: `weighted_attention_entropy(ξ, X, β, r)`, `effective_num_patterns(r)`, `Random.randperm`.
- Produces:
  - `transition_statistics(log_βs::AbstractVector, Hs::AbstractVector) -> (i_steepest, i_onset, dH, d2H)`; `i_steepest`/`i_onset` index into `log_βs`.
  - `find_entropy_transition(X̂::Matrix{Float64}, r::Vector{Float64}=ones(size(X̂,2)); α=0.01, n_betas=60, β_range=(0.1,500.0), n_probes=20, seed=0) -> (β_steepest, β_onset, K_eff, βs, Hs)`.

- [ ] **Step 1: Write the failing test**

Create `code/test/test_transition_statistic.jl`:
```julia
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random, LinearAlgebra

unitcols(d, K, seed) = (X = randn(MersenneTwister(seed), d, K);
                        for k in 1:K; X[:, k] ./= norm(X[:, k]); end; X)

@testset "transition_statistics locates the steepest drop of an analytic sigmoid" begin
    logβ = collect(range(-3.0, 5.0, length=400))
    c, k, A = 1.7, 6.0, 3.0
    Hs = A ./ (1 .+ exp.(k .* (logβ .- c)))          # high at low β, drops through c
    st = transition_statistics(logβ, Hs)
    @test isapprox(logβ[st.i_steepest], c; atol=0.05) # steepest slope at the midpoint
    @test logβ[st.i_onset] < logβ[st.i_steepest]      # onset precedes the midpoint
end

@testset "transition_statistics validation" begin
    @test_throws DimensionMismatch transition_statistics([1.0, 2.0], [1.0])
    @test_throws ArgumentError transition_statistics([1.0, 2.0], [1.0, 2.0])
end

@testset "find_entropy_transition with all probes is permutation invariant" begin
    X = unitcols(10, 16, 77)
    a = find_entropy_transition(X; n_betas=40, n_probes=16, seed=1)
    b = find_entropy_transition(X[:, 16:-1:1]; n_betas=40, n_probes=16, seed=1)
    @test isapprox(a.β_steepest, b.β_steepest; rtol=1e-8)
    @test isapprox(a.Hs, b.Hs; rtol=1e-8)             # curve independent of column order
    @test a.β_steepest > 0
    @test length(a.Hs) == 40
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd code && julia test/test_transition_statistic.jl
```
Expected: FAIL with `UndefVarError: transition_statistics not defined`.

- [ ] **Step 3: Write minimal implementation**

In `code/src/Protein.jl`, after `find_entropy_inflection` (around line 411), add:
```julia
"""
    transition_statistics(log_βs, Hs) -> NamedTuple

Descriptive transition statistics for an entropy curve `Hs` sampled at increasing
log-inverse-temperatures `log_βs`:

- `i_steepest`: index of the steepest entropy DROP (most negative dH/d(logβ)). For a
  sigmoid-like decay this is the transition midpoint.
- `i_onset`: index of maximum downward curvature (most negative d²H/d(logβ)²). This is
  the transition ONSET. It is NOT an inflection point: a true inflection is a zero
  crossing of the second derivative. The v1 `find_entropy_inflection` reported this
  onset while calling it an inflection.

Indices refer to positions in `log_βs`. Also returns the first/second derivative arrays.
"""
function transition_statistics(log_βs::AbstractVector, Hs::AbstractVector)
    length(log_βs) == length(Hs) || throw(DimensionMismatch(
        "log_βs and Hs must have equal length"))
    length(Hs) >= 3 || throw(ArgumentError("need at least 3 points"))
    dH  = diff(Hs) ./ diff(log_βs)
    d2H = diff(dH) ./ diff(log_βs[1:end-1])
    return (i_steepest = argmin(dH) + 1,
            i_onset    = argmin(d2H) + 1,
            dH = dH, d2H = d2H)
end

"""
    find_entropy_transition(X̂, r=ones(size(X̂,2)); α=0.01, n_betas=60,
                            β_range=(0.1,500.0), n_probes=20, seed=0)

Descriptive transition analysis for the (optionally multiplicity-weighted) attention
entropy. Probes `n_probes` RANDOMLY chosen memory columns (seeded), so unlike the v1
`find_entropy_inflection` (which used the first columns) the result is order
independent. Returns `β_steepest` (steepest entropy drop, the transition midpoint),
`β_onset` (maximum downward curvature), `K_eff`, and the β/H curves.

Entropy is evaluated at stored memories, so this probes the retrieval basins and is a
descriptive crossover statistic, not a finite-size phase-transition estimate.
"""
function find_entropy_transition(X̂::Matrix{Float64},
                                 r::Vector{Float64}=ones(size(X̂, 2));
                                 α::Float64=0.01, n_betas::Int=60,
                                 β_range::Tuple{Float64,Float64}=(0.1, 500.0),
                                 n_probes::Int=20, seed::Int=0)
    d, K = size(X̂)
    βs = 10 .^ range(log10(β_range[1]), log10(β_range[2]), length=n_betas)
    probes = randperm(MersenneTwister(seed), K)[1:min(n_probes, K)]
    Hs = [mean(weighted_attention_entropy(X̂[:, k], X̂, β, r) for k in probes) for β in βs]
    st = transition_statistics(log.(βs), Hs)
    return (β_steepest = βs[st.i_steepest], β_onset = βs[st.i_onset],
            K_eff = effective_num_patterns(r), βs = βs, Hs = Hs)
end
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd code && julia test/test_transition_statistic.jl
```
Expected: PASS, all 3 testsets.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/src/Protein.jl code/test/test_transition_statistic.jl
git commit -m "feat: order-independent transition statistic (steepest drop and onset)"
```

---

### Task 5: Aggregate test suite

**Files:**
- Create: `code/test/runtests.jl`

**Interfaces:**
- Consumes: the four test files from Tasks 1-4 plus the existing `test_mask.jl`.
- Produces: a single entry point runnable via `julia test/runtests.jl`.

- [ ] **Step 1: Create the aggregator**

Create `code/test/runtests.jl`:
```julia
# Full unit suite for the SA Binding Generation Study.
# Each included file self-includes Include.jl (idempotent) and defines its own testsets.
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test

@testset "SA Binding Generation Study" begin
    include("test_score_gradient.jl")
    include("test_gmm_identity.jl")
    include("test_entropy_identities.jl")
    include("test_transition_statistic.jl")
    include("test_mask.jl")
end
```

- [ ] **Step 2: Run the full suite**

Run:
```bash
cd code && julia test/runtests.jl
```
Expected: PASS. Final line reports the aggregate `SA Binding Generation Study` testset with zero failures and zero errors across all included files.

- [ ] **Step 3: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/test/runtests.jl
git commit -m "test: add runtests.jl aggregating the unit suite"
```

---

### Task 6: Exact-vs-ULA baseline table

**Files:**
- Create: `code/experiments/run_gmm_baseline.jl`
- Produces data: `code/data/kunitz/gmm_baseline_comparison.csv`

**Interfaces:**
- Consumes: `parse_stockholm`, `clean_alignment`, `build_memory_matrix`, `multiplicity_vector`, `effective_binder_fraction`, `find_weighted_entropy_inflection`, `generate_weighted_sequences`, `decode_sample`, `aa_composition_kl` (all existing), and `exact_gmm_sample` (Task 1).
- Produces: a CSV with columns `rho, beta_star, f_eff, mass_exact, mass_ula, p1_exact, p1_ula, aa_kl`. The script self-checks that ULA tracks the exact equilibrium.

- [ ] **Step 1: Write the baseline experiment**

Create `code/experiments/run_gmm_baseline.jl`:
```julia
# run_gmm_baseline.jl
# Baseline: the exact Gaussian-mixture sampler vs the ULA generator, on Kunitz.
# Confirms ULA reproduces the exact equilibrium (designated mass, decoded phenotype,
# amino-acid composition) at the same β. Reviewer-insurance for the GMM identity.
_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))
using Random, LinearAlgebra, Statistics

const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
raw = parse_stockholm(joinpath(CACHE_DIR, "PF00014_seed.sto"))
char_mat, names = clean_alignment(raw; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)
lys = [count(i -> char_mat[i, j] == 'K', 1:K_total) /
       max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K_total)) for j in 1:L]
p1 = argmax(lys)
strong = findall(i -> char_mat[i, p1] in ('K', 'R'), 1:K_total)
X̂, pca, _, _ = build_memory_matrix(char_mat; pratio=0.95)

assign(ξ, r, β) = argmax(β .* (X̂' * ξ) .+ log.(r))
p1kr(seqs) = count(s -> length(s) >= p1 && s[p1] in ('K', 'R'), seqs) / length(seqs)

rows = DataFrame(rho=Float64[], beta_star=Float64[], f_eff=Float64[],
                 mass_exact=Float64[], mass_ula=Float64[],
                 p1_exact=Float64[], p1_ula=Float64[], aa_kl=Float64[])

for ρ in [1.0, 10.0, 500.0]
    r = multiplicity_vector(K_total, strong; ρ=ρ)
    f_eff = effective_binder_fraction(r, strong)
    β = find_weighted_entropy_inflection(X̂, r; n_betas=60).β_star   # paper's operating β
    ula_seqs, ula_pca = generate_weighted_sequences(X̂, pca, L, r;
        β=β, n_chains=40, T=6000, α=0.01, burnin=2000, thin=100, seed=42)
    N = length(ula_seqs)
    ex = exact_gmm_sample(X̂, r, β, N; rng=MersenneTwister(7))
    exact_seqs = [decode_sample(ex.Ξ[:, i], pca, L) for i in 1:N]
    push!(rows, (ρ, β, f_eff,
        mean(assign(ex.Ξ[:, i], r, β) in strong for i in 1:N),
        mean(assign(ula_pca[i],  r, β) in strong for i in 1:N),
        p1kr(exact_seqs), p1kr(ula_seqs),
        aa_composition_kl(ula_seqs, exact_seqs)))
end

show(stdout, rows); println()
CSV.write(joinpath(CACHE_DIR, "gmm_baseline_comparison.csv"), rows)

# self-check: ULA tracks the exact equilibrium
@assert all(abs.(rows.mass_exact .- rows.mass_ula) .< 0.05) "designated mass mismatch"
@assert all(rows.aa_kl .< 0.01) "AA composition mismatch"
@info "GMM baseline written to gmm_baseline_comparison.csv; ULA matches exact GMM."
```

- [ ] **Step 2: Run the baseline and verify the self-check passes**

Run:
```bash
cd code && julia experiments/run_gmm_baseline.jl
```
Expected: a printed 3-row table, the CSV written, and the final info line "ULA matches exact GMM." No `AssertionError`. (`mass_exact` and `mass_ula` agree within 0.05; `aa_kl` < 0.01 on every row.)

- [ ] **Step 3: Confirm the CSV exists**

Run:
```bash
cd code && head -1 data/kunitz/gmm_baseline_comparison.csv && wc -l data/kunitz/gmm_baseline_comparison.csv
```
Expected: header `rho,beta_star,f_eff,mass_exact,mass_ula,p1_exact,p1_ula,aa_kl` and `4` lines total (header + 3 rows).

- [ ] **Step 4: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/experiments/run_gmm_baseline.jl code/data/kunitz/gmm_baseline_comparison.csv
git commit -m "feat: exact-GMM vs ULA baseline table on Kunitz"
```

---

### Task 7: SI derivation of the Gaussian-mixture identity

**Files:**
- Create: `paper-arxiv/sections/si_gmm_derivation.tex`
- Modify: `paper-arxiv/sections/appendix.tex` (append one `\input` line)

**Interfaces:**
- Consumes: amsmath/amssymb (already loaded by `Paper_v1.tex`).
- Produces: a labeled SI subsection `\label{app:gmm}` with the full derivation and corollaries.

- [ ] **Step 1: Write the SI fragment**

Create `paper-arxiv/sections/si_gmm_derivation.tex`:
```latex
\subsection{The stationary distribution is an exact Gaussian mixture}\label{app:gmm}

The stochastic attention sampler evolves an unconstrained state
$\boldsymbol{\xi}\in\mathbb{R}^{d}$ (the update in Algorithm~1 applies no
projection back to the sphere; only the stored memories are unit norm). Its
stationary law is the Gibbs distribution
$p_\beta(\boldsymbol{\xi})\propto e^{-\beta E_r(\boldsymbol{\xi})}$ of the
multiplicity-weighted Hopfield energy
\begin{equation}
  E_r(\boldsymbol{\xi})
  = \tfrac12\lVert\boldsymbol{\xi}\rVert^2
    - \tfrac1\beta\log\sum_{k=1}^{K} r_k\, e^{\beta\, \mathbf{m}_k^\top\boldsymbol{\xi}},
  \qquad \lVert\mathbf{m}_k\rVert = 1 .
\end{equation}
We show this law is exactly a Gaussian mixture.

\paragraph{Exponentiation.}
Multiplying by $-\beta$, the $-\tfrac1\beta$ inside the energy cancels the
$-\beta$ outside, and $e^{\log S}=S$ removes the logarithm:
\begin{equation}
  e^{-\beta E_r(\boldsymbol{\xi})}
  = e^{-\frac\beta2\lVert\boldsymbol{\xi}\rVert^2}
    \sum_{k=1}^{K} r_k\, e^{\beta\, \mathbf{m}_k^\top\boldsymbol{\xi}} .
\end{equation}

\paragraph{Completing the square.}
For each term,
$-\tfrac\beta2\lVert\boldsymbol{\xi}\rVert^2 + \beta\,\mathbf{m}_k^\top\boldsymbol{\xi}
 = -\tfrac\beta2\lVert\boldsymbol{\xi}-\mathbf{m}_k\rVert^2
   + \tfrac\beta2\lVert\mathbf{m}_k\rVert^2$, hence
\begin{equation}
  e^{-\beta E_r(\boldsymbol{\xi})}
  = \sum_{k=1}^{K} r_k\, e^{\frac\beta2\lVert\mathbf{m}_k\rVert^2}\,
    e^{-\frac\beta2\lVert\boldsymbol{\xi}-\mathbf{m}_k\rVert^2} .
\end{equation}

\paragraph{Unit norm and normalization.}
Because $\lVert\mathbf{m}_k\rVert = 1$ for all $k$, the factor
$e^{\beta/2}$ is common to every term and cancels against the partition function.
Recognizing each summand as an unnormalized isotropic Gaussian
$\mathcal{N}(\boldsymbol{\xi};\mathbf{m}_k,\beta^{-1}\mathbf{I})$ and integrating
over $\mathbb{R}^d$ gives
\begin{equation}
  \boxed{\;
  p_\beta(\boldsymbol{\xi})
  = \sum_{k=1}^{K} w_k\, \mathcal{N}(\boldsymbol{\xi};\,\mathbf{m}_k,\,\beta^{-1}\mathbf{I}),
  \qquad w_k = \frac{r_k}{\sum_j r_j} \; }
\end{equation}
an exact Gaussian mixture whose components are centered on the stored memories,
share the isotropic covariance $\beta^{-1}\mathbf{I}$, and carry mixture weights
equal to the normalized multiplicities. Had the memories not been unit norm, the
weights would instead be $r_k\,e^{\frac\beta2\lVert\mathbf{m}_k\rVert^2}$; the
result would still be a Gaussian mixture, but the weights would no longer reduce
to $w_k$.

\paragraph{Corollaries.}
\emph{(i) Exact sampling.} Draw $k\sim\mathrm{Categorical}(w)$ and
$\boldsymbol{\xi}=\mathbf{m}_k+\beta^{-1/2}\mathbf{z}$, $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$;
no Markov chain is required. Unadjusted Langevin dynamics target the same
$p_\beta$ subject to finite-step discretization and mixing error.
\emph{(ii) Exact designated control.} For a designated subset $B$, the latent
probability mass on $B$ is $\sum_{k\in B} w_k = \sum_{k\in B} r_k / \sum_j r_j
\equiv f_{\mathrm{eff}}$; with $r_k=\rho$ on $B$ and $1$ otherwise,
$f_{\mathrm{eff}} = \rho K_B / (\rho K_B + K_{NB})$, with no equal-similarity
assumption.
\emph{(iii) Score.} $\nabla_{\boldsymbol{\xi}}\log p_\beta
= -\beta\nabla E_r
= \beta\bigl[\mathbf{X}\,\mathrm{softmax}(\beta\mathbf{X}^\top\boldsymbol{\xi}+\log\mathbf{r})
  - \boldsymbol{\xi}\bigr]$, whose Euler-Maruyama discretization is the Algorithm~1
update.
```

- [ ] **Step 2: Hook the fragment into the appendix**

Run (appends one line; deterministic, no manual edit needed):
```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
printf '\n\\input{sections/si_gmm_derivation}\n' >> paper-arxiv/sections/appendix.tex
```

- [ ] **Step 3: Build the arXiv PDF and confirm the fragment processed**

Run:
```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/paper-arxiv && ./Build.sh Paper_v1
grep -c 'app:gmm' Paper_v1.aux
```
Expected: the build completes and produces `Paper_v1.pdf`; `grep -c` returns `1` (the new label `app:gmm` was written to the aux file, so the subsection compiled). If the build has pre-existing warnings unrelated to `si_gmm_derivation`, ignore them (build hygiene is Plan 6); only a `si_gmm_derivation`-referencing error blocks this task.

- [ ] **Step 4: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-arxiv/sections/si_gmm_derivation.tex paper-arxiv/sections/appendix.tex
git commit -m "docs: SI derivation of the exact Gaussian-mixture identity"
```

---

## Coverage against the spec (self-review)

- SI derivation (spec 2.1, W1): Task 7.
- Exact GMM sampler + corollary B control (spec 2.1 A/B): Tasks 1, 6.
- Score with the missing beta (spec 2.1 D, math item 1): Task 2.
- Shannon vs Renyi-2 entropy (math item 4): Task 3.
- Transition statistic redefinition + random probes (math item 5): Task 4.
- Real test suite scaffolding (W5): Tasks 1-5 + `runtests.jl`.
- Exact-vs-ULA baseline table (W3 core): Task 6.

Deferred by design (later plans, tracked): paper-prose corrections that need surgical edits to `theory.tex`/`discussion.tex`/`appendix.tex` ("no approximation" wording, the score proposition text, the entropy-baseline figure, four-vs-six families, `d_full/K` range, the "unit sphere" phrasing) live in Plan 5 (prose), which will read those files first. The PCA-radius ablation, decoder transfer-function study, canonical-CSV reconciliation, biology labels, docking provenance, and build/rename are Plans 2-6.

## Notes for the executor

- Every new randomized test seeds its own `MersenneTwister`; results are deterministic.
- Do not run `git add -A`; stage only the paths listed in each task's commit step (the tree carries an uncommitted `paper/` -> `paper-arxiv/` rename to leave untouched).
- Numeric tolerances were chosen with margin (finite-diff `< 1e-6`; 200k-sample Monte Carlo means `< 0.05`); if a stochastic test flakes, it indicates a real regression, not a tolerance that is too tight.
