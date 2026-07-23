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
