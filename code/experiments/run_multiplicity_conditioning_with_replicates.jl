# ──────────────────────────────────────────────────────────────────────────────
# run_multiplicity_conditioning_with_replicates.jl
#
# Modified version that includes proper uncertainty quantification
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "multiplicity")
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Load data (same as original)
# ══════════════════════════════════════════════════════════════════════════════
@info "Loading Kunitz domain data"
sto_file = download_pfam_seed("PF00014"; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)
stored_seqs = [String(char_mat[i, :]) for i in 1:K_total]

# identify binders
lys_fracs = [count(i -> char_mat[i, j] == 'K', 1:K_total) /
             max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K_total))
             for j in 1:L]
p1_pos = argmax(lys_fracs)
binding_loop = collect(max(1, p1_pos - 4):min(L, p1_pos + 4))
strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
strong_seqs = stored_seqs[strong_idx]
K_b = length(strong_idx)
K_nb = K_total - K_b
natural_frac = K_b / K_total

@info "  K=$K_total | K_b=$K_b binders | K_nb=$K_nb non-binders | natural fraction=$(round(natural_frac, digits=3))"

# build the full-family memory matrix (shared across all ρ values)
X̂, pca_model, _, _ = build_memory_matrix(char_mat; pratio=0.95)
d = size(X̂, 1)
@info "  Memory matrix: $d × $K_total"

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 (MODIFIED): Generation quality with replicates
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT 2: Generation quality across ρ (WITH REPLICATES)"
@info "="^70

ρ_gen_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
n_reps = 5  # Number of independent replicates

# DataFrame to store ALL individual replicate results
gen_results_raw = DataFrame(
    ρ=Float64[], replicate=Int[], f_eff=Float64[], K_eff=Float64[], β_star=Float64[],
    p1_kr_frac=Float64[], diversity=Float64[], mean_novelty=Float64[],
    mean_seqid=Float64[], mean_valid=Float64[], kl_aa=Float64[],
    loop_entropy=Float64[], n_generated=Int[],
)

for (ρ_idx, ρ) in enumerate(ρ_gen_values)
    @info "  ρ = $ρ"
    r = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    f_eff = effective_binder_fraction(r, strong_idx)
    K_eff_val = effective_num_patterns(r)

    # find β* for this ρ (this is deterministic, compute once)
    pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)
    β_star = pt.β_star

    # Run multiple replicates with different seeds
    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 10000 + (ρ_idx - 1) * n_reps + rep  # Collision-free seed
        Random.seed!(seed)

        # generate sequences with unique seed
        seqs, pca_vecs = generate_weighted_sequences(X̂, pca_model, L, r;
            β=β_star, n_chains=30, T=5000, seed=seed)

        # evaluate metrics for this replicate
        n = length(seqs)
        p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n

        # pairwise diversity (sample subset for speed)
        n_pairs = min(500, n * (n - 1) ÷ 2)
        pair_ids = Float64[]
        for _ in 1:n_pairs
            i, j = rand(1:n), rand(1:n)
            while i == j; j = rand(1:n); end
            push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
        end
        diversity = 1.0 - mean(pair_ids)

        seq_ids = [nearest_sequence_identity(s, stored_seqs) for s in seqs]
        mean_novelty = 1.0 - mean(seq_ids)
        mean_seqid = mean(seq_ids)
        mean_valid = mean(valid_residue_fraction.(seqs))
        kl = aa_composition_kl(seqs, stored_seqs)

        # binding loop entropy
        freq = aa_freq_matrix(seqs, L)
        loop_ent = 0.0
        for pos in binding_loop
            for aa in 1:N_AA
                p = freq[aa, pos]
                p > 1e-10 && (loop_ent -= p * log(p))
            end
        end
        loop_ent /= length(binding_loop)

        # Store this replicate
        push!(gen_results_raw, (ρ, rep, f_eff, K_eff_val, β_star,
                               p1_kr, diversity, mean_novelty, mean_seqid,
                               mean_valid, kl, loop_ent, n))
    end
end

# Aggregate statistics across replicates
gen_results_agg = combine(groupby(gen_results_raw, :ρ),
    :f_eff => first => :f_eff,          # These are deterministic
    :K_eff => first => :K_eff,
    :β_star => first => :β_star,
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :diversity => mean => :diversity_mean,
    :diversity => std => :diversity_std,
    :mean_novelty => mean => :novelty_mean,
    :mean_novelty => std => :novelty_std,
    :mean_valid => mean => :valid_mean,
    :mean_valid => std => :valid_std,
    :kl_aa => mean => :kl_mean,
    :kl_aa => std => :kl_std,
    :loop_entropy => mean => :loop_ent_mean,
    :loop_entropy => std => :loop_ent_std,
    :n_generated => first => :n_generated
)

@info "\nGeneration results with uncertainty:"
show(stdout, gen_results_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 (MODIFIED): Calibration with replicates
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT 3: f_target → f_observed calibration (WITH REPLICATES)"
@info "="^70

f_targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

calibration_results_raw = DataFrame(
    f_target=Float64[], replicate=Int[], ρ=Float64[], f_eff=Float64[],
    f_observed=Float64[], K_eff=Float64[], β_star=Float64[],
)

for (ft_idx, ft) in enumerate(f_targets)
    @info "  f_target = $ft"
    result = build_multiplicity_conditioned_memory(char_mat, strong_idx;
        f_target=ft)

    pt = find_weighted_entropy_inflection(result.X̂, result.r; n_betas=50)

    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 20000 + (ft_idx - 1) * n_reps + rep  # Collision-free seed
        Random.seed!(seed)

        seqs, _ = generate_weighted_sequences(result.X̂, result.pca_model, L, result.r;
            β=pt.β_star, n_chains=20, T=5000, seed=seed)

        f_obs = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / length(seqs)

        push!(calibration_results_raw, (ft, rep, result.ρ, result.f_eff, f_obs,
                                       result.K_eff, pt.β_star))
    end
end

# Aggregate calibration results
calibration_results_agg = combine(groupby(calibration_results_raw, :f_target),
    :ρ => first => :ρ,
    :f_eff => first => :f_eff,
    :K_eff => first => :K_eff,
    :β_star => first => :β_star,
    :f_observed => mean => :f_obs_mean,
    :f_observed => std => :f_obs_std
)

@info "\nCalibration results with uncertainty:"
show(stdout, calibration_results_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES WITH ERROR BARS
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "Generating figures with error bars"
@info "="^70

# --- Figure 2: Generation quality with error bars ---
p2 = plot(layout=(2, 2), size=(1000, 800), margin=10Plots.mm)

plot!(p2[1], log10.(gen_results_agg.ρ), gen_results_agg.p1_kr_mean,
    ribbon=gen_results_agg.p1_kr_std, fillalpha=0.3,
    marker=:circle, linewidth=2.5, color=:steelblue, label="",
    xlabel="log₁₀(ρ)", ylabel="Fraction K/R at P1", ylim=(0, 1.05),
    title="Phenotype Fidelity")
hline!(p2[1], [natural_frac], linestyle=:dash, color=:gray, label="natural (ρ=1)")

plot!(p2[2], log10.(gen_results_agg.ρ), gen_results_agg.diversity_mean,
    ribbon=gen_results_agg.diversity_std, fillalpha=0.3,
    marker=:circle, linewidth=2.5, color=:coral, label="",
    xlabel="log₁₀(ρ)", ylabel="Pairwise diversity",
    title="Sequence Diversity")

plot!(p2[3], log10.(gen_results_agg.ρ), gen_results_agg.kl_mean,
    ribbon=gen_results_agg.kl_std, fillalpha=0.3,
    marker=:circle, linewidth=2.5, color=:forestgreen, label="",
    xlabel="log₁₀(ρ)", ylabel="KL(AA)",
    title="AA Composition Divergence")

plot!(p2[4], log10.(gen_results_agg.ρ), gen_results_agg.loop_ent_mean,
    ribbon=gen_results_agg.loop_ent_std, fillalpha=0.3,
    marker=:circle, linewidth=2.5, color=:purple, label="",
    xlabel="log₁₀(ρ)", ylabel="Per-position entropy (nats)",
    title="Binding Loop Constraint")

savefig(p2, joinpath(FIG_DIR, "fig2_rho_sweep_with_errors.pdf"))
savefig(p2, joinpath(FIG_DIR, "fig2_rho_sweep_with_errors.png"))
@info "  Saved fig2_rho_sweep_with_errors"

# --- Figure 3: Calibration curve with error bars ---
p3 = plot(size=(600, 500), margin=10Plots.mm,
    xlabel="Target effective binder fraction (f_target)",
    ylabel="Observed P1 K/R fraction (f_observed)",
    title="Multiplicity Conditioning Calibration",
    legend=:topleft)

plot!(p3, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)", linewidth=1.5)
plot!(p3, calibration_results_agg.f_target, calibration_results_agg.f_obs_mean,
    ribbon=calibration_results_agg.f_obs_std, fillalpha=0.3,
    marker=:circle, linewidth=2.5, color=:steelblue, label="observed ± SD", markersize=5)

savefig(p3, joinpath(FIG_DIR, "fig3_calibration_with_errors.pdf"))
savefig(p3, joinpath(FIG_DIR, "fig3_calibration_with_errors.png"))
@info "  Saved fig3_calibration_with_errors"

# --- Figure 4: Pareto front with error bars ---
p4 = plot(size=(700, 500), margin=10Plots.mm,
    xlabel="Pairwise sequence diversity",
    ylabel="P1 K/R fraction (phenotype fidelity)",
    title="Fidelity–Diversity Pareto Front\n(with uncertainty bounds)",
    legend=:bottomleft)

# Plot with error bars in both dimensions
scatter!(p4, gen_results_agg.diversity_mean, gen_results_agg.p1_kr_mean,
        xerror=gen_results_agg.diversity_std,
        yerror=gen_results_agg.p1_kr_std,
        marker=:circle, markersize=8, color=:steelblue,
        label="Multiplicity (ρ sweep) ± SD")

# annotate ρ values
for row in eachrow(gen_results_agg)
    annotate!(p4, row.diversity_mean + 0.003, row.p1_kr_mean - 0.02,
        text("ρ=$(Int(row.ρ))", 7, :left))
end

savefig(p4, joinpath(FIG_DIR, "fig4_pareto_front_with_errors.pdf"))
savefig(p4, joinpath(FIG_DIR, "fig4_pareto_front_with_errors.png"))
@info "  Saved fig4_pareto_front_with_errors"

# ══════════════════════════════════════════════════════════════════════════════
# Save results (both raw and aggregated)
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "multiplicity_generation_raw_replicates.csv"), gen_results_raw)
CSV.write(joinpath(CACHE_DIR, "multiplicity_generation_aggregated.csv"), gen_results_agg)
CSV.write(joinpath(CACHE_DIR, "multiplicity_calibration_raw_replicates.csv"), calibration_results_raw)
CSV.write(joinpath(CACHE_DIR, "multiplicity_calibration_aggregated.csv"), calibration_results_agg)

@info "\n" * "="^70
@info "Multiplicity conditioning experiment complete (WITH UNCERTAINTY)!"
@info "="^70
@info "\nKey improvements:"
@info "  1. Each condition run with $n_reps independent replicates"
@info "  2. Standard deviations computed across replicates"
@info "  3. All figures include error bars (ribbon plots)"
@info "  4. Both raw and aggregated data saved for analysis"
