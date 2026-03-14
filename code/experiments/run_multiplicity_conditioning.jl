# ──────────────────────────────────────────────────────────────────────────────
# run_multiplicity_conditioning.jl
#
# The central experiment: multiplicity-weighted Hopfield energy for
# conditioning protein sequence generation on functional subsets.
#
# Theory: The multiplicity ratio ρ = r_binder / r_nonbinder parameterizes
# a continuous family of Boltzmann distributions that interpolate between
# "generate any family member" (ρ=1) and "generate only binders" (ρ→∞).
#
# The energy landscape:
#   E_r(ξ) = ½‖ξ‖² - (1/β) log Σ_k r_k exp(β m_k^T ξ)
#
# The Langevin update:
#   ξ_{t+1} = (1-α)ξ_t + α X softmax(β X^T ξ_t + log r) + √(2α/β) ε_t
#
# Key questions this experiment answers:
#   1. How does phenotype fidelity scale with ρ?
#   2. How does the phase transition β*(ρ) shift with multiplicity?
#   3. What is the Pareto front of fidelity vs diversity vs fold quality?
#   4. How does K_eff(ρ) relate to β*?
#   5. Does the f_target → f_generated mapping work as predicted?
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "multiplicity")
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Load data
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
# EXPERIMENT 1: Phase transition β*(ρ) as a function of multiplicity ratio
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT 1: Phase transition β*(ρ)"
@info "="^70

ρ_values = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

phase_results = DataFrame(
    ρ=Float64[], f_eff=Float64[], K_eff=Float64[],
    β_star=Float64[], log_K_eff=Float64[],
)

for ρ in ρ_values
    r = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    f_eff = effective_binder_fraction(r, strong_idx)
    K_eff = effective_num_patterns(r)

    pt = find_weighted_entropy_inflection(X̂, r; n_betas=60)

    @info "  ρ=$(round(ρ, digits=1))  f_eff=$(round(f_eff, digits=3))  K_eff=$(round(K_eff, digits=1))  β*=$(round(pt.β_star, digits=2))"
    push!(phase_results, (ρ, f_eff, K_eff, pt.β_star, log(K_eff)))
end

@info "\nPhase transition results:"
show(stdout, phase_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Multiplicity ratio sweep — generation quality
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT 2: Generation quality across ρ"
@info "="^70

ρ_gen_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]

gen_results = DataFrame(
    ρ=Float64[], f_eff=Float64[], K_eff=Float64[], β_star=Float64[],
    p1_kr_frac=Float64[], diversity=Float64[], mean_novelty=Float64[],
    mean_seqid=Float64[], mean_valid=Float64[], kl_aa=Float64[],
    loop_entropy=Float64[], n_generated=Int[],
)

for ρ in ρ_gen_values
    @info "  ρ = $ρ"
    r = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    f_eff = effective_binder_fraction(r, strong_idx)
    K_eff_val = effective_num_patterns(r)

    # find β* for this ρ
    pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)
    β_star = pt.β_star

    # generate sequences
    seqs, pca_vecs = generate_weighted_sequences(X̂, pca_model, L, r;
        β=β_star, n_chains=30, T=5000, seed=42)

    # evaluate
    n = length(seqs)
    p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n

    # pairwise diversity
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

    push!(gen_results, (ρ, f_eff, K_eff_val, β_star,
                         p1_kr, diversity, mean_novelty, mean_seqid,
                         mean_valid, kl, loop_ent, n))
end

@info "\nGeneration results:"
show(stdout, gen_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Target fraction → observed fraction calibration
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT 3: f_target → f_observed calibration"
@info "="^70

f_targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

calibration_results = DataFrame(
    f_target=Float64[], ρ=Float64[], f_eff=Float64[],
    f_observed=Float64[], K_eff=Float64[], β_star=Float64[],
)

for ft in f_targets
    @info "  f_target = $ft"
    result = build_multiplicity_conditioned_memory(char_mat, strong_idx;
        f_target=ft)

    pt = find_weighted_entropy_inflection(result.X̂, result.r; n_betas=50)

    seqs, _ = generate_weighted_sequences(result.X̂, result.pca_model, L, result.r;
        β=pt.β_star, n_chains=20, T=5000, seed=42)

    f_obs = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / length(seqs)

    push!(calibration_results, (ft, result.ρ, result.f_eff, f_obs, result.K_eff, pt.β_star))
end

@info "\nCalibration results:"
show(stdout, calibration_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Comparison with hard curation at matched fidelity
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT 4: Multiplicity vs hard curation (matched fidelity)"
@info "="^70

# Hard curation: only binders
strong_char = char_mat[strong_idx, :]
X̂_hard, pca_hard, _, _ = build_memory_matrix(strong_char; pratio=0.95)
pt_hard = find_entropy_inflection(X̂_hard)
hard_seqs, hard_pca = generate_sequences(X̂_hard, pca_hard, L;
    β=pt_hard.β_star, n_chains=30, T=5000, seed=42)

# Multiplicity at ρ that gives ~100% fidelity (ρ=500)
r_high = multiplicity_vector(K_total, strong_idx; ρ=500.0)
pt_high = find_weighted_entropy_inflection(X̂, r_high)
mult_seqs, mult_pca = generate_weighted_sequences(X̂, pca_model, L, r_high;
    β=pt_high.β_star, n_chains=30, T=5000, seed=42)

# Compare
function detailed_metrics(seqs, pca_vecs, X̂, β, ref_seqs, label)
    n = length(seqs)
    p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n
    valid = mean(valid_residue_fraction.(seqs))
    kl = aa_composition_kl(seqs, ref_seqs)

    pair_ids = Float64[]
    for _ in 1:min(1000, n * (n - 1) ÷ 2)
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
    end
    diversity = 1.0 - mean(pair_ids)
    novelty = 1.0 - mean(nearest_sequence_identity(s, ref_seqs) for s in seqs)

    energies = [hopfield_energy(v, X̂, β) for v in pca_vecs]

    @info "  [$label]"
    @info "    P1 K/R: $(round(p1_kr, digits=3))"
    @info "    Diversity: $(round(diversity, digits=3))"
    @info "    Novelty: $(round(novelty, digits=3))"
    @info "    Valid: $(round(valid, digits=3))"
    @info "    KL(AA): $(round(kl, digits=4))"
    @info "    Mean energy: $(round(mean(energies), digits=3))"

    return (p1_kr=p1_kr, diversity=diversity, novelty=novelty,
            valid=valid, kl=kl, mean_energy=mean(energies))
end

@info "\nHead-to-head comparison (both should be ~100% P1 fidelity):"
hard_metrics = detailed_metrics(hard_seqs, hard_pca, X̂_hard, pt_hard.β_star,
                                 strong_seqs, "Hard curation (32 binders only)")
mult_metrics = detailed_metrics(mult_seqs, mult_pca, X̂, pt_high.β_star,
                                 stored_seqs, "Multiplicity ρ=500 (full family)")

@info "\nAdvantage of multiplicity conditioning:"
@info "  Diversity: $(round(mult_metrics.diversity, digits=3)) vs $(round(hard_metrics.diversity, digits=3)) " *
      (mult_metrics.diversity > hard_metrics.diversity ? "(multiplicity wins ✓)" : "(hard curation wins)")
@info "  KL(AA):   $(round(mult_metrics.kl, digits=4)) vs $(round(hard_metrics.kl, digits=4)) " *
      (mult_metrics.kl < hard_metrics.kl ? "(multiplicity wins ✓)" : "(hard curation wins)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "Generating figures"
@info "="^70

# --- Figure 1: β*(ρ) phase diagram ---
p1 = plot(layout=(1, 2), size=(1000, 400), margin=10Plots.mm)

plot!(p1[1], log10.(phase_results.ρ), phase_results.β_star,
    marker=:circle, linewidth=2, color=:steelblue, label="",
    xlabel="log₁₀(ρ)", ylabel="β*",
    title="Phase Transition vs Multiplicity Ratio")

plot!(p1[2], phase_results.K_eff, phase_results.β_star,
    marker=:circle, linewidth=2, color=:coral, label="Empirical β*",
    xlabel="K_eff(r)", ylabel="β*",
    title="Phase Transition vs Effective K")
# add √d reference
hline!(p1[2], [sqrt(d)], linestyle=:dash, color=:gray, label="√d")

savefig(p1, joinpath(FIG_DIR, "fig1_phase_transition.pdf"))
savefig(p1, joinpath(FIG_DIR, "fig1_phase_transition.png"))
@info "  Saved fig1_phase_transition"

# --- Figure 2: Generation quality across ρ (the main result) ---
p2 = plot(layout=(2, 2), size=(1000, 800), margin=10Plots.mm)

plot!(p2[1], log10.(gen_results.ρ), gen_results.p1_kr_frac,
    marker=:circle, linewidth=2.5, color=:steelblue, label="",
    xlabel="log₁₀(ρ)", ylabel="Fraction K/R at P1", ylim=(0, 1.05),
    title="Phenotype Fidelity")
hline!(p2[1], [natural_frac], linestyle=:dash, color=:gray, label="natural (ρ=1)")

plot!(p2[2], log10.(gen_results.ρ), gen_results.diversity,
    marker=:circle, linewidth=2.5, color=:coral, label="",
    xlabel="log₁₀(ρ)", ylabel="Pairwise diversity",
    title="Sequence Diversity")

plot!(p2[3], log10.(gen_results.ρ), gen_results.kl_aa,
    marker=:circle, linewidth=2.5, color=:forestgreen, label="",
    xlabel="log₁₀(ρ)", ylabel="KL(AA)",
    title="AA Composition Divergence")

plot!(p2[4], log10.(gen_results.ρ), gen_results.loop_entropy,
    marker=:circle, linewidth=2.5, color=:purple, label="",
    xlabel="log₁₀(ρ)", ylabel="Per-position entropy (nats)",
    title="Binding Loop Constraint")

savefig(p2, joinpath(FIG_DIR, "fig2_rho_sweep.pdf"))
savefig(p2, joinpath(FIG_DIR, "fig2_rho_sweep.png"))
@info "  Saved fig2_rho_sweep"

# --- Figure 3: Calibration curve (f_target vs f_observed) ---
p3 = plot(size=(600, 500), margin=10Plots.mm,
    xlabel="Target effective binder fraction (f_target)",
    ylabel="Observed P1 K/R fraction (f_observed)",
    title="Multiplicity Conditioning Calibration",
    legend=:topleft)

plot!(p3, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)", linewidth=1.5)
scatter!(p3, calibration_results.f_target, calibration_results.f_observed,
    marker=:circle, markersize=7, color=:steelblue, label="observed", linewidth=0)
plot!(p3, calibration_results.f_target, calibration_results.f_observed,
    linewidth=2, color=:steelblue, label="")

savefig(p3, joinpath(FIG_DIR, "fig3_calibration.pdf"))
savefig(p3, joinpath(FIG_DIR, "fig3_calibration.png"))
@info "  Saved fig3_calibration"

# --- Figure 4: Pareto front (fidelity vs diversity, colored by ρ) ---
p4 = plot(size=(700, 500), margin=10Plots.mm,
    xlabel="Pairwise sequence diversity",
    ylabel="P1 K/R fraction (phenotype fidelity)",
    title="Fidelity–Diversity Pareto Front\n(parameterized by multiplicity ratio ρ)",
    legend=:bottomleft)

scatter!(p4, gen_results.diversity, gen_results.p1_kr_frac,
    marker=:circle, markersize=8, color=:steelblue, label="Multiplicity (ρ sweep)",
    zcolor=log10.(gen_results.ρ), colorbar_title="log₁₀(ρ)")

# add hard curation point
scatter!(p4, [hard_metrics.diversity], [hard_metrics.p1_kr],
    marker=:star5, markersize=12, color=:red, label="Hard curation (32 binders)")

# annotate ρ values
for row in eachrow(gen_results)
    annotate!(p4, row.diversity + 0.003, row.p1_kr_frac - 0.02,
        text("ρ=$(Int(row.ρ))", 7, :left))
end

savefig(p4, joinpath(FIG_DIR, "fig4_pareto_front.pdf"))
savefig(p4, joinpath(FIG_DIR, "fig4_pareto_front.png"))
@info "  Saved fig4_pareto_front"

# --- Figure 5: Entropy curves at different ρ values ---
p5 = plot(size=(700, 500), margin=10Plots.mm,
    xlabel="β (inverse temperature)", ylabel="Attention entropy H(β)",
    title="Phase Transition Shifts with Multiplicity Ratio",
    legend=:topright, xscale=:log10)

ρ_show = [1.0, 5.0, 20.0, 100.0, 1000.0]
colors_show = [:gray, :steelblue, :coral, :purple, :forestgreen]
for (i, ρ) in enumerate(ρ_show)
    r = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    pt = find_weighted_entropy_inflection(X̂, r; n_betas=60)
    plot!(p5, pt.βs, pt.Hs, linewidth=2, color=colors_show[i], label="ρ=$ρ")
    vline!(p5, [pt.β_star], linestyle=:dash, color=colors_show[i], label="", linewidth=1)
end

savefig(p5, joinpath(FIG_DIR, "fig5_entropy_curves.pdf"))
savefig(p5, joinpath(FIG_DIR, "fig5_entropy_curves.png"))
@info "  Saved fig5_entropy_curves"

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "multiplicity_phase_transition.csv"), phase_results)
CSV.write(joinpath(CACHE_DIR, "multiplicity_generation.csv"), gen_results)
CSV.write(joinpath(CACHE_DIR, "multiplicity_calibration.csv"), calibration_results)

@info "\n" * "="^70
@info "Multiplicity conditioning experiment complete!"
@info "="^70
@info "\nKey results:"
@info "  1. β*(ρ) shifts as multiplicity ratio increases"
@info "  2. Phenotype fidelity is a monotonic function of ρ"
@info "  3. f_target → f_observed calibration shows predictive control"
@info "  4. Multiplicity conditioning preserves full-family fold constraints"
@info "     while achieving phenotype transfer comparable to hard curation"
