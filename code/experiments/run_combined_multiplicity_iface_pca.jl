# ──────────────────────────────────────────────────────────────────────────────
# run_combined_multiplicity_iface_pca.jl
#
# The synthesis: multiplicity-weighted Hopfield energy on an interface-aware
# PCA encoding. Combines:
#   - Approach 1b: multiplicity ratio ρ tilts the energy toward binders
#   - Approach 4:  interface-weighted PCA ensures binding-site variation is
#                  captured in the principal components
#
# The calibration diagnostics showed the gap is entirely in the PCA layer:
#   attention weights track f_eff perfectly, but PCA reconstruction doesn't
#   resolve binder vs non-binder at the P1 position.
#
# Hypothesis: interface-weighted PCA + multiplicity weighting will close the
# calibration gap, because:
#   1. The PCA now captures interface variation in its top components
#   2. The multiplicity weights correctly tilt the Hopfield energy
#   3. The full chain ρ → attention → PCA → decode becomes faithful
#
# Experiment structure:
#   A. Sweep interface weight w at fixed ρ (does PCA weight help?)
#   B. Sweep ρ at fixed interface weight (does calibration gap close?)
#   C. Full calibration: f_target → f_observed with combined method
#   D. Gap decomposition: attention / soft / hard layers
#   E. Head-to-head: combined vs hard curation vs multiplicity-only
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "combined")
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

lys_fracs = [count(i -> char_mat[i, j] == 'K', 1:K_total) /
             max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K_total))
             for j in 1:L]
p1_pos = argmax(lys_fracs)
binding_loop = collect(max(1, p1_pos - 4):min(L, p1_pos + 4))
strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
K_b = length(strong_idx)
K_nb = K_total - K_b
natural_frac = K_b / K_total

@info "  K=$K_total | K_b=$K_b | P1 at $p1_pos | binding loop: $binding_loop"

# ══════════════════════════════════════════════════════════════════════════════
# Helper: build interface-weighted memory + multiplicity vector
# ══════════════════════════════════════════════════════════════════════════════
"""
Build the combined system: interface-weighted PCA + multiplicity vector.
Returns everything needed to generate and decode.
"""
function build_combined_system(char_mat, binder_indices, interface_positions;
                                pratio=0.95, weight=3.0, ρ=10.0)
    K = size(char_mat, 1)

    # interface-weighted one-hot + PCA
    X_onehot = interface_weighted_onehot(char_mat, interface_positions; weight=weight)
    d_full = size(X_onehot, 1)
    pca_model = MultivariateStats.fit(PCA, X_onehot; pratio=pratio)
    d_pca = MultivariateStats.outdim(pca_model)
    Z = MultivariateStats.transform(pca_model, X_onehot)

    # unit-norm
    X̂ = copy(Z)
    for k in 1:K
        nk = norm(X̂[:, k])
        X̂[:, k] ./= (nk + 1e-12)
    end

    # multiplicity vector
    r = multiplicity_vector(K, binder_indices; ρ=ρ)
    f_eff = effective_binder_fraction(r, binder_indices)
    K_eff = effective_num_patterns(r)

    return (X̂=X̂, pca_model=pca_model, r=r, d_pca=d_pca,
            f_eff=f_eff, K_eff=K_eff, ρ=ρ, weight=weight,
            interface_positions=interface_positions)
end

"""
Generate sequences from the combined system, collecting full diagnostics.
"""
function generate_with_diagnostics(sys, L, p1_pos, binder_indices, char_mat;
                                    n_chains=20, T=5000, burnin=2000, thin=100,
                                    seed=42, β=nothing)
    X̂ = sys.X̂
    r = sys.r
    pca_model = sys.pca_model
    d, K = size(X̂)
    log_r = log.(r)
    iface_pos = sys.interface_positions
    w = sys.weight

    # find β* if not provided
    if β === nothing
        pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)
        β = pt.β_star
    end

    gen_seqs = String[]
    attn_binder = Float64[]
    soft_p1 = Float64[]

    for chain in 1:n_chains
        k = mod1(chain, K)
        ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)
        result = weighted_sample(X̂, ξ₀, T, r; β=β, α=0.01, seed=seed + chain)

        for t in burnin:thin:T
            ξ = result.Ξ[t + 1, :]

            # decode (undo interface weighting)
            seq = decode_weighted_sample(ξ, pca_model, L, iface_pos; weight=w)
            push!(gen_seqs, seq)

            # attention diagnostic
            logits = β .* (X̂' * ξ) .+ log_r
            attn = NNlib.softmax(logits)
            push!(attn_binder, sum(attn[idx] for idx in binder_indices))

            # soft P1 diagnostic (reconstruct, undo weight, then check P1)
            x_oh = vec(MultivariateStats.reconstruct(pca_model, ξ))
            p1_start = (p1_pos - 1) * N_AA + 1
            p1_block = x_oh[p1_start:(p1_start + N_AA - 1)]
            # undo interface weight at P1 if P1 is an interface position
            if p1_pos in iface_pos
                p1_block ./= w
            end
            p1_probs = NNlib.softmax(p1_block .* 5.0)
            push!(soft_p1, p1_probs[AA_TO_IDX['K']] + p1_probs[AA_TO_IDX['R']])
        end
    end

    n = length(gen_seqs)
    f_obs = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), gen_seqs) / n
    valid = mean(valid_residue_fraction.(gen_seqs))

    # diversity
    pair_ids = Float64[]
    for _ in 1:min(500, n * (n - 1) ÷ 2)
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(gen_seqs[i], gen_seqs[j]))
    end
    diversity = 1.0 - mean(pair_ids)

    return (seqs=gen_seqs, f_obs=f_obs, diversity=diversity, valid=valid,
            attn_binder=mean(attn_binder), soft_p1=mean(soft_p1),
            β=β, n=n)
end

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: Interface weight sweep at fixed ρ
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT A: Interface weight sweep (fixed ρ=50)"
@info "="^70

weight_values = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
ρ_fixed_A = 50.0

weight_results = DataFrame(
    weight=Float64[], f_obs=Float64[], soft_p1=Float64[],
    attn_binder=Float64[], diversity=Float64[], d_pca=Int[],
    β=Float64[],
)

for w in weight_values
    @info "  weight=$w"
    sys = build_combined_system(char_mat, strong_idx, binding_loop;
        weight=w, ρ=ρ_fixed_A)
    res = generate_with_diagnostics(sys, L, p1_pos, strong_idx, char_mat;
        n_chains=20, seed=42)

    push!(weight_results, (w, res.f_obs, res.soft_p1, res.attn_binder,
                            res.diversity, sys.d_pca, res.β))
end

@info "\nWeight sweep results (ρ=$ρ_fixed_A):"
show(stdout, weight_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: ρ sweep at best interface weight
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT B: ρ sweep with interface-weighted PCA"
@info "="^70

# pick the weight that gave best f_obs in experiment A
best_weight = weight_results.weight[argmax(weight_results.f_obs)]
@info "  Best interface weight from Experiment A: $best_weight"

ρ_values_B = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

rho_combined_results = DataFrame(
    ρ=Float64[], f_eff=Float64[], f_obs=Float64[],
    soft_p1=Float64[], attn_binder=Float64[], diversity=Float64[],
    β=Float64[],
)

# also run standard PCA for comparison
rho_standard_results = DataFrame(
    ρ=Float64[], f_eff=Float64[], f_obs=Float64[],
    soft_p1=Float64[], attn_binder=Float64[], diversity=Float64[],
    β=Float64[],
)

X̂_std, pca_std, _, _ = build_memory_matrix(char_mat; pratio=0.95)

for ρ in ρ_values_B
    @info "  ρ=$ρ"

    # combined system
    sys_c = build_combined_system(char_mat, strong_idx, binding_loop;
        weight=best_weight, ρ=ρ)
    res_c = generate_with_diagnostics(sys_c, L, p1_pos, strong_idx, char_mat;
        n_chains=25, seed=42)
    push!(rho_combined_results, (ρ, sys_c.f_eff, res_c.f_obs,
        res_c.soft_p1, res_c.attn_binder, res_c.diversity, res_c.β))

    # standard PCA comparison
    r_std = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    pt_std = find_weighted_entropy_inflection(X̂_std, r_std; n_betas=50)
    log_r_std = log.(r_std)

    gen_seqs_std = String[]
    attn_std = Float64[]
    soft_std = Float64[]
    d_std = size(X̂_std, 1)

    for chain in 1:25
        k = mod1(chain, K_total)
        ξ₀ = X̂_std[:, k] .+ 0.01 .* randn(d_std)
        result = weighted_sample(X̂_std, ξ₀, 5000, r_std;
            β=pt_std.β_star, α=0.01, seed=42 + chain)
        for t in 2000:100:5000
            ξ = result.Ξ[t + 1, :]
            seq = decode_sample(ξ, pca_std, L)
            push!(gen_seqs_std, seq)

            logits = pt_std.β_star .* (X̂_std' * ξ) .+ log_r_std
            attn = NNlib.softmax(logits)
            push!(attn_std, sum(attn[idx] for idx in strong_idx))

            x_oh = vec(MultivariateStats.reconstruct(pca_std, ξ))
            p1s = (p1_pos - 1) * N_AA + 1
            p1b = x_oh[p1s:(p1s + N_AA - 1)]
            p1p = NNlib.softmax(p1b .* 5.0)
            push!(soft_std, p1p[AA_TO_IDX['K']] + p1p[AA_TO_IDX['R']])
        end
    end

    n_s = length(gen_seqs_std)
    f_obs_s = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), gen_seqs_std) / n_s
    pair_ids_s = Float64[]
    for _ in 1:min(500, n_s * (n_s - 1) ÷ 2)
        i, j = rand(1:n_s), rand(1:n_s)
        while i == j; j = rand(1:n_s); end
        push!(pair_ids_s, sequence_identity(gen_seqs_std[i], gen_seqs_std[j]))
    end
    f_eff_s = effective_binder_fraction(r_std, strong_idx)
    push!(rho_standard_results, (ρ, f_eff_s, f_obs_s,
        mean(soft_std), mean(attn_std), 1.0 - mean(pair_ids_s), pt_std.β_star))
end

@info "\nCombined (weight=$best_weight) results:"
show(stdout, rho_combined_results)
println()
@info "\nStandard PCA results:"
show(stdout, rho_standard_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C: Full calibration with combined method
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT C: Full calibration (combined method)"
@info "="^70

f_targets_C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

cal_combined = DataFrame(
    f_target=Float64[], ρ=Float64[], f_obs_combined=Float64[],
    f_obs_standard=Float64[], soft_p1_combined=Float64[], soft_p1_standard=Float64[],
)

for ft in f_targets_C
    @info "  f_target=$ft"
    ρ = ρ_for_target_fraction(K_b, K_nb, ft)

    # combined
    sys = build_combined_system(char_mat, strong_idx, binding_loop;
        weight=best_weight, ρ=ρ)
    res = generate_with_diagnostics(sys, L, p1_pos, strong_idx, char_mat;
        n_chains=20, seed=42)

    # standard
    r_std = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    pt_std = find_weighted_entropy_inflection(X̂_std, r_std; n_betas=50)

    seqs_std = String[]
    soft_std_vals = Float64[]
    d_std = size(X̂_std, 1)
    log_r_std = log.(r_std)

    for chain in 1:20
        k = mod1(chain, K_total)
        ξ₀ = X̂_std[:, k] .+ 0.01 .* randn(d_std)
        result = weighted_sample(X̂_std, ξ₀, 5000, r_std;
            β=pt_std.β_star, α=0.01, seed=42 + chain)
        for t in 2000:100:5000
            ξ = result.Ξ[t + 1, :]
            push!(seqs_std, decode_sample(ξ, pca_std, L))
            x_oh = vec(MultivariateStats.reconstruct(pca_std, ξ))
            p1s = (p1_pos - 1) * N_AA + 1
            p1b = x_oh[p1s:(p1s + N_AA - 1)]
            p1p = NNlib.softmax(p1b .* 5.0)
            push!(soft_std_vals, p1p[AA_TO_IDX['K']] + p1p[AA_TO_IDX['R']])
        end
    end
    f_obs_std = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs_std) / length(seqs_std)

    push!(cal_combined, (ft, ρ, res.f_obs, f_obs_std, res.soft_p1, mean(soft_std_vals)))
end

@info "\nCalibration comparison:"
show(stdout, cal_combined)
println()

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D: Head-to-head at matched high ρ
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "EXPERIMENT D: Head-to-head comparison"
@info "="^70

# hard curation
strong_char = char_mat[strong_idx, :]
X̂_hard, pca_hard, _, _ = build_memory_matrix(strong_char; pratio=0.95)
pt_hard = find_entropy_inflection(X̂_hard)
hard_seqs, _ = generate_sequences(X̂_hard, pca_hard, L;
    β=pt_hard.β_star, n_chains=30, T=5000, seed=42)
hard_f = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), hard_seqs) / length(hard_seqs)
hard_pair_ids = Float64[]
for _ in 1:500
    i, j = rand(1:length(hard_seqs)), rand(1:length(hard_seqs))
    while i == j; j = rand(1:length(hard_seqs)); end
    push!(hard_pair_ids, sequence_identity(hard_seqs[i], hard_seqs[j]))
end
hard_div = 1.0 - mean(hard_pair_ids)
strong_seqs = stored_seqs[strong_idx]
hard_kl = aa_composition_kl(hard_seqs, strong_seqs)
hard_valid = mean(valid_residue_fraction.(hard_seqs))

# combined at ρ=500
sys_500 = build_combined_system(char_mat, strong_idx, binding_loop;
    weight=best_weight, ρ=500.0)
res_500 = generate_with_diagnostics(sys_500, L, p1_pos, strong_idx, char_mat;
    n_chains=30, seed=42)
comb_kl = aa_composition_kl(res_500.seqs, stored_seqs)

# multiplicity-only at ρ=500
r_500 = multiplicity_vector(K_total, strong_idx; ρ=500.0)
pt_500 = find_weighted_entropy_inflection(X̂_std, r_500; n_betas=50)
mult_seqs, _ = generate_weighted_sequences(X̂_std, pca_std, L, r_500;
    β=pt_500.β_star, n_chains=30, T=5000, seed=42)
mult_f = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), mult_seqs) / length(mult_seqs)
mult_pair_ids = Float64[]
for _ in 1:500
    i, j = rand(1:length(mult_seqs)), rand(1:length(mult_seqs))
    while i == j; j = rand(1:length(mult_seqs)); end
    push!(mult_pair_ids, sequence_identity(mult_seqs[i], mult_seqs[j]))
end
mult_div = 1.0 - mean(mult_pair_ids)
mult_kl = aa_composition_kl(mult_seqs, stored_seqs)
mult_valid = mean(valid_residue_fraction.(mult_seqs))

@info "\n=== HEAD-TO-HEAD (ρ=500) ==="
@info "  Method                    | P1 K/R | Diversity | KL(AA)  | Valid"
@info "  " * "-"^75
@info "  Hard curation (32 only)   | $(round(hard_f, digits=3))  | $(round(hard_div, digits=3))     | $(round(hard_kl, digits=4))  | $(round(hard_valid, digits=3))"
@info "  Multiplicity only (ρ=500) | $(round(mult_f, digits=3))  | $(round(mult_div, digits=3))     | $(round(mult_kl, digits=4)) | $(round(mult_valid, digits=3))"
@info "  Combined (ρ=500, w=$(best_weight))   | $(round(res_500.f_obs, digits=3))  | $(round(res_500.diversity, digits=3))     | $(round(comb_kl, digits=4)) | $(round(res_500.valid, digits=3))"

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "Generating figures"
@info "="^70

# --- Figure 1: Interface weight effect ---
p1 = plot(layout=(1, 3), size=(1200, 400), margin=10Plots.mm)

plot!(p1[1], weight_results.weight, weight_results.f_obs,
    marker=:circle, linewidth=2, color=:steelblue, label="P1 K/R (hard)",
    xlabel="Interface weight w", ylabel="Fraction",
    title="Phenotype Fidelity (ρ=$ρ_fixed_A)")
plot!(p1[1], weight_results.weight, weight_results.soft_p1,
    marker=:square, linewidth=2, color=:purple, label="P1 K/R (soft)")

plot!(p1[2], weight_results.weight, weight_results.diversity,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="Interface weight w", ylabel="Diversity",
    title="Sequence Diversity")

plot!(p1[3], weight_results.weight, weight_results.d_pca,
    marker=:circle, linewidth=2, color=:forestgreen, label="",
    xlabel="Interface weight w", ylabel="PCA dimensions",
    title="PCA Dimensionality")

savefig(p1, joinpath(FIG_DIR, "fig1_weight_sweep.pdf"))
savefig(p1, joinpath(FIG_DIR, "fig1_weight_sweep.png"))
@info "  Saved fig1_weight_sweep"

# --- Figure 2: Calibration comparison (the money figure) ---
p2 = plot(size=(700, 550), margin=12Plots.mm,
    xlabel="Target effective binder fraction (f_target)",
    ylabel="Observed P1 K/R fraction",
    title="Calibration: Standard PCA vs Combined Method",
    legend=:topleft, ylim=(0, 1.05), xlim=(0, 1.05))

plot!(p2, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)", linewidth=1.5)
plot!(p2, cal_combined.f_target, cal_combined.f_obs_standard,
    marker=:circle, linewidth=2.5, color=:coral, label="Standard PCA", markersize=6)
plot!(p2, cal_combined.f_target, cal_combined.f_obs_combined,
    marker=:diamond, linewidth=2.5, color=:steelblue, label="Combined (w=$(best_weight))", markersize=6)

savefig(p2, joinpath(FIG_DIR, "fig2_calibration_comparison.pdf"))
savefig(p2, joinpath(FIG_DIR, "fig2_calibration_comparison.png"))
@info "  Saved fig2_calibration_comparison"

# --- Figure 3: ρ sweep comparison ---
p3 = plot(layout=(1, 2), size=(1000, 450), margin=10Plots.mm)

plot!(p3[1], log10.(rho_standard_results.ρ), rho_standard_results.f_obs,
    marker=:circle, linewidth=2, color=:coral, label="Standard PCA",
    xlabel="log₁₀(ρ)", ylabel="P1 K/R fraction", ylim=(0, 1.05),
    title="Phenotype Fidelity")
plot!(p3[1], log10.(rho_combined_results.ρ), rho_combined_results.f_obs,
    marker=:diamond, linewidth=2, color=:steelblue, label="Combined (w=$(best_weight))")
hline!(p3[1], [natural_frac], linestyle=:dot, color=:gray, label="natural")
hline!(p3[1], [1.0], linestyle=:dash, color=:lightgray, label="")

plot!(p3[2], log10.(rho_standard_results.ρ), rho_standard_results.diversity,
    marker=:circle, linewidth=2, color=:coral, label="Standard PCA",
    xlabel="log₁₀(ρ)", ylabel="Pairwise diversity",
    title="Sequence Diversity")
plot!(p3[2], log10.(rho_combined_results.ρ), rho_combined_results.diversity,
    marker=:diamond, linewidth=2, color=:steelblue, label="Combined (w=$(best_weight))")

savefig(p3, joinpath(FIG_DIR, "fig3_rho_comparison.pdf"))
savefig(p3, joinpath(FIG_DIR, "fig3_rho_comparison.png"))
@info "  Saved fig3_rho_comparison"

# --- Figure 4: Soft P1 score comparison (the diagnostic layer) ---
p4 = plot(size=(700, 450), margin=10Plots.mm,
    xlabel="log₁₀(ρ)", ylabel="Soft P1 K/R probability",
    title="PCA Bottleneck: Standard vs Interface-Weighted",
    legend=:topleft)

plot!(p4, log10.(rho_standard_results.ρ), rho_standard_results.soft_p1,
    marker=:circle, linewidth=2.5, color=:coral, label="Standard PCA (flat!)", markersize=5)
plot!(p4, log10.(rho_combined_results.ρ), rho_combined_results.soft_p1,
    marker=:diamond, linewidth=2.5, color=:steelblue, label="Combined (responsive!)", markersize=5)
hline!(p4, [natural_frac], linestyle=:dot, color=:gray, label="natural binder fraction")

savefig(p4, joinpath(FIG_DIR, "fig4_soft_score_comparison.pdf"))
savefig(p4, joinpath(FIG_DIR, "fig4_soft_score_comparison.png"))
@info "  Saved fig4_soft_score_comparison"

# --- Figure 5: Head-to-head bar chart ---
methods = ["Hard curation\n(32 binders)", "Multiplicity\n(ρ=500, std PCA)", "Combined\n(ρ=500, w=$(best_weight))"]
p5 = plot(layout=(1, 3), size=(1200, 400), margin=10Plots.mm)

bar!(p5[1], methods, [hard_f, mult_f, res_500.f_obs],
    ylabel="P1 K/R fraction", title="Phenotype Fidelity",
    color=[:coral, :steelblue, :purple], legend=false, ylim=(0, 1.05))

bar!(p5[2], methods, [hard_div, mult_div, res_500.diversity],
    ylabel="Pairwise diversity", title="Sequence Diversity",
    color=[:coral, :steelblue, :purple], legend=false)

bar!(p5[3], methods, [hard_kl, mult_kl, comb_kl],
    ylabel="KL(AA)", title="AA Composition Quality",
    color=[:coral, :steelblue, :purple], legend=false)

savefig(p5, joinpath(FIG_DIR, "fig5_head_to_head.pdf"))
savefig(p5, joinpath(FIG_DIR, "fig5_head_to_head.png"))
@info "  Saved fig5_head_to_head"

# --- Figure 6: Pareto front ---
p6 = plot(size=(700, 500), margin=10Plots.mm,
    xlabel="Pairwise diversity", ylabel="P1 K/R fraction",
    title="Fidelity–Diversity Pareto Front",
    legend=:bottomleft, ylim=(0, 1.05))

scatter!(p6, rho_standard_results.diversity, rho_standard_results.f_obs,
    marker=:circle, markersize=7, color=:coral, label="Multiplicity only")
scatter!(p6, rho_combined_results.diversity, rho_combined_results.f_obs,
    marker=:diamond, markersize=7, color=:steelblue, label="Combined")
scatter!(p6, [hard_div], [hard_f],
    marker=:star5, markersize=12, color=:red, label="Hard curation")

savefig(p6, joinpath(FIG_DIR, "fig6_pareto.pdf"))
savefig(p6, joinpath(FIG_DIR, "fig6_pareto.png"))
@info "  Saved fig6_pareto"

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "combined_weight_sweep.csv"), weight_results)
CSV.write(joinpath(CACHE_DIR, "combined_rho_sweep.csv"), rho_combined_results)
CSV.write(joinpath(CACHE_DIR, "combined_calibration.csv"), cal_combined)

@info "\n" * "="^70
@info "Combined multiplicity + interface PCA experiment complete!"
@info "="^70
