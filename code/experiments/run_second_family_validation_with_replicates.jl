# ──────────────────────────────────────────────────────────────────────────────
# run_second_family_validation_with_replicates.jl
#
# Updated version with proper uncertainty quantification.
# Test the multiplicity-weighted Hopfield theory on multiple protein families
# to validate that the calibration gap depends on PCA-space separation.
#
# Prediction: families where functional subsets cluster tightly in PCA space
# will have smaller calibration gaps than families where they're interleaved.
#
# Families tested:
#   1. Kunitz (PF00014) — 32/99 strong binders, P1 residue defines binding
#   2. SH3 domains (PF00018) — conserved binding groove, split by aromatic content
#   3. WW domains (PF00397) — small, two binding classes (Group I vs Group II/III)
#
# Key improvements:
#   - Multiple independent replicates (n_reps = 5) for each ρ sweep
#   - Standard deviations computed for all calibration measurements
#   - Error bars in calibration gap analysis
#   - Both raw and aggregated data saved
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const CACHE_DIR = joinpath(_CODE_DIR, "data")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "multi_family")
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

"""
Analyze a protein family with replicates: load, split, measure separation, run ρ sweep.
Returns a results NamedTuple with uncertainty quantification.
"""
function analyze_family_with_replicates(pfam_id::String, family_name::String;
                         split_function, split_label::String,
                         ρ_values=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
                         n_chains=20, n_reps=5)

    cache_dir = joinpath(CACHE_DIR, lowercase(family_name))
    mkpath(cache_dir)

    @info "\n" * "="^60
    @info "FAMILY: $family_name ($pfam_id) - WITH REPLICATES"
    @info "="^60

    # --- Load and clean alignment ---
    sto_file = download_pfam_seed(pfam_id; cache_dir=cache_dir)
    raw_seqs = parse_stockholm(sto_file)
    char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
    K_total, L = size(char_mat)
    stored_seqs = [String(char_mat[i, :]) for i in 1:K_total]
    @info "  Alignment: $K_total × $L"

    # --- Apply the functional split ---
    group_A_idx, group_B_idx, marker_pos = split_function(char_mat, stored_seqs)
    K_A = length(group_A_idx)
    K_B = length(group_B_idx)
    natural_frac = K_A / K_total
    @info "  Split: $(split_label)"
    @info "    Group A: $K_A | Group B: $K_B | Natural fraction A: $(round(natural_frac, digits=3))"

    if K_A < 3 || K_B < 3
        @warn "  Too few sequences in one group, skipping family"
        return nothing
    end

    # --- Build memory matrix ---
    X̂, pca_model, _, _ = build_memory_matrix(char_mat; pratio=0.95)
    d = size(X̂, 1)
    @info "  Memory: $d × $K_total"

    # --- PCA-space separation (deterministic) ---
    within_A = Float64[]
    for i in group_A_idx, j in group_A_idx
        i >= j && continue
        push!(within_A, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
    end

    within_B = Float64[]
    for i in group_B_idx, j in group_B_idx
        i >= j && continue
        push!(within_B, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
    end

    between = Float64[]
    for i in group_A_idx, j in group_B_idx
        push!(between, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
    end

    mean_within = (mean(within_A) * length(within_A) + mean(within_B) * length(within_B)) /
                  (length(within_A) + length(within_B))
    std_within = sqrt((var(within_A) * length(within_A) + var(within_B) * length(within_B)) /
                      (length(within_A) + length(within_B)))
    separation_index = (mean_within - mean(between)) / (0.5 * (std_within + std(between)))

    @info "  PCA separation:"
    @info "    Within-A cosine: $(round(mean(within_A), digits=4)) ± $(round(std(within_A), digits=4))"
    @info "    Within-B cosine: $(round(mean(within_B), digits=4)) ± $(round(std(within_B), digits=4))"
    @info "    Between cosine:  $(round(mean(between), digits=4)) ± $(round(std(between), digits=4))"
    @info "    Fisher separation index: $(round(separation_index, digits=3))"

    # --- Hard curation baseline with replicates ---
    group_A_char = char_mat[group_A_idx, :]
    X̂_hard, pca_hard, _, _ = build_memory_matrix(group_A_char; pratio=0.95)
    pt_hard = find_entropy_inflection(X̂_hard)

    group_A_residues = Set(char_mat[idx, marker_pos] for idx in group_A_idx
                           if !(char_mat[idx, marker_pos] in ('-', '.')))

    hard_marker_fracs = Float64[]
    for rep in 1:n_reps
        seed = 10000 + rep
        hard_seqs, _ = generate_sequences(X̂_hard, pca_hard, L;
            β=pt_hard.β_star, n_chains=n_chains, T=5000, seed=seed)

        hard_marker_frac = count(s -> length(s) >= marker_pos && s[marker_pos] in group_A_residues,
                                  hard_seqs) / length(hard_seqs)
        push!(hard_marker_fracs, hard_marker_frac)
    end

    hard_curation_mean = mean(hard_marker_fracs)
    hard_curation_std = std(hard_marker_fracs)
    @info "  Hard curation: marker fraction = $(round(hard_curation_mean, digits=3)) ± $(round(hard_curation_std, digits=3))"

    # --- ρ sweep with replicates ---
    sweep_results_raw = DataFrame(
        ρ=Float64[], replicate=Int[], f_eff=Float64[], f_obs=Float64[],
        attn_A=Float64[], diversity=Float64[]
    )

    for (idx, ρ) in enumerate(ρ_values)
        @info "  ρ = $ρ"
        r = multiplicity_vector(K_total, group_A_idx; ρ=ρ)
        f_eff = effective_binder_fraction(r, group_A_idx)
        log_r = log.(r)
        pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)
        β = pt.β_star

        for rep in 1:n_reps
            @info "    replicate $rep/$n_reps"
            seed = 20000 + (idx - 1) * n_reps + rep
            Random.seed!(seed)

            gen_seqs = String[]
            attn_A_vals = Float64[]

            for chain in 1:n_chains
                k = mod1(chain, K_total)
                ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)
                result = weighted_sample(X̂, ξ₀, 5000, r; β=β, α=0.01,
                    seed=seed + chain)

                for t in 2000:100:5000
                    ξ = result.Ξ[t + 1, :]
                    push!(gen_seqs, decode_sample(ξ, pca_model, L))

                    logits = β .* (X̂' * ξ) .+ log_r
                    attn = NNlib.softmax(logits)
                    push!(attn_A_vals, sum(attn[idx] for idx in group_A_idx))
                end
            end

            n = length(gen_seqs)
            f_obs = count(s -> length(s) >= marker_pos && s[marker_pos] in group_A_residues,
                           gen_seqs) / n

            pair_ids = Float64[]
            for _ in 1:min(300, n * (n - 1) ÷ 2)
                i, j = rand(1:n), rand(1:n)
                while i == j; j = rand(1:n); end
                push!(pair_ids, sequence_identity(gen_seqs[i], gen_seqs[j]))
            end
            div = 1.0 - mean(pair_ids)

            push!(sweep_results_raw, (ρ, rep, f_eff, f_obs, mean(attn_A_vals), div))
        end
    end

    # Aggregate ρ sweep results
    sweep_results_agg = combine(groupby(sweep_results_raw, :ρ),
        :f_eff => first => :f_eff,  # deterministic
        :f_obs => mean => :f_obs_mean,
        :f_obs => std => :f_obs_std,
        :attn_A => mean => :attn_A_mean,
        :attn_A => std => :attn_A_std,
        :diversity => mean => :diversity_mean,
        :diversity => std => :diversity_std
    )

    # calibration gap at high ρ with uncertainty
    high_ρ_data = filter(row -> row.ρ == last(ρ_values), sweep_results_raw)
    cal_gaps = high_ρ_data.f_eff .- high_ρ_data.f_obs
    cal_gap_mean = mean(cal_gaps)
    cal_gap_std = std(cal_gaps)

    @info "\n  Summary for $family_name:"
    @info "    Separation index: $(round(separation_index, digits=3))"
    @info "    Hard curation marker frac: $(round(hard_curation_mean, digits=3)) ± $(round(hard_curation_std, digits=3))"
    @info "    Calibration gap at ρ=$(last(ρ_values)): $(round(cal_gap_mean, digits=3)) ± $(round(cal_gap_std, digits=3))"

    return (family=family_name, pfam_id=pfam_id,
            K=K_total, L=L, K_A=K_A, K_B=K_B,
            d_pca=d, separation_index=separation_index,
            natural_frac=natural_frac,
            hard_curation_mean=hard_curation_mean,
            hard_curation_std=hard_curation_std,
            cal_gap_mean=cal_gap_mean,
            cal_gap_std=cal_gap_std,
            sweep_raw=sweep_results_raw,
            sweep_agg=sweep_results_agg,
            marker_pos=marker_pos,
            group_A_residues=group_A_residues)
end

# ══════════════════════════════════════════════════════════════════════════════
# Family-specific split functions
# ══════════════════════════════════════════════════════════════════════════════

# Kunitz: K/R at P1 (strong trypsin inhibitors)
function kunitz_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    lys_fracs = [count(i -> char_mat[i, j] == 'K', 1:K) /
                 max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K))
                 for j in 1:L]
    p1_pos = argmax(lys_fracs)
    strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K)
    weak_idx = findall(i -> !(char_mat[i, p1_pos] in ('K', 'R')) &&
                             !(char_mat[i, p1_pos] in ('-', '.')), 1:K)
    return strong_idx, weak_idx, p1_pos
end

# SH3: W-rich vs W-poor (aromatic content in binding groove)
function sh3_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    # look for highly conserved W position in binding groove
    w_fracs = [count(i -> char_mat[i, j] == 'W', 1:K) /
               max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K))
               for j in 1:L]
    best_pos = argmax(w_fracs)

    w_rich = findall(i -> char_mat[i, best_pos] == 'W', 1:K)
    w_poor = findall(i -> char_mat[i, best_pos] != 'W' &&
                          !(char_mat[i, best_pos] in ('-', '.')), 1:K)
    return w_rich, w_poor, best_pos
end

# WW: Group I vs II/III specificity (simplified by conserved residue)
function ww_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    # find the most informative position by entropy
    entropies = Float64[]
    for j in 1:L
        counts = Dict{Char, Int}()
        for i in 1:K
            c = char_mat[i, j]
            c in ('-', '.') && continue
            counts[c] = get(counts, c, 0) + 1
        end
        if length(counts) <= 1
            push!(entropies, 0.0)
        else
            total = sum(values(counts))
            entropy = -sum(p/total * log(p/total) for p in values(counts))
            push!(entropies, entropy)
        end
    end

    # find position with intermediate entropy (good for splits)
    target_entropy = log(2) * 0.8  # ~80% of max entropy for binary split
    best_pos = argmin(abs.(entropies .- target_entropy))

    # split by most common residue at that position
    residue_counts = StatsBase.countmap([char_mat[i, best_pos] for i in 1:K
                                       if !(char_mat[i, best_pos] in ('-', '.'))])
    if isempty(residue_counts)
        return Int[], collect(1:K), 1
    end

    most_common = first(sort(collect(residue_counts), by=x -> -x[2]))[1]
    group_A = findall(i -> char_mat[i, best_pos] == most_common, 1:K)
    group_B = findall(i -> char_mat[i, best_pos] != most_common &&
                           !(char_mat[i, best_pos] in ('-', '.')), 1:K)
    return group_A, group_B, best_pos
end

# ══════════════════════════════════════════════════════════════════════════════
# Run all families with replicates
# ══════════════════════════════════════════════════════════════════════════════

families = [
    ("PF00014", "Kunitz", kunitz_split, "K/R at P1 (trypsin binding)"),
    ("PF00018", "SH3", sh3_split, "W at binding groove (aromatic split)"),
    ("PF00397", "WW", ww_split, "Specificity loop residue"),
]

all_results = []
for (pfam_id, name, split_fn, label) in families
    result = analyze_family_with_replicates(pfam_id, name;
        split_function=split_fn, split_label=label,
        ρ_values=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0],
        n_chains=20, n_reps=5)
    if result !== nothing
        push!(all_results, result)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Cross-family comparison with uncertainty
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "CROSS-FAMILY COMPARISON (WITH UNCERTAINTY)"
@info "="^70

comparison = DataFrame(
    family=String[], K=Int[], d_pca=Int[],
    K_A=Int[], natural_frac=Float64[],
    separation_index=Float64[],
    hard_curation_mean=Float64[], hard_curation_std=Float64[],
    cal_gap_mean=Float64[], cal_gap_std=Float64[]
)

for r in all_results
    push!(comparison, (r.family, r.K, r.d_pca, r.K_A, r.natural_frac,
                        r.separation_index, r.hard_curation_mean, r.hard_curation_std,
                        r.cal_gap_mean, r.cal_gap_std))
end

@info "\nCross-family comparison with uncertainty:"
show(stdout, comparison)
println()

# ══════════════════════════════════════════════════════════════════════════════
# Figures with error bars
# ══════════════════════════════════════════════════════════════════════════════
@info "\nGenerating figures with error bars"

# --- Figure 1: Multi-family calibration curves ---
n_fam = length(all_results)
p1 = plot(layout=(1, n_fam), size=(400 * n_fam, 400), margin=10Plots.mm)

for (i, r) in enumerate(all_results)
    plot!(p1[i], log10.(r.sweep_agg.ρ), r.sweep_agg.f_obs_mean,
        ribbon=r.sweep_agg.f_obs_std, fillalpha=0.3,
        marker=:circle, linewidth=2, color=:steelblue, label="f_obs ± SD",
        xlabel="log₁₀(ρ)", ylabel="Marker fraction", ylim=(0, 1.05),
        title="$(r.family)\n(sep=$(round(r.separation_index, digits=2)))")

    plot!(p1[i], log10.(r.sweep_agg.ρ), r.sweep_agg.attn_A_mean,
        ribbon=r.sweep_agg.attn_A_std, fillalpha=0.3,
        marker=:diamond, linewidth=2, color=:coral, label="attention ± SD")

    hline!(p1[i], [r.natural_frac], linestyle=:dot, color=:gray, linewidth=2, label="natural")
    hline!(p1[i], [r.hard_curation_mean], linestyle=:dash, color=:forestgreen, linewidth=2, label="hard curation")

    # Add error band for hard curation
    hline!(p1[i], [r.hard_curation_mean + r.hard_curation_std],
           linestyle=:dash, color=:forestgreen, alpha=0.3, linewidth=1, label="")
    hline!(p1[i], [r.hard_curation_mean - r.hard_curation_std],
           linestyle=:dash, color=:forestgreen, alpha=0.3, linewidth=1, label="")
end

savefig(p1, joinpath(FIG_DIR, "fig1_multi_family_calibration_with_errors.pdf"))
savefig(p1, joinpath(FIG_DIR, "fig1_multi_family_calibration_with_errors.png"))
@info "  Saved fig1_multi_family_calibration_with_errors"

# --- Figure 2: Separation index vs calibration gap with error bars ---
if length(all_results) >= 2
    p2 = plot(size=(600, 450), margin=10Plots.mm,
        xlabel="Fisher separation index",
        ylabel="Calibration gap at ρ=500\n(f_eff - f_obs)",
        title="PCA Separation Predicts Calibration Gap\n(with uncertainty bounds)",
        legend=:topright)

    sep_vals = [r.separation_index for r in all_results]
    gap_means = [r.cal_gap_mean for r in all_results]
    gap_stds = [r.cal_gap_std for r in all_results]

    scatter!(p2, sep_vals, gap_means,
            yerror=gap_stds,
            marker=:circle, markersize=10, color=:steelblue, label="Families ± SD")

    # label points
    for r in all_results
        annotate!(p2, r.separation_index + 0.02, r.cal_gap_mean + 0.01,
            text(r.family, 10, :left))
    end

    # trend line if enough points
    if length(all_results) >= 3
        # simple linear fit using mean values
        x = sep_vals
        y = gap_means
        x_mean = mean(x)
        y_mean = mean(y)
        slope = sum((xi - x_mean) * (yi - y_mean) for (xi, yi) in zip(x, y)) /
                sum((xi - x_mean)^2 for xi in x)
        intercept = y_mean - slope * x_mean
        x_line = range(minimum(x) - 0.05, maximum(x) + 0.05, length=50)
        plot!(p2, x_line, intercept .+ slope .* x_line,
            linestyle=:dash, color=:coral, linewidth=2, label="Linear fit")

        r_sq = 1 - sum((yi - (intercept + slope * xi))^2 for (xi, yi) in zip(x, y)) /
                   sum((yi - y_mean)^2 for yi in y)

        # Show correlation and uncertainty
        annotate!(p2, minimum(x), maximum(y) - 0.02,
            text("R² = $(round(r_sq, digits=2))\nSlope = $(round(slope, digits=2))", 10, :left))
    end

    savefig(p2, joinpath(FIG_DIR, "fig2_separation_vs_gap_with_errors.pdf"))
    savefig(p2, joinpath(FIG_DIR, "fig2_separation_vs_gap_with_errors.png"))
    @info "  Saved fig2_separation_vs_gap_with_errors"
end

# --- Figure 3: Detailed per-family comparison ---
p3 = plot(layout=(2, 2), size=(1000, 800), margin=12Plots.mm)

if length(all_results) >= 2
    # Plot 1: Separation index comparison
    bar!(p3[1], 1:length(all_results), [r.separation_index for r in all_results],
        ylabel="Fisher separation index", title="PCA Space Separation",
        xticks=(1:length(all_results), [r.family for r in all_results]),
        color=:steelblue, legend=false)

    # Plot 2: Calibration gap comparison with error bars
    bar!(p3[2], 1:length(all_results), [r.cal_gap_mean for r in all_results],
        yerror=[r.cal_gap_std for r in all_results],
        ylabel="Calibration gap", title="Calibration Gap (ρ=500)",
        xticks=(1:length(all_results), [r.family for r in all_results]),
        color=:coral, legend=false)

    # Plot 3: Hard curation performance with error bars
    bar!(p3[3], 1:length(all_results), [r.hard_curation_mean for r in all_results],
        yerror=[r.hard_curation_std for r in all_results],
        ylabel="Hard curation marker fraction", title="Hard Curation Performance",
        xticks=(1:length(all_results), [r.family for r in all_results]),
        color=:forestgreen, legend=false)

    # Plot 4: Family characteristics
    scatter!(p3[4], [r.K for r in all_results], [r.cal_gap_mean for r in all_results],
            yerror=[r.cal_gap_std for r in all_results],
            xlabel="Family size (K)", ylabel="Calibration gap",
            title="Gap vs Family Size", marker=:circle, markersize=8,
            color=:purple, legend=false)
end

savefig(p3, joinpath(FIG_DIR, "fig3_detailed_comparison_with_errors.pdf"))
savefig(p3, joinpath(FIG_DIR, "fig3_detailed_comparison_with_errors.png"))
@info "  Saved fig3_detailed_comparison_with_errors"

# ══════════════════════════════════════════════════════════════════════════════
# Save results (both raw and aggregated)
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "multi_family_comparison_with_uncertainty.csv"), comparison)

for r in all_results
    family_dir = joinpath(CACHE_DIR, lowercase(r.family))
    mkpath(family_dir)

    # Save raw replicate data
    CSV.write(joinpath(family_dir, "multiplicity_sweep_raw_replicates.csv"), r.sweep_raw)

    # Save aggregated data
    CSV.write(joinpath(family_dir, "multiplicity_sweep_aggregated.csv"), r.sweep_agg)
end

@info "\n" * "="^70
@info "Multi-family validation complete (WITH UNCERTAINTY)!"
@info "="^70
@info "\nKey improvements:"
@info "  1. Each family's ρ sweep run with 5 independent replicates"
@info "  2. Standard deviations computed for all calibration gaps"
@info "  3. Error bars in separation vs gap correlation"
@info "  4. Hard curation performance quantified with uncertainty"
@info "  5. Both raw and aggregated data saved per family"
@info "\nPrediction: higher separation index → smaller calibration gap (with uncertainty bounds)"

# Statistical test of separation-gap correlation
if length(all_results) >= 3
    sep_vals = [r.separation_index for r in all_results]
    gap_means = [r.cal_gap_mean for r in all_results]

    # Pearson correlation
    r_corr = cor(sep_vals, gap_means)

    @info "\nStatistical Analysis:"
    @info "  Pearson correlation (separation vs gap): $(round(r_corr, digits=3))"

    if abs(r_corr) > 0.5
        @info "  Strong correlation detected!"
    elseif abs(r_corr) > 0.3
        @info "  Moderate correlation detected."
    else
        @info "  Weak correlation - may need more families to establish trend."
    end
end
