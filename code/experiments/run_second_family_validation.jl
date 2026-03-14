# ──────────────────────────────────────────────────────────────────────────────
# run_second_family_validation.jl
#
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
# For each family:
#   A. Identify a functional split (analogous to K/R at P1 for Kunitz)
#   B. Measure PCA-space Fisher separation
#   C. Run ρ sweep with calibration diagnostics
#   D. Compare gap magnitude to separation index
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
Analyze a protein family: load, split, measure separation, run ρ sweep.
Returns a results NamedTuple.
"""
function analyze_family(pfam_id::String, family_name::String;
                         split_function, split_label::String,
                         ρ_values=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
                         n_chains=20, seed=42)

    cache_dir = joinpath(CACHE_DIR, lowercase(family_name))
    mkpath(cache_dir)

    @info "\n" * "="^60
    @info "FAMILY: $family_name ($pfam_id)"
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

    # --- PCA-space separation ---
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

    # --- Hard curation baseline ---
    group_A_char = char_mat[group_A_idx, :]
    X̂_hard, pca_hard, _, _ = build_memory_matrix(group_A_char; pratio=0.95)
    pt_hard = find_entropy_inflection(X̂_hard)
    hard_seqs, _ = generate_sequences(X̂_hard, pca_hard, L;
        β=pt_hard.β_star, n_chains=n_chains, T=5000, seed=seed)

    hard_marker_frac = count(s -> length(s) >= marker_pos &&
        s[marker_pos] == char_mat[group_A_idx[1], marker_pos], hard_seqs) / length(hard_seqs)

    # More general: fraction matching ANY Group A marker residue at marker_pos
    group_A_residues = Set(char_mat[idx, marker_pos] for idx in group_A_idx
                           if !(char_mat[idx, marker_pos] in ('-', '.')))
    hard_marker_frac = count(s -> length(s) >= marker_pos && s[marker_pos] in group_A_residues,
                              hard_seqs) / length(hard_seqs)
    @info "  Hard curation: marker fraction = $(round(hard_marker_frac, digits=3))"

    # --- ρ sweep ---
    sweep_results = DataFrame(
        ρ=Float64[], f_eff=Float64[], f_obs=Float64[],
        attn_A=Float64[], diversity=Float64[],
    )

    log_r_cache = Dict{Float64, Vector{Float64}}()

    for ρ in ρ_values
        r = multiplicity_vector(K_total, group_A_idx; ρ=ρ)
        f_eff = effective_binder_fraction(r, group_A_idx)
        log_r = log.(r)
        pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)
        β = pt.β_star

        gen_seqs = String[]
        attn_A_vals = Float64[]

        for chain in 1:n_chains
            k = mod1(chain, K_total)
            ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)
            result = weighted_sample(X̂, ξ₀, 5000, r; β=β, α=0.01, seed=seed + chain)

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

        push!(sweep_results, (ρ, f_eff, f_obs, mean(attn_A_vals), div))
        @info "  ρ=$(round(ρ, digits=0)): f_eff=$(round(f_eff, digits=3)), f_obs=$(round(f_obs, digits=3)), attn=$(round(mean(attn_A_vals), digits=3))"
    end

    # calibration gap at high ρ
    high_ρ_row = last(sweep_results)
    cal_gap = high_ρ_row.f_eff - high_ρ_row.f_obs

    @info "\n  Summary for $family_name:"
    @info "    Separation index: $(round(separation_index, digits=3))"
    @info "    Hard curation marker frac: $(round(hard_marker_frac, digits=3))"
    @info "    Calibration gap at ρ=$(last(ρ_values)): $(round(cal_gap, digits=3))"

    return (family=family_name, pfam_id=pfam_id,
            K=K_total, L=L, K_A=K_A, K_B=K_B,
            d_pca=d, separation_index=separation_index,
            natural_frac=natural_frac,
            hard_curation_frac=hard_marker_frac,
            cal_gap=cal_gap,
            sweep=sweep_results,
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
    group_A = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K)
    group_B = findall(i -> !(char_mat[i, p1_pos] in ('K', 'R')) &&
                           !(char_mat[i, p1_pos] in ('-', '.')), 1:K)
    return group_A, group_B, p1_pos
end

# SH3: split by conserved tryptophan at the RT-Src loop
# SH3 domains have a conserved W in the binding groove; some variants have F or Y
function sh3_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    # find the most conserved W position (should be in the hydrophobic groove)
    trp_fracs = [count(i -> char_mat[i, j] == 'W', 1:K) /
                 max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K))
                 for j in 1:L]
    # find positions with W frequency between 0.2 and 0.8 (variable, not fixed)
    variable_W = findall(f -> 0.15 < f < 0.85, trp_fracs)
    if isempty(variable_W)
        # fallback: use the position with W fraction closest to 0.5
        target = 0.5
        dists = abs.(trp_fracs .- target)
        marker = argmin(dists)
    else
        marker = variable_W[argmax(trp_fracs[variable_W])]
    end
    group_A = findall(i -> char_mat[i, marker] == 'W', 1:K)
    group_B = findall(i -> char_mat[i, marker] != 'W' &&
                           !(char_mat[i, marker] in ('-', '.')), 1:K)
    return group_A, group_B, marker
end

# WW: split by first residue of the binding specificity loop
# Group I WW domains bind PPxY motifs; Group II/III bind different ligands
# The xW loop has a conserved position that distinguishes groups
function ww_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    # WW domains have two conserved W residues. The specificity is in the loop between them.
    # Split by the most variable position in the middle third of the alignment
    mid_start = max(1, L ÷ 3)
    mid_end = min(L, 2 * L ÷ 3)

    # find most variable position (highest entropy) in the middle
    best_pos = mid_start
    best_entropy = 0.0
    for j in mid_start:mid_end
        counts = Dict{Char, Int}()
        total = 0
        for i in 1:K
            c = char_mat[i, j]
            c in ('-', '.') && continue
            counts[c] = get(counts, c, 0) + 1
            total += 1
        end
        total < K ÷ 2 && continue  # skip high-gap columns
        H = 0.0
        for (_, n) in counts
            p = n / total
            H -= p * log(p)
        end
        if H > best_entropy
            best_entropy = H
            best_pos = j
        end
    end

    # split by most common residue at that position
    counts = Dict{Char, Int}()
    for i in 1:K
        c = char_mat[i, best_pos]
        c in ('-', '.') && continue
        counts[c] = get(counts, c, 0) + 1
    end
    sorted = sort(collect(counts), by=x -> -x[2])
    top_aa = sorted[1][1]

    group_A = findall(i -> char_mat[i, best_pos] == top_aa, 1:K)
    group_B = findall(i -> char_mat[i, best_pos] != top_aa &&
                           !(char_mat[i, best_pos] in ('-', '.')), 1:K)
    return group_A, group_B, best_pos
end

# ══════════════════════════════════════════════════════════════════════════════
# Run all families
# ══════════════════════════════════════════════════════════════════════════════

families = [
    ("PF00014", "Kunitz", kunitz_split, "K/R at P1 (trypsin binding)"),
    ("PF00018", "SH3", sh3_split, "W at binding groove (aromatic split)"),
    ("PF00397", "WW", ww_split, "Specificity loop residue"),
]

all_results = []
for (pfam_id, name, split_fn, label) in families
    result = analyze_family(pfam_id, name;
        split_function=split_fn, split_label=label,
        ρ_values=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0],
        n_chains=20, seed=42)
    if result !== nothing
        push!(all_results, result)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Cross-family comparison
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "CROSS-FAMILY COMPARISON"
@info "="^70

comparison = DataFrame(
    family=String[], K=Int[], d_pca=Int[],
    K_A=Int[], natural_frac=Float64[],
    separation_index=Float64[],
    hard_curation_frac=Float64[],
    cal_gap_rho500=Float64[],
)

for r in all_results
    push!(comparison, (r.family, r.K, r.d_pca, r.K_A, r.natural_frac,
                        r.separation_index, r.hard_curation_frac, r.cal_gap))
end

@info "\nCross-family comparison:"
show(stdout, comparison)
println()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
@info "\nGenerating figures"

# --- Figure 1: Calibration curves for each family ---
p1 = plot(size=(700, 500), margin=10Plots.mm,
    xlabel="Effective binder fraction (f_eff)",
    ylabel="Observed marker fraction (f_obs)",
    title="Multiplicity Calibration Across Families",
    legend=:topleft, ylim=(0, 1.05), xlim=(0, 1.05))
plot!(p1, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal", linewidth=1.5)

colors_fam = [:steelblue, :coral, :forestgreen, :purple, :orange]
for (i, r) in enumerate(all_results)
    plot!(p1, r.sweep.f_eff, r.sweep.f_obs,
        marker=:circle, linewidth=2, color=colors_fam[i],
        label="$(r.family) (sep=$(round(r.separation_index, digits=2)))", markersize=5)
end

savefig(p1, joinpath(FIG_DIR, "fig1_multi_family_calibration.pdf"))
savefig(p1, joinpath(FIG_DIR, "fig1_multi_family_calibration.png"))
@info "  Saved fig1_multi_family_calibration"

# --- Figure 2: Separation index vs calibration gap ---
if length(all_results) >= 2
    p2 = plot(size=(600, 450), margin=10Plots.mm,
        xlabel="Fisher separation index",
        ylabel="Calibration gap at ρ=500\n(f_eff - f_obs)",
        title="PCA Separation Predicts Calibration Gap",
        legend=false)

    sep_vals = [r.separation_index for r in all_results]
    gap_vals = [r.cal_gap for r in all_results]
    scatter!(p2, sep_vals, gap_vals,
        marker=:circle, markersize=10, color=:steelblue)

    # label points
    for r in all_results
        annotate!(p2, r.separation_index + 0.02, r.cal_gap + 0.01,
            text(r.family, 10, :left))
    end

    # trend line if enough points
    if length(all_results) >= 3
        # simple linear fit
        x = sep_vals
        y = gap_vals
        x_mean = mean(x)
        y_mean = mean(y)
        slope = sum((xi - x_mean) * (yi - y_mean) for (xi, yi) in zip(x, y)) /
                sum((xi - x_mean)^2 for xi in x)
        intercept = y_mean - slope * x_mean
        x_line = range(minimum(x) - 0.05, maximum(x) + 0.05, length=50)
        plot!(p2, x_line, intercept .+ slope .* x_line,
            linestyle=:dash, color=:coral, linewidth=2)
        r_sq = 1 - sum((yi - (intercept + slope * xi))^2 for (xi, yi) in zip(x, y)) /
                   sum((yi - y_mean)^2 for yi in y)
        annotate!(p2, minimum(x), maximum(y),
            text("R² = $(round(r_sq, digits=2))", 10, :left))
    end

    savefig(p2, joinpath(FIG_DIR, "fig2_separation_vs_gap.pdf"))
    savefig(p2, joinpath(FIG_DIR, "fig2_separation_vs_gap.png"))
    @info "  Saved fig2_separation_vs_gap"
end

# --- Figure 3: Per-family ρ sweep panels ---
n_fam = length(all_results)
p3 = plot(layout=(1, n_fam), size=(400 * n_fam, 400), margin=10Plots.mm)
for (i, r) in enumerate(all_results)
    plot!(p3[i], log10.(r.sweep.ρ), r.sweep.f_obs,
        marker=:circle, linewidth=2, color=:steelblue, label="f_obs",
        xlabel="log₁₀(ρ)", ylabel="Fraction", ylim=(0, 1.05),
        title="$(r.family) (sep=$(round(r.separation_index, digits=2)))")
    plot!(p3[i], log10.(r.sweep.ρ), r.sweep.attn_A,
        marker=:diamond, linewidth=2, color=:coral, label="attention")
    hline!(p3[i], [r.natural_frac], linestyle=:dot, color=:gray, label="natural")
    hline!(p3[i], [r.hard_curation_frac], linestyle=:dash, color=:forestgreen, label="hard cur.")
end

savefig(p3, joinpath(FIG_DIR, "fig3_per_family_rho.pdf"))
savefig(p3, joinpath(FIG_DIR, "fig3_per_family_rho.png"))
@info "  Saved fig3_per_family_rho"

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "multi_family_comparison.csv"), comparison)
for r in all_results
    CSV.write(joinpath(CACHE_DIR, lowercase(r.family), "multiplicity_sweep.csv"), r.sweep)
end

@info "\n" * "="^70
@info "Multi-family validation complete!"
@info "="^70
@info "\nPrediction: higher separation index → smaller calibration gap"
@info "Does the data support this? Check fig2_separation_vs_gap"
