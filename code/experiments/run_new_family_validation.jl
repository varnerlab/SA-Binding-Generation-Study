# ──────────────────────────────────────────────────────────────────────────────
# run_new_family_validation.jl
#
# Extend the cross-family S–Δ regression with two additional Pfam families:
#   4. Homeobox (PF00046) — Gln at position 50 (TAAT-binding) vs other specificities
#   5. Forkhead (PF00250) — H/N at recognition helix H3 (base-specific contacts) vs other
#
# Also re-runs the original three families so all five are analyzed with
# identical pipeline settings and saved to a single comparison CSV.
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const CACHE_DIR = joinpath(_CODE_DIR, "data")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "multi_family_5")
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Reuse the generic analyze_family() from run_second_family_validation.jl
# (copy the function here so this script is self-contained)
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

    # fraction matching ANY Group A marker residue at marker_pos
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
        @info "  ρ=$(round(ρ, digits=0)): f_eff=$(round(f_eff, digits=3)), f_obs=$(round(f_obs, digits=3)), attn=$(round(mean(attn_A_vals), digits=3)), div=$(round(div, digits=3))"
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

# --- Original three families ---

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

# SH3: split by conserved tryptophan at the binding groove
function sh3_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    trp_fracs = [count(i -> char_mat[i, j] == 'W', 1:K) /
                 max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K))
                 for j in 1:L]
    variable_W = findall(f -> 0.15 < f < 0.85, trp_fracs)
    if isempty(variable_W)
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

# WW: split by specificity-loop residue (highest entropy in middle third)
function ww_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    mid_start = max(1, L ÷ 3)
    mid_end = min(L, 2 * L ÷ 3)

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
        total < K ÷ 2 && continue
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

# --- New families ---

# Homeobox: Gln at position 50 in the recognition helix (TAAT-binding)
# Position 50 is in the conserved WFxNRR motif — the most conserved Gln
# in the C-terminal third of the alignment (helix 3).
# Group A = Gln (Class I/typical homeodomains), Group B = non-Gln (POU, Bicoid, CUT, TALE)
function homeobox_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    # Position 50 maps to the most conserved Q in the C-terminal half of the domain.
    # Find the column with highest Gln frequency in the last third of the alignment.
    last_third_start = max(1, 2 * L ÷ 3)
    gln_fracs = zeros(L)
    for j in last_third_start:L
        total = count(i -> !(char_mat[i, j] in ('-', '.')), 1:K)
        total == 0 && continue
        gln_fracs[j] = count(i -> char_mat[i, j] == 'Q', 1:K) / total
    end
    marker = argmax(gln_fracs)
    @info "  Homeobox: position 50 marker at alignment column $marker (Q fraction = $(round(gln_fracs[marker], digits=3)))"

    group_A = findall(i -> char_mat[i, marker] == 'Q', 1:K)
    group_B = findall(i -> char_mat[i, marker] != 'Q' &&
                           !(char_mat[i, marker] in ('-', '.')), 1:K)
    return group_A, group_B, marker
end

# Forkhead: H or N at recognition helix H3 position 7 (base-specific DNA contacts)
# Group A = His or Asn (H-bond donor/acceptor, base-specific contacts in major groove)
# Group B = other residues (Arg, Lys, Thr, Ser — charged/polar, altered specificity)
# The marker column is the position with the highest combined H+N frequency
# in the middle third of the alignment (where helix H3 sits).
function forkhead_split(char_mat, stored_seqs)
    K, L = size(char_mat)
    # Helix H3 is roughly in the middle third of the ~100-residue domain.
    mid_start = max(1, L ÷ 3)
    mid_end = min(L, 2 * L ÷ 3)

    hn_fracs = zeros(L)
    for j in mid_start:mid_end
        total = count(i -> !(char_mat[i, j] in ('-', '.')), 1:K)
        total == 0 && continue
        hn_count = count(i -> char_mat[i, j] in ('H', 'N'), 1:K)
        hn_fracs[j] = hn_count / total
    end

    # We want the column closest to 50% H+N (best binary split), not the highest.
    # Filter to columns with H+N fraction between 0.3 and 0.7 for a balanced split.
    candidates = findall(j -> 0.3 < hn_fracs[j] < 0.7, mid_start:mid_end) .+ (mid_start - 1)
    if !isempty(candidates)
        # Among balanced candidates, pick the one closest to 0.5
        marker = candidates[argmin(abs.(hn_fracs[candidates] .- 0.5))]
    else
        # Fallback: highest H+N frequency in the middle third
        marker = argmax(hn_fracs)
    end
    @info "  Forkhead: H3 marker at alignment column $marker (H+N fraction = $(round(hn_fracs[marker], digits=3)))"

    group_A = findall(i -> char_mat[i, marker] in ('H', 'N'), 1:K)
    group_B = findall(i -> !(char_mat[i, marker] in ('H', 'N')) &&
                           !(char_mat[i, marker] in ('-', '.')), 1:K)
    return group_A, group_B, marker
end

# ══════════════════════════════════════════════════════════════════════════════
# Run all five families
# ══════════════════════════════════════════════════════════════════════════════

families = [
    ("PF00014", "Kunitz",    kunitz_split,    "K/R at P1 (trypsin binding)"),
    ("PF00018", "SH3",       sh3_split,       "W at binding groove"),
    ("PF00397", "WW",        ww_split,        "Specificity loop residue"),
    ("PF00046", "Homeobox",  homeobox_split,  "Q at position 50 (TAAT-binding)"),
    ("PF00250", "Forkhead",  forkhead_split,  "H/N at H3 recognition helix"),
]

all_results = []
for (pfam_id, name, split_fn, label) in families
    result = analyze_family(pfam_id, name;
        split_function=split_fn, split_label=label,
        ρ_values=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
        n_chains=30, seed=42)
    if result !== nothing
        push!(all_results, result)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Cross-family comparison
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "CROSS-FAMILY COMPARISON (5 Pfam families)"
@info "="^70

comparison = DataFrame(
    family=String[], pfam_id=String[], K=Int[], L=Int[], d_pca=Int[],
    K_A=Int[], K_B=Int[], natural_frac=Float64[],
    separation_index=Float64[],
    hard_curation_frac=Float64[],
    cal_gap_rho500=Float64[],
)

for r in all_results
    # Get gap at ρ=500 specifically
    row_500 = filter(row -> row.ρ == 500.0, r.sweep)
    gap_500 = isempty(row_500) ? r.cal_gap : first(row_500).f_eff - first(row_500).f_obs
    push!(comparison, (r.family, r.pfam_id, r.K, r.L, r.d_pca,
                        r.K_A, r.K_B, r.natural_frac,
                        r.separation_index, r.hard_curation_frac, gap_500))
end

@info "\nCross-family comparison:"
show(stdout, comparison)
println()

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "multi_family_comparison_5fam.csv"), comparison)
for r in all_results
    CSV.write(joinpath(CACHE_DIR, lowercase(r.family), "multiplicity_sweep.csv"), r.sweep)
end

# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════
@info "\nGenerating figures"

# --- Figure 1: Calibration curves for each family ---
p1 = plot(size=(800, 550), margin=10Plots.mm,
    xlabel="Effective designated fraction (f_eff)",
    ylabel="Observed marker fraction (f_obs)",
    title="Multiplicity Calibration: 5 Pfam Families",
    legend=:topleft, ylim=(0, 1.05), xlim=(0, 1.05))
plot!(p1, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal", linewidth=1.5)

colors_fam = [:steelblue, :coral, :forestgreen, :purple, :goldenrod]
for (i, r) in enumerate(all_results)
    plot!(p1, r.sweep.f_eff, r.sweep.f_obs,
        marker=:circle, linewidth=2, color=colors_fam[i],
        label="$(r.family) (S=$(round(r.separation_index, digits=2)))", markersize=5)
end

savefig(p1, joinpath(FIG_DIR, "fig_calibration_5fam.pdf"))
@info "  Saved fig_calibration_5fam"

# --- Figure 2: Separation index vs calibration gap (the key regression) ---
p2 = plot(size=(600, 450), margin=10Plots.mm,
    xlabel="Fisher separation index (S)",
    ylabel="Calibration gap (Δ) at ρ=500",
    title="PCA Separation Predicts Calibration Gap",
    legend=false)

sep_vals = [r.separation_index for r in all_results]
gap_vals = [Float64(comparison[i, :cal_gap_rho500]) for i in 1:nrow(comparison)]
scatter!(p2, sep_vals, gap_vals,
    marker=:circle, markersize=10, color=:steelblue)

for (i, r) in enumerate(all_results)
    annotate!(p2, r.separation_index + 0.02, gap_vals[i] + 0.01,
        text(r.family, 10, :left))
end

# Linear fit
if length(all_results) >= 3
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
    @info "  Linear fit: Δ ≈ $(round(intercept, digits=2)) + $(round(slope, digits=2))·S, R² = $(round(r_sq, digits=3))"
    annotate!(p2, minimum(x), maximum(y),
        text("Δ ≈ $(round(intercept, digits=2)) + $(round(slope, digits=2))·S\nR² = $(round(r_sq, digits=3))", 10, :left))
end

savefig(p2, joinpath(FIG_DIR, "fig_separation_vs_gap_5fam.pdf"))
@info "  Saved fig_separation_vs_gap_5fam"

# --- Figure 3: Per-family ρ sweep panels ---
n_fam = length(all_results)
p3 = plot(layout=(1, n_fam), size=(350 * n_fam, 400), margin=8Plots.mm)
for (i, r) in enumerate(all_results)
    plot!(p3[i], log10.(r.sweep.ρ), r.sweep.f_obs,
        marker=:circle, linewidth=2, color=:steelblue, label="f_obs",
        xlabel="log₁₀(ρ)", ylabel="Fraction", ylim=(0, 1.05),
        title="$(r.family)\n(S=$(round(r.separation_index, digits=2)))")
    plot!(p3[i], log10.(r.sweep.ρ), r.sweep.attn_A,
        marker=:diamond, linewidth=2, color=:coral, label="attention")
    hline!(p3[i], [r.natural_frac], linestyle=:dot, color=:gray, label="natural")
    hline!(p3[i], [r.hard_curation_frac], linestyle=:dash, color=:forestgreen, label="hard cur.")
end

savefig(p3, joinpath(FIG_DIR, "fig_per_family_rho_5fam.pdf"))
@info "  Saved fig_per_family_rho_5fam"

# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "5-family validation complete!"
@info "="^70
@info "\nResults saved to: $(joinpath(CACHE_DIR, "multi_family_comparison_5fam.csv"))"
@info "Figures saved to: $FIG_DIR"
