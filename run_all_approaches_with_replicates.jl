# ──────────────────────────────────────────────────────────────────────────────
# run_all_approaches_with_replicates.jl
#
# Updated version with proper uncertainty quantification.
# Demonstrates all four binding approaches on Kunitz domains with replicates:
#   1. Curated memory matrix
#   2. Biased energy landscape
#   3. Post-hoc filtering
#   4. Interface-aware PCA
#
# Key improvements:
#   - Multiple independent replicates (n_reps = 5) for each approach
#   - Standard deviations computed across replicates
#   - Error bars in summary figures
#   - Both raw and aggregated data saved
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = joinpath(_SCRIPT_DIR, "code")
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const PFAM_ID = "PF00014"  # Kunitz/BPTI
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "approaches")
mkpath(CACHE_DIR)
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Load and prepare data (shared across all approaches)
# ══════════════════════════════════════════════════════════════════════════════
@info "Loading Kunitz domain alignment"
sto_file = download_pfam_seed(PFAM_ID; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)
stored_seqs = [String(char_mat[i, :]) for i in 1:K_total]

# identify P1 and interface positions
lys_fracs = [count(i -> char_mat[i, j] == 'K', 1:K_total) /
             max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K_total))
             for j in 1:L]
p1_pos = argmax(lys_fracs)
binding_loop = collect(max(1, p1_pos - 4):min(L, p1_pos + 4))
interface_positions = binding_loop

# identify strong binders (K/R at P1)
strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
strong_seqs = stored_seqs[strong_idx]
@info "  P1 position: $p1_pos | Interface: $interface_positions"
@info "  Strong binders: $(length(strong_idx)) / $K_total"

# build shared models
X̂_full, pca_full, L_full, d_full = build_memory_matrix(char_mat; pratio=0.95)
pt_full = find_entropy_inflection(X̂_full)
β_full = pt_full.β_star

result_curated = build_binder_memory(char_mat, strong_idx; pratio=0.95)
X̂_curated = result_curated.X̂
pca_curated = result_curated.pca_model
pt_curated = find_entropy_inflection(X̂_curated)
β_curated = pt_curated.β_star

iface_profile = build_interface_profile(strong_seqs, stored_seqs,
                                         interface_positions, pca_full, L)

@info "  Shared models built successfully"

# ══════════════════════════════════════════════════════════════════════════════
# Experiment parameters
# ══════════════════════════════════════════════════════════════════════════════
n_reps = 5
λ_values = [0.05, 0.1, 0.2, 0.5]
weight_values = [2.0, 3.0, 5.0]

# DataFrame to store ALL individual replicate results
results_raw = DataFrame(
    approach=String[], replicate=Int[], parameter=String[],
    n_generated=Int[], p1_kr_frac=Float64[], mean_valid_frac=Float64[],
    kl_aa=Float64[], mean_novelty=Float64[], diversity=Float64[],
)

# Helper function to evaluate metrics
function evaluate_sequences(seqs, reference_seqs, p1_pos)
    n = length(seqs)
    n == 0 && return (0.0, 0.0, 0.0, 0.0, 0.0)

    # P1 K/R fraction
    p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n

    # Validity
    valid = mean(valid_residue_fraction.(seqs))

    # KL divergence
    kl = aa_composition_kl(seqs, reference_seqs)

    # Novelty (distance from input)
    seq_ids = [nearest_sequence_identity(s, reference_seqs) for s in seqs]
    novelty = 1.0 - mean(seq_ids)

    # Pairwise diversity (sample subset for speed)
    n_pairs = min(300, n * (n - 1) ÷ 2)
    pair_ids = Float64[]
    for _ in 1:n_pairs
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
    end
    diversity = 1.0 - mean(pair_ids)

    return (p1_kr, valid, kl, novelty, diversity)
end

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE: Standard SA on full family (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "BASELINE: Standard SA on full family (WITH REPLICATES)"
@info "="^70

for rep in 1:n_reps
    @info "  Baseline replicate $rep/$n_reps"
    seed = 10000 + rep

    seqs, pca_vecs = generate_sequences(X̂_full, pca_full, L;
        β=β_full, n_chains=30, T=5000, seed=seed)

    metrics = evaluate_sequences(seqs, stored_seqs, p1_pos)
    push!(results_raw, ("Baseline: Standard SA", rep, "",
                        length(seqs), metrics...))
end

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 1: Curated Memory Matrix (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "APPROACH 1: Curated Memory Matrix (WITH REPLICATES)"
@info "="^70

for rep in 1:n_reps
    @info "  Approach 1 replicate $rep/$n_reps"
    seed = 20000 + rep

    seqs, pca_vecs = generate_sequences(X̂_curated, pca_curated, L;
        β=β_curated, n_chains=30, T=5000, seed=seed)

    metrics = evaluate_sequences(seqs, strong_seqs, p1_pos)
    push!(results_raw, ("Approach 1: Curated memory", rep, "",
                        length(seqs), metrics...))
end

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2: Biased Energy Landscape (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "APPROACH 2: Biased Energy Landscape (WITH REPLICATES)"
@info "="^70

for (idx, λ) in enumerate(λ_values)
    @info "  λ = $λ"
    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 30000 + (idx - 1) * n_reps + rep

        seqs, pca_vecs = generate_biased_sequences(X̂_full, pca_full, L, iface_profile;
            β=β_full, λ=λ, n_chains=20, T=5000,
            seed=seed)

        metrics = evaluate_sequences(seqs, stored_seqs, p1_pos)
        push!(results_raw, ("Approach 2: Biased energy", rep, "λ=$λ",
                            length(seqs), metrics...))
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 3: Post-hoc Filtering (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "APPROACH 3: Post-hoc Filtering (WITH REPLICATES)"
@info "="^70

for rep in 1:n_reps
    @info "  Approach 3 replicate $rep/$n_reps"
    seed = 40000 + rep

    # generate large pool
    pool_seqs, pool_pca = generate_sequences(X̂_full, pca_full, L;
        β=β_full, n_chains=60, T=5000, seed=seed)

    # filter and rank
    top_candidates = filter_and_rank(pool_seqs, pool_pca, iface_profile;
        stored_seqs=stored_seqs, X̂=X̂_full, β=β_full,
        top_k=50, min_novelty=0.02, min_valid_frac=0.8)

    if nrow(top_candidates) > 0
        filtered_seqs = top_candidates.sequence
        metrics = evaluate_sequences(filtered_seqs, stored_seqs, p1_pos)
        push!(results_raw, ("Approach 3: Post-hoc filter", rep, "",
                            length(filtered_seqs), metrics...))
    else
        # No sequences passed filter
        push!(results_raw, ("Approach 3: Post-hoc filter", rep, "",
                            0, 0.0, 0.0, 0.0, 0.0, 0.0))
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 4: Interface-Aware PCA (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "APPROACH 4: Interface-Aware PCA (WITH REPLICATES)"
@info "="^70

for (idx, w) in enumerate(weight_values)
    @info "  Interface weight = $w"

    # Build interface-weighted memory once per weight
    result_w = build_interface_weighted_memory(char_mat, interface_positions;
        pratio=0.95, weight=w)
    X̂_w = result_w.X̂
    pca_w = result_w.pca_model
    pt_w = find_entropy_inflection(X̂_w)
    β_w = pt_w.β_star
    d_w, K_w = size(X̂_w)

    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 50000 + (idx - 1) * n_reps + rep

        # generate sequences
        seqs_w = String[]
        pca_w_vecs = Vector{Float64}[]

        for chain in 1:20
            k = mod1(chain, K_w)
            ξ₀ = X̂_w[:, k] .+ 0.01 .* randn(d_w)
            res = sample(X̂_w, ξ₀, 5000; β=β_w, α=0.01,
                        seed=seed + chain)

            for t in 2000:100:5000
                ξ = res.Ξ[t + 1, :]
                seq = decode_weighted_sample(ξ, pca_w, L, interface_positions; weight=w)
                push!(seqs_w, seq)
                push!(pca_w_vecs, ξ)
            end
        end

        metrics = evaluate_sequences(seqs_w, stored_seqs, p1_pos)
        push!(results_raw, ("Approach 4: Interface PCA", rep, "weight=$w",
                            length(seqs_w), metrics...))
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE RESULTS WITH UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "AGGREGATING RESULTS WITH UNCERTAINTY"
@info "="^70

# Create combined approach+parameter identifier for grouping
results_raw.approach_param = [
    isempty(row.parameter) ? row.approach : "$(row.approach) ($(row.parameter))"
    for row in eachrow(results_raw)
]

# Aggregate statistics across replicates
results_agg = combine(groupby(results_raw, :approach_param),
    :n_generated => mean => :n_generated,  # Approach 3 (post-hoc filter) can produce variable n_generated
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :mean_valid_frac => mean => :valid_mean,
    :mean_valid_frac => std => :valid_std,
    :kl_aa => mean => :kl_mean,
    :kl_aa => std => :kl_std,
    :mean_novelty => mean => :novelty_mean,
    :mean_novelty => std => :novelty_std,
    :diversity => mean => :diversity_mean,
    :diversity => std => :diversity_std,
)

@info "\nResults with uncertainty:"
show(stdout, results_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURES WITH ERROR BARS
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "Generating summary figures with error bars"
@info "="^70

# Figure 1: P1 K/R fraction comparison with error bars
p1 = bar(1:nrow(results_agg), results_agg.p1_kr_mean,
    yerror=results_agg.p1_kr_std,
    ylabel="Fraction K/R at P1",
    title="Binding Phenotype Inheritance (All Approaches)",
    legend=false, xticks=(1:nrow(results_agg), results_agg.approach_param),
    rotation=45, bar_width=0.7, ylim=(0, 1.1),
    size=(1200, 600), margin=20Plots.mm, color=:steelblue)

# Add reference lines
hline!(p1, [count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), stored_seqs) / length(stored_seqs)],
       linestyle=:dash, color=:red, linewidth=2, label="Full family")
hline!(p1, [count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), strong_seqs) / length(strong_seqs)],
       linestyle=:dash, color=:green, linewidth=2, label="Strong binders")

savefig(p1, joinpath(FIG_DIR, "all_approaches_p1_with_errors.pdf"))
savefig(p1, joinpath(FIG_DIR, "all_approaches_p1_with_errors.png"))
@info "  Saved all_approaches_p1_with_errors"

# Figure 2: Multi-metric comparison
p2 = plot(layout=(2, 2), size=(1200, 1000), margin=15Plots.mm)

# P1 K/R fraction
bar!(p2[1], 1:nrow(results_agg), results_agg.p1_kr_mean,
    yerror=results_agg.p1_kr_std,
    ylabel="P1 K/R fraction", title="Phenotype Fidelity",
    xticks=(1:nrow(results_agg), [split(x, " ")[end-1:end] |> x -> join(x, " ") for x in results_agg.approach_param]),
    rotation=45, color=:steelblue, legend=false)

# Validity
bar!(p2[2], 1:nrow(results_agg), results_agg.valid_mean,
    yerror=results_agg.valid_std,
    ylabel="Valid residue fraction", title="Sequence Validity",
    xticks=(1:nrow(results_agg), [split(x, " ")[end-1:end] |> x -> join(x, " ") for x in results_agg.approach_param]),
    rotation=45, color=:coral, legend=false, ylim=(0.8, 1.05))

# Diversity
bar!(p2[3], 1:nrow(results_agg), results_agg.diversity_mean,
    yerror=results_agg.diversity_std,
    ylabel="Pairwise diversity", title="Sequence Diversity",
    xticks=(1:nrow(results_agg), [split(x, " ")[end-1:end] |> x -> join(x, " ") for x in results_agg.approach_param]),
    rotation=45, color=:forestgreen, legend=false)

# KL divergence
bar!(p2[4], 1:nrow(results_agg), results_agg.kl_mean,
    yerror=results_agg.kl_std,
    ylabel="KL(AA composition)", title="Composition Divergence",
    xticks=(1:nrow(results_agg), [split(x, " ")[end-1:end] |> x -> join(x, " ") for x in results_agg.approach_param]),
    rotation=45, color=:purple, legend=false)

savefig(p2, joinpath(FIG_DIR, "all_approaches_multimetric_with_errors.pdf"))
savefig(p2, joinpath(FIG_DIR, "all_approaches_multimetric_with_errors.png"))
@info "  Saved all_approaches_multimetric_with_errors"

# Figure 3: Fidelity vs Diversity scatter plot with error bars
p3 = plot(size=(700, 500), margin=15Plots.mm,
    xlabel="Pairwise sequence diversity",
    ylabel="P1 K/R fraction (phenotype fidelity)",
    title="Fidelity–Diversity Trade-off\n(with uncertainty bounds)",
    legend=:bottomleft)

scatter!(p3, results_agg.diversity_mean, results_agg.p1_kr_mean,
        xerror=results_agg.diversity_std,
        yerror=results_agg.p1_kr_std,
        marker=:circle, markersize=8, color=:steelblue,
        label="All approaches ± SD")

# Annotate points
for (i, row) in enumerate(eachrow(results_agg))
    approach_short = split(row.approach_param, ":")[1] |> x -> split(x, " ")[end]
    annotate!(p3, row.diversity_mean + 0.002, row.p1_kr_mean - 0.03,
        text(approach_short, 7, :left))
end

savefig(p3, joinpath(FIG_DIR, "all_approaches_fidelity_diversity_with_errors.pdf"))
savefig(p3, joinpath(FIG_DIR, "all_approaches_fidelity_diversity_with_errors.png"))
@info "  Saved all_approaches_fidelity_diversity_with_errors"

# ══════════════════════════════════════════════════════════════════════════════
# Save results (both raw and aggregated)
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "approach_comparison_raw_replicates.csv"), results_raw)
CSV.write(joinpath(CACHE_DIR, "approach_comparison_aggregated.csv"), results_agg)

@info "\n" * "="^70
@info "All approaches experiment complete (WITH UNCERTAINTY)!"
@info "="^70
@info "\nKey improvements:"
@info "  1. Each approach/parameter run with $n_reps independent replicates"
@info "  2. Standard deviations computed across replicates"
@info "  3. All figures include error bars"
@info "  4. Both raw and aggregated data saved for analysis"
@info "  5. Multi-metric evaluation with uncertainty bounds"