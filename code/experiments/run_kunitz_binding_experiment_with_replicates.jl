# ──────────────────────────────────────────────────────────────────────────────
# run_kunitz_binding_experiment_with_replicates.jl
#
# Updated version with proper uncertainty quantification.
# The "cleanest follow-up experiment" from binding.md:
#   Take Kunitz domains (PF00014) with binding data.
#   Split into strong vs. weak binders.
#   Run SA on each subset with multiple replicates.
#   Show generated sequences inherit the binding phenotype of the input set.
#
# Key improvements:
#   - Multiple independent replicates (n_reps = 5) for each condition
#   - Standard deviations computed across replicates
#   - Error bars in all figures
#   - Both raw and aggregated data saved
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const PFAM_ID = "PF00014"  # Kunitz/BPTI family
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "kunitz")
mkpath(CACHE_DIR)
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Download and parse Kunitz seed alignment
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 1: Loading Kunitz domain alignment (PF00014)"
sto_file = download_pfam_seed(PFAM_ID; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)
stored_seqs = [String(char_mat[i, :]) for i in 1:K_total]
@info "  Kunitz family: $K_total sequences × $L positions"

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Define binding phenotype split
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 2: Splitting by binding phenotype (P1 residue)"

# Find the P1 position: look for the column with highest Lys frequency
lys_fracs = zeros(L)
for j in 1:L
    n_lys = count(i -> char_mat[i, j] == 'K', 1:K_total)
    n_valid = count(i -> char_mat[i, j] != '-' && char_mat[i, j] != '.', 1:K_total)
    lys_fracs[j] = n_valid > 0 ? n_lys / n_valid : 0.0
end

# P1 is the position with highest Lys enrichment (should be >30%)
p1_candidates = findall(f -> f > 0.2, lys_fracs)
if isempty(p1_candidates)
    p1_pos = argmax(lys_fracs)
else
    p1_pos = p1_candidates[argmax(lys_fracs[p1_candidates])]
end
@info "  P1 position identified: column $p1_pos (Lys fraction: $(round(lys_fracs[p1_pos], digits=2)))"

# Define the binding loop region around P1
binding_loop = collect(max(1, p1_pos - 4):min(L, p1_pos + 4))

# Split: strong binders have K or R at P1
strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
weak_idx = findall(i -> !(char_mat[i, p1_pos] in ('K', 'R')) &&
                         char_mat[i, p1_pos] != '-' && char_mat[i, p1_pos] != '.', 1:K_total)

@info "  Strong binders (K/R at P1): $(length(strong_idx)) sequences"
@info "  Weak/non-binders (other at P1): $(length(weak_idx)) sequences"

# show P1 residue distribution
p1_residues = [char_mat[i, p1_pos] for i in 1:K_total]
p1_counts = sort(collect(StatsBase.countmap(p1_residues)), by=x -> -x[2])
@info "  P1 residue distribution: $p1_counts"

strong_seqs = stored_seqs[strong_idx]
weak_seqs = stored_seqs[weak_idx]

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Build memory matrices for each subset
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 3: Building memory matrices"

# Full family (baseline)
X̂_all, pca_all, L_all, d_full_all = build_memory_matrix(char_mat; pratio=0.95)

# Strong binders only
strong_char = char_mat[strong_idx, :]
if length(strong_idx) >= 5
    X̂_strong, pca_strong, _, _ = build_memory_matrix(strong_char; pratio=0.95)
else
    @warn "Too few strong binders ($(length(strong_idx))), using full family PCA"
    X̂_strong, pca_strong = X̂_all, pca_all
end

# Weak binders only
weak_char = char_mat[weak_idx, :]
if length(weak_idx) >= 5
    X̂_weak, pca_weak, _, _ = build_memory_matrix(weak_char; pratio=0.95)
else
    @warn "Too few weak binders ($(length(weak_idx))), using full family PCA"
    X̂_weak, pca_weak = X̂_all, pca_all
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Find β* for each subset
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 4: Finding β* for each subset"

pt_all = find_entropy_inflection(X̂_all)
β_all = pt_all.β_star
@info "  Full family:    β* = $(round(β_all, digits=2))"

if length(strong_idx) >= 5
    pt_strong = find_entropy_inflection(X̂_strong)
    β_strong = pt_strong.β_star
else
    β_strong = β_all
end
@info "  Strong binders: β* = $(round(β_strong, digits=2))"

if length(weak_idx) >= 5
    pt_weak = find_entropy_inflection(X̂_weak)
    β_weak = pt_weak.β_star
else
    β_weak = β_all
end
@info "  Weak binders:   β* = $(round(β_weak, digits=2))"

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Generate sequences with replicates
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 5: Generating sequences with replicates"

n_reps = 5

# DataFrame to store all individual replicate results
results_raw = DataFrame(
    condition=String[], replicate=Int[],
    n_generated=Int[], p1_kr_frac=Float64[], mean_valid_frac=Float64[],
    kl_aa=Float64[], mean_novelty=Float64[], mean_seqid=Float64[],
    diversity=Float64[]
)

# Helper function for metrics evaluation
function evaluate_binding_metrics(seqs, reference_seqs, pca_vecs, X̂, p1_pos)
    n = length(seqs)
    n == 0 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # P1 K/R fraction
    p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n

    # Validity
    valid = mean(valid_residue_fraction.(seqs))

    # KL divergence
    kl = aa_composition_kl(seqs, reference_seqs)

    # Novelty using PCA vectors
    novelties = [sample_novelty(v, X̂) for v in pca_vecs]
    mean_novelty = mean(novelties)

    # Sequence identity
    seq_ids = [nearest_sequence_identity(s, reference_seqs) for s in seqs]
    mean_seqid = mean(seq_ids)

    # Pairwise diversity (sample subset for speed)
    n_pairs = min(300, n * (n - 1) ÷ 2)
    pair_ids = Float64[]
    for _ in 1:n_pairs
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
    end
    diversity = 1.0 - mean(pair_ids)

    return (p1_kr, valid, kl, mean_novelty, mean_seqid, diversity)
end

# Generate from full family with replicates
@info "  Generating from full family"
for rep in 1:n_reps
    @info "    replicate $rep/$n_reps"
    seed = 10000 + rep

    seqs, pca_vecs = generate_sequences(X̂_all, pca_all, L;
        β=β_all, n_chains=30, T=5000, seed=seed)

    metrics = evaluate_binding_metrics(seqs, stored_seqs, pca_vecs, X̂_all, p1_pos)
    push!(results_raw, ("Full family", rep, length(seqs), metrics...))
end

# Generate from strong binders with replicates
if length(strong_idx) >= 5
    @info "  Generating from strong binders"
    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 20000 + rep

        seqs, pca_vecs = generate_sequences(X̂_strong, pca_strong, L;
            β=β_strong, n_chains=30, T=5000, seed=seed)

        metrics = evaluate_binding_metrics(seqs, strong_seqs, pca_vecs, X̂_strong, p1_pos)
        push!(results_raw, ("Strong binders", rep, length(seqs), metrics...))
    end
else
    @warn "Skipping strong-binder generation (too few patterns)"
end

# Generate from weak binders with replicates
if length(weak_idx) >= 5
    @info "  Generating from weak binders"
    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 30000 + rep

        seqs, pca_vecs = generate_sequences(X̂_weak, pca_weak, L;
            β=β_weak, n_chains=30, T=5000, seed=seed)

        metrics = evaluate_binding_metrics(seqs, weak_seqs, pca_vecs, X̂_weak, p1_pos)
        push!(results_raw, ("Weak binders", rep, length(seqs), metrics...))
    end
else
    @warn "Skipping weak-binder generation (too few patterns)"
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Aggregate results with uncertainty
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 6: Aggregating results with uncertainty"

# Aggregate statistics across replicates
results_agg = combine(groupby(results_raw, :condition),
    :n_generated => first => :n_generated,
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :mean_valid_frac => mean => :valid_mean,
    :mean_valid_frac => std => :valid_std,
    :kl_aa => mean => :kl_mean,
    :kl_aa => std => :kl_std,
    :mean_novelty => mean => :novelty_mean,
    :mean_novelty => std => :novelty_std,
    :mean_seqid => mean => :seqid_mean,
    :mean_seqid => std => :seqid_std,
    :diversity => mean => :diversity_mean,
    :diversity => std => :diversity_std
)

@info "\nResults with uncertainty:"
show(stdout, results_agg)
println()

# Add input reference data for comparison
input_strong_p1 = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), strong_seqs) / length(strong_seqs)
input_weak_p1 = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), weak_seqs) / length(weak_seqs)
input_all_p1 = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), stored_seqs) / length(stored_seqs)

@info "\nInput reference P1 K/R fractions:"
@info "  Strong binders: $(round(input_strong_p1, digits=3))"
@info "  Weak binders:   $(round(input_weak_p1, digits=3))"
@info "  Full family:    $(round(input_all_p1, digits=3))"

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Visualization with error bars
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 7: Generating figures with error bars"

# Figure 1: P1 K/R fraction comparison with error bars
p1 = bar(1:nrow(results_agg), results_agg.p1_kr_mean,
    yerror=results_agg.p1_kr_std,
    ylabel="Fraction K/R at P1 (position $p1_pos)",
    title="P1 Binding Phenotype Inheritance",
    legend=false,
    xticks=(1:nrow(results_agg), results_agg.condition),
    rotation=15, bar_width=0.6,
    fillcolor=[:steelblue, :coral, :forestgreen][1:nrow(results_agg)],
    ylim=(0, 1.1), size=(800, 500), margin=15Plots.mm)

# Add reference lines for input data
hline!(p1, [input_all_p1], linestyle=:dash, color=:gray, linewidth=2, label="Input: Full family")
if length(strong_idx) >= 5
    hline!(p1, [input_strong_p1], linestyle=:dash, color=:red, linewidth=2, label="Input: Strong binders")
end
if length(weak_idx) >= 5
    hline!(p1, [input_weak_p1], linestyle=:dash, color=:blue, linewidth=2, label="Input: Weak binders")
end

savefig(p1, joinpath(FIG_DIR, "p1_phenotype_inheritance_with_errors.pdf"))
savefig(p1, joinpath(FIG_DIR, "p1_phenotype_inheritance_with_errors.png"))
@info "  Saved p1_phenotype_inheritance_with_errors"

# Figure 2: Multi-metric comparison with error bars
p2 = plot(layout=(2, 2), size=(1200, 1000), margin=15Plots.mm)

# P1 K/R fraction
bar!(p2[1], 1:nrow(results_agg), results_agg.p1_kr_mean,
    yerror=results_agg.p1_kr_std,
    ylabel="P1 K/R fraction", title="Phenotype Fidelity",
    xticks=(1:nrow(results_agg), results_agg.condition),
    rotation=45, color=:steelblue, legend=false)

# Validity
bar!(p2[2], 1:nrow(results_agg), results_agg.valid_mean,
    yerror=results_agg.valid_std,
    ylabel="Valid residue fraction", title="Sequence Validity",
    xticks=(1:nrow(results_agg), results_agg.condition),
    rotation=45, color=:coral, legend=false, ylim=(0.8, 1.05))

# Diversity
bar!(p2[3], 1:nrow(results_agg), results_agg.diversity_mean,
    yerror=results_agg.diversity_std,
    ylabel="Pairwise diversity", title="Sequence Diversity",
    xticks=(1:nrow(results_agg), results_agg.condition),
    rotation=45, color=:forestgreen, legend=false)

# Novelty
bar!(p2[4], 1:nrow(results_agg), results_agg.novelty_mean,
    yerror=results_agg.novelty_std,
    ylabel="Mean novelty", title="Novelty vs Training",
    xticks=(1:nrow(results_agg), results_agg.condition),
    rotation=45, color=:purple, legend=false)

savefig(p2, joinpath(FIG_DIR, "binding_experiment_multimetric_with_errors.pdf"))
savefig(p2, joinpath(FIG_DIR, "binding_experiment_multimetric_with_errors.png"))
@info "  Saved binding_experiment_multimetric_with_errors"

# Figure 3: Phenotype transfer effectiveness
p3 = plot(size=(700, 500), margin=15Plots.mm,
    xlabel="Generated sequence diversity",
    ylabel="P1 K/R fraction (phenotype fidelity)",
    title="Phenotype Transfer: Fidelity vs Diversity\n(with uncertainty bounds)",
    legend=:bottomleft)

# Plot with error bars in both dimensions
if nrow(results_agg) > 0
    scatter!(p3, results_agg.diversity_mean, results_agg.p1_kr_mean,
            xerror=results_agg.diversity_std,
            yerror=results_agg.p1_kr_std,
            marker=:circle, markersize=10,
            color=[:steelblue, :coral, :forestgreen][1:nrow(results_agg)],
            label="Generated ± SD")

    # Annotate points
    for (i, row) in enumerate(eachrow(results_agg))
        annotate!(p3, row.diversity_mean + 0.005, row.p1_kr_mean - 0.04,
            text(row.condition, 8, :left))
    end
end

# Add input references
scatter!(p3, [0.0], [input_strong_p1], marker=:star, markersize=12, color=:red, label="Input: Strong binders")
scatter!(p3, [0.0], [input_weak_p1], marker=:star, markersize=12, color=:blue, label="Input: Weak binders")
scatter!(p3, [0.0], [input_all_p1], marker=:star, markersize=12, color=:gray, label="Input: Full family")

savefig(p3, joinpath(FIG_DIR, "phenotype_transfer_fidelity_diversity_with_errors.pdf"))
savefig(p3, joinpath(FIG_DIR, "phenotype_transfer_fidelity_diversity_with_errors.png"))
@info "  Saved phenotype_transfer_fidelity_diversity_with_errors"

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Statistical significance testing
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 8: Statistical significance testing"

# Test if strong-conditioned sequences have significantly higher P1 K/R fraction
# than weak-conditioned sequences
strong_data = filter(row -> row.condition == "Strong binders", results_raw)
weak_data = filter(row -> row.condition == "Weak binders", results_raw)
full_data = filter(row -> row.condition == "Full family", results_raw)

if nrow(strong_data) > 0 && nrow(weak_data) > 0
    # Two-sample t-test for P1 K/R fraction
    strong_p1_vals = strong_data.p1_kr_frac
    weak_p1_vals = weak_data.p1_kr_frac

    # Simple two-sample test
    pooled_var = ((length(strong_p1_vals) - 1) * var(strong_p1_vals) +
                  (length(weak_p1_vals) - 1) * var(weak_p1_vals)) /
                 (length(strong_p1_vals) + length(weak_p1_vals) - 2)
    se_diff = sqrt(pooled_var * (1/length(strong_p1_vals) + 1/length(weak_p1_vals)))
    t_stat = (mean(strong_p1_vals) - mean(weak_p1_vals)) / se_diff
    df = length(strong_p1_vals) + length(weak_p1_vals) - 2

    @info "  Strong vs Weak P1 K/R fraction:"
    @info "    Strong: $(round(mean(strong_p1_vals), digits=3)) ± $(round(std(strong_p1_vals), digits=3))"
    @info "    Weak:   $(round(mean(weak_p1_vals), digits=3)) ± $(round(std(weak_p1_vals), digits=3))"
    @info "    t-statistic: $(round(t_stat, digits=3)) (df=$df)"
    @info "    Effect size (Cohen's d): $(round((mean(strong_p1_vals) - mean(weak_p1_vals)) / sqrt(pooled_var), digits=3))"
end

if nrow(strong_data) > 0 && nrow(full_data) > 0
    strong_p1_vals = strong_data.p1_kr_frac
    full_p1_vals = full_data.p1_kr_frac

    @info "  Strong vs Full family P1 K/R fraction:"
    @info "    Strong: $(round(mean(strong_p1_vals), digits=3)) ± $(round(std(strong_p1_vals), digits=3))"
    @info "    Full:   $(round(mean(full_p1_vals), digits=3)) ± $(round(std(full_p1_vals), digits=3))"
    @info "    Difference: $(round(mean(strong_p1_vals) - mean(full_p1_vals), digits=3))"
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Save results
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 9: Saving results"

# Save raw and aggregated data
CSV.write(joinpath(CACHE_DIR, "binding_experiment_raw_replicates.csv"), results_raw)
CSV.write(joinpath(CACHE_DIR, "binding_experiment_aggregated.csv"), results_agg)

# Save one example set of sequences per condition for visualization
if nrow(full_data) > 0
    rep1_full = generate_sequences(X̂_all, pca_all, L; β=β_all, n_chains=30, T=5000, seed=42)[1]

    function save_fasta(seqs, filepath, prefix)
        open(filepath, "w") do io
            for (i, seq) in enumerate(seqs[1:min(50, end)])  # Save up to 50 sequences
                println(io, ">$(prefix)_$(lpad(i, 4, '0'))")
                println(io, seq)
            end
        end
        @info "  Saved $(min(50, length(seqs))) sequences to $filepath"
    end

    save_fasta(rep1_full, joinpath(CACHE_DIR, "generated_full_family_example.fasta"), "SA_full")
end

if nrow(strong_data) > 0 && length(strong_idx) >= 5
    rep1_strong = generate_sequences(X̂_strong, pca_strong, L; β=β_strong, n_chains=30, T=5000, seed=42)[1]
    save_fasta(rep1_strong, joinpath(CACHE_DIR, "generated_strong_conditioned_example.fasta"), "SA_strong")
end

if nrow(weak_data) > 0 && length(weak_idx) >= 5
    rep1_weak = generate_sequences(X̂_weak, pca_weak, L; β=β_weak, n_chains=30, T=5000, seed=42)[1]
    save_fasta(rep1_weak, joinpath(CACHE_DIR, "generated_weak_conditioned_example.fasta"), "SA_weak")
end

@info "\n" * "="^70
@info "Kunitz binding experiment complete (WITH UNCERTAINTY)!"
@info "="^70
@info "\nKey improvements:"
@info "  1. Each condition run with $n_reps independent replicates"
@info "  2. Standard deviations computed across replicates"
@info "  3. All figures include error bars"
@info "  4. Statistical significance testing included"
@info "  5. Both raw and aggregated data saved for analysis"
@info ""
@info "Key question: Do strong-conditioned sequences have higher K/R fraction"
@info "at P1 than weak-conditioned sequences (with statistical significance)?"
