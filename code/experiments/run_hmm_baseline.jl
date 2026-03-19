# ──────────────────────────────────────────────────────────────────────────────
# run_hmm_baseline.jl — HMM baseline comparison for SA binding generation
#
# Generates sequences using HMMER3 (hmmbuild + hmmemit) and compares them
# against SA-generated sequences (full family, strong-conditioned, weak-conditioned)
# using the same evaluation metrics: KL divergence, novelty, sequence identity,
# diversity, and P1 K/R fraction.
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

using HypothesisTests

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
const PFAM_ID = "PF00014"
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "baselines")
const DATA_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const N_HMM_SEQS = 150
const N_REPS = 5           # replicates for SA generation
const P1_POSITION = 25     # P1 residue position in cleaned Kunitz alignment
const STRONG_BINDER_AA = Set(['K', 'R'])

mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load alignment and build models
# ══════════════════════════════════════════════════════════════════════════════
@info "Loading Kunitz domain alignment"
sto_file = download_pfam_seed(PFAM_ID; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs)
K, L = size(char_mat)

# stored sequences as strings
stored_seqs = [String(char_mat[i, :]) for i in 1:K]

# identify binders
binder_indices = findall(i -> char_mat[i, P1_POSITION] in STRONG_BINDER_AA, 1:K)
nonbinder_indices = setdiff(1:K, binder_indices)
@info "  P1 position: $P1_POSITION | Strong binders: $(length(binder_indices)) / $K"

# build memory matrices
X̂, pca_model, L_out, d_full = build_memory_matrix(char_mat; pratio=0.95)
pt = find_entropy_inflection(X̂)
β_star = pt.β_star

# curated binder memory
binder_result = build_binder_memory(char_mat, binder_indices)
pt_binder = find_entropy_inflection(binder_result.X̂)
β_binder = pt_binder.β_star

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Generate HMM baseline sequences
# ══════════════════════════════════════════════════════════════════════════════
@info "Generating HMM baseline sequences"
hmm_file = joinpath(CACHE_DIR, "kunitz.hmm")
hmm_fasta = joinpath(CACHE_DIR, "hmm_generated.fasta")

# Build HMM from Stockholm alignment
run(`hmmbuild --amino $hmm_file $sto_file`)

# Emit sequences
run(`hmmemit -N $N_HMM_SEQS --seed 42 -o $hmm_fasta $hmm_file`)

# Parse and clean HMM sequences
hmm_raw = parse_fasta(hmm_fasta)
@info "  HMM emitted $(length(hmm_raw)) raw sequences"

hmm_seqs = String[]
for (name, seq) in hmm_raw
    # keep only standard amino acids
    cleaned = filter(c -> c in AA_ALPHABET, seq)
    # align to reference length
    if length(cleaned) >= L
        push!(hmm_seqs, cleaned[1:L])
    else
        # pad with most common AA at each position (use alanine as fallback)
        padded = cleaned * repeat("A", L - length(cleaned))
        push!(hmm_seqs, padded)
    end
end
@info "  HMM processed: $(length(hmm_seqs)) sequences of length $L"

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Generate SA sequences with replicates
# ══════════════════════════════════════════════════════════════════════════════
@info "Generating SA sequences with $N_REPS replicates"

# Helper: project sequences into the full-family PCA space
function project_seqs_to_full_pca(seqs, full_pca_model, L_full)
    pca_vecs = Vector{Float64}[]
    for seq in seqs
        x = zeros(N_AA * L_full)
        for pos in 1:min(L_full, length(seq))
            idx = get(AA_TO_IDX, seq[pos], 0)
            idx > 0 && (x[(pos-1)*N_AA + idx] = 1.0)
        end
        push!(pca_vecs, vec(MultivariateStats.transform(full_pca_model, x)))
    end
    return pca_vecs
end

# Helper to compute metrics for a set of sequences
# pca_vecs must be in the full-family PCA space (for consistent novelty/diversity)
function compute_metrics(seqs, pca_vecs, stored_seqs, X̂)
    p1_kr = mean(s -> s[P1_POSITION] in STRONG_BINDER_AA ? 1.0 : 0.0, seqs)
    kl = aa_composition_kl(seqs, stored_seqs)
    nov = mean(sample_novelty(v, X̂) for v in pca_vecs)
    seqid = mean(nearest_sequence_identity(s, stored_seqs) for s in seqs)
    div = sample_diversity(pca_vecs)
    valid = mean(valid_residue_fraction(s) for s in seqs)
    return (p1_kr=p1_kr, kl=kl, novelty=nov, seqid=seqid, diversity=div, valid=valid)
end

# SA full family replicates
sa_full_metrics = []
for rep in 1:N_REPS
    seed = 10000 + rep
    seqs, pca_vecs = generate_sequences(X̂, pca_model, L; β=β_star, seed=seed)
    push!(sa_full_metrics, compute_metrics(seqs, pca_vecs, stored_seqs, X̂))
end

# SA strong binder replicates
sa_strong_metrics = []
for rep in 1:N_REPS
    seed = 20000 + rep
    seqs, _ = generate_sequences(binder_result.X̂, binder_result.pca_model,
                                  binder_result.L; β=β_binder, seed=seed)
    # Re-project into full-family PCA space for consistent metrics
    pca_vecs = project_seqs_to_full_pca(seqs, pca_model, L)
    push!(sa_strong_metrics, compute_metrics(seqs, pca_vecs, stored_seqs, X̂))
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Compute HMM metrics
# ══════════════════════════════════════════════════════════════════════════════
@info "Computing HMM baseline metrics"

# Project HMM sequences into PCA space for novelty/diversity
hmm_pca_vecs = Vector{Float64}[]
for seq in hmm_seqs
    # one-hot encode
    x = zeros(d_full)
    for pos in 1:L
        idx = get(AA_TO_IDX, seq[pos], 0)
        idx > 0 && (x[(pos-1)*N_AA + idx] = 1.0)
    end
    # project to PCA space
    ξ = vec(MultivariateStats.transform(pca_model, x))
    push!(hmm_pca_vecs, ξ)
end

hmm_metrics = compute_metrics(hmm_seqs, hmm_pca_vecs, stored_seqs, X̂)

# Per-chain metrics for HMM (split into groups of 5 for SE estimation)
n_per_group = 5
n_groups = div(length(hmm_seqs), n_per_group)
hmm_chain_p1 = Float64[]
hmm_chain_kl = Float64[]
hmm_chain_nov = Float64[]
hmm_chain_seqid = Float64[]
for g in 1:n_groups
    idx_range = ((g-1)*n_per_group + 1):(g*n_per_group)
    group_seqs = hmm_seqs[idx_range]
    group_pca = hmm_pca_vecs[idx_range]
    push!(hmm_chain_p1, mean(s -> s[P1_POSITION] in STRONG_BINDER_AA ? 1.0 : 0.0, group_seqs))
    push!(hmm_chain_kl, aa_composition_kl(group_seqs, stored_seqs))
    push!(hmm_chain_nov, mean(sample_novelty(v, X̂) for v in group_pca))
    push!(hmm_chain_seqid, mean(nearest_sequence_identity(s, stored_seqs) for s in group_seqs))
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Bootstrap baseline — resample from stored sequences
# ══════════════════════════════════════════════════════════════════════════════
@info "Computing bootstrap baseline"
bootstrap_metrics = []
for rep in 1:N_REPS
    Random.seed!(30000 + rep)
    boot_idx = rand(1:K, N_HMM_SEQS)
    boot_seqs = stored_seqs[boot_idx]
    boot_pca = [X̂[:, i] for i in boot_idx]
    push!(bootstrap_metrics, compute_metrics(boot_seqs, boot_pca, stored_seqs, X̂))
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Aggregate results
# ══════════════════════════════════════════════════════════════════════════════
@info "Aggregating results"

function summarize(metrics_list)
    fields = keys(first(metrics_list))
    result = Dict{Symbol, Tuple{Float64, Float64}}()
    for f in fields
        vals = [getfield(m, f) for m in metrics_list]
        result[f] = (mean(vals), std(vals))
    end
    return result
end

sa_full_summary = summarize(sa_full_metrics)
sa_strong_summary = summarize(sa_strong_metrics)
bootstrap_summary = summarize(bootstrap_metrics)

# HMM uses chain-based SE
hmm_summary = Dict(
    :p1_kr => (mean(hmm_chain_p1), std(hmm_chain_p1) / sqrt(n_groups)),
    :kl => (hmm_metrics.kl, std(hmm_chain_kl) / sqrt(n_groups)),
    :novelty => (mean(hmm_chain_nov), std(hmm_chain_nov) / sqrt(n_groups)),
    :seqid => (mean(hmm_chain_seqid), std(hmm_chain_seqid) / sqrt(n_groups)),
    :diversity => (hmm_metrics.diversity, 0.0),
    :valid => (hmm_metrics.valid, 0.0),
)

# Build results DataFrame
methods = ["SA (full family)", "SA (strong binders)", "HMM emit", "Bootstrap"]
summaries = [sa_full_summary, sa_strong_summary, hmm_summary, bootstrap_summary]

results_df = DataFrame(
    method = String[],
    p1_kr_mean = Float64[], p1_kr_std = Float64[],
    kl_mean = Float64[], kl_std = Float64[],
    novelty_mean = Float64[], novelty_std = Float64[],
    seqid_mean = Float64[], seqid_std = Float64[],
    diversity_mean = Float64[], diversity_std = Float64[],
)

for (name, s) in zip(methods, summaries)
    push!(results_df, (
        method = name,
        p1_kr_mean = s[:p1_kr][1], p1_kr_std = s[:p1_kr][2],
        kl_mean = s[:kl][1], kl_std = s[:kl][2],
        novelty_mean = s[:novelty][1], novelty_std = s[:novelty][2],
        seqid_mean = s[:seqid][1], seqid_std = s[:seqid][2],
        diversity_mean = s[:diversity][1], diversity_std = s[:diversity][2],
    ))
end

@info "\nBaseline comparison results:"
pretty_table(results_df)
CSV.write(joinpath(DATA_DIR, "baseline_comparison.csv"), results_df)

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Statistical tests (SA full vs HMM)
# ══════════════════════════════════════════════════════════════════════════════
@info "Statistical significance tests"

# Collect per-replicate values for SA
sa_full_p1_vals = [m.p1_kr for m in sa_full_metrics]
sa_full_nov_vals = [m.novelty for m in sa_full_metrics]
sa_full_seqid_vals = [m.seqid for m in sa_full_metrics]

# Compare SA vs HMM (using chain-level values for HMM, replicate-level for SA)
function sig_marker(p)
    p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "n.s."
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Generate figures
# ══════════════════════════════════════════════════════════════════════════════
@info "Generating baseline comparison figures"

# Color scheme
sa_color = RGB(0.20, 0.47, 0.69)
sa_strong_color = RGB(0.80, 0.20, 0.20)
hmm_color = RGB(0.30, 0.69, 0.29)
boot_color = RGB(0.93, 0.68, 0.20)

# Figure 1: P1 K/R fraction comparison
method_labels = ["SA (full)", "SA (strong)", "HMM emit", "Bootstrap"]
p1_means = [s[:p1_kr][1] for s in summaries]
p1_stds = [s[:p1_kr][2] for s in summaries]
colors = [sa_color, sa_strong_color, hmm_color, boot_color]

p1 = bar(method_labels, p1_means; yerror=p1_stds,
    ylabel="P1 K/R fraction", title="Binding Marker: SA vs Baselines",
    color=reshape(colors, 1, :), legend=false, bar_width=0.6,
    ylims=(0, 1.15), linewidth=0, size=(600, 400), dpi=300,
    xrotation=15, margin=5Plots.mm)
hline!([length(binder_indices)/K], linestyle=:dash, color=:gray, label="Natural fraction")
savefig(p1, joinpath(FIG_DIR, "baseline_p1_comparison.png"))
@info "  Saved baseline_p1_comparison"

# Figure 2: Multi-metric comparison (4 panels)
p_kl = bar(method_labels, [s[:kl][1] for s in summaries];
    yerror=[s[:kl][2] for s in summaries],
    ylabel="KL divergence", title="AA Composition",
    color=reshape(colors, 1, :), legend=false, bar_width=0.6,
    linewidth=0, xrotation=15, margin=3Plots.mm)

p_nov = bar(method_labels, [s[:novelty][1] for s in summaries];
    yerror=[s[:novelty][2] for s in summaries],
    ylabel="Novelty", title="Novelty (PCA space)",
    color=reshape(colors, 1, :), legend=false, bar_width=0.6,
    linewidth=0, xrotation=15, margin=3Plots.mm)

p_seqid = bar(method_labels, [s[:seqid][1] for s in summaries];
    yerror=[s[:seqid][2] for s in summaries],
    ylabel="Sequence identity", title="Nearest Seq. Identity",
    color=reshape(colors, 1, :), legend=false, bar_width=0.6,
    linewidth=0, xrotation=15, margin=3Plots.mm)

p_div = bar(method_labels, [s[:diversity][1] for s in summaries];
    yerror=[s[:diversity][2] for s in summaries],
    ylabel="Diversity", title="Diversity (PCA space)",
    color=reshape(colors, 1, :), legend=false, bar_width=0.6,
    linewidth=0, xrotation=15, margin=3Plots.mm)

p_multi = plot(p_kl, p_nov, p_seqid, p_div;
    layout=(2, 2), size=(900, 700), dpi=300,
    plot_title="SA vs Baselines: Multi-Metric Comparison")
savefig(p_multi, joinpath(FIG_DIR, "baseline_multimetric_comparison.png"))
@info "  Saved baseline_multimetric_comparison"

# Figure 3: Novelty vs Sequence Identity scatter (fidelity-diversity)
p_scatter = scatter(
    [sa_full_summary[:seqid][1]], [sa_full_summary[:novelty][1]];
    xerror=[sa_full_summary[:seqid][2]], yerror=[sa_full_summary[:novelty][2]],
    label="SA (full)", color=sa_color, markersize=8, markerstrokewidth=0)
scatter!(
    [sa_strong_summary[:seqid][1]], [sa_strong_summary[:novelty][1]];
    xerror=[sa_strong_summary[:seqid][2]], yerror=[sa_strong_summary[:novelty][2]],
    label="SA (strong)", color=sa_strong_color, markersize=8, markerstrokewidth=0)
scatter!(
    [hmm_summary[:seqid][1]], [hmm_summary[:novelty][1]];
    xerror=[hmm_summary[:seqid][2]], yerror=[hmm_summary[:novelty][2]],
    label="HMM emit", color=hmm_color, markersize=8, markerstrokewidth=0)
scatter!(
    [bootstrap_summary[:seqid][1]], [bootstrap_summary[:novelty][1]];
    xerror=[bootstrap_summary[:seqid][2]], yerror=[bootstrap_summary[:novelty][2]],
    label="Bootstrap", color=boot_color, markersize=8, markerstrokewidth=0)
plot!(p_scatter; xlabel="Nearest Sequence Identity", ylabel="Novelty (PCA space)",
    title="Fidelity-Novelty Trade-off", size=(600, 500), dpi=300,
    legend=:topright, margin=5Plots.mm)
savefig(p_scatter, joinpath(FIG_DIR, "baseline_fidelity_novelty.png"))
@info "  Saved baseline_fidelity_novelty"

@info "\n======================================================================="
@info "HMM baseline comparison complete!"
@info "======================================================================="
