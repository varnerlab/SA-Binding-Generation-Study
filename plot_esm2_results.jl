# ──────────────────────────────────────────────────────────────────────────────
# plot_esm2_results.jl — Generate figures from ESM2 pseudo-perplexity scores
#
# Reads the CSV output from score_esm2_perplexity.py and generates
# publication-quality figures using Plots.jl.
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = joinpath(_SCRIPT_DIR, "code")
include(joinpath(_CODE_DIR, "Include.jl"))

const DATA_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "esm_validation")
mkpath(FIG_DIR)

# Load results
raw_csv = joinpath(DATA_DIR, "esm2_pseudo_perplexity.csv")
if !isfile(raw_csv)
    @error "ESM2 results not found. Run score_esm2_perplexity.py first."
    exit(1)
end

df = CSV.read(raw_csv, DataFrame)
@info "Loaded $(nrow(df)) scored sequences from $(length(unique(df.source))) sources"

# ══════════════════════════════════════════════════════════════════════════════
# Compute summary statistics
# ══════════════════════════════════════════════════════════════════════════════
summary_df = combine(groupby(df, :source),
    :pseudo_perplexity => mean => :ppl_mean,
    :pseudo_perplexity => std => :ppl_std,
    :mean_log_likelihood => mean => :ll_mean,
    :mean_log_likelihood => std => :ll_std,
    nrow => :n,
)

@info "\nESM2 Pseudo-Perplexity Summary:"
pretty_table(summary_df)

# ══════════════════════════════════════════════════════════════════════════════
# Color scheme and ordering
# ══════════════════════════════════════════════════════════════════════════════
source_order = ["stored", "SA_full", "SA_strong", "SA_weak", "HMM_emit"]
source_labels = ["Stored", "SA (full)", "SA (strong)", "SA (weak)", "HMM emit"]
source_colors = Dict(
    "stored" => RGB(0.50, 0.50, 0.50),
    "SA_full" => RGB(0.20, 0.47, 0.69),
    "SA_strong" => RGB(0.80, 0.20, 0.20),
    "SA_weak" => RGB(0.20, 0.60, 0.20),
    "HMM_emit" => RGB(0.93, 0.68, 0.20),
)

available = [s for s in source_order if s in unique(df.source)]
avail_labels = [source_labels[findfirst(==(s), source_order)] for s in available]
avail_colors = [source_colors[s] for s in available]

# Get summary values
ppl_means = Float64[]
ppl_stds = Float64[]
for s in available
    row = filter(r -> r.source == s, summary_df)
    push!(ppl_means, row.ppl_mean[1])
    push!(ppl_stds, row.ppl_std[1])
end

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Bar chart of pseudo-perplexity
# ══════════════════════════════════════════════════════════════════════════════
p_bar = bar(avail_labels, ppl_means; yerror=ppl_stds,
    ylabel="Pseudo-perplexity", title="ESM2 Sequence Plausibility (lower = better)",
    color=reshape(avail_colors, 1, :), legend=false, bar_width=0.6,
    linewidth=0, size=(650, 450), dpi=300,
    xrotation=15, margin=5Plots.mm)
savefig(p_bar, joinpath(FIG_DIR, "esm2_perplexity_comparison.png"))
@info "  Saved esm2_perplexity_comparison"

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Box/violin plot of per-sequence perplexity distributions
# ══════════════════════════════════════════════════════════════════════════════
# Build grouped data for violin plot
group_labels = String[]
group_values = Float64[]
for (i, s) in enumerate(available)
    sdf = filter(r -> r.source == s, df)
    for ppl in sdf.pseudo_perplexity
        push!(group_labels, avail_labels[i])
        push!(group_values, ppl)
    end
end

# Use categorical array for ordering
cat_labels = CategoricalArray(group_labels; levels=avail_labels)

p_violin = violin(cat_labels, group_values;
    ylabel="Pseudo-perplexity", title="ESM2 Per-Sequence Perplexity Distribution",
    legend=false, linewidth=0.5, size=(650, 450), dpi=300,
    xrotation=15, margin=5Plots.mm, alpha=0.7)
boxplot!(cat_labels, group_values; fillalpha=0.3, linewidth=1, markersize=2, legend=false)
savefig(p_violin, joinpath(FIG_DIR, "esm2_perplexity_distribution.png"))
@info "  Saved esm2_perplexity_distribution"

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Mean log-likelihood bar chart
# ══════════════════════════════════════════════════════════════════════════════
ll_means = Float64[]
ll_stds = Float64[]
for s in available
    row = filter(r -> r.source == s, summary_df)
    push!(ll_means, row.ll_mean[1])
    push!(ll_stds, row.ll_std[1])
end

p_ll = bar(avail_labels, ll_means; yerror=ll_stds,
    ylabel="Mean log-likelihood", title="ESM2 Mean Log-Likelihood (higher = better)",
    color=reshape(avail_colors, 1, :), legend=false, bar_width=0.6,
    linewidth=0, size=(650, 450), dpi=300,
    xrotation=15, margin=5Plots.mm)
savefig(p_ll, joinpath(FIG_DIR, "esm2_loglikelihood_comparison.png"))
@info "  Saved esm2_loglikelihood_comparison"

@info "\nESM2 figure generation complete!"
