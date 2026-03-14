# ──────────────────────────────────────────────────────────────────────────────
# run_all_approaches.jl
#
# Demonstrates all four binding approaches on a single protein family.
# Uses Kunitz domains (PF00014) as the test case:
#   1. Curated memory matrix
#   2. Biased energy landscape
#   3. Post-hoc filtering
#   4. Interface-aware PCA
#
# Compares generated sequences across approaches.
# ──────────────────────────────────────────────────────────────────────────────

# --- setup ---
_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

# --- configuration ---
const PFAM_ID = "PF00014"  # Kunitz/BPTI
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "approaches")
mkpath(CACHE_DIR)
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Load and prepare data
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
interface_positions = binding_loop  # use binding loop as interface

# identify strong binders (K/R at P1)
strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
strong_seqs = stored_seqs[strong_idx]
@info "  P1 position: $p1_pos | Interface: $interface_positions"
@info "  Strong binders: $(length(strong_idx)) / $K_total"

# ══════════════════════════════════════════════════════════════════════════════
# Baseline: Full family SA
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^60
@info "BASELINE: Standard SA on full family"
@info "="^60

X̂_full, pca_full, L_full, d_full = build_memory_matrix(char_mat; pratio=0.95)
pt_full = find_entropy_inflection(X̂_full)
β_full = pt_full.β_star

gen_baseline_seqs, gen_baseline_pca = generate_sequences(X̂_full, pca_full, L;
    β=β_full, n_chains=30, T=5000, seed=42)

# ══════════════════════════════════════════════════════════════════════════════
# Approach 1: Curated Memory Matrix
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^60
@info "APPROACH 1: Curated Memory Matrix (strong binders only)"
@info "="^60

result_curated = build_binder_memory(char_mat, strong_idx; pratio=0.95)
X̂_curated = result_curated.X̂
pca_curated = result_curated.pca_model

pt_curated = find_entropy_inflection(X̂_curated)
β_curated = pt_curated.β_star

gen_curated_seqs, gen_curated_pca = generate_sequences(X̂_curated, pca_curated, L;
    β=β_curated, n_chains=30, T=5000, seed=42)

# ══════════════════════════════════════════════════════════════════════════════
# Approach 2: Biased Energy Landscape
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^60
@info "APPROACH 2: Biased Energy Landscape (E_Hopfield + λ·E_bind)"
@info "="^60

iface_profile = build_interface_profile(strong_seqs, stored_seqs,
                                         interface_positions, pca_full, L)

# try multiple λ values
λ_values = [0.05, 0.1, 0.2, 0.5]
biased_results = Dict{Float64, Tuple{Vector{String}, Vector{Vector{Float64}}}}()

for λ in λ_values
    @info "  λ = $λ"
    seqs, pca_vecs = generate_biased_sequences(X̂_full, pca_full, L, iface_profile;
        β=β_full, λ=λ, n_chains=20, T=5000, seed=42)
    biased_results[λ] = (seqs, pca_vecs)
end

# ══════════════════════════════════════════════════════════════════════════════
# Approach 3: Post-hoc Filtering
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^60
@info "APPROACH 3: Post-hoc Filtering"
@info "="^60

# generate a large pool using baseline SA
gen_pool_seqs, gen_pool_pca = generate_sequences(X̂_full, pca_full, L;
    β=β_full, n_chains=60, T=5000, seed=100)

# filter and rank
top_candidates = filter_and_rank(gen_pool_seqs, gen_pool_pca, iface_profile;
    stored_seqs=stored_seqs, X̂=X̂_full, β=β_full,
    top_k=50, min_novelty=0.02, min_valid_frac=0.8)

@info "  Top 10 candidates:"
if nrow(top_candidates) >= 10
    show(stdout, first(top_candidates[!, [:rank, :interface_score, :novelty, :nearest_seqid, :composite_score]], 10))
    println()
end

# ══════════════════════════════════════════════════════════════════════════════
# Approach 4: Interface-Aware PCA
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^60
@info "APPROACH 4: Interface-Aware PCA Encoding"
@info "="^60

# try multiple weights
weight_values = [2.0, 3.0, 5.0]
iface_pca_results = Dict{Float64, Tuple{Vector{String}, Vector{Vector{Float64}}}}()

for w in weight_values
    @info "  Interface weight = $w"
    result_w = build_interface_weighted_memory(char_mat, interface_positions;
        pratio=0.95, weight=w)
    X̂_w = result_w.X̂
    pca_w = result_w.pca_model

    pt_w = find_entropy_inflection(X̂_w)
    β_w = pt_w.β_star

    # generate and decode (using weighted decoder)
    d_w, K_w = size(X̂_w)
    seqs_w = String[]
    pca_w_vecs = Vector{Float64}[]

    for chain in 1:20
        k = mod1(chain, K_w)
        ξ₀ = X̂_w[:, k] .+ 0.01 .* randn(d_w)
        res = sample(X̂_w, ξ₀, 5000; β=β_w, α=0.01, seed=42 + chain)
        for t in 2000:100:5000
            ξ = res.Ξ[t + 1, :]
            seq = decode_weighted_sample(ξ, pca_w, L, interface_positions; weight=w)
            push!(seqs_w, seq)
            push!(pca_w_vecs, ξ)
        end
    end

    iface_pca_results[w] = (seqs_w, pca_w_vecs)
    @info "    Generated $(length(seqs_w)) sequences"
end

# ══════════════════════════════════════════════════════════════════════════════
# Comparison: P1 phenotype across all approaches
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^60
@info "COMPARISON: P1 residue (K/R) fraction across approaches"
@info "="^60

function p1_frac(seqs, p1_pos)
    n = length(seqs)
    n == 0 && return 0.0
    count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n
end

comparison_data = DataFrame(
    approach = String[],
    n_generated = Int[],
    p1_kr_frac = Float64[],
    mean_valid_frac = Float64[],
    kl_aa = Float64[],
)

# input reference
push!(comparison_data, (
    "Input: Strong binders",
    length(strong_seqs),
    p1_frac(strong_seqs, p1_pos),
    mean(valid_residue_fraction.(strong_seqs)),
    0.0
))

push!(comparison_data, (
    "Input: Full family",
    K_total,
    p1_frac(stored_seqs, p1_pos),
    mean(valid_residue_fraction.(stored_seqs)),
    0.0
))

# baseline
push!(comparison_data, (
    "Baseline: Standard SA",
    length(gen_baseline_seqs),
    p1_frac(gen_baseline_seqs, p1_pos),
    mean(valid_residue_fraction.(gen_baseline_seqs)),
    aa_composition_kl(gen_baseline_seqs, stored_seqs)
))

# approach 1
push!(comparison_data, (
    "Approach 1: Curated memory",
    length(gen_curated_seqs),
    p1_frac(gen_curated_seqs, p1_pos),
    mean(valid_residue_fraction.(gen_curated_seqs)),
    aa_composition_kl(gen_curated_seqs, strong_seqs)
))

# approach 2
for λ in λ_values
    seqs_λ = biased_results[λ][1]
    push!(comparison_data, (
        "Approach 2: λ=$λ",
        length(seqs_λ),
        p1_frac(seqs_λ, p1_pos),
        mean(valid_residue_fraction.(seqs_λ)),
        aa_composition_kl(seqs_λ, stored_seqs)
    ))
end

# approach 3
if nrow(top_candidates) > 0
    filtered_seqs = top_candidates.sequence
    push!(comparison_data, (
        "Approach 3: Post-hoc filter",
        length(filtered_seqs),
        p1_frac(filtered_seqs, p1_pos),
        mean(valid_residue_fraction.(filtered_seqs)),
        aa_composition_kl(filtered_seqs, stored_seqs)
    ))
end

# approach 4
for w in weight_values
    seqs_w = iface_pca_results[w][1]
    push!(comparison_data, (
        "Approach 4: weight=$w",
        length(seqs_w),
        p1_frac(seqs_w, p1_pos),
        mean(valid_residue_fraction.(seqs_w)),
        aa_composition_kl(seqs_w, stored_seqs)
    ))
end

@info "\nResults:"
show(stdout, comparison_data)
println()

# save comparison table
CSV.write(joinpath(CACHE_DIR, "approach_comparison.csv"), comparison_data)
@info "\nSaved comparison to $(joinpath(CACHE_DIR, "approach_comparison.csv"))"

# ══════════════════════════════════════════════════════════════════════════════
# Summary figure
# ══════════════════════════════════════════════════════════════════════════════
@info "\nGenerating summary figure"

p = bar(comparison_data.approach, comparison_data.p1_kr_frac,
    ylabel="Fraction K/R at P1",
    title="Binding Phenotype Inheritance: All Approaches",
    legend=false, rotation=30, bar_width=0.7,
    ylim=(0, 1.05), size=(1000, 500), margin=15Plots.mm,
    color=:steelblue)
hline!([p1_frac(stored_seqs, p1_pos)], linestyle=:dash, color=:red,
    label="Full family baseline", linewidth=2)

savefig(p, joinpath(FIG_DIR, "all_approaches_comparison.pdf"))
savefig(p, joinpath(FIG_DIR, "all_approaches_comparison.png"))
@info "Saved all_approaches_comparison.{pdf,png}"

@info "\n" * "="^60
@info "All four approaches complete!"
@info "="^60
