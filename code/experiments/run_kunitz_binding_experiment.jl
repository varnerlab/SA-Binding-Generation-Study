# ──────────────────────────────────────────────────────────────────────────────
# run_kunitz_binding_experiment.jl
#
# The "cleanest follow-up experiment" from binding.md:
#   Take Kunitz domains (PF00014) with binding data.
#   Split into strong vs. weak binders.
#   Run SA on each subset.
#   Show generated sequences inherit the binding phenotype of the input set.
#
# Kunitz domains are serine protease inhibitors. Many have measured Ki values
# against trypsin. BPTI (bovine pancreatic trypsin inhibitor) is the archetype.
#
# Key interface positions (P1 loop & contact residues, BPTI numbering):
#   Positions 11-18 (binding loop), 34-39 (secondary contact region)
#   P1 residue (position 15): Lys for trypsin, Arg, or other basic residue
# ──────────────────────────────────────────────────────────────────────────────

# --- setup ---
_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

# --- configuration ---
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
# In the Kunitz domain, the P1 residue (contact with trypsin S1 pocket) is the
# primary determinant of inhibitory specificity:
#   - K/R at P1 → strong trypsin inhibitors (Ki in nM range)
#   - Other residues at P1 → weak or non-inhibitors of trypsin
#
# We identify P1 by finding the most conserved Lys position in the alignment
# (BPTI Lys15 equivalent). For the seed alignment, this is typically in the
# binding loop region.

@info "Step 2: Splitting by binding phenotype (P1 residue)"

# Find the P1 position: look for the column with highest Lys frequency
# in the binding loop region (roughly positions 10-25 of cleaned alignment)
lys_fracs = zeros(L)
for j in 1:L
    n_lys = count(i -> char_mat[i, j] == 'K', 1:K_total)
    n_valid = count(i -> char_mat[i, j] != '-' && char_mat[i, j] != '.', 1:K_total)
    lys_fracs[j] = n_valid > 0 ? n_lys / n_valid : 0.0
end

# P1 is the position with highest Lys enrichment (should be >30%)
p1_candidates = findall(f -> f > 0.2, lys_fracs)
if isempty(p1_candidates)
    # fallback: just use the column with max Lys frequency
    p1_pos = argmax(lys_fracs)
else
    p1_pos = p1_candidates[argmax(lys_fracs[p1_candidates])]
end
@info "  P1 position identified: column $p1_pos (Lys fraction: $(round(lys_fracs[p1_pos], digits=2)))"

# Define the binding loop region around P1
binding_loop = collect(max(1, p1_pos - 4):min(L, p1_pos + 4))

# Split: strong binders have K or R at P1 (basic residues → trypsin inhibition)
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
# Step 5: Generate sequences from each subset
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 5: Generating sequences"

gen_all_seqs, gen_all_pca = generate_sequences(X̂_all, pca_all, L;
    β=β_all, n_chains=30, T=5000, seed=42)

if length(strong_idx) >= 5
    gen_strong_seqs, gen_strong_pca = generate_sequences(X̂_strong, pca_strong, L;
        β=β_strong, n_chains=30, T=5000, seed=42)
else
    gen_strong_seqs, gen_strong_pca = String[], Vector{Float64}[]
    @warn "Skipping strong-binder generation (too few patterns)"
end

if length(weak_idx) >= 5
    gen_weak_seqs, gen_weak_pca = generate_sequences(X̂_weak, pca_weak, L;
        β=β_weak, n_chains=30, T=5000, seed=42)
else
    gen_weak_seqs, gen_weak_pca = String[], Vector{Float64}[]
    @warn "Skipping weak-binder generation (too few patterns)"
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Evaluate — do generated sequences inherit the binding phenotype?
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 6: Evaluating binding phenotype inheritance"

function p1_phenotype_analysis(seqs::Vector{String}, label::String, p1_pos::Int)
    n = length(seqs)
    n == 0 && return nothing

    p1_residues = [length(s) >= p1_pos ? s[p1_pos] : '-' for s in seqs]
    p1_counts = sort(collect(StatsBase.countmap(p1_residues)), by=x -> -x[2])

    n_basic = count(r -> r in ('K', 'R'), p1_residues)
    frac_basic = n_basic / n

    @info "  [$label] P1 distribution: $p1_counts"
    @info "  [$label] Fraction K/R at P1: $(round(frac_basic, digits=3)) ($n_basic / $n)"

    return (label=label, n=n, frac_basic=frac_basic, p1_counts=p1_counts)
end

function binding_loop_analysis(seqs::Vector{String}, label::String,
                                binding_loop::Vector{Int})
    n = length(seqs)
    n == 0 && return nothing

    # amino acid composition at binding loop positions
    freq = aa_freq_matrix(seqs, maximum(binding_loop))
    loop_freq = freq[:, binding_loop]

    @info "  [$label] Binding loop composition (positions $binding_loop):"
    for (i, pos) in enumerate(binding_loop)
        top_aa = sortperm(loop_freq[:, i], rev=true)[1:3]
        top_str = join(["$(AA_ALPHABET[a])=$(round(loop_freq[a, i], digits=2))" for a in top_aa], ", ")
        @info "    Position $pos: $top_str"
    end

    return loop_freq
end

# analyze each generation set
results = []

@info "\n=== P1 Phenotype Analysis ==="
push!(results, p1_phenotype_analysis(strong_seqs, "Input: Strong binders", p1_pos))
push!(results, p1_phenotype_analysis(weak_seqs, "Input: Weak binders", p1_pos))
push!(results, p1_phenotype_analysis(gen_all_seqs, "Generated: Full family", p1_pos))

if !isempty(gen_strong_seqs)
    push!(results, p1_phenotype_analysis(gen_strong_seqs, "Generated: Strong-conditioned", p1_pos))
end
if !isempty(gen_weak_seqs)
    push!(results, p1_phenotype_analysis(gen_weak_seqs, "Generated: Weak-conditioned", p1_pos))
end

@info "\n=== Binding Loop Analysis ==="
binding_loop_analysis(strong_seqs, "Input: Strong", binding_loop)
binding_loop_analysis(weak_seqs, "Input: Weak", binding_loop)
binding_loop_analysis(gen_all_seqs, "Gen: Full", binding_loop)
if !isempty(gen_strong_seqs)
    binding_loop_analysis(gen_strong_seqs, "Gen: Strong-conditioned", binding_loop)
end
if !isempty(gen_weak_seqs)
    binding_loop_analysis(gen_weak_seqs, "Gen: Weak-conditioned", binding_loop)
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Standard quality metrics
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 7: Standard quality metrics"

function report_quality(seqs, pca_vecs, X̂, β, stored_seqs, label)
    n = length(seqs)
    n == 0 && return nothing

    kl = aa_composition_kl(seqs, stored_seqs)
    novelties = [sample_novelty(v, X̂) for v in pca_vecs]
    seq_ids = [nearest_sequence_identity(s, stored_seqs) for s in seqs]
    valid_fracs = [valid_residue_fraction(s) for s in seqs]

    @info "  [$label]"
    @info "    KL(AA): $(round(kl, digits=4))"
    @info "    Novelty: $(round(mean(novelties), digits=3)) ± $(round(std(novelties), digits=3))"
    @info "    SeqID:   $(round(mean(seq_ids), digits=3)) ± $(round(std(seq_ids), digits=3))"
    @info "    Valid:   $(round(mean(valid_fracs), digits=3))"

    return (label=label, kl=kl, mean_novelty=mean(novelties),
            mean_seqid=mean(seq_ids), mean_valid=mean(valid_fracs))
end

report_quality(gen_all_seqs, gen_all_pca, X̂_all, β_all, stored_seqs, "Full family")
if !isempty(gen_strong_seqs)
    report_quality(gen_strong_seqs, gen_strong_pca, X̂_strong, β_strong, strong_seqs, "Strong-conditioned")
end
if !isempty(gen_weak_seqs)
    report_quality(gen_weak_seqs, gen_weak_pca, X̂_weak, β_weak, weak_seqs, "Weak-conditioned")
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Visualization
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 8: Generating figures"

# --- Figure 1: P1 residue composition bar chart ---
function plot_p1_comparison(results, p1_pos)
    valid_results = filter(!isnothing, results)
    isempty(valid_results) && return nothing

    labels = [r.label for r in valid_results]
    fracs = [r.frac_basic for r in valid_results]

    p = bar(labels, fracs,
        ylabel="Fraction K/R at P1 (pos $p1_pos)",
        title="P1 Residue Phenotype Inheritance",
        legend=false, rotation=15, bar_width=0.6,
        fillcolor=[:steelblue, :coral, :gray, :steelblue, :coral][1:length(fracs)],
        ylim=(0, 1.05), size=(800, 400), margin=10Plots.mm)
    hline!([0.5], linestyle=:dash, color=:gray, label="")

    savefig(p, joinpath(FIG_DIR, "p1_phenotype_inheritance.pdf"))
    savefig(p, joinpath(FIG_DIR, "p1_phenotype_inheritance.png"))
    @info "  Saved p1_phenotype_inheritance.{pdf,png}"
    return p
end

plot_p1_comparison(results, p1_pos)

# --- Figure 2: Binding loop heatmaps ---
function plot_loop_heatmap(seqs, label, binding_loop, fig_name)
    isempty(seqs) && return nothing
    freq = aa_freq_matrix(seqs, maximum(binding_loop))
    loop_freq = freq[:, binding_loop]

    p = heatmap(1:length(binding_loop), 1:N_AA, loop_freq,
        xticks=(1:length(binding_loop), string.(binding_loop)),
        yticks=(1:N_AA, string.(AA_ALPHABET)),
        xlabel="Binding loop position", ylabel="Amino acid",
        title="$label — Binding loop composition",
        color=:YlOrRd, clims=(0, 1), size=(600, 500))

    savefig(p, joinpath(FIG_DIR, "$(fig_name).pdf"))
    savefig(p, joinpath(FIG_DIR, "$(fig_name).png"))
    @info "  Saved $(fig_name).{pdf,png}"
    return p
end

plot_loop_heatmap(strong_seqs, "Input: Strong Binders", binding_loop, "loop_input_strong")
plot_loop_heatmap(weak_seqs, "Input: Weak Binders", binding_loop, "loop_input_weak")
plot_loop_heatmap(gen_all_seqs, "Generated: Full Family", binding_loop, "loop_gen_all")
if !isempty(gen_strong_seqs)
    plot_loop_heatmap(gen_strong_seqs, "Generated: Strong-Conditioned", binding_loop, "loop_gen_strong")
end
if !isempty(gen_weak_seqs)
    plot_loop_heatmap(gen_weak_seqs, "Generated: Weak-Conditioned", binding_loop, "loop_gen_weak")
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Save results
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 9: Saving results"

# save generated sequences as FASTA
function save_fasta(seqs, filepath, prefix)
    open(filepath, "w") do io
        for (i, seq) in enumerate(seqs)
            println(io, ">$(prefix)_$(lpad(i, 4, '0'))")
            println(io, seq)
        end
    end
    @info "  Saved $(length(seqs)) sequences to $filepath"
end

save_fasta(gen_all_seqs, joinpath(CACHE_DIR, "generated_full_family.fasta"), "SA_full")
if !isempty(gen_strong_seqs)
    save_fasta(gen_strong_seqs, joinpath(CACHE_DIR, "generated_strong_conditioned.fasta"), "SA_strong")
end
if !isempty(gen_weak_seqs)
    save_fasta(gen_weak_seqs, joinpath(CACHE_DIR, "generated_weak_conditioned.fasta"), "SA_weak")
end

@info "\n" * "="^70
@info "Kunitz binding experiment complete!"
@info "="^70
@info "Key question: Do strong-conditioned sequences have higher K/R fraction"
@info "at P1 than weak-conditioned sequences?"
@info ""
@info "If yes → SA inherits binding phenotype from curated memory."
@info "This validates Approach 1 (curated memory matrix) from binding.md."
