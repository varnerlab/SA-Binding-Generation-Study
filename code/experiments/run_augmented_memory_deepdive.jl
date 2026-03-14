# ──────────────────────────────────────────────────────────────────────────────
# run_augmented_memory_deepdive.jl
#
# Deep dive into the augmented memory matrix approach for binder design.
#
# Studies four strategies along two axes:
#   1. How many binders do you need? (scaling)
#   2. How should binder information enter the memory? (strategy)
#
# Strategies:
#   A. Hard curation — only binders in the memory (Approach 1, baseline)
#   B. Weighted memory — full family, binders get higher softmax weight
#   C. Interpolation augmentation — binders + convex combos fill subspace
#   D. Mixed memory — tunable blend of binder + non-binder patterns
#   E. Consensus seeding — add explicit binder centroid to full memory
#
# Evaluation metrics for each:
#   - P1 K/R fraction (phenotype fidelity)
#   - Pairwise sequence diversity among generated sequences
#   - Novelty (distance from input)
#   - Valid residue fraction
#   - KL divergence of AA composition
#   - Binding loop composition entropy (how constrained is the interface?)
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "augmented_memory")
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Load Kunitz data (shared across all experiments)
# ══════════════════════════════════════════════════════════════════════════════
@info "Loading Kunitz domain data"
sto_file = download_pfam_seed("PF00014"; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)
stored_seqs = [String(char_mat[i, :]) for i in 1:K_total]

# identify P1 and binders
lys_fracs = [count(i -> char_mat[i, j] == 'K', 1:K_total) /
             max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K_total))
             for j in 1:L]
p1_pos = argmax(lys_fracs)
binding_loop = collect(max(1, p1_pos - 4):min(L, p1_pos + 4))
strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
strong_seqs = stored_seqs[strong_idx]
@info "  $K_total sequences × $L positions | P1 at $p1_pos | $(length(strong_idx)) strong binders"

# build full-family baseline
X̂_full, pca_full, _, _ = build_memory_matrix(char_mat; pratio=0.95)
pt_full = find_entropy_inflection(X̂_full)
β_full = pt_full.β_star

# ══════════════════════════════════════════════════════════════════════════════
# Helper: evaluate a generation run
# ══════════════════════════════════════════════════════════════════════════════
function evaluate_generation(seqs::Vector{String}, pca_vecs::Vector{Vector{Float64}},
                              X̂::Matrix{Float64}, β::Float64,
                              ref_seqs::Vector{String}, p1_pos::Int,
                              binding_loop::Vector{Int})
    n = length(seqs)
    n == 0 && return nothing

    # P1 phenotype
    p1_kr_frac = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n

    # sequence diversity (pairwise identity among generated)
    n_pairs = min(500, n * (n - 1) ÷ 2)  # cap for speed
    pair_ids = Float64[]
    for _ in 1:n_pairs
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
    end
    mean_pairwise_id = mean(pair_ids)
    diversity = 1.0 - mean_pairwise_id

    # novelty (nearest SeqID to ANY input)
    seq_ids = [nearest_sequence_identity(s, ref_seqs) for s in seqs]
    mean_seqid = mean(seq_ids)
    mean_novelty = 1.0 - mean_seqid

    # valid residues
    mean_valid = mean(valid_residue_fraction.(seqs))

    # KL
    kl = aa_composition_kl(seqs, ref_seqs)

    # binding loop entropy — how constrained is the interface?
    freq = aa_freq_matrix(seqs, L)
    loop_entropy = 0.0
    for pos in binding_loop
        pos > L && continue
        for aa in 1:N_AA
            p = freq[aa, pos]
            p > 1e-10 && (loop_entropy -= p * log(p))
        end
    end
    loop_entropy /= length(binding_loop)  # per-position

    return (p1_kr_frac=p1_kr_frac, diversity=diversity,
            mean_novelty=mean_novelty, mean_seqid=mean_seqid,
            mean_valid=mean_valid, kl_aa=kl, loop_entropy=loop_entropy,
            n_generated=n)
end

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 1: Binder Scaling — How many binders do you need?
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 1: Binder Scaling (hard curation)"
@info "="^70

n_binder_values = [3, 5, 8, 10, 15, 20, 25, 32]
n_replicates = 3
scaling_results = DataFrame(
    n_binders=Int[], replicate=Int[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_seqid=Float64[],
    mean_valid=Float64[], kl_aa=Float64[], loop_entropy=Float64[],
)

for n_bind in n_binder_values
    n_use = min(n_bind, length(strong_idx))
    for rep in 1:n_replicates
        @info "  n_binders=$n_use, replicate=$rep"

        # subsample binders
        Random.seed!(1000 * n_bind + rep)
        if n_use >= length(strong_idx)
            sub_idx = strong_idx
        else
            sub_idx = strong_idx[randperm(length(strong_idx))[1:n_use]]
        end

        # build memory from subset
        sub_char = char_mat[sub_idx, :]
        pratio = n_use <= 5 ? 0.99 : 0.95  # keep more variance for tiny sets
        X̂_sub, pca_sub, _, _ = build_memory_matrix(sub_char; pratio=pratio)
        pt_sub = find_entropy_inflection(X̂_sub)
        β_sub = pt_sub.β_star

        # generate
        n_chains = max(10, n_use)
        seqs, pca_vecs = generate_sequences(X̂_sub, pca_sub, L;
            β=β_sub, n_chains=n_chains, T=5000, seed=42 + rep)

        # evaluate
        ev = evaluate_generation(seqs, pca_vecs, X̂_sub, β_sub,
                                  strong_seqs, p1_pos, binding_loop)
        if ev !== nothing
            push!(scaling_results, (n_use, rep, ev.p1_kr_frac, ev.diversity,
                                     ev.mean_novelty, ev.mean_seqid,
                                     ev.mean_valid, ev.kl_aa, ev.loop_entropy))
        end
    end
end

@info "\nScaling results:"
# aggregate by n_binders
scaling_agg = combine(groupby(scaling_results, :n_binders),
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :diversity => mean => :diversity_mean,
    :mean_novelty => mean => :novelty_mean,
    :kl_aa => mean => :kl_mean,
    :loop_entropy => mean => :entropy_mean,
)
show(stdout, scaling_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 2: Weighted Memory — soft curation via softmax bias
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 2: Weighted Memory (soft curation)"
@info "="^70

weight_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
weighted_results = DataFrame(
    binder_weight=Float64[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_seqid=Float64[],
    mean_valid=Float64[], kl_aa=Float64[], loop_entropy=Float64[],
)

for bw in weight_values
    @info "  binder_weight = $bw"
    wmem = build_weighted_memory(char_mat, strong_idx; binder_weight=bw)

    seqs, pca_vecs = generate_weighted_sequences(wmem.X̂, wmem.pca_model, L, wmem.weights;
        β=β_full, n_chains=20, T=5000, seed=42)

    ev = evaluate_generation(seqs, pca_vecs, wmem.X̂, β_full,
                              stored_seqs, p1_pos, binding_loop)
    if ev !== nothing
        push!(weighted_results, (bw, ev.p1_kr_frac, ev.diversity,
                                  ev.mean_novelty, ev.mean_seqid,
                                  ev.mean_valid, ev.kl_aa, ev.loop_entropy))
    end
end

@info "\nWeighted memory results:"
show(stdout, weighted_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 3: Interpolation Augmentation
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 3: Interpolation Augmentation"
@info "="^70

# Start with curated binder memory, then augment with interpolations
strong_char = char_mat[strong_idx, :]
X̂_strong, pca_strong, _, _ = build_memory_matrix(strong_char; pratio=0.95)
pt_strong = find_entropy_inflection(X̂_strong)
β_strong = pt_strong.β_star

n_interp_values = [0, 10, 25, 50, 100, 200]
interp_results = DataFrame(
    n_interp=Int[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_seqid=Float64[],
    mean_valid=Float64[], kl_aa=Float64[], loop_entropy=Float64[],
)

for n_int in n_interp_values
    @info "  n_interpolations = $n_int"

    if n_int == 0
        X̂_use = X̂_strong
    else
        result_aug = augment_memory_interpolations(X̂_strong, collect(1:length(strong_idx));
                                                     n_interp=n_int, seed=42)
        X̂_use = result_aug.X̂_aug
    end

    # re-compute β* for augmented memory
    pt_aug = find_entropy_inflection(X̂_use)
    β_aug = pt_aug.β_star

    seqs, pca_vecs = generate_sequences(X̂_use, pca_strong, L;
        β=β_aug, n_chains=20, T=5000, seed=42)

    ev = evaluate_generation(seqs, pca_vecs, X̂_use, β_aug,
                              strong_seqs, p1_pos, binding_loop)
    if ev !== nothing
        push!(interp_results, (n_int, ev.p1_kr_frac, ev.diversity,
                                ev.mean_novelty, ev.mean_seqid,
                                ev.mean_valid, ev.kl_aa, ev.loop_entropy))
    end
end

@info "\nInterpolation augmentation results:"
show(stdout, interp_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 4: Mixed Memory — binder fraction sweep
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 4: Mixed Memory (binder fraction sweep)"
@info "="^70

binder_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mixed_results = DataFrame(
    binder_frac_target=Float64[], binder_frac_actual=Float64[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_seqid=Float64[],
    mean_valid=Float64[], kl_aa=Float64[], loop_entropy=Float64[],
)

for bf in binder_fractions
    @info "  binder_fraction = $bf"
    mmem = build_mixed_memory(char_mat, strong_idx;
        binder_fraction=bf, target_K=K_total, seed=42)

    pt_mix = find_entropy_inflection(mmem.X̂)
    β_mix = pt_mix.β_star

    seqs, pca_vecs = generate_sequences(mmem.X̂, mmem.pca_model, L;
        β=β_mix, n_chains=20, T=5000, seed=42)

    ev = evaluate_generation(seqs, pca_vecs, mmem.X̂, β_mix,
                              stored_seqs, p1_pos, binding_loop)
    if ev !== nothing
        push!(mixed_results, (bf, mmem.binder_frac, ev.p1_kr_frac, ev.diversity,
                               ev.mean_novelty, ev.mean_seqid,
                               ev.mean_valid, ev.kl_aa, ev.loop_entropy))
    end
end

@info "\nMixed memory results:"
show(stdout, mixed_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 5: Consensus Seeding
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 5: Consensus Seeding"
@info "="^70

consensus_counts = [0, 1, 3, 5, 10, 20]
consensus_results = DataFrame(
    n_consensus=Int[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_seqid=Float64[],
    mean_valid=Float64[], kl_aa=Float64[], loop_entropy=Float64[],
)

for nc in consensus_counts
    @info "  n_consensus = $nc"

    if nc == 0
        X̂_use = X̂_full
    else
        result_cons = build_consensus_seeded_memory(X̂_full, strong_idx;
                                                      n_consensus=nc, seed=42)
        X̂_use = result_cons.X̂_aug
    end

    pt_c = find_entropy_inflection(X̂_use)
    β_c = pt_c.β_star

    seqs, pca_vecs = generate_sequences(X̂_use, pca_full, L;
        β=β_c, n_chains=20, T=5000, seed=42)

    ev = evaluate_generation(seqs, pca_vecs, X̂_use, β_c,
                              stored_seqs, p1_pos, binding_loop)
    if ev !== nothing
        push!(consensus_results, (nc, ev.p1_kr_frac, ev.diversity,
                                   ev.mean_novelty, ev.mean_seqid,
                                   ev.mean_valid, ev.kl_aa, ev.loop_entropy))
    end
end

@info "\nConsensus seeding results:"
show(stdout, consensus_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "Generating figures"
@info "="^70

# --- Figure 1: Scaling study ---
p1 = plot(layout=(1, 3), size=(1200, 400), margin=8Plots.mm,
    title=["P1 Phenotype Fidelity" "Sequence Diversity" "AA KL Divergence"])

plot!(p1[1], scaling_agg.n_binders, scaling_agg.p1_kr_mean,
    ribbon=scaling_agg.p1_kr_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:steelblue, label="",
    xlabel="Number of input binders", ylabel="Fraction K/R at P1", ylim=(0, 1.05))

plot!(p1[2], scaling_agg.n_binders, scaling_agg.diversity_mean,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="Number of input binders", ylabel="Pairwise diversity")

plot!(p1[3], scaling_agg.n_binders, scaling_agg.kl_mean,
    marker=:circle, linewidth=2, color=:forestgreen, label="",
    xlabel="Number of input binders", ylabel="KL(AA)")

savefig(p1, joinpath(FIG_DIR, "study1_binder_scaling.pdf"))
savefig(p1, joinpath(FIG_DIR, "study1_binder_scaling.png"))
@info "  Saved study1_binder_scaling"

# --- Figure 2: Weighted memory ---
p2 = plot(layout=(1, 3), size=(1200, 400), margin=8Plots.mm,
    title=["P1 Phenotype Fidelity" "Sequence Diversity" "AA KL Divergence"])

plot!(p2[1], log10.(weighted_results.binder_weight), weighted_results.p1_kr_frac,
    marker=:circle, linewidth=2, color=:steelblue, label="",
    xlabel="log₁₀(binder weight)", ylabel="Fraction K/R at P1", ylim=(0, 1.05))
hline!(p2[1], [length(strong_idx) / K_total], linestyle=:dash, color=:gray, label="full family")

plot!(p2[2], log10.(weighted_results.binder_weight), weighted_results.diversity,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="log₁₀(binder weight)", ylabel="Pairwise diversity")

plot!(p2[3], log10.(weighted_results.binder_weight), weighted_results.kl_aa,
    marker=:circle, linewidth=2, color=:forestgreen, label="",
    xlabel="log₁₀(binder weight)", ylabel="KL(AA)")

savefig(p2, joinpath(FIG_DIR, "study2_weighted_memory.pdf"))
savefig(p2, joinpath(FIG_DIR, "study2_weighted_memory.png"))
@info "  Saved study2_weighted_memory"

# --- Figure 3: Interpolation augmentation ---
p3 = plot(layout=(1, 3), size=(1200, 400), margin=8Plots.mm,
    title=["P1 Phenotype Fidelity" "Sequence Diversity" "Novelty"])

plot!(p3[1], interp_results.n_interp, interp_results.p1_kr_frac,
    marker=:circle, linewidth=2, color=:steelblue, label="",
    xlabel="N interpolated patterns", ylabel="Fraction K/R at P1", ylim=(0, 1.05))

plot!(p3[2], interp_results.n_interp, interp_results.diversity,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="N interpolated patterns", ylabel="Pairwise diversity")

plot!(p3[3], interp_results.n_interp, interp_results.mean_novelty,
    marker=:circle, linewidth=2, color=:purple, label="",
    xlabel="N interpolated patterns", ylabel="Mean novelty")

savefig(p3, joinpath(FIG_DIR, "study3_interpolation.pdf"))
savefig(p3, joinpath(FIG_DIR, "study3_interpolation.png"))
@info "  Saved study3_interpolation"

# --- Figure 4: Mixed memory ---
p4 = plot(layout=(1, 3), size=(1200, 400), margin=8Plots.mm,
    title=["P1 Phenotype Fidelity" "Sequence Diversity" "AA KL Divergence"])

plot!(p4[1], mixed_results.binder_frac_actual, mixed_results.p1_kr_frac,
    marker=:circle, linewidth=2, color=:steelblue, label="",
    xlabel="Binder fraction in memory", ylabel="Fraction K/R at P1", ylim=(0, 1.05))

plot!(p4[2], mixed_results.binder_frac_actual, mixed_results.diversity,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="Binder fraction in memory", ylabel="Pairwise diversity")

plot!(p4[3], mixed_results.binder_frac_actual, mixed_results.kl_aa,
    marker=:circle, linewidth=2, color=:forestgreen, label="",
    xlabel="Binder fraction in memory", ylabel="KL(AA)")

savefig(p4, joinpath(FIG_DIR, "study4_mixed_memory.pdf"))
savefig(p4, joinpath(FIG_DIR, "study4_mixed_memory.png"))
@info "  Saved study4_mixed_memory"

# --- Figure 5: Consensus seeding ---
p5 = plot(layout=(1, 3), size=(1200, 400), margin=8Plots.mm,
    title=["P1 Phenotype Fidelity" "Sequence Diversity" "Novelty"])

plot!(p5[1], consensus_results.n_consensus, consensus_results.p1_kr_frac,
    marker=:circle, linewidth=2, color=:steelblue, label="",
    xlabel="N consensus patterns added", ylabel="Fraction K/R at P1", ylim=(0, 1.05))

plot!(p5[2], consensus_results.n_consensus, consensus_results.diversity,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="N consensus patterns added", ylabel="Pairwise diversity")

plot!(p5[3], consensus_results.n_consensus, consensus_results.mean_novelty,
    marker=:circle, linewidth=2, color=:purple, label="",
    xlabel="N consensus patterns added", ylabel="Mean novelty")

savefig(p5, joinpath(FIG_DIR, "study5_consensus.pdf"))
savefig(p5, joinpath(FIG_DIR, "study5_consensus.png"))
@info "  Saved study5_consensus"

# --- Figure 6: Grand comparison (Pareto front: phenotype fidelity vs diversity) ---
@info "\nGenerating Pareto front figure"
p6 = plot(size=(700, 500), margin=10Plots.mm,
    xlabel="Pairwise sequence diversity", ylabel="P1 K/R fraction (phenotype fidelity)",
    title="Phenotype Fidelity vs. Diversity\n(Pareto front across strategies)",
    legend=:bottomleft)

# plot each study
scatter!(p6, scaling_agg.diversity_mean, scaling_agg.p1_kr_mean,
    label="Hard curation (N binders)", marker=:circle, markersize=6, color=:steelblue)

scatter!(p6, weighted_results.diversity, weighted_results.p1_kr_frac,
    label="Weighted memory", marker=:diamond, markersize=6, color=:coral)

scatter!(p6, interp_results.diversity, interp_results.p1_kr_frac,
    label="Interpolation augmented", marker=:star5, markersize=7, color=:purple)

scatter!(p6, mixed_results.diversity, mixed_results.p1_kr_frac,
    label="Mixed memory", marker=:utriangle, markersize=6, color=:forestgreen)

scatter!(p6, consensus_results.diversity, consensus_results.p1_kr_frac,
    label="Consensus seeded", marker=:square, markersize=5, color=:orange)

savefig(p6, joinpath(FIG_DIR, "study6_pareto_front.pdf"))
savefig(p6, joinpath(FIG_DIR, "study6_pareto_front.png"))
@info "  Saved study6_pareto_front"

# ══════════════════════════════════════════════════════════════════════════════
# Save all results
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "deepdive_scaling.csv"), scaling_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_weighted.csv"), weighted_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_interpolation.csv"), interp_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_mixed.csv"), mixed_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_consensus.csv"), consensus_results)

@info "\n" * "="^70
@info "Augmented memory deep-dive complete!"
@info "="^70
@info "All results saved to $CACHE_DIR"
@info "All figures saved to $FIG_DIR"
