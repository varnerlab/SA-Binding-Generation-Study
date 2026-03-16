# ──────────────────────────────────────────────────────────────────────────────
# run_augmented_memory_deepdive_with_replicates.jl
#
# Updated version with proper uncertainty quantification across ALL studies.
# Deep dive into the augmented memory matrix approach for binder design.
#
# Studies five strategies along two axes:
#   1. How many binders do you need? (scaling)
#   2. How should binder information enter the memory? (strategy)
#
# Strategies:
#   A. Hard curation — only binders in the memory (Study 1: scaling)
#   B. Weighted memory — full family, binders get higher softmax weight (Study 2)
#   C. Interpolation augmentation — binders + convex combos fill subspace (Study 3)
#   D. Mixed memory — tunable blend of binder + non-binder patterns (Study 4)
#   E. Consensus seeding — add explicit binder centroid to full memory (Study 5)
#
# Key improvements:
#   - Multiple independent replicates (n_reps = 5) for ALL studies
#   - Standard deviations computed across replicates
#   - Error bars in all summary figures
#   - Both raw and aggregated data saved
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = joinpath(_SCRIPT_DIR, "code")
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
    n == 0 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # P1 phenotype
    p1_kr_frac = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n

    # sequence diversity (pairwise identity among generated)
    n_pairs = min(300, n * (n - 1) ÷ 2)  # cap for speed
    pair_ids = Float64[]
    for _ in 1:n_pairs
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
    end
    diversity = 1.0 - mean(pair_ids)

    # novelty (nearest SeqID to ANY input)
    seq_ids = [nearest_sequence_identity(s, ref_seqs) for s in seqs]
    mean_novelty = 1.0 - mean(seq_ids)

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

    return (p1_kr_frac, diversity, mean_novelty, mean_valid, kl, loop_entropy)
end

# ══════════════════════════════════════════════════════════════════════════════
# Experiment parameters
# ══════════════════════════════════════════════════════════════════════════════
n_reps = 5

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 1: Binder Scaling — How many binders do you need? (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 1: Binder Scaling (hard curation) - WITH REPLICATES"
@info "="^70

n_binder_values = [3, 5, 8, 10, 15, 20, 25, 32]
scaling_results = DataFrame(
    n_binders=Int[], replicate=Int[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_valid=Float64[],
    kl_aa=Float64[], loop_entropy=Float64[]
)

for (idx, n_bind) in enumerate(n_binder_values)
    n_use = min(n_bind, length(strong_idx))
    for rep in 1:n_reps
        @info "  n_binders=$n_use, replicate=$rep/$n_reps"

        # subsample binders with unique seed
        seed = 10000 + (idx - 1) * n_reps + rep
        Random.seed!(seed)
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

        # generate with unique seed
        n_chains = max(10, n_use)
        seqs, pca_vecs = generate_sequences(X̂_sub, pca_sub, L;
            β=β_sub, n_chains=n_chains, T=5000, seed=seed)

        # evaluate
        metrics = evaluate_generation(seqs, pca_vecs, X̂_sub, β_sub,
                                      strong_seqs, p1_pos, binding_loop)
        push!(scaling_results, (n_use, rep, metrics...))
    end
end

# Aggregate Study 1 results
scaling_agg = combine(groupby(scaling_results, :n_binders),
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :diversity => mean => :diversity_mean,
    :diversity => std => :diversity_std,
    :mean_novelty => mean => :novelty_mean,
    :mean_novelty => std => :novelty_std,
    :mean_valid => mean => :valid_mean,
    :mean_valid => std => :valid_std,
    :kl_aa => mean => :kl_mean,
    :kl_aa => std => :kl_std,
    :loop_entropy => mean => :entropy_mean,
    :loop_entropy => std => :entropy_std
)

@info "\nStudy 1 (Scaling) results with uncertainty:"
show(stdout, scaling_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 2: Weighted Memory — soft curation via softmax bias (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 2: Weighted Memory (soft curation) - WITH REPLICATES"
@info "="^70

weight_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
weighted_results = DataFrame(
    binder_weight=Float64[], replicate=Int[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_valid=Float64[],
    kl_aa=Float64[], loop_entropy=Float64[]
)

for (idx, bw) in enumerate(weight_values)
    @info "  binder_weight = $bw"
    wmem = build_weighted_memory(char_mat, strong_idx; binder_weight=bw)

    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 20000 + (idx - 1) * n_reps + rep
        Random.seed!(seed)

        seqs, pca_vecs = generate_weighted_sequences(wmem.X̂, wmem.pca_model, L, wmem.weights;
            β=β_full, n_chains=20, T=5000, seed=seed)

        metrics = evaluate_generation(seqs, pca_vecs, wmem.X̂, β_full,
                                      stored_seqs, p1_pos, binding_loop)
        push!(weighted_results, (bw, rep, metrics...))
    end
end

# Aggregate Study 2 results
weighted_agg = combine(groupby(weighted_results, :binder_weight),
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :diversity => mean => :diversity_mean,
    :diversity => std => :diversity_std,
    :mean_novelty => mean => :novelty_mean,
    :mean_novelty => std => :novelty_std,
    :mean_valid => mean => :valid_mean,
    :mean_valid => std => :valid_std,
    :kl_aa => mean => :kl_mean,
    :kl_aa => std => :kl_std,
    :loop_entropy => mean => :entropy_mean,
    :loop_entropy => std => :entropy_std
)

@info "\nStudy 2 (Weighted) results with uncertainty:"
show(stdout, weighted_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 3: Interpolation Augmentation (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 3: Interpolation Augmentation - WITH REPLICATES"
@info "="^70

n_interpolants = [0, 5, 10, 20, 30, 50]
interp_results = DataFrame(
    n_interpolants=Int[], replicate=Int[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_valid=Float64[],
    kl_aa=Float64[], loop_entropy=Float64[]
)

for (idx, n_int) in enumerate(n_interpolants)
    @info "  n_interpolants = $n_int"

    # build base set (strong binders + interpolants)
    augmented_idx = copy(strong_idx)
    if n_int > 0
        # add interpolated patterns between random binder pairs
        Random.seed!(100 + n_int)  # deterministic interpolants
        aug_patterns = Matrix{Char}(undef, n_int, L)
        for i in 1:n_int
            # pick two random binders
            a, b = rand(strong_idx, 2)
            α = rand(0.2:0.1:0.8)  # interpolation weight
            # simple character-level interpolation (pick from a or b per position)
            for j in 1:L
                aug_patterns[i, j] = rand() < α ? char_mat[a, j] : char_mat[b, j]
            end
        end
        # add to char_mat temporarily
        full_mat = vcat(char_mat, aug_patterns)
        augmented_idx = vcat(strong_idx, (K_total + 1):(K_total + n_int))
        char_use = full_mat
    else
        char_use = char_mat
    end

    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 30000 + (idx - 1) * n_reps + rep
        Random.seed!(seed)

        # build memory from augmented set
        aug_char = char_use[augmented_idx, :]
        X̂_aug, pca_aug, _, _ = build_memory_matrix(aug_char; pratio=0.95)
        pt_aug = find_entropy_inflection(X̂_aug)
        β_aug = pt_aug.β_star

        # generate
        seqs, pca_vecs = generate_sequences(X̂_aug, pca_aug, L;
            β=β_aug, n_chains=20, T=5000, seed=seed)

        metrics = evaluate_generation(seqs, pca_vecs, X̂_aug, β_aug,
                                      strong_seqs, p1_pos, binding_loop)
        push!(interp_results, (n_int, rep, metrics...))
    end
end

# Aggregate Study 3 results
interp_agg = combine(groupby(interp_results, :n_interpolants),
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :diversity => mean => :diversity_mean,
    :diversity => std => :diversity_std,
    :mean_novelty => mean => :novelty_mean,
    :mean_novelty => std => :novelty_std,
    :mean_valid => mean => :valid_mean,
    :mean_valid => std => :valid_std,
    :kl_aa => mean => :kl_mean,
    :kl_aa => std => :kl_std,
    :loop_entropy => mean => :entropy_mean,
    :loop_entropy => std => :entropy_std
)

@info "\nStudy 3 (Interpolation) results with uncertainty:"
show(stdout, interp_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 4: Mixed Memory — binder fraction sweep (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 4: Mixed Memory (binder fraction sweep) - WITH REPLICATES"
@info "="^70

binder_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mixed_results = DataFrame(
    binder_frac_target=Float64[], replicate=Int[],
    binder_frac_actual=Float64[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_valid=Float64[],
    kl_aa=Float64[], loop_entropy=Float64[]
)

for (idx, bf) in enumerate(binder_fractions)
    @info "  binder_fraction = $bf"

    # build mixed memory
    n_bind_target = round(Int, bf * 30)  # target ~30 total patterns
    n_nonb_target = 30 - n_bind_target

    n_bind_actual = min(n_bind_target, length(strong_idx))
    weak_idx = setdiff(1:K_total, strong_idx)
    n_nonb_actual = min(n_nonb_target, length(weak_idx))

    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 40000 + (idx - 1) * n_reps + rep
        Random.seed!(seed)

        # subsample both groups with unique seeds
        bind_sub = n_bind_actual >= length(strong_idx) ? strong_idx :
                   strong_idx[randperm(length(strong_idx))[1:n_bind_actual]]
        nonb_sub = n_nonb_actual >= length(weak_idx) ? weak_idx[1:n_nonb_actual] :
                   weak_idx[randperm(length(weak_idx))[1:n_nonb_actual]]

        mixed_idx = vcat(bind_sub, nonb_sub)
        actual_bf = length(bind_sub) / length(mixed_idx)

        # build memory
        mixed_char = char_mat[mixed_idx, :]
        X̂_mix, pca_mix, _, _ = build_memory_matrix(mixed_char; pratio=0.95)
        pt_mix = find_entropy_inflection(X̂_mix)
        β_mix = pt_mix.β_star

        # generate
        seqs, pca_vecs = generate_sequences(X̂_mix, pca_mix, L;
            β=β_mix, n_chains=20, T=5000, seed=seed)

        metrics = evaluate_generation(seqs, pca_vecs, X̂_mix, β_mix,
                                      stored_seqs, p1_pos, binding_loop)
        push!(mixed_results, (bf, rep, actual_bf, metrics...))
    end
end

# Aggregate Study 4 results
mixed_agg = combine(groupby(mixed_results, :binder_frac_target),
    :binder_frac_actual => mean => :binder_frac_actual_mean,
    :binder_frac_actual => std => :binder_frac_actual_std,
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :diversity => mean => :diversity_mean,
    :diversity => std => :diversity_std,
    :mean_novelty => mean => :novelty_mean,
    :mean_novelty => std => :novelty_std,
    :mean_valid => mean => :valid_mean,
    :mean_valid => std => :valid_std,
    :kl_aa => mean => :kl_mean,
    :kl_aa => std => :kl_std,
    :loop_entropy => mean => :entropy_mean,
    :loop_entropy => std => :entropy_std
)

@info "\nStudy 4 (Mixed Memory) results with uncertainty:"
show(stdout, mixed_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# STUDY 5: Consensus Seeding (WITH REPLICATES)
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "STUDY 5: Consensus Seeding - WITH REPLICATES"
@info "="^70

n_consensus_patterns = [0, 1, 2, 3, 5]
consensus_results = DataFrame(
    n_consensus=Int[], replicate=Int[],
    p1_kr_frac=Float64[], diversity=Float64[],
    mean_novelty=Float64[], mean_valid=Float64[],
    kl_aa=Float64[], loop_entropy=Float64[]
)

for (idx, n_cons) in enumerate(n_consensus_patterns)
    @info "  n_consensus = $n_cons"

    # build consensus patterns (deterministic)
    consensus_patterns = Matrix{Char}(undef, max(1, n_cons), L)
    if n_cons > 0
        # simple consensus: most frequent residue at each position among binders
        for j in 1:L
            counts = Dict{Char, Int}()
            for i in strong_idx
                c = char_mat[i, j]
                c in ('-', '.') && continue
                counts[c] = get(counts, c, 0) + 1
            end
            consensus_aa = isempty(counts) ? 'A' : first(sort(collect(counts), by=x -> -x[2]))[1]

            # add some variation for multiple consensus patterns
            for k in 1:n_cons
                if k == 1
                    consensus_patterns[k, j] = consensus_aa
                else
                    # slight variation: second/third most common, or random
                    if length(counts) >= k
                        consensus_patterns[k, j] = sort(collect(counts), by=x -> -x[2])[k][1]
                    else
                        consensus_patterns[k, j] = consensus_aa
                    end
                end
            end
        end
    else
        consensus_patterns = Matrix{Char}(undef, 0, L)  # empty for n_cons=0
    end

    for rep in 1:n_reps
        @info "    replicate $rep/$n_reps"
        seed = 50000 + (idx - 1) * n_reps + rep
        Random.seed!(seed)

        # build memory: full family + consensus patterns
        if n_cons > 0
            full_mat = vcat(char_mat, consensus_patterns)
            # weight consensus patterns highly in weighted memory
            weights = vcat(ones(K_total), fill(10.0, n_cons))

            X̂_cons, pca_cons, _, _ = build_memory_matrix(full_mat; pratio=0.95)

            # generate with weighted sampling
            seqs, pca_vecs = generate_weighted_sequences(X̂_cons, pca_cons, L, weights;
                β=β_full, n_chains=20, T=5000, seed=seed)
        else
            # baseline: just full family
            seqs, pca_vecs = generate_sequences(X̂_full, pca_full, L;
                β=β_full, n_chains=20, T=5000, seed=seed)
        end

        metrics = evaluate_generation(seqs, pca_vecs, X̂_full, β_full,
                                      stored_seqs, p1_pos, binding_loop)
        push!(consensus_results, (n_cons, rep, metrics...))
    end
end

# Aggregate Study 5 results
consensus_agg = combine(groupby(consensus_results, :n_consensus),
    :p1_kr_frac => mean => :p1_kr_mean,
    :p1_kr_frac => std => :p1_kr_std,
    :diversity => mean => :diversity_mean,
    :diversity => std => :diversity_std,
    :mean_novelty => mean => :novelty_mean,
    :mean_novelty => std => :novelty_std,
    :mean_valid => mean => :valid_mean,
    :mean_valid => std => :valid_std,
    :kl_aa => mean => :kl_mean,
    :kl_aa => std => :kl_std,
    :loop_entropy => mean => :entropy_mean,
    :loop_entropy => std => :entropy_std
)

@info "\nStudy 5 (Consensus Seeding) results with uncertainty:"
show(stdout, consensus_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES WITH ERROR BARS
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "Generating figures with error bars"
@info "="^70

# Figure 1: Study 1 (Scaling)
p1 = plot(layout=(2, 2), size=(1000, 800), margin=10Plots.mm)

plot!(p1[1], scaling_agg.n_binders, scaling_agg.p1_kr_mean,
    ribbon=scaling_agg.p1_kr_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:steelblue, label="",
    xlabel="N binders in memory", ylabel="P1 K/R fraction",
    title="Study 1: Hard Curation Scaling")

plot!(p1[2], scaling_agg.n_binders, scaling_agg.diversity_mean,
    ribbon=scaling_agg.diversity_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="N binders in memory", ylabel="Sequence diversity",
    title="Diversity vs Memory Size")

plot!(p1[3], scaling_agg.n_binders, scaling_agg.kl_mean,
    ribbon=scaling_agg.kl_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:forestgreen, label="",
    xlabel="N binders in memory", ylabel="KL(AA composition)",
    title="Composition Divergence")

plot!(p1[4], scaling_agg.n_binders, scaling_agg.entropy_mean,
    ribbon=scaling_agg.entropy_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:purple, label="",
    xlabel="N binders in memory", ylabel="Binding loop entropy",
    title="Interface Constraint")

savefig(p1, joinpath(FIG_DIR, "study1_scaling_with_errors.pdf"))
savefig(p1, joinpath(FIG_DIR, "study1_scaling_with_errors.png"))
@info "  Saved study1_scaling_with_errors"

# Figure 2: Study 2 (Weighted Memory)
p2 = plot(layout=(2, 2), size=(1000, 800), margin=10Plots.mm)

plot!(p2[1], log10.(weighted_agg.binder_weight), weighted_agg.p1_kr_mean,
    ribbon=weighted_agg.p1_kr_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:steelblue, label="",
    xlabel="log₁₀(binder weight)", ylabel="P1 K/R fraction",
    title="Study 2: Weighted Memory")

plot!(p2[2], log10.(weighted_agg.binder_weight), weighted_agg.diversity_mean,
    ribbon=weighted_agg.diversity_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="log₁₀(binder weight)", ylabel="Sequence diversity",
    title="Diversity vs Weight")

plot!(p2[3], log10.(weighted_agg.binder_weight), weighted_agg.kl_mean,
    ribbon=weighted_agg.kl_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:forestgreen, label="",
    xlabel="log₁₀(binder weight)", ylabel="KL(AA composition)",
    title="Composition Divergence")

plot!(p2[4], log10.(weighted_agg.binder_weight), weighted_agg.entropy_mean,
    ribbon=weighted_agg.entropy_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:purple, label="",
    xlabel="log₁₀(binder weight)", ylabel="Binding loop entropy",
    title="Interface Constraint")

savefig(p2, joinpath(FIG_DIR, "study2_weighted_with_errors.pdf"))
savefig(p2, joinpath(FIG_DIR, "study2_weighted_with_errors.png"))
@info "  Saved study2_weighted_with_errors"

# Figure 3: Studies 3-5 comparison
p3 = plot(layout=(1, 3), size=(1200, 400), margin=10Plots.mm)

# Study 3: Interpolation
plot!(p3[1], interp_agg.n_interpolants, interp_agg.p1_kr_mean,
    ribbon=interp_agg.p1_kr_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:purple, label="",
    xlabel="N interpolants", ylabel="P1 K/R fraction",
    title="Study 3: Interpolation Augmentation")

# Study 4: Mixed Memory
plot!(p3[2], mixed_agg.binder_frac_target, mixed_agg.p1_kr_mean,
    ribbon=mixed_agg.p1_kr_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:forestgreen, label="",
    xlabel="Target binder fraction", ylabel="P1 K/R fraction",
    title="Study 4: Mixed Memory")

# Study 5: Consensus Seeding
plot!(p3[3], consensus_agg.n_consensus, consensus_agg.p1_kr_mean,
    ribbon=consensus_agg.p1_kr_std, fillalpha=0.3,
    marker=:circle, linewidth=2, color=:orange, label="",
    xlabel="N consensus patterns", ylabel="P1 K/R fraction",
    title="Study 5: Consensus Seeding")

savefig(p3, joinpath(FIG_DIR, "studies345_comparison_with_errors.pdf"))
savefig(p3, joinpath(FIG_DIR, "studies345_comparison_with_errors.png"))
@info "  Saved studies345_comparison_with_errors"

# Figure 4: Grand Pareto front with error bars
p4 = plot(size=(800, 600), margin=15Plots.mm,
    xlabel="Pairwise sequence diversity", ylabel="P1 K/R fraction (phenotype fidelity)",
    title="Phenotype Fidelity vs. Diversity\n(Pareto front across all strategies)",
    legend=:bottomleft)

# Plot each study with error bars
scatter!(p4, scaling_agg.diversity_mean, scaling_agg.p1_kr_mean,
        xerror=scaling_agg.diversity_std, yerror=scaling_agg.p1_kr_std,
        label="Hard curation (N binders)", marker=:circle, markersize=6, color=:steelblue)

scatter!(p4, weighted_agg.diversity_mean, weighted_agg.p1_kr_mean,
        xerror=weighted_agg.diversity_std, yerror=weighted_agg.p1_kr_std,
        label="Weighted memory", marker=:diamond, markersize=6, color=:coral)

scatter!(p4, interp_agg.diversity_mean, interp_agg.p1_kr_mean,
        xerror=interp_agg.diversity_std, yerror=interp_agg.p1_kr_std,
        label="Interpolation augmented", marker=:star5, markersize=7, color=:purple)

scatter!(p4, mixed_agg.diversity_mean, mixed_agg.p1_kr_mean,
        xerror=mixed_agg.diversity_std, yerror=mixed_agg.p1_kr_std,
        label="Mixed memory", marker=:utriangle, markersize=6, color=:forestgreen)

scatter!(p4, consensus_agg.diversity_mean, consensus_agg.p1_kr_mean,
        xerror=consensus_agg.diversity_std, yerror=consensus_agg.p1_kr_std,
        label="Consensus seeded", marker=:square, markersize=5, color=:orange)

savefig(p4, joinpath(FIG_DIR, "grand_pareto_front_with_errors.pdf"))
savefig(p4, joinpath(FIG_DIR, "grand_pareto_front_with_errors.png"))
@info "  Saved grand_pareto_front_with_errors"

# ══════════════════════════════════════════════════════════════════════════════
# Save all results (both raw and aggregated)
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "deepdive_scaling_raw_replicates.csv"), scaling_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_scaling_aggregated.csv"), scaling_agg)

CSV.write(joinpath(CACHE_DIR, "deepdive_weighted_raw_replicates.csv"), weighted_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_weighted_aggregated.csv"), weighted_agg)

CSV.write(joinpath(CACHE_DIR, "deepdive_interpolation_raw_replicates.csv"), interp_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_interpolation_aggregated.csv"), interp_agg)

CSV.write(joinpath(CACHE_DIR, "deepdive_mixed_raw_replicates.csv"), mixed_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_mixed_aggregated.csv"), mixed_agg)

CSV.write(joinpath(CACHE_DIR, "deepdive_consensus_raw_replicates.csv"), consensus_results)
CSV.write(joinpath(CACHE_DIR, "deepdive_consensus_aggregated.csv"), consensus_agg)

@info "\n" * "="^70
@info "Augmented memory deep-dive complete (WITH UNCERTAINTY)!"
@info "="^70
@info "\nKey improvements:"
@info "  1. ALL 5 studies now run with $n_reps independent replicates"
@info "  2. Standard deviations computed across replicates for all metrics"
@info "  3. All figures include error bars (ribbons and error bars)"
@info "  4. Both raw and aggregated data saved for all studies"
@info "  5. Comprehensive Pareto front analysis with uncertainty bounds"
@info "\nAll results saved to $CACHE_DIR"