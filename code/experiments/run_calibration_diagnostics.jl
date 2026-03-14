# ──────────────────────────────────────────────────────────────────────────────
# run_calibration_diagnostics.jl
#
# Tighten the f_target → f_observed calibration for multiplicity conditioning.
#
# The calibration curve from run_multiplicity_conditioning.jl showed a
# systematic lag: f_observed < f_target. This experiment diagnoses WHY and
# builds a quantitative model of the gap.
#
# Diagnostic axes:
#   A. Fine ρ sweep with replicates (tight error bars)
#   B. Attention weight audit (does the chain actually attend to binders?)
#   C. Soft P1 score (continuous-valued, before argmax decoding)
#   D. β sweep at fixed ρ (is β* optimal for phenotype transfer?)
#   E. PCA separation analysis (how far apart are binders in PCA space?)
#   F. Position-resolved transfer (which positions transfer well, which don't?)
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "calibration")
mkpath(FIG_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Load data (same as multiplicity experiment)
# ══════════════════════════════════════════════════════════════════════════════
@info "Loading Kunitz domain data"
sto_file = download_pfam_seed("PF00014"; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)
stored_seqs = [String(char_mat[i, :]) for i in 1:K_total]

lys_fracs = [count(i -> char_mat[i, j] == 'K', 1:K_total) /
             max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K_total))
             for j in 1:L]
p1_pos = argmax(lys_fracs)
binding_loop = collect(max(1, p1_pos - 4):min(L, p1_pos + 4))
strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
K_b = length(strong_idx)
K_nb = K_total - K_b
natural_frac = K_b / K_total

X̂, pca_model, _, d_full = build_memory_matrix(char_mat; pratio=0.95)
d = size(X̂, 1)
@info "  K=$K_total | K_b=$K_b | d=$d | P1 at position $p1_pos"

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC A: Fine ρ sweep with replicates
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "DIAGNOSTIC A: Fine ρ sweep with replicates"
@info "="^70

ρ_fine = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0,
          75.0, 100.0, 150.0, 200.0, 500.0, 1000.0]
n_reps = 5

fine_results = DataFrame(
    ρ=Float64[], f_eff=Float64[], K_eff=Float64[], β_star=Float64[],
    replicate=Int[], f_observed=Float64[],
    mean_attention_on_binders=Float64[],  # diagnostic B
    soft_p1_kr_score=Float64[],            # diagnostic C
)

for ρ in ρ_fine
    r = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    f_eff = effective_binder_fraction(r, strong_idx)
    K_eff_val = effective_num_patterns(r)
    pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)
    β = pt.β_star
    log_r = log.(r)

    for rep in 1:n_reps
        @info "  ρ=$(round(ρ, digits=1)), rep=$rep"
        Random.seed!(42 * 1000 + rep * 100 + round(Int, ρ))

        # run chains and collect both sequences and attention diagnostics
        gen_seqs = String[]
        attention_on_binders = Float64[]
        soft_p1_scores = Float64[]
        binder_set = Set(strong_idx)

        n_chains = 20
        for chain in 1:n_chains
            k = mod1(chain, K_total)
            ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)

            result = weighted_sample(X̂, ξ₀, 5000, r; β=β, α=0.01,
                                      seed=42 * 1000 + rep * 100 + round(Int, ρ) + chain)

            for t in 2000:100:5000
                ξ = result.Ξ[t + 1, :]
                seq = decode_sample(ξ, pca_model, L)
                push!(gen_seqs, seq)

                # DIAGNOSTIC B: measure attention weight on binder patterns
                logits = β .* (X̂' * ξ) .+ log_r
                attn = NNlib.softmax(logits)
                binder_attn = sum(attn[idx] for idx in strong_idx)
                push!(attention_on_binders, binder_attn)

                # DIAGNOSTIC C: soft P1 score (continuous, before argmax)
                x_onehot = vec(MultivariateStats.reconstruct(pca_model, ξ))
                p1_start = (p1_pos - 1) * N_AA + 1
                p1_block = x_onehot[p1_start:(p1_start + N_AA - 1)]
                # softmax to get probabilities
                p1_probs = NNlib.softmax(p1_block .* 5.0)  # temperature-scaled
                k_idx = AA_TO_IDX['K']
                r_idx = AA_TO_IDX['R']
                soft_kr = p1_probs[k_idx] + p1_probs[r_idx]
                push!(soft_p1_scores, soft_kr)
            end
        end

        # hard P1 fraction
        f_obs = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), gen_seqs) / length(gen_seqs)

        push!(fine_results, (ρ, f_eff, K_eff_val, β, rep, f_obs,
                              mean(attention_on_binders), mean(soft_p1_scores)))
    end
end

# aggregate
fine_agg = combine(groupby(fine_results, :ρ),
    :f_eff => first => :f_eff,
    :K_eff => first => :K_eff,
    :β_star => first => :β_star,
    :f_observed => mean => :f_obs_mean,
    :f_observed => std => :f_obs_std,
    :mean_attention_on_binders => mean => :attn_binder_mean,
    :soft_p1_kr_score => mean => :soft_p1_mean,
)

@info "\nFine calibration results:"
show(stdout, fine_agg)
println()

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC D: β sweep at fixed ρ
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "DIAGNOSTIC D: β sweep at fixed ρ values"
@info "="^70

ρ_fixed = [10.0, 50.0, 200.0]
β_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

beta_sweep_results = DataFrame(
    ρ=Float64[], β_mult=Float64[], β_used=Float64[], β_star=Float64[],
    f_observed=Float64[], diversity=Float64[], mean_valid=Float64[],
)

for ρ in ρ_fixed
    r = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)
    β_star = pt.β_star

    for bm in β_multipliers
        β_use = β_star * bm
        @info "  ρ=$ρ, β=$(round(β_use, digits=2)) ($(bm)× β*)"

        seqs, pca_vecs = generate_weighted_sequences(X̂, pca_model, L, r;
            β=β_use, n_chains=20, T=5000, seed=42)

        n = length(seqs)
        f_obs = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n
        valid = mean(valid_residue_fraction.(seqs))

        # quick diversity
        pair_ids = Float64[]
        for _ in 1:min(300, n * (n - 1) ÷ 2)
            i, j = rand(1:n), rand(1:n)
            while i == j; j = rand(1:n); end
            push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
        end
        div = 1.0 - mean(pair_ids)

        push!(beta_sweep_results, (ρ, bm, β_use, β_star, f_obs, div, valid))
    end
end

@info "\nβ sweep results:"
show(stdout, beta_sweep_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC E: PCA separation analysis
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "DIAGNOSTIC E: PCA separation of binders vs non-binders"
@info "="^70

# cosine similarity between binder centroid and each pattern
binder_centroid = mean(X̂[:, strong_idx], dims=2) |> vec
binder_centroid ./= norm(binder_centroid)

nonbinder_idx = setdiff(1:K_total, strong_idx)
nonbinder_centroid = mean(X̂[:, nonbinder_idx], dims=2) |> vec
nonbinder_centroid ./= norm(nonbinder_centroid)

# centroid separation
centroid_cos = dot(binder_centroid, nonbinder_centroid)
@info "  Binder–nonbinder centroid cosine similarity: $(round(centroid_cos, digits=4))"
@info "  Centroid angular separation: $(round(acos(clamp(centroid_cos, -1, 1)) * 180 / π, digits=1))°"

# within-group vs between-group similarity
within_binder = Float64[]
for i in strong_idx, j in strong_idx
    i >= j && continue
    push!(within_binder, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
end

within_nonbinder = Float64[]
for i in nonbinder_idx, j in nonbinder_idx
    i >= j && continue
    push!(within_nonbinder, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
end

between_groups = Float64[]
for i in strong_idx, j in nonbinder_idx
    push!(between_groups, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
end

@info "  Within-binder cosine sim:     $(round(mean(within_binder), digits=4)) ± $(round(std(within_binder), digits=4))"
@info "  Within-nonbinder cosine sim:  $(round(mean(within_nonbinder), digits=4)) ± $(round(std(within_nonbinder), digits=4))"
@info "  Between-group cosine sim:     $(round(mean(between_groups), digits=4)) ± $(round(std(between_groups), digits=4))"

separation_index = (mean(within_binder) - mean(between_groups)) /
                   (0.5 * (std(within_binder) + std(between_groups)))
@info "  Separation index (Fisher-like): $(round(separation_index, digits=3))"

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC F: Position-resolved phenotype transfer
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "DIAGNOSTIC F: Position-resolved transfer"
@info "="^70

# For each position in the binding loop, measure how well the binder consensus
# transfers at different ρ values

# binder consensus residue at each binding loop position
binder_consensus = Char[]
for pos in binding_loop
    counts = Dict{Char, Int}()
    for idx in strong_idx
        c = char_mat[idx, pos]
        c in ('-', '.') && continue
        counts[c] = get(counts, c, 0) + 1
    end
    if isempty(counts)
        push!(binder_consensus, '-')
    else
        push!(binder_consensus, first(sort(collect(counts), by=x -> -x[2]))[1])
    end
end
@info "  Binder consensus at binding loop: $(String(binder_consensus))"
@info "  Positions: $binding_loop"

# measure transfer at each position for selected ρ values
ρ_pos_sweep = [1.0, 10.0, 100.0, 500.0]
position_results = DataFrame(
    ρ=Float64[], position=Int[], consensus_residue=Char[],
    binder_input_freq=Float64[], generated_freq=Float64[],
    transfer_ratio=Float64[],
)

for ρ in ρ_pos_sweep
    r = multiplicity_vector(K_total, strong_idx; ρ=ρ)
    pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)

    seqs, _ = generate_weighted_sequences(X̂, pca_model, L, r;
        β=pt.β_star, n_chains=30, T=5000, seed=42)

    for (i, pos) in enumerate(binding_loop)
        consensus_aa = binder_consensus[i]
        consensus_aa == '-' && continue

        # frequency in input binders
        input_freq = count(idx -> char_mat[idx, pos] == consensus_aa, strong_idx) / K_b

        # frequency in generated
        gen_freq = count(s -> length(s) >= pos && s[pos] == consensus_aa, seqs) / length(seqs)

        # frequency in full family (background)
        bg_freq = count(idx -> char_mat[idx, pos] == consensus_aa, 1:K_total) / K_total

        # transfer ratio: how much of the binder enrichment is captured?
        # (gen_freq - bg_freq) / (input_freq - bg_freq)
        enrichment = input_freq - bg_freq
        transfer = enrichment > 0.01 ? (gen_freq - bg_freq) / enrichment : NaN

        push!(position_results, (ρ, pos, consensus_aa, input_freq, gen_freq,
                                  isnan(transfer) ? 0.0 : clamp(transfer, -1, 2)))
    end
end

@info "\nPosition-resolved transfer:"
show(stdout, position_results)
println()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "Generating figures"
@info "="^70

# --- Figure 1: Calibration curve with error bars + attention + soft score ---
p1 = plot(layout=(1, 3), size=(1400, 450), margin=10Plots.mm)

# 1a: f_target vs f_observed (tight)
plot!(p1[1], [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal", linewidth=1.5)
plot!(p1[1], fine_agg.f_eff, fine_agg.f_obs_mean,
    ribbon=fine_agg.f_obs_std, fillalpha=0.3,
    marker=:circle, markersize=5, linewidth=2, color=:steelblue, label="observed (hard decode)",
    xlabel="Effective binder fraction (f_eff)", ylabel="Observed P1 K/R fraction",
    title="Calibration: Hard Decode", ylim=(0, 1.05), xlim=(0, 1.05))

# 1b: f_target vs attention on binders
plot!(p1[2], [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal", linewidth=1.5)
plot!(p1[2], fine_agg.f_eff, fine_agg.attn_binder_mean,
    marker=:circle, markersize=5, linewidth=2, color=:coral, label="attention on binders",
    xlabel="Effective binder fraction (f_eff)", ylabel="Mean attention weight on binders",
    title="Calibration: Attention Weights", ylim=(0, 1.05), xlim=(0, 1.05))

# 1c: f_target vs soft P1 score
plot!(p1[3], [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal", linewidth=1.5)
plot!(p1[3], fine_agg.f_eff, fine_agg.soft_p1_mean,
    marker=:circle, markersize=5, linewidth=2, color=:purple, label="soft P1 K/R score",
    xlabel="Effective binder fraction (f_eff)", ylabel="Soft P1 K/R probability",
    title="Calibration: Soft Decode", ylim=(0, 1.05), xlim=(0, 1.05))

savefig(p1, joinpath(FIG_DIR, "fig1_calibration_triptych.pdf"))
savefig(p1, joinpath(FIG_DIR, "fig1_calibration_triptych.png"))
@info "  Saved fig1_calibration_triptych"

# --- Figure 2: Three-layer calibration overlay ---
p2 = plot(size=(700, 500), margin=10Plots.mm,
    xlabel="Effective binder fraction (f_eff)",
    ylabel="Measured fraction / score",
    title="Calibration Gap Decomposition",
    legend=:topleft, ylim=(0, 1.05), xlim=(0, 1.05))

plot!(p2, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)", linewidth=1.5)
plot!(p2, fine_agg.f_eff, fine_agg.attn_binder_mean,
    marker=:diamond, linewidth=2, color=:coral, label="Attention on binders")
plot!(p2, fine_agg.f_eff, fine_agg.soft_p1_mean,
    marker=:square, linewidth=2, color=:purple, label="Soft P1 K/R score")
plot!(p2, fine_agg.f_eff, fine_agg.f_obs_mean,
    ribbon=fine_agg.f_obs_std, fillalpha=0.2,
    marker=:circle, linewidth=2, color=:steelblue, label="Hard P1 K/R fraction")

# annotate the gaps
if nrow(fine_agg) > 0
    mid_idx = nrow(fine_agg) ÷ 2 + 1
    x_mid = fine_agg.f_eff[mid_idx]
    annotate!(p2, x_mid + 0.05, (fine_agg.attn_binder_mean[mid_idx] + fine_agg.soft_p1_mean[mid_idx]) / 2,
        text("PCA\nblur", 8, :left, :red))
    annotate!(p2, x_mid + 0.05, (fine_agg.soft_p1_mean[mid_idx] + fine_agg.f_obs_mean[mid_idx]) / 2,
        text("argmax\nloss", 8, :left, :blue))
end

savefig(p2, joinpath(FIG_DIR, "fig2_gap_decomposition.pdf"))
savefig(p2, joinpath(FIG_DIR, "fig2_gap_decomposition.png"))
@info "  Saved fig2_gap_decomposition"

# --- Figure 3: β sweep ---
p3 = plot(layout=(1, length(ρ_fixed)), size=(400 * length(ρ_fixed), 400), margin=10Plots.mm)
for (i, ρ) in enumerate(ρ_fixed)
    sub = filter(row -> row.ρ == ρ, beta_sweep_results)
    plot!(p3[i], sub.β_mult, sub.f_observed,
        marker=:circle, linewidth=2, color=:steelblue, label="P1 K/R",
        xlabel="β / β*", ylabel="Fraction", ylim=(0, 1.05),
        title="ρ = $ρ")
    plot!(p3[i], sub.β_mult, sub.diversity,
        marker=:diamond, linewidth=2, color=:coral, label="diversity")
    vline!(p3[i], [1.0], linestyle=:dash, color=:gray, label="β*")
end

savefig(p3, joinpath(FIG_DIR, "fig3_beta_sweep.pdf"))
savefig(p3, joinpath(FIG_DIR, "fig3_beta_sweep.png"))
@info "  Saved fig3_beta_sweep"

# --- Figure 4: PCA separation ---
p4 = plot(layout=(1, 2), size=(1000, 400), margin=10Plots.mm)

histogram!(p4[1], within_binder, bins=30, alpha=0.6, label="within binder",
    color=:steelblue, normalize=:pdf)
histogram!(p4[1], between_groups, bins=30, alpha=0.6, label="between groups",
    color=:coral, normalize=:pdf)
histogram!(p4[1], within_nonbinder, bins=30, alpha=0.4, label="within nonbinder",
    color=:gray, normalize=:pdf)
plot!(p4[1], xlabel="Cosine similarity", ylabel="Density",
    title="PCA Space: Binder vs Non-binder Separation")

# 2D PCA projection of patterns colored by binder status
if d >= 2
    scatter!(p4[2], X̂[1, strong_idx], X̂[2, strong_idx],
        marker=:circle, markersize=6, color=:steelblue, label="binders")
    scatter!(p4[2], X̂[1, nonbinder_idx], X̂[2, nonbinder_idx],
        marker=:diamond, markersize=4, color=:coral, label="non-binders", alpha=0.6)
    plot!(p4[2], xlabel="PC1", ylabel="PC2",
        title="First 2 PCs: Binder vs Non-binder")
end

savefig(p4, joinpath(FIG_DIR, "fig4_pca_separation.pdf"))
savefig(p4, joinpath(FIG_DIR, "fig4_pca_separation.png"))
@info "  Saved fig4_pca_separation"

# --- Figure 5: Position-resolved transfer ---
p5 = plot(size=(800, 500), margin=10Plots.mm,
    xlabel="Binding loop position", ylabel="Transfer ratio",
    title="Position-Resolved Phenotype Transfer",
    legend=:topright)

hline!(p5, [1.0], linestyle=:dash, color=:gray, label="perfect transfer")
hline!(p5, [0.0], linestyle=:dot, color=:lightgray, label="")

colors_pos = [:gray, :steelblue, :coral, :purple]
for (i, ρ) in enumerate(ρ_pos_sweep)
    sub = filter(row -> row.ρ == ρ, position_results)
    plot!(p5, sub.position, sub.transfer_ratio,
        marker=:circle, linewidth=2, color=colors_pos[i], label="ρ=$ρ", markersize=5)
end

# mark P1 position
vline!(p5, [p1_pos], linestyle=:dash, color=:red, label="P1", linewidth=2)

savefig(p5, joinpath(FIG_DIR, "fig5_position_transfer.pdf"))
savefig(p5, joinpath(FIG_DIR, "fig5_position_transfer.png"))
@info "  Saved fig5_position_transfer"

# ══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE GAP MODEL
# ══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^70
@info "Quantitative gap model"
@info "="^70

# The calibration gap has two components:
# Gap 1: f_eff → attention_on_binders (should be ≈ identity if well-mixed)
# Gap 2: attention_on_binders → soft_p1 (PCA blurring)
# Gap 3: soft_p1 → f_observed (argmax discretization)

@info "\nGap decomposition at each ρ:"
@info "  ρ       | f_eff | attn  | soft_P1 | hard_P1 | gap_attn | gap_PCA | gap_argmax"
@info "  " * "-"^85

for row in eachrow(fine_agg)
    gap_attn = row.f_eff - row.attn_binder_mean
    gap_pca = row.attn_binder_mean - row.soft_p1_mean
    gap_argmax = row.soft_p1_mean - row.f_obs_mean
    @info "  $(lpad(round(row.ρ, digits=0), 7)) | $(round(row.f_eff, digits=3)) | $(round(row.attn_binder_mean, digits=3)) | $(round(row.soft_p1_mean, digits=3))   | $(round(row.f_obs_mean, digits=3))   | $(round(gap_attn, digits=3))    | $(round(gap_pca, digits=3))   | $(round(gap_argmax, digits=3))"
end

# total gap
total_gaps = fine_agg.f_eff .- fine_agg.f_obs_mean
attn_gaps = fine_agg.f_eff .- fine_agg.attn_binder_mean
pca_gaps = fine_agg.attn_binder_mean .- fine_agg.soft_p1_mean
argmax_gaps = fine_agg.soft_p1_mean .- fine_agg.f_obs_mean

@info "\nMean gap components (averaged across ρ):"
@info "  Total gap:     $(round(mean(total_gaps), digits=3))"
@info "  Attention gap:  $(round(mean(attn_gaps), digits=3)) ($(round(100*mean(attn_gaps)/mean(total_gaps), digits=0))%)"
@info "  PCA blur gap:   $(round(mean(pca_gaps), digits=3)) ($(round(100*mean(pca_gaps)/mean(total_gaps), digits=0))%)"
@info "  Argmax gap:     $(round(mean(argmax_gaps), digits=3)) ($(round(100*mean(argmax_gaps)/mean(total_gaps), digits=0))%)"

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════
CSV.write(joinpath(CACHE_DIR, "calibration_fine.csv"), fine_results)
CSV.write(joinpath(CACHE_DIR, "calibration_fine_agg.csv"), fine_agg)
CSV.write(joinpath(CACHE_DIR, "calibration_beta_sweep.csv"), beta_sweep_results)
CSV.write(joinpath(CACHE_DIR, "calibration_position_transfer.csv"), position_results)

@info "\n" * "="^70
@info "Calibration diagnostics complete!"
@info "="^70
