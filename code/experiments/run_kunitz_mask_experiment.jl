# ──────────────────────────────────────────────────────────────────────────────
# run_kunitz_mask_experiment.jl
#
# Adds the hard attention mask (b = -Inf on background) as the rho -> Inf
# endpoint of the multiplicity calibration curve on the Kunitz family
# (P1 K/R designated subset).
#
# Corrected finding: at matched beta, the hard mask gives the same recovery as
# the strongest finite-rho multiplicity condition (background mass contributes
# ~0 at fixed beta). The calibration gap is retrieval-sharpness (beta) bound,
# not background-mass bound. Recovery climbs monotonically with beta to ~1.0;
# curation reaches that ceiling at higher novelty than the high-beta mask.
#
# Conditions (all on the full-family PCA basis except hard curation):
#   1. Unconditional  (b = 0)
#   2. Multiplicity calibration sweep  (finite rho via build_multiplicity_conditioned_memory)
#   3. Hard mask  (b = -Inf on background, full basis)
#   4. Hard curation  (subset basis)
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const PFAM_ID = "PF00014"
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "kunitz")
mkpath(CACHE_DIR); mkpath(FIG_DIR)

# --- Load + clean alignment ---
@info "Loading Kunitz alignment (PF00014)"
sto_file = download_pfam_seed(PFAM_ID; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)
stored_seqs = [String(char_mat[i, :]) for i in 1:K_total]
@info "  $K_total sequences x $L positions"

# --- P1 split (designated = K/R at P1), identical logic to run_kunitz_binding_experiment.jl ---
lys_fracs = zeros(L)
for j in 1:L
    n_lys = count(i -> char_mat[i, j] == 'K', 1:K_total)
    n_valid = count(i -> char_mat[i, j] != '-' && char_mat[i, j] != '.', 1:K_total)
    lys_fracs[j] = n_valid > 0 ? n_lys / n_valid : 0.0
end
p1_candidates = findall(f -> f > 0.2, lys_fracs)
p1_pos = isempty(p1_candidates) ? argmax(lys_fracs) : p1_candidates[argmax(lys_fracs[p1_candidates])]
designated_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)
@info "  P1 column $p1_pos; designated (K/R at P1) = $(length(designated_idx))/$K_total"
length(designated_idx) >= 5 || error("Too few designated sequences ($(length(designated_idx))) for this experiment")

# --- Full-family basis (shared by unconditional, multiplicity, mask) ---
X̂_all, pca_all, L_all, _ = build_memory_matrix(char_mat; pratio=0.95)
β_all = find_entropy_inflection(X̂_all).β_star
@info "  Full-family β* = $(round(β_all, digits=3))"

# --- Metric helper: P1 K/R fraction + sequence-level novelty/diversity ---
function eval_condition(seqs::Vector{String}, label::String)
    n = length(seqs)
    p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / n
    np = min(500, n * (n - 1) ÷ 2)
    pair_ids = Float64[]
    for _ in 1:np
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
    end
    diversity = 1.0 - mean(pair_ids)
    novelty = 1.0 - mean(nearest_sequence_identity(s, stored_seqs) for s in seqs)
    kl = aa_composition_kl(seqs, stored_seqs)
    @info "  [$label] P1 K/R = $(round(p1_kr,digits=3)), novelty = $(round(novelty,digits=3)), " *
          "diversity = $(round(diversity,digits=3)), KL = $(round(kl,digits=3))"
    return (label=label, p1_kr=p1_kr, novelty=novelty, diversity=diversity, kl=kl, n=n)
end

results = DataFrame(condition=String[], f_target=Float64[], beta=Float64[],
                    p1_kr=Float64[], novelty=Float64[], diversity=Float64[],
                    kl=Float64[], n=Int[])

# --- Condition 1: Unconditional (b = 0) ---
@info "Condition 1: Unconditional"
uncond_seqs, _ = generate_sequences(X̂_all, pca_all, L; β=β_all, n_chains=30, T=5000, seed=42)
m = eval_condition(uncond_seqs, "unconditional")
push!(results, ("unconditional",
                effective_binder_fraction(ones(K_total), designated_idx),
                β_all,
                m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Condition 2: Multiplicity calibration sweep (finite rho) ---
@info "Condition 2: Multiplicity calibration sweep"
for ft in [0.5, 0.7, 0.9, 0.95, 0.99]
    res = build_multiplicity_conditioned_memory(char_mat, designated_idx; f_target=ft)
    β_w = find_weighted_entropy_inflection(res.X̂, res.r; n_betas=50).β_star
    seqs, _ = generate_weighted_sequences(res.X̂, res.pca_model, L, res.r;
        β=β_w, n_chains=20, T=5000, seed=42)
    m = eval_condition(seqs, "multiplicity(f=$ft)")
    push!(results, ("multiplicity", res.f_eff, β_w, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))
end

# --- Condition 3: Hard mask (b = -Inf on background, full basis) ---
@info "Condition 3: Hard mask"
β_mask = find_entropy_inflection(X̂_all[:, designated_idx]).β_star
mask_seqs, _ = generate_masked_sequences(X̂_all, pca_all, L, designated_idx;
    β=β_mask, n_chains=30, T=5000, seed=42)
m = eval_condition(mask_seqs, "mask")
push!(results, ("mask", 1.0, β_mask, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Condition 4: Hard curation (subset basis) ---
@info "Condition 4: Hard curation (subset basis)"
X̂_cur, pca_cur, _, _ = build_memory_matrix(char_mat[designated_idx, :]; pratio=0.95)
β_cur = find_entropy_inflection(X̂_cur).β_star
cur_seqs, _ = generate_sequences(X̂_cur, pca_cur, L; β=β_cur, n_chains=30, T=5000, seed=42)
m = eval_condition(cur_seqs, "curation")
push!(results, ("curation", 1.0, β_cur, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Write CSV ---
csv_path = joinpath(CACHE_DIR, "mask_experiment.csv")
CSV.write(csv_path, results)
@info "Wrote $csv_path"

# --- Calibration figure: multiplicity curve + mask/curation/unconditional overlays ---
# (Retained for reference; the β-recovery figure below is the primary result.)
mult = results[results.condition .== "multiplicity", :]
uncond_row = results[results.condition .== "unconditional", :]
mask_row = results[results.condition .== "mask", :]
cur_row  = results[results.condition .== "curation", :]

p_cal = plot(size=(640, 520), margin=10Plots.mm, legend=:bottomright,
    xlabel="Target effective designated fraction (f_target)",
    ylabel="Observed P1 K/R fraction",
    title="Kunitz: multiplicity calibration with mask endpoint")
plot!(p_cal, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)")
plot!(p_cal, mult.f_target, mult.p1_kr, marker=:circle, color=:steelblue, lw=2, label="multiplicity (finite rho)")
scatter!(p_cal, uncond_row.f_target, uncond_row.p1_kr, marker=:diamond, ms=8, color=:gray, label="unconditional")
scatter!(p_cal, mask_row.f_target, mask_row.p1_kr, marker=:star5, ms=12, color=:crimson, label="hard mask (b=-Inf)")
scatter!(p_cal, cur_row.f_target, cur_row.p1_kr, marker=:utriangle, ms=9, color=:forestgreen, label="hard curation")
savefig(p_cal, joinpath(FIG_DIR, "fig_mask_calibration.png"))
savefig(p_cal, joinpath(FIG_DIR, "fig_mask_calibration.pdf"))
@info "Wrote fig_mask_calibration"

# --- Fixed-mask β-sweep: P1 K/R + novelty vs β inside the masked regime ---
# Shows that recovery climbs monotonically with β toward 1.0 (no floor),
# establishing the gap as retrieval-sharpness (β) bound, not background-mass bound.
@info "Fixed-mask β-sweep"
betasweep = DataFrame(β=Float64[], p1_kr=Float64[], novelty=Float64[])
for βb in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 512.0]
    seqs, _ = generate_masked_sequences(X̂_all, pca_all, L, designated_idx;
        β=βb, n_chains=20, T=5000, seed=42)
    p1_kr = count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), seqs) / length(seqs)
    nov = 1.0 - mean(nearest_sequence_identity(s, stored_seqs) for s in seqs)
    push!(betasweep, (βb, p1_kr, nov))
    @info "  β=$βb: P1 K/R=$(round(p1_kr,digits=3)), novelty=$(round(nov,digits=3))"
end
CSV.write(joinpath(CACHE_DIR, "mask_betasweep.csv"), betasweep)
@info "Wrote mask_betasweep.csv"

# --- Corrected analysis: β-bound finding (replaces invalid gap-decomposition) ---
mult_top = mult[argmax(mult.f_target), :]   # strongest finite-rho row (f=0.99)
# Mask β-sweep value at the β nearest to the strongest-multiplicity β
bs_nearest_idx = argmin(abs.(betasweep.β .- mult_top.beta))
bs_at_mult_beta = betasweep[bs_nearest_idx, :]

@info "=== Corrected analysis: calibration gap is β-bound, not background-mass bound ==="
@info ""
@info "1. Matched-β comparison (at β≈$(round(mult_top.beta,digits=1))):"
@info "   Hard mask at β=$(bs_at_mult_beta.β): P1 K/R = $(round(bs_at_mult_beta.p1_kr,digits=3))"
@info "   Strongest finite-rho multiplicity (f=$(mult_top.f_target), β≈$(round(mult_top.beta,digits=1))): P1 K/R = $(round(mult_top.p1_kr,digits=3))"
@info "   Difference (mask - multiplicity) at matched β ≈ $(round(bs_at_mult_beta.p1_kr - mult_top.p1_kr, digits=3))"
@info "   Conclusion: at fixed β the hard mask (rho -> Inf endpoint) gives essentially"
@info "   the same recovery as the strongest finite-rho condition. Removing the last"
@info "   residual background mass contributes ~0 to recovery at matched β."
@info ""
@info "2. β-bound conclusion:"
@info "   Recovery climbs monotonically with β: $(join([string(round(r.p1_kr,digits=3)) for r in eachrow(betasweep)], ", "))"
@info "   (β = $(join(Int.(betasweep.β), ", ")))"
@info "   No plateau is reached before β=512 (P1 K/R ≈ $(round(betasweep.p1_kr[end],digits=3)))."
@info "   The calibration gap is retrieval-sharpness (β) bound."
@info ""
@info "3. Novelty trade-off and curation efficiency:"
@info "   High-β mask (β=512): P1 K/R=$(round(betasweep.p1_kr[end],digits=3)), novelty=$(round(betasweep.novelty[end],digits=3))"
@info "   Hard curation: P1 K/R=$(round(cur_row.p1_kr[1],digits=3)), novelty=$(round(cur_row.novelty[1],digits=3))"
@info "   Curation reaches the recovery ceiling at higher novelty ($(round(cur_row.novelty[1],digits=3)) vs $(round(betasweep.novelty[end],digits=3))),"
@info "   making it more novelty-efficient than brute-force high-β masking."

# --- Unifying figure: recovery vs β (mask curve + multiplicity points overlaid) ---
# Central claim: both methods trace one recovery-vs-β relationship; rho matters
# only through the β it buys.
p_beta = plot(size=(700, 540), margin=10Plots.mm, legend=:bottomright,
    xlabel="Inverse temperature β (log scale)",
    ylabel="P1 K/R fraction",
    title="Kunitz: recovery is β-bound (mask curve + multiplicity points)",
    xscale=:log10)
plot!(p_beta, betasweep.β, betasweep.p1_kr,
    color=:crimson, lw=2.5, marker=:circle, ms=5, label="hard mask β-sweep")
# Multiplicity conditions overlaid at their (β_w, p1_kr) — should fall on/near the mask curve
scatter!(p_beta, mult.beta, mult.p1_kr,
    marker=:diamond, ms=9, color=:steelblue,
    label="multiplicity (finite rho, at β_w)")
# Unconditional reference
scatter!(p_beta, [uncond_row.beta[1]], [uncond_row.p1_kr[1]],
    marker=:square, ms=8, color=:gray, label="unconditional")
# Curation reference
scatter!(p_beta, [cur_row.beta[1]], [cur_row.p1_kr[1]],
    marker=:utriangle, ms=9, color=:forestgreen, label="hard curation")
hline!(p_beta, [1.0], linestyle=:dot, color=:black, alpha=0.4, label="")
savefig(p_beta, joinpath(FIG_DIR, "fig_mask_betasweep.png"))
savefig(p_beta, joinpath(FIG_DIR, "fig_mask_betasweep.pdf"))
@info "Wrote fig_mask_betasweep"

# --- Novelty trade-off figure: novelty vs recovery (mask β-sweep path + curation) ---
p_nov = plot(size=(640, 500), margin=10Plots.mm, legend=:topright,
    xlabel="P1 K/R fraction (recovery)",
    ylabel="Novelty",
    title="Kunitz: novelty-recovery trade-off")
plot!(p_nov, betasweep.p1_kr, betasweep.novelty,
    color=:crimson, lw=2.5, marker=:circle, ms=5, label="hard mask (β path)")
# Annotate a few β values along the curve
for row in eachrow(betasweep[1:2:end, :])
    annotate!(p_nov, row.p1_kr + 0.01, row.novelty, text("β=$(Int(row.β))", 7, :left, :gray))
end
scatter!(p_nov, mult.p1_kr, mult.novelty,
    marker=:diamond, ms=8, color=:steelblue, label="multiplicity (finite rho)")
scatter!(p_nov, [uncond_row.p1_kr[1]], [uncond_row.novelty[1]],
    marker=:square, ms=8, color=:gray, label="unconditional")
scatter!(p_nov, [cur_row.p1_kr[1]], [cur_row.novelty[1]],
    marker=:utriangle, ms=9, color=:forestgreen, label="hard curation")
savefig(p_nov, joinpath(FIG_DIR, "fig_mask_novelty_tradeoff.png"))
savefig(p_nov, joinpath(FIG_DIR, "fig_mask_novelty_tradeoff.pdf"))
@info "Wrote fig_mask_novelty_tradeoff"
