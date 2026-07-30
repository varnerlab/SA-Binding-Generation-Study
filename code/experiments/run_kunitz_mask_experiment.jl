# ──────────────────────────────────────────────────────────────────────────────
# run_kunitz_mask_experiment.jl
#
# Hard attention masking (b = -Inf on background) as the rho -> Inf endpoint of
# the multiplicity calibration axis on the Kunitz family (P1 K/R designated subset).
#
# Framing (gap decomposition; see theory.tex eq:gap-decomp):
#   The hard mask is the literal Delta_attn = 0 slice (f_eff = 1, all attention on
#   designated patterns). The fixed-mask beta-sweep traces the decode-limited
#   ceiling: the recovery reachable when attention is perfect and only the
#   Delta_PCA + Delta_argmax terms remain, which contract as beta rises.
#   Finite-rho multiplicity conditions sit BELOW this envelope at matched beta: at equal
#   beta, hard masking recovers more phenotype than soft weighting, a margin that shrinks to
#   0 as rho -> Inf. Since attention tracks f_eff (Delta_attn ~ 0), this advantage is a DECODE
#   effect, not Delta_attn: recovery depends on both retrieval sharpness (beta) and
#   conditioning strength (rho), both acting through the decode gap.
#
# All conditions use an identical neutral warm-start (mod1(chain,K)) and 30 chains,
# so the mask vs multiplicity vs unconditional comparison is unconfounded. Recovery
# (P1 K/R) and novelty are reported as chain-level mean +/- SE.
#
# Conditions (all on the full-family PCA basis except hard curation):
#   1. Unconditional  (b = 0)
#   2. Multiplicity calibration sweep  (finite rho via build_multiplicity_conditioned_memory)
#   3. Hard mask  (b = -Inf on background, full basis) at its own beta*
#   4. Hard curation  (subset basis)
#   + exact matched-beta mask runs at each multiplicity beta_w (matched-beta masking advantage)
#   + fixed-mask beta-sweep (the Delta_attn = 0 envelope)
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

Random.seed!(42)   # global determinism: makes the whole run reproducible end-to-end

const PFAM_ID = "PF00014"
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "kunitz")
const N_CHAINS = 30          # equalized across every condition
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
β_all = all_memory_onset(X̂_all)
@info "  Full-family β* = $(round(β_all, digits=3))"

# ──────────────────────────────────────────────────────────────────────────────
# Chain-level metric helpers (each Langevin chain is one independent replicate)
# ──────────────────────────────────────────────────────────────────────────────
# Generators emit a flat sequence array ordered chain-by-chain with a fixed
# samples_per_chain = length(burnin:thin:T), so we can recover per-chain values by
# slicing without changing any generator signature.
function per_chain_p1_nov(seqs::Vector{String}, n_chains::Int)
    spc = length(seqs) ÷ n_chains
    p1 = Float64[]; nov = Float64[]
    for c in 1:n_chains
        chunk = seqs[(c - 1) * spc + 1 : c * spc]
        push!(p1, count(s -> length(s) >= p1_pos && s[p1_pos] in ('K', 'R'), chunk) / length(chunk))
        push!(nov, 1.0 - mean(nearest_sequence_identity(s, stored_seqs) for s in chunk))
    end
    return p1, nov
end

mean_se(v::Vector{Float64}) = (mean(v), length(v) > 1 ? std(v) / sqrt(length(v)) : NaN)

# --- Metric helper: chain-level P1 K/R + novelty (mean ± SE), plus diversity/KL ---
function eval_condition(seqs::Vector{String}, label::String, n_chains::Int)
    n = length(seqs)
    if n < 2
        @warn "eval_condition: n=$n < 2 for [$label]; metrics undefined"
        return (label=label, p1_kr=NaN, p1_kr_se=NaN, novelty=NaN, novelty_se=NaN,
                diversity=NaN, kl=NaN, n=n)
    end
    p1_chain, nov_chain = per_chain_p1_nov(seqs, n_chains)
    p1_kr, p1_kr_se = mean_se(p1_chain)
    novelty, novelty_se = mean_se(nov_chain)
    np = min(500, n * (n - 1) ÷ 2)
    pair_ids = Float64[]
    for _ in 1:np
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(seqs[i], seqs[j]))
    end
    diversity = 1.0 - mean(pair_ids)
    kl = aa_composition_kl(seqs, stored_seqs)
    @info "  [$label] P1 K/R = $(round(p1_kr,digits=3))±$(round(p1_kr_se,digits=3)), " *
          "novelty = $(round(novelty,digits=3))±$(round(novelty_se,digits=3)), " *
          "diversity = $(round(diversity,digits=3)), KL = $(round(kl,digits=3))"
    return (label=label, p1_kr=p1_kr, p1_kr_se=p1_kr_se, novelty=novelty,
            novelty_se=novelty_se, diversity=diversity, kl=kl, n=n)
end

results = DataFrame(condition=String[], f_target=Float64[], beta=Float64[],
                    p1_kr=Float64[], p1_kr_se=Float64[], novelty=Float64[], novelty_se=Float64[],
                    diversity=Float64[], kl=Float64[], n=Int[])

# --- Condition 1: Unconditional (b = 0) ---
@info "Condition 1: Unconditional"
uncond_seqs, _ = generate_sequences(X̂_all, pca_all, L; β=β_all, n_chains=N_CHAINS, T=5000, seed=42)
m = eval_condition(uncond_seqs, "unconditional", N_CHAINS)
push!(results, ("unconditional", effective_binder_fraction(ones(K_total), designated_idx), β_all,
                m.p1_kr, m.p1_kr_se, m.novelty, m.novelty_se, m.diversity, m.kl, m.n))

# --- Condition 2: Multiplicity calibration sweep (finite rho) ---
# Also record (f_eff, β_w, P1 K/R ± SE) so we can run the mask at exactly each β_w.
@info "Condition 2: Multiplicity calibration sweep"
mult_rows = NamedTuple[]
for ft in [0.5, 0.7, 0.9, 0.95, 0.99]
    res = build_multiplicity_conditioned_memory(char_mat, designated_idx; f_target=ft)
    β_w = all_memory_onset(res.X̂, res.r; n_betas=50)
    seqs, _ = generate_weighted_sequences(res.X̂, res.pca_model, L, res.r;
        β=β_w, n_chains=N_CHAINS, T=5000, seed=42)
    local m = eval_condition(seqs, "multiplicity(f=$ft)", N_CHAINS)
    push!(results, ("multiplicity", res.f_eff, β_w, m.p1_kr, m.p1_kr_se, m.novelty, m.novelty_se, m.diversity, m.kl, m.n))
    push!(mult_rows, (f_target=ft, f_eff=res.f_eff, beta_w=β_w, p1_kr=m.p1_kr, p1_kr_se=m.p1_kr_se))
end

# --- Condition 3: Hard mask (b = -Inf on background, full basis) at its own β* ---
@info "Condition 3: Hard mask"
β_mask = all_memory_onset(X̂_all[:, designated_idx])
mask_seqs, _ = generate_masked_sequences(X̂_all, pca_all, L, designated_idx;
    β=β_mask, n_chains=N_CHAINS, T=5000, seed=42)
m = eval_condition(mask_seqs, "mask", N_CHAINS)
push!(results, ("mask", 1.0, β_mask, m.p1_kr, m.p1_kr_se, m.novelty, m.novelty_se, m.diversity, m.kl, m.n))

# --- Condition 4: Hard curation (subset basis) ---
@info "Condition 4: Hard curation (subset basis)"
X̂_cur, pca_cur, _, _ = build_memory_matrix(char_mat[designated_idx, :]; pratio=0.95)
β_cur = all_memory_onset(X̂_cur)
cur_seqs, _ = generate_sequences(X̂_cur, pca_cur, L; β=β_cur, n_chains=N_CHAINS, T=5000, seed=42)
m = eval_condition(cur_seqs, "curation", N_CHAINS)
push!(results, ("curation", 1.0, β_cur, m.p1_kr, m.p1_kr_se, m.novelty, m.novelty_se, m.diversity, m.kl, m.n))

# --- Write main CSV ---
csv_path = joinpath(CACHE_DIR, "mask_experiment.csv")
CSV.write(csv_path, results)
@info "Wrote $csv_path"

# ──────────────────────────────────────────────────────────────────────────────
# Exact matched-β residuals: run the mask at each multiplicity β_w (no interpolation,
# no grid quantization). residual = mult P1 K/R − mask P1 K/R at the same β = the matched-β
# masking advantage (NOT Δ_attn, which ≈ 0; the advantage is a decode effect, see note below).
# ──────────────────────────────────────────────────────────────────────────────
@info "Exact matched-β mask runs (matched-β masking advantage per condition)"
residuals = DataFrame(f_target=Float64[], f_eff=Float64[], beta_w=Float64[],
                      mult_p1kr=Float64[], mult_se=Float64[],
                      mask_p1kr=Float64[], mask_se=Float64[],
                      residual=Float64[], residual_se=Float64[])
for r in mult_rows
    ms, _ = generate_masked_sequences(X̂_all, pca_all, L, designated_idx;
        β=r.beta_w, n_chains=N_CHAINS, T=5000, seed=42)
    mm = eval_condition(ms, "mask@β=$(round(r.beta_w,digits=2))", N_CHAINS)
    resid = r.p1_kr - mm.p1_kr
    resid_se = sqrt(r.p1_kr_se^2 + mm.p1_kr_se^2)
    push!(residuals, (r.f_target, r.f_eff, r.beta_w, r.p1_kr, r.p1_kr_se,
                      mm.p1_kr, mm.p1_kr_se, resid, resid_se))
end
CSV.write(joinpath(CACHE_DIR, "mask_residuals.csv"), residuals)
@info "Wrote mask_residuals.csv"

# --- Fixed-mask β-sweep: the Δ_attn = 0 envelope, with chain-level SE ---
@info "Fixed-mask β-sweep"
betasweep = DataFrame(β=Float64[], p1_kr=Float64[], p1_kr_se=Float64[],
                      novelty=Float64[], novelty_se=Float64[])
for βb in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 512.0]
    seqs, _ = generate_masked_sequences(X̂_all, pca_all, L, designated_idx;
        β=βb, n_chains=N_CHAINS, T=5000, seed=42)
    p1c, novc = per_chain_p1_nov(seqs, N_CHAINS)
    p1m, p1se = mean_se(p1c)
    novm, novse = mean_se(novc)
    push!(betasweep, (βb, p1m, p1se, novm, novse))
    @info "  β=$βb: P1 K/R=$(round(p1m,digits=3))±$(round(p1se,digits=3)), novelty=$(round(novm,digits=3))±$(round(novse,digits=3))"
end
CSV.write(joinpath(CACHE_DIR, "mask_betasweep.csv"), betasweep)
@info "Wrote mask_betasweep.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Corrected analysis: gap decomposition (mask = Δ_attn = 0 envelope)
# ──────────────────────────────────────────────────────────────────────────────
mult = results[results.condition .== "multiplicity", :]
uncond_row = results[results.condition .== "unconditional", :]
mask_row = results[results.condition .== "mask", :]
cur_row  = results[results.condition .== "curation", :]

@info "=== Corrected analysis: gap decomposition (mask = Δ_attn=0 envelope; ρ acts via decode) ==="
@info ""
@info "1. Matched-β masking advantage (mult P1 K/R − mask P1 K/R at the SAME β_w):"
for r in eachrow(residuals)
    flag = abs(r.residual) <= r.residual_se ? "within 1 SE of 0" : "> 1 SE from 0"
    @info "   f=$(r.f_target) (β_w=$(round(r.beta_w,digits=2))): " *
          "mult=$(round(r.mult_p1kr,digits=3))±$(round(r.mult_se,digits=3)), " *
          "mask=$(round(r.mask_p1kr,digits=3))±$(round(r.mask_se,digits=3)), " *
          "residual=$(round(r.residual,digits=3))±$(round(r.residual_se,digits=3))  ($flag)"
end
monotone = all(diff(residuals.residual) .>= 0)   # residuals should rise toward 0 as f_target↑
@info "   Residuals monotone increasing toward 0 with f_target: $monotone"
@info ""
@info "2. Decode-limited ceiling (Δ_attn = 0 envelope): recovery climbs with β to"
@info "   $(round(betasweep.p1_kr[end],digits=3))±$(round(betasweep.p1_kr_se[end],digits=3)) at β=$(Int(betasweep.β[end]))."
@info ""
@info "3. Interpretation:"
@info "   - Nonzero advantage at finite ρ ⇒ recovery depends on ρ as well as β (two-factor)."
@info "   - Since attention tracks f_eff (Δ_attn≈0), this is a DECODE effect: concentrating the"
@info "     weighted superposition on designated patterns decodes more reliably, not an attention gap."
@info "   - Advantage → 0 as ρ→∞, where mask and strongest multiplicity coincide."
@info "   - Novelty cost: mask β* novelty=$(round(mask_row.novelty[1],digits=3)), "
@info "     high-β (β=$(Int(betasweep.β[end]))) novelty=$(round(betasweep.novelty[end],digits=3)), "
@info "     curation novelty=$(round(cur_row.novelty[1],digits=3)) at P1 K/R=$(round(cur_row.p1_kr[1],digits=3))."

# ──────────────────────────────────────────────────────────────────────────────
# Figures (all with chain-level error bars)
# ──────────────────────────────────────────────────────────────────────────────

# --- Unifying figure: recovery surface (mask envelope + finite-ρ points) ---
p_beta = plot(size=(720, 540), margin=10Plots.mm, legend=:bottomright,
    xlabel="Inverse temperature β (log scale)",
    ylabel="P1 K/R fraction",
    title="Kunitz: P1 K/R vs β (mask envelope + finite-ρ points)",
    xscale=:log10)
plot!(p_beta, betasweep.β, betasweep.p1_kr, yerror=betasweep.p1_kr_se,
    color=:crimson, lw=2.5, marker=:circle, ms=5, label="hard mask β-sweep (Δ_attn=0 envelope)")
# exact-β_w mask points (faint) so the vertical residual to the multiplicity points reads directly
scatter!(p_beta, residuals.beta_w, residuals.mask_p1kr, yerror=residuals.mask_se,
    marker=:circle, ms=4, color=:crimson, alpha=0.35, label="mask at matched β_w")
scatter!(p_beta, mult.beta, mult.p1_kr, yerror=mult.p1_kr_se,
    marker=:diamond, ms=9, color=:steelblue, label="multiplicity (finite ρ, at β_w)")
scatter!(p_beta, [uncond_row.beta[1]], [uncond_row.p1_kr[1]], yerror=[uncond_row.p1_kr_se[1]],
    marker=:square, ms=8, color=:gray, label="unconditional")
scatter!(p_beta, [cur_row.beta[1]], [cur_row.p1_kr[1]], yerror=[cur_row.p1_kr_se[1]],
    marker=:utriangle, ms=9, color=:forestgreen, label="hard curation")
hline!(p_beta, [1.0], linestyle=:dot, color=:black, alpha=0.4, label="")
savefig(p_beta, joinpath(FIG_DIR, "fig_mask_betasweep.png"))
savefig(p_beta, joinpath(FIG_DIR, "fig_mask_betasweep.pdf"))
@info "Wrote fig_mask_betasweep"

# --- Calibration figure: observed vs target effective fraction (with mask/curation/uncond) ---
p_cal = plot(size=(640, 520), margin=10Plots.mm, legend=:bottomright,
    xlabel="Target effective designated fraction (f_eff)",
    ylabel="Observed P1 K/R fraction",
    title="Kunitz: multiplicity calibration with mask endpoint")
plot!(p_cal, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)")
plot!(p_cal, mult.f_target, mult.p1_kr, yerror=mult.p1_kr_se, marker=:circle, color=:steelblue, lw=2, label="multiplicity (finite ρ)")
scatter!(p_cal, uncond_row.f_target, uncond_row.p1_kr, yerror=uncond_row.p1_kr_se, marker=:diamond, ms=8, color=:gray, label="unconditional")
scatter!(p_cal, mask_row.f_target, mask_row.p1_kr, yerror=mask_row.p1_kr_se, marker=:star5, ms=12, color=:crimson, label="hard mask (b=-Inf)")
scatter!(p_cal, cur_row.f_target, cur_row.p1_kr, yerror=cur_row.p1_kr_se, marker=:utriangle, ms=9, color=:forestgreen, label="hard curation")
savefig(p_cal, joinpath(FIG_DIR, "fig_mask_calibration.png"))
savefig(p_cal, joinpath(FIG_DIR, "fig_mask_calibration.pdf"))
@info "Wrote fig_mask_calibration"

# --- Novelty-recovery trade-off (mask β path + multiplicity + curation) ---
p_nov = plot(size=(640, 500), margin=10Plots.mm, legend=:topright,
    xlabel="P1 K/R fraction (recovery)",
    ylabel="Novelty",
    title="Kunitz: novelty-recovery trade-off")
plot!(p_nov, betasweep.p1_kr, betasweep.novelty, xerror=betasweep.p1_kr_se, yerror=betasweep.novelty_se,
    color=:crimson, lw=2.5, marker=:circle, ms=5, label="hard mask (β path)")
for row in eachrow(betasweep[1:2:end, :])
    annotate!(p_nov, row.p1_kr + 0.01, row.novelty, text("β=$(Int(row.β))", 7, :left, :gray))
end
scatter!(p_nov, mult.p1_kr, mult.novelty, xerror=mult.p1_kr_se, yerror=mult.novelty_se,
    marker=:diamond, ms=8, color=:steelblue, label="multiplicity (finite ρ)")
scatter!(p_nov, [uncond_row.p1_kr[1]], [uncond_row.novelty[1]], marker=:square, ms=8, color=:gray, label="unconditional")
scatter!(p_nov, [cur_row.p1_kr[1]], [cur_row.novelty[1]], marker=:utriangle, ms=9, color=:forestgreen, label="hard curation")
savefig(p_nov, joinpath(FIG_DIR, "fig_mask_novelty_tradeoff.png"))
savefig(p_nov, joinpath(FIG_DIR, "fig_mask_novelty_tradeoff.pdf"))
@info "Wrote fig_mask_novelty_tradeoff"
