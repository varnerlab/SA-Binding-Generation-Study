# ──────────────────────────────────────────────────────────────────────────────
# run_kunitz_mask_experiment.jl
#
# Adds the hard attention mask (b = -Inf on background) as the rho -> Inf
# endpoint of the multiplicity calibration curve on the Kunitz family
# (P1 K/R designated subset), and decomposes the calibration gap into
# residual-background-mass (closed by the mask) vs decode/geometry floor
# (irreducible).
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

results = DataFrame(condition=String[], f_target=Float64[], p1_kr=Float64[],
                    novelty=Float64[], diversity=Float64[], kl=Float64[], n=Int[])

# --- Condition 1: Unconditional (b = 0) ---
@info "Condition 1: Unconditional"
uncond_seqs, _ = generate_sequences(X̂_all, pca_all, L; β=β_all, n_chains=30, T=5000, seed=42)
m = eval_condition(uncond_seqs, "unconditional")
push!(results, ("unconditional",
                effective_binder_fraction(ones(K_total), designated_idx),
                m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Condition 2: Multiplicity calibration sweep (finite rho) ---
@info "Condition 2: Multiplicity calibration sweep"
for ft in [0.5, 0.7, 0.9, 0.95, 0.99]
    res = build_multiplicity_conditioned_memory(char_mat, designated_idx; f_target=ft)
    β_w = find_weighted_entropy_inflection(res.X̂, res.r; n_betas=50).β_star
    seqs, _ = generate_weighted_sequences(res.X̂, res.pca_model, L, res.r;
        β=β_w, n_chains=20, T=5000, seed=42)
    m = eval_condition(seqs, "multiplicity(f=$ft)")
    push!(results, ("multiplicity", res.f_eff, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))
end

# --- Condition 3: Hard mask (b = -Inf on background, full basis) ---
@info "Condition 3: Hard mask"
β_mask = find_entropy_inflection(X̂_all[:, designated_idx]).β_star
mask_seqs, _ = generate_masked_sequences(X̂_all, pca_all, L, designated_idx;
    β=β_mask, n_chains=30, T=5000, seed=42)
m = eval_condition(mask_seqs, "mask")
push!(results, ("mask", 1.0, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Condition 4: Hard curation (subset basis) ---
@info "Condition 4: Hard curation (subset basis)"
X̂_cur, pca_cur, _, _ = build_memory_matrix(char_mat[designated_idx, :]; pratio=0.95)
β_cur = find_entropy_inflection(X̂_cur).β_star
cur_seqs, _ = generate_sequences(X̂_cur, pca_cur, L; β=β_cur, n_chains=30, T=5000, seed=42)
m = eval_condition(cur_seqs, "curation")
push!(results, ("curation", 1.0, m.p1_kr, m.novelty, m.diversity, m.kl, m.n))

# --- Write CSV ---
csv_path = joinpath(CACHE_DIR, "mask_experiment.csv")
CSV.write(csv_path, results)
@info "Wrote $csv_path"

# --- Calibration figure: multiplicity curve + mask/curation/unconditional overlays ---
mult = results[results.condition .== "multiplicity", :]
uncond_row = results[results.condition .== "unconditional", :]
mask_row = results[results.condition .== "mask", :]
cur_row  = results[results.condition .== "curation", :]

p = plot(size=(640, 520), margin=10Plots.mm, legend=:bottomright,
    xlabel="Target effective designated fraction (f_target)",
    ylabel="Observed P1 K/R fraction",
    title="Kunitz: multiplicity calibration with mask endpoint")
plot!(p, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)")
plot!(p, mult.f_target, mult.p1_kr, marker=:circle, color=:steelblue, lw=2, label="multiplicity (finite rho)")
scatter!(p, uncond_row.f_target, uncond_row.p1_kr, marker=:diamond, ms=8, color=:gray, label="unconditional")
scatter!(p, mask_row.f_target, mask_row.p1_kr, marker=:star5, ms=12, color=:crimson, label="hard mask (b=-Inf)")
scatter!(p, cur_row.f_target, cur_row.p1_kr, marker=:utriangle, ms=9, color=:forestgreen, label="hard curation")
savefig(p, joinpath(FIG_DIR, "fig_mask_calibration.png"))
savefig(p, joinpath(FIG_DIR, "fig_mask_calibration.pdf"))
@info "Wrote fig_mask_calibration"

# --- Calibration-gap decomposition ---
mult_top = mult[argmax(mult.f_target), :]
gap_finite = mult_top.f_target - mult_top.p1_kr      # residual at the strongest finite rho
gap_mask   = 1.0 - mask_row.p1_kr[1]                 # remaining under the mask = decode/geometry floor
@info "=== Calibration-gap decomposition ==="
@info "  Strongest finite rho (f_target=$(mult_top.f_target)): observed P1 K/R = $(round(mult_top.p1_kr,digits=3)), gap = $(round(gap_finite,digits=3))"
@info "  Hard mask: observed P1 K/R = $(round(mask_row.p1_kr[1],digits=3)), residual (decode/geometry) = $(round(gap_mask,digits=3))"
@info "  Residual background mass closed by the mask ~= $(round(gap_finite - gap_mask, digits=3))"
@info "  Scaffold/quality: mask novelty=$(round(mask_row.novelty[1],digits=3)) vs curation novelty=$(round(cur_row.novelty[1],digits=3)); " *
      "mask KL=$(round(mask_row.kl[1],digits=3)) vs curation KL=$(round(cur_row.kl[1],digits=3))"
