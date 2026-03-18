# ──────────────────────────────────────────────────────────────────────────────
# run_conotoxin_multiplicity_sweep.jl
#
# Run the same multiplicity-weighted ρ sweep on ω-conotoxin that was done for
# Kunitz, SH3, and WW in run_second_family_validation.jl.
#
# This computes:
#   - Fisher separation index S between 23 strong Cav2.2 binders and 51 others
#   - f_eff and f_obs across ρ = [1, 2, 5, 10, 20, 50, 100, 500]
#   - Calibration gap Δ = f_eff - f_obs at ρ=500
#
# The marker phenotype is Tyr (Y) at the pharmacophore position (Tyr13 in
# MVIIA numbering), matching the conotoxin experiment.
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

using DataFrames, CSV

const DATA_DIR = joinpath(_CODE_DIR, "data", "omega_conotoxin")
const CACHE_DIR = joinpath(_CODE_DIR, "data")

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load alignments and identify strong binders
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 1: Loading ω-conotoxin alignments"

full_fasta   = joinpath(DATA_DIR, "omega_conotoxin_full_family_aligned.fasta")
strong_fasta = joinpath(DATA_DIR, "strong_cav22_binders_aligned.fasta")

raw_full   = parse_fasta(full_fasta)
raw_strong = parse_fasta(strong_fasta)

char_full, names_full = clean_alignment(raw_full; max_gap_frac_col=0.5, max_gap_frac_seq=0.4)
K_total, L = size(char_full)
stored_seqs = [String(char_full[i, :]) for i in 1:K_total]
@info "  Full family: $K_total × $L"

# identify strong binder indices by matching UniProt IDs
strong_ids = Set{String}()
for (name, _) in raw_strong
    # extract UniProt ID (first field before |)
    uid = split(name, "|")[1]
    push!(strong_ids, uid)
end

group_A_idx = Int[]  # strong binders
group_B_idx = Int[]  # background
for (i, name) in enumerate(names_full)
    uid = split(name, "|")[1]
    if uid in strong_ids
        push!(group_A_idx, i)
    else
        push!(group_B_idx, i)
    end
end

K_A = length(group_A_idx)
K_B = length(group_B_idx)
natural_frac = K_A / K_total
@info "  Strong binders (Group A): $K_A"
@info "  Background (Group B): $K_B"
@info "  Natural fraction: $(round(natural_frac, digits=3))"

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Find pharmacophore marker position (Tyr13)
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 2: Finding Tyr pharmacophore position"

tyr_fracs = zeros(L)
for j in 1:L
    n_tyr   = count(i -> char_full[i, j] == 'Y', 1:K_total)
    n_valid = count(i -> !(char_full[i, j] in ('-', '.', '~')), 1:K_total)
    tyr_fracs[j] = n_valid > 0 ? n_tyr / n_valid : 0.0
end
marker_pos = argmax(tyr_fracs)
@info "  Tyr pharmacophore at column $marker_pos (freq=$(round(tyr_fracs[marker_pos], digits=3)))"

# marker residue set: Y at the pharmacophore position
group_A_residues = Set(['Y'])

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Build memory matrix and compute Fisher separation
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 3: Building memory matrix and computing separation index"

X̂, pca_model, _, _ = build_memory_matrix(char_full; pratio=0.95)
d = size(X̂, 1)
@info "  Memory: $d × $K_total"

# PCA-space cosine similarity analysis
within_A = Float64[]
for i in group_A_idx, j in group_A_idx
    i >= j && continue
    push!(within_A, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
end

within_B = Float64[]
for i in group_B_idx, j in group_B_idx
    i >= j && continue
    push!(within_B, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
end

between_AB = Float64[]
for i in group_A_idx, j in group_B_idx
    push!(between_AB, dot(X̂[:, i], X̂[:, j]) / (norm(X̂[:, i]) * norm(X̂[:, j])))
end

mean_within = (mean(within_A) * length(within_A) + mean(within_B) * length(within_B)) /
              (length(within_A) + length(within_B))
std_within = sqrt((var(within_A) * length(within_A) + var(within_B) * length(within_B)) /
                  (length(within_A) + length(within_B)))
separation_index = (mean_within - mean(between_AB)) / (0.5 * (std_within + std(between_AB)))

@info "  Within-A cosine: $(round(mean(within_A), digits=4)) ± $(round(std(within_A), digits=4))"
@info "  Within-B cosine: $(round(mean(within_B), digits=4)) ± $(round(std(within_B), digits=4))"
@info "  Between cosine:  $(round(mean(between_AB), digits=4)) ± $(round(std(between_AB), digits=4))"
@info "  Fisher separation index S = $(round(separation_index, digits=4))"

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: ρ sweep
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 4: Running ρ sweep"

ρ_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
n_chains = 20
seed = 42

sweep_results = DataFrame(
    ρ=Float64[], f_eff=Float64[], f_obs=Float64[],
    attn_A=Float64[], diversity=Float64[],
)

for ρ in ρ_values
    r = multiplicity_vector(K_total, group_A_idx; ρ=ρ)
    f_eff = effective_binder_fraction(r, group_A_idx)
    log_r = log.(r)
    pt = find_weighted_entropy_inflection(X̂, r; n_betas=50)
    β = pt.β_star

    gen_seqs = String[]
    attn_A_vals = Float64[]

    for chain in 1:n_chains
        k = mod1(chain, K_total)
        ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)
        result = weighted_sample(X̂, ξ₀, 5000, r; β=β, α=0.01, seed=seed + chain)

        for t in 2000:100:5000
            ξ = result.Ξ[t + 1, :]
            push!(gen_seqs, decode_sample(ξ, pca_model, L))

            logits = β .* (X̂' * ξ) .+ log_r
            attn = NNlib.softmax(logits)
            push!(attn_A_vals, sum(attn[idx] for idx in group_A_idx))
        end
    end

    n = length(gen_seqs)
    f_obs = count(s -> length(s) >= marker_pos && s[marker_pos] in group_A_residues,
                   gen_seqs) / n

    pair_ids = Float64[]
    for _ in 1:min(300, n * (n - 1) ÷ 2)
        i, j = rand(1:n), rand(1:n)
        while i == j; j = rand(1:n); end
        push!(pair_ids, sequence_identity(gen_seqs[i], gen_seqs[j]))
    end
    div = 1.0 - mean(pair_ids)

    push!(sweep_results, (ρ, f_eff, f_obs, mean(attn_A_vals), div))
    @info "  ρ=$(round(ρ, digits=0)): f_eff=$(round(f_eff, digits=3)), f_obs=$(round(f_obs, digits=3)), attn=$(round(mean(attn_A_vals), digits=3))"
end

# calibration gap at ρ=500
cal_gap = last(sweep_results).f_eff - last(sweep_results).f_obs

@info "\n" * "="^60
@info "SUMMARY: ω-Conotoxin multiplicity sweep"
@info "="^60
@info "  Separation index S = $(round(separation_index, digits=4))"
@info "  Calibration gap Δ = $(round(cal_gap, digits=4)) (at ρ=500)"
@info "  Natural binder fraction = $(round(natural_frac, digits=3))"

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Save results
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 5: Saving results"

# Save sweep data
sweep_path = joinpath(DATA_DIR, "multiplicity_sweep.csv")
CSV.write(sweep_path, sweep_results)
@info "  Saved ρ sweep → $sweep_path"

# Update multi-family comparison
comparison_path = joinpath(CACHE_DIR, "multi_family_comparison.csv")
existing = CSV.read(comparison_path, DataFrame)

# remove old conotoxin row if present
filter!(row -> row.family != "Conotoxin", existing)

new_row = DataFrame(
    family=["Conotoxin"],
    K=[K_total], d_pca=[d], K_A=[K_A],
    natural_frac=[natural_frac],
    separation_index=[separation_index],
    hard_curation_frac=[1.0],  # from conotoxin experiment: ~98.3% Tyr
    cal_gap_rho500=[cal_gap],
)
updated = vcat(existing, new_row)
CSV.write(comparison_path, updated)
@info "  Updated multi_family_comparison.csv with Conotoxin row"

@info "\nDone! Now re-run render_separation_gap_figure.py to update the figure."
