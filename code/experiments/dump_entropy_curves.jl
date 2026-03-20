# Dump entropy curves for the Kunitz domain at multiple ρ values to CSV
# for rendering with matplotlib.

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")

# --- Load Kunitz alignment ---
sto_file = download_pfam_seed("PF00014"; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K_total, L = size(char_mat)

# --- Find strong binders (K/R at P1) ---
lys_fracs = [count(i -> char_mat[i, j] == 'K', 1:K_total) /
             max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K_total))
             for j in 1:L]
p1_pos = argmax(lys_fracs)
strong_idx = findall(i -> char_mat[i, p1_pos] in ('K', 'R'), 1:K_total)

# --- Build memory matrix ---
X̂, pca_model, _, _ = build_memory_matrix(char_mat; pratio=0.95)
d, K = size(X̂)
@info "Memory: $d × $K, strong binders: $(length(strong_idx))"

# --- Compute entropy curves ---
ρ_values = [1.0, 5.0, 20.0, 100.0, 1000.0]
n_betas = 80

open(joinpath(CACHE_DIR, "entropy_curves.csv"), "w") do io
    println(io, "rho,K_eff,beta,H,beta_star")
    for ρ in ρ_values
        r = multiplicity_vector(K_total, strong_idx; ρ=ρ)
        K_eff = effective_num_patterns(r)
        pt = find_weighted_entropy_inflection(X̂, r; n_betas=n_betas)
        for (β, H) in zip(pt.βs, pt.Hs)
            println(io, "$ρ,$K_eff,$β,$H,$(pt.β_star)")
        end
        @info "  ρ=$ρ: K_eff=$(round(K_eff, digits=1)), β*=$(round(pt.β_star, digits=2))"
    end
end

@info "Saved entropy_curves.csv"
@info "log(K) = $(round(log(K), digits=3))"
