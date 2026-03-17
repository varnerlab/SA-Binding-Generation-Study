# ──────────────────────────────────────────────────────────────────────────────
# compute_conotoxin_sar_agreement.jl
#
# Compute agreement between SA-generated conotoxin sequences and published
# SAR data for ω-conotoxin binding to Cav2.2.
#
# Published SAR critical residues (MVIIA numbering → alignment column):
#   Tyr13 — primary pharmacophore (hydroxyl contacts pore)
#   Lys2  — stabilizes loop 2 conformation
#   Arg10 — loop 2, critical for binding
#   Leu11 — loop 2, critical for binding
#   Cys1,8,15,16,20,25 — disulfide framework (structural integrity)
#
# Sources:
#   Kim et al. 1995 (BBRC) — Tyr13→Ala abolishes activity
#   Sato et al. 1993 (BBRC) — Lys2→Ala: 40× potency loss (GVIA)
#   Sato et al. 2000 (FEBS Lett) — Thr11,Tyr13,Lys2 essential (MVIIC)
#   Schroeder et al. 2012 (Biopolymers) — Lys2→Ala destabilizes loop 2
#   Lewis et al. 2012 (Mar Drugs) — Arg10, Leu11 crucial for binding
#   Schroeder et al. 1999 (Biochemistry) — D-Tyr13: 1000× loss
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

using StatsBase

const DATA_DIR = joinpath(_CODE_DIR, "data", "omega_conotoxin")

# ── Load aligned stored sequences to find column mappings ────────────────────
raw_full = parse_fasta(joinpath(DATA_DIR, "omega_conotoxin_full_family_aligned.fasta"))
char_full, names_full = clean_alignment(raw_full; max_gap_frac_col=0.5, max_gap_frac_seq=0.4)
K, L = size(char_full)
@info "Full family: $K sequences × $L positions"

# Find MVIIA in the alignment to establish numbering
mviia_idx = findfirst(n -> contains(n, "MVIIA"), names_full)
if mviia_idx !== nothing
    mviia_aligned = String(char_full[mviia_idx, :])
    @info "MVIIA aligned: $mviia_aligned"
    @info "MVIIA name: $(names_full[mviia_idx])"
else
    @warn "MVIIA not found in alignment"
end

# MVIIA sequence: CKGKGAKCSRLMYDCCTGSCRSGKCG (26 residues)
# After gap filtering to 26 columns, the alignment positions map directly
# to MVIIA numbering for the core conserved positions.

# ── Define SAR-critical residues ─────────────────────────────────────────────
# These are alignment column positions, mapped from MVIIA numbering
# MVIIA: C1-K2-G3-K4-G5-A6-K7-C8-S9-R10-L11-M12-Y13-D14-C15-C16-T17-G18-S19-C20-R21-S22-G23-K24-C25-G26

sar_residues = [
    # (column, MVIIA_pos, wt_residue, role, effect, citation)
    (13, 13, 'Y', "Primary pharmacophore", "Ala: abolishes activity; Phe: 100× loss", "Kim et al. 1995; Schroeder et al. 1999"),
    (2,   2, 'K', "Loop 2 stabilization", "Ala: 40× potency loss (GVIA)", "Sato et al. 1993; Schroeder et al. 2012"),
    (10, 10, 'R', "Loop 2 binding", "Critical for channel interaction", "Lewis et al. 2012"),
    (11, 11, 'L', "Loop 2 binding", "Critical for channel interaction", "Lewis et al. 2012"),
    (1,   1, 'C', "Disulfide framework", "Required for fold", "structural"),
    (8,   8, 'C', "Disulfide framework", "Required for fold", "structural"),
    (15, 15, 'C', "Disulfide framework", "Required for fold", "structural"),
    (16, 16, 'C', "Disulfide framework", "Required for fold", "structural"),
    (20, 20, 'C', "Disulfide framework", "Required for fold", "structural"),
    (25, 25, 'C', "Disulfide framework", "Required for fold", "structural"),
    (21, 21, 'R', "Electrostatic complementarity", "Ala: reduced potency", "Lewis et al. 2012"),
    (4,   4, 'K', "P/Q selectivity", "Ala: important for P/Q binding", "Sato et al. 2000"),
]

# ── Load generated sequences ────────────────────────────────────────────────
strong_seqs = parse_fasta(joinpath(DATA_DIR, "generated_strong_seeded.fasta"))
full_seqs = parse_fasta(joinpath(DATA_DIR, "generated_full_seeded.fasta"))
@info "Strong-seeded: $(length(strong_seqs)) sequences"
@info "Full-seeded: $(length(full_seqs)) sequences"

# Also load strong binder input
raw_strong = parse_fasta(joinpath(DATA_DIR, "strong_cav22_binders.fasta"))
@info "Strong binder input: $(length(raw_strong)) sequences"

# ── Compute agreement ───────────────────────────────────────────────────────
function compute_sar_agreement(seqs, sar_residues, label)
    n = length(seqs)
    @info "\n=== $label ($n sequences) ==="

    results = []
    for (col, mviia_pos, wt, role, effect, cite) in sar_residues
        # Count sequences with the wild-type residue at this position
        n_wt = count(x -> length(x[2]) >= col && x[2][col] == wt, seqs)
        frac = n_wt / n

        if wt in ('K', 'R')
            n_basic = count(x -> length(x[2]) >= col && x[2][col] in ('K', 'R'), seqs)
            frac_basic = n_basic / n
            @info "  Col $col (MVIIA pos $mviia_pos, $wt): exact=$(round(frac,digits=3)), K/R=$(round(frac_basic,digits=3)) — $role"
            push!(results, (col=col, mviia_pos=mviia_pos, wt=wt, role=role, effect=effect, cite=cite,
                           frac_exact=frac, frac_permissive=frac_basic))
        else
            @info "  Col $col (MVIIA pos $mviia_pos, $wt): $(round(frac,digits=3)) — $role"
            push!(results, (col=col, mviia_pos=mviia_pos, wt=wt, role=role, effect=effect, cite=cite,
                           frac_exact=frac, frac_permissive=frac))
        end
    end

    return results
end

results_strong_input = compute_sar_agreement(raw_strong, sar_residues, "Strong binder input (23 seqs)")
results_strong_gen = compute_sar_agreement(strong_seqs, sar_residues, "SA strong-seeded (1550 seqs)")
results_full_gen = compute_sar_agreement(full_seqs, sar_residues, "SA full-seeded (1550 seqs)")

# ── Build summary table ─────────────────────────────────────────────────────
@info "\n" * "="^80
@info "SAR Agreement Summary Table"
@info "="^80

summary_df = DataFrame(
    Position = [r.mviia_pos for r in results_strong_gen],
    WT_Residue = [string(r.wt) for r in results_strong_gen],
    Role = [r.role for r in results_strong_gen],
    Effect_of_mutation = [r.effect for r in results_strong_gen],
    Input_strong = [round(r.frac_permissive, digits=3) for r in results_strong_input],
    SA_strong = [round(r.frac_permissive, digits=3) for r in results_strong_gen],
    SA_full = [round(r.frac_permissive, digits=3) for r in results_full_gen],
    Citation = [r.cite for r in results_strong_gen],
)

pretty_table(summary_df)
CSV.write(joinpath(DATA_DIR, "sar_agreement.csv"), summary_df)
@info "Saved SAR agreement table to $(joinpath(DATA_DIR, "sar_agreement.csv"))"
