# ──────────────────────────────────────────────────────────────────────────────
# run_omega_conotoxin_experiment.jl
#
# ω-Conotoxin binder generation experiment.
#
# ω-Conotoxins are disulfide-rich peptides from cone snails that block
# voltage-gated N-type Ca²⁺ channels (Cav2.2) in sensory neurons.
# Ziconotide (MVIIA) is an FDA-approved non-opioid analgesic derived from
# this family. Motivated by prior modeling of ATP-gated pain signaling in
# DRG neurons (Song & Varner, PLoS ONE 2009, PMID 19750220).
#
# Experiment design (mirrors Kunitz→trypsin from binding.md):
#   - Full family (74 seqs): all SwissProt ω-conotoxins, MAFFT-aligned
#   - Strong binders (23 seqs): confirmed/predicted Cav2.2 N-type blockers
#   - Weak/non-selective (2 seqs): MVIIC/MVIID (P/Q-type preference)
#
# Key pharmacophore positions (MVIIA numbering):
#   Tyr13: most critical binding determinant (hydroxyl contacts channel)
#   Lys2, Lys7: basic residues important for channel interaction
#   Loop 2 (positions 8-13): primary contact region with Cav2.2
#
# Run both seed conditions and compare:
#   (A) SA seeded with full family (74)    → broad sequence diversity
#   (B) SA seeded with strong binders (23) → enriched pharmacophore inheritance
# ──────────────────────────────────────────────────────────────────────────────

# --- setup ---
_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

# --- configuration ---
const DATA_DIR  = joinpath(_CODE_DIR, "data", "omega_conotoxin")
const FIG_DIR   = joinpath(_CODE_DIR, "figs", "omega_conotoxin")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

# aligned FASTA files (produced by MAFFT --auto)
const FULL_FAMILY_FASTA   = joinpath(DATA_DIR, "omega_conotoxin_full_family_aligned.fasta")
const STRONG_BINDERS_FASTA = joinpath(DATA_DIR, "strong_cav22_binders_aligned.fasta")

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load and clean alignments
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 1: Loading ω-conotoxin alignments"

function fasta_to_stockholm_tuples(filepath::String)
    # parse_fasta returns (name, seq); same format as parse_stockholm
    return parse_fasta(filepath)
end

raw_full   = fasta_to_stockholm_tuples(FULL_FAMILY_FASTA)
raw_strong = fasta_to_stockholm_tuples(STRONG_BINDERS_FASTA)

char_full,   names_full   = clean_alignment(raw_full;   max_gap_frac_col=0.5, max_gap_frac_seq=0.4)
char_strong, names_strong = clean_alignment(raw_strong; max_gap_frac_col=0.5, max_gap_frac_seq=0.4)

K_full,   L_full   = size(char_full)
K_strong, L_strong = size(char_strong)

@info "  Full family:    $K_full sequences × $L_full positions"
@info "  Strong binders: $K_strong sequences × $L_strong positions"

stored_full   = [String(char_full[i, :])   for i in 1:K_full]
stored_strong = [String(char_strong[i, :]) for i in 1:K_strong]

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Identify pharmacophore positions
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 2: Identifying pharmacophore positions"

# Tyr13 (MVIIA numbering) is the primary Cav2.2 binding determinant.
# In the alignment, find the column with the highest Tyr (Y) frequency.
function find_pharmacophore_pos(char_mat::Matrix{Char}, aa::Char)
    K, L = size(char_mat)
    freqs = zeros(L)
    for j in 1:L
        n_aa    = count(i -> char_mat[i, j] == aa,                          1:K)
        n_valid = count(i -> !(char_mat[i, j] in ('.', '-', '~')),           1:K)
        freqs[j] = n_valid > 0 ? n_aa / n_valid : 0.0
    end
    return freqs
end

# Tyr (Y) position — primary pharmacophore
tyr_freqs_full   = find_pharmacophore_pos(char_full,   'Y')
tyr_freqs_strong = find_pharmacophore_pos(char_strong, 'Y')

tyr_pos_full   = argmax(tyr_freqs_full)
tyr_pos_strong = argmax(tyr_freqs_strong)

@info "  Full family   — Tyr (pharmacophore) position: col $tyr_pos_full " *
      "(freq=$(round(tyr_freqs_full[tyr_pos_full], digits=2)))"
@info "  Strong binders — Tyr (pharmacophore) position: col $tyr_pos_strong " *
      "(freq=$(round(tyr_freqs_strong[tyr_pos_strong], digits=2)))"

# define binding loop window (±4 around Tyr position)
loop_full   = collect(max(1, tyr_pos_full   - 4):min(L_full,   tyr_pos_full   + 4))
loop_strong = collect(max(1, tyr_pos_strong - 4):min(L_strong, tyr_pos_strong + 4))

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Build memory matrices
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 3: Building memory matrices"

@info "  [Full family — 74 seqs]"
X̂_full, pca_full, _, _ = build_memory_matrix(char_full; pratio=0.95)

@info "  [Strong binders — 23 seqs]"
X̂_strong, pca_strong, _, _ = build_memory_matrix(char_strong; pratio=0.95)

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Find β* for each seed condition
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 4: Finding β* (phase transition) for each seed"

pt_full   = find_entropy_inflection(X̂_full)
pt_strong = find_entropy_inflection(X̂_strong)

β_full   = pt_full.β_star
β_strong = pt_strong.β_star

@info "  Full family    β* = $(round(β_full,   digits=2))"
@info "  Strong binders β* = $(round(β_strong, digits=2))"

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Generate sequences from each seed condition
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 5: Generating sequences"

gen_full_seqs, gen_full_pca = generate_sequences(X̂_full, pca_full, L_full;
    β=β_full, n_chains=50, T=5000, seed=42)
@info "  Generated $(length(gen_full_seqs)) sequences from full family seed"

gen_strong_seqs, gen_strong_pca = generate_sequences(X̂_strong, pca_strong, L_strong;
    β=β_strong, n_chains=50, T=5000, seed=42)
@info "  Generated $(length(gen_strong_seqs)) sequences from strong-binder seed"

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Pharmacophore inheritance analysis
# ══════════════════════════════════════════════════════════════════════════════
@info "Step 6: Pharmacophore inheritance analysis"

function pharmacophore_analysis(seqs::Vector{String}, label::String,
                                 tyr_pos::Int, loop::Vector{Int})
    n = length(seqs)
    n == 0 && return nothing

    # Tyr fraction at pharmacophore position
    tyr_res = [length(s) >= tyr_pos ? s[tyr_pos] : '-' for s in seqs]
    n_tyr   = count(r -> r == 'Y', tyr_res)
    frac_tyr = n_tyr / n

    # basic residue (K/R) frequency across the whole sequence (electrostatic
    # complementarity with the negatively charged Cav2.2 pore entrance)
    total_res = sum(length(s) for s in seqs)
    n_basic   = sum(count(c -> c in ('K', 'R'), s) for s in seqs)
    frac_basic = n_basic / total_res

    @info "  [$label]"
    @info "    Tyr at pharmacophore pos $tyr_pos: $(round(frac_tyr,  digits=3)) ($n_tyr / $n)"
    @info "    Basic residue (K/R) fraction:     $(round(frac_basic, digits=3))"

    return (label=label, n=n, frac_tyr=frac_tyr, frac_basic=frac_basic,
            tyr_pos=tyr_pos, tyr_res=tyr_res)
end

results = []

@info "\n=== Input sequences ==="
push!(results, pharmacophore_analysis(stored_full,   "Input: Full family (74)",    tyr_pos_full,   loop_full))
push!(results, pharmacophore_analysis(stored_strong, "Input: Strong binders (23)", tyr_pos_strong, loop_strong))

@info "\n=== Generated sequences ==="
push!(results, pharmacophore_analysis(gen_full_seqs,   "Generated: Full-seeded",   tyr_pos_full,   loop_full))
push!(results, pharmacophore_analysis(gen_strong_seqs, "Generated: Strong-seeded", tyr_pos_strong, loop_strong))

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Standard quality metrics
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 7: Quality metrics"

function report_quality(seqs, pca_vecs, X̂, stored_seqs, label)
    n = length(seqs)
    n == 0 && return nothing

    kl         = aa_composition_kl(seqs, stored_seqs)
    novelties  = [sample_novelty(v, X̂) for v in pca_vecs]
    seq_ids    = [nearest_sequence_identity(s, stored_seqs) for s in seqs]
    valid_fracs = [valid_residue_fraction(s) for s in seqs]

    @info "  [$label]"
    @info "    KL(AA):   $(round(kl,               digits=4))"
    @info "    Novelty:  $(round(mean(novelties),  digits=3)) ± $(round(std(novelties),   digits=3))"
    @info "    SeqID:    $(round(mean(seq_ids),    digits=3)) ± $(round(std(seq_ids),     digits=3))"
    @info "    Valid:    $(round(mean(valid_fracs),digits=3))"

    return (label=label, kl=kl, mean_novelty=mean(novelties),
            mean_seqid=mean(seq_ids), mean_valid=mean(valid_fracs))
end

q_full   = report_quality(gen_full_seqs,   gen_full_pca,   X̂_full,   stored_full,   "Generated: Full-seeded")
q_strong = report_quality(gen_strong_seqs, gen_strong_pca, X̂_strong, stored_strong, "Generated: Strong-seeded")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Visualization
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 8: Generating figures"

# --- Helper: compute loop frequency matrix ---
function loop_freq_matrix(seqs, loop)
    isempty(seqs) && return nothing
    L_max = maximum(length(s) for s in seqs)
    loop_clipped = filter(p -> p <= L_max, loop)
    isempty(loop_clipped) && return nothing
    freq = aa_freq_matrix(seqs, maximum(loop_clipped))
    return freq[:, loop_clipped], loop_clipped
end

# --- Redesigned Figure: Binding-loop heatmap with residual ---
# Panel (a): Input strong binder heatmap (reference)
# Panel (b): Residual (Generated − Input) with diverging colormap
@info "  Generating redesigned heatmap + residual figure"

freq_input_strong, loop_s   = loop_freq_matrix(stored_strong, loop_strong)
freq_gen_strong, _          = loop_freq_matrix(gen_strong_seqs, loop_strong)
freq_input_full, loop_f     = loop_freq_matrix(stored_full, loop_full)
freq_gen_full, _            = loop_freq_matrix(gen_full_seqs, loop_full)

# Panel (a): Strong binder input
pa = heatmap(1:length(loop_s), 1:N_AA, freq_input_strong;
    xticks=(1:length(loop_s), string.(loop_s)),
    yticks=(1:N_AA, string.(AA_ALPHABET)),
    xlabel="Position", ylabel="Amino acid",
    title="(a) Input: strong Cav2.2 binders (n=23)",
    color=:YlOrRd, clims=(0, 1), colorbar_title="Frequency")

# Panel (b): Strong residual (generated - input)
residual_strong = freq_gen_strong .- freq_input_strong
max_res = max(maximum(abs.(residual_strong)), 0.15)  # floor at 0.15 for scale
pb = heatmap(1:length(loop_s), 1:N_AA, residual_strong;
    xticks=(1:length(loop_s), string.(loop_s)),
    yticks=(1:N_AA, string.(AA_ALPHABET)),
    xlabel="Position", ylabel="Amino acid",
    title="(b) Δ: generated − input (strong-seeded)",
    color=:RdBu, clims=(-max_res, max_res), colorbar_title="Δ freq")

# Panel (c): Full family input
pc = heatmap(1:length(loop_f), 1:N_AA, freq_input_full;
    xticks=(1:length(loop_f), string.(loop_f)),
    yticks=(1:N_AA, string.(AA_ALPHABET)),
    xlabel="Position", ylabel="Amino acid",
    title="(c) Input: full O-superfamily (n=74)",
    color=:YlOrRd, clims=(0, 1), colorbar_title="Frequency")

# Panel (d): Full family residual (generated - input)
residual_full = freq_gen_full .- freq_input_full
max_res_f = max(maximum(abs.(residual_full)), 0.15)
pd = heatmap(1:length(loop_f), 1:N_AA, residual_full;
    xticks=(1:length(loop_f), string.(loop_f)),
    yticks=(1:N_AA, string.(AA_ALPHABET)),
    xlabel="Position", ylabel="Amino acid",
    title="(d) Δ: generated − input (full-seeded)",
    color=:RdBu, clims=(-max_res_f, max_res_f), colorbar_title="Δ freq")

p_combined = plot(pa, pb, pc, pd; layout=(2, 2), size=(1100, 900), dpi=300,
    margin=5Plots.mm)
savefig(p_combined, joinpath(FIG_DIR, "loop_heatmap_with_residuals.pdf"))
savefig(p_combined, joinpath(FIG_DIR, "loop_heatmap_with_residuals.png"))
@info "  Saved loop_heatmap_with_residuals.{pdf,png}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Save generated sequences
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 9: Saving generated sequences"

function save_fasta(seqs, filepath, prefix)
    open(filepath, "w") do io
        for (i, seq) in enumerate(seqs)
            println(io, ">$(prefix)_$(lpad(i, 4, '0'))")
            println(io, seq)
        end
    end
    @info "  Saved $(length(seqs)) sequences → $filepath"
end

save_fasta(gen_full_seqs,   joinpath(DATA_DIR, "generated_full_seeded.fasta"),   "SA_full")
save_fasta(gen_strong_seqs, joinpath(DATA_DIR, "generated_strong_seeded.fasta"), "SA_strong")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# Step 10: Copy figures to paper/sections/figs
# ══════════════════════════════════════════════════════════════════════════════
@info "\nStep 10: Copying figures to paper"

const PAPER_FIG_DIR = joinpath(_CODE_DIR, "..", "paper", "sections", "figs")
mkpath(PAPER_FIG_DIR)

for f in readdir(FIG_DIR; join=true)
    endswith(f, ".pdf") || endswith(f, ".png") || continue
    dest = joinpath(PAPER_FIG_DIR, basename(f))
    cp(f, dest; force=true)
end
@info "  Copied figures → $PAPER_FIG_DIR"

@info "\n" * "="^70
@info "ω-Conotoxin experiment complete!"
@info "="^70
@info ""
@info "Key question: Does strong-binder seeding enrich Tyr at the"
@info "pharmacophore position relative to full-family seeding?"
@info ""
@info "If yes → SA inherits Cav2.2 binding pharmacophore from curated seed."
@info "This extends the Kunitz result to a therapeutic peptide family"
@info "with direct relevance to pain (Song & Varner 2009, PMID 19750220)."
@info ""
@info "Generated files:"
@info "  $(joinpath(DATA_DIR, "generated_full_seeded.fasta"))"
@info "  $(joinpath(DATA_DIR, "generated_strong_seeded.fasta"))"
