# ──────────────────────────────────────────────────────────────────────────────
# render_sequence_analysis_figure.jl
#
# Sequence-level analysis of SA-generated Kunitz domain sequences.
# Analogous to Figure 6 in arxiv:2603.14717, but adapted for binding study.
#
# Panel A: Alignment of family consensus with top 5 SA strong-binder-conditioned
#          sequences (ranked by ESMFold pLDDT). Positions colored by conservation.
#          P1 residue highlighted. Dots = matches to consensus, letters = substitutions.
#
# Panel B: Per-position Shannon entropy comparison:
#          stored MSA vs SA full-family vs SA strong-conditioned
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

using StatsBase

# ── Configuration ────────────────────────────────────────────────────────────
const PFAM_ID = "PF00014"
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "sequence_analysis")
const PAPER_FIG_DIR = joinpath(_CODE_DIR, "..", "paper", "sections", "figs")
mkpath(FIG_DIR)

# ── Load stored MSA ─────────────────────────────────────────────────────────
@info "Loading stored alignment"
sto_file = download_pfam_seed(PFAM_ID; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, seq_names = clean_alignment(raw_seqs; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
K, L = size(char_mat)
@info "  $K sequences × $L positions"

# ── Identify P1 position ────────────────────────────────────────────────────
lys_fracs = zeros(L)
for j in 1:L
    n_lys = count(i -> char_mat[i, j] == 'K', 1:K)
    n_valid = count(i -> char_mat[i, j] ∉ ('-', '.'), 1:K)
    lys_fracs[j] = n_valid > 0 ? n_lys / n_valid : 0.0
end
p1_candidates = findall(f -> f > 0.2, lys_fracs)
p1_pos = isempty(p1_candidates) ? argmax(lys_fracs) :
         p1_candidates[argmax(lys_fracs[p1_candidates])]
@info "  P1 position: column $p1_pos (Lys fraction: $(round(lys_fracs[p1_pos], digits=2)))"

# ── Per-position conservation and consensus ──────────────────────────────────
conservation = zeros(L)
consensus = fill('-', L)
stored_entropy = zeros(L)

for j in 1:L
    col = [char_mat[i, j] for i in 1:K if char_mat[i, j] ∉ ('-', '.', '~')]
    if isempty(col)
        continue
    end
    counts = countmap(col)
    n = length(col)
    # Find most frequent residue
    max_count = 0
    max_aa = '-'
    for (aa, c) in counts
        if c > max_count
            max_count = c
            max_aa = aa
        end
    end
    conservation[j] = max_count / n
    consensus[j] = max_aa
    # Shannon entropy
    H = -sum(c/n * log2(c/n) for (_, c) in counts)
    stored_entropy[j] = H
end

consensus_str = String(consensus)
cys_positions = findall(c -> c == 'C', consensus)
@info "  Consensus: $consensus_str"
@info "  Cysteine positions: $cys_positions"
@info "  P1 consensus residue: $(consensus[p1_pos])"

# ── Load generated sequences ────────────────────────────────────────────────
@info "Loading generated sequences"
strong_seqs_raw = parse_fasta(joinpath(CACHE_DIR, "generated_strong_conditioned.fasta"))
full_seqs_raw = parse_fasta(joinpath(CACHE_DIR, "generated_full_family.fasta"))
@info "  Strong-conditioned: $(length(strong_seqs_raw)) sequences"
@info "  Full-family: $(length(full_seqs_raw)) sequences"

# ── Load pLDDT scores from structure validation ─────────────────────────────
@info "Loading structure validation scores"
val_df = CSV.read(joinpath(CACHE_DIR, "structure_validation_raw.csv"), DataFrame)
strong_val = filter(r -> r.source == "SA_strong" && r.success, val_df)
full_val = filter(r -> r.source == "SA_full" && r.success, val_df)

# Build name→pLDDT lookup
strong_plddt = Dict(r.name => r.plddt for r in eachrow(strong_val))
full_plddt = Dict(r.name => r.plddt for r in eachrow(full_val))

# ── Select top 5 strong-conditioned by pLDDT ────────────────────────────────
# Match fasta headers to validation names
strong_with_plddt = []
for (name, seq) in strong_seqs_raw
    if haskey(strong_plddt, name)
        push!(strong_with_plddt, (name=name, seq=seq, plddt=strong_plddt[name]))
    end
end
sort!(strong_with_plddt, by=x -> -x.plddt)
top5_strong = strong_with_plddt[1:min(5, length(strong_with_plddt))]
@info "  Top 5 strong-conditioned by pLDDT:"
for t in top5_strong
    @info "    $(t.name): pLDDT=$(round(t.plddt, digits=1))"
end

# ── Compute per-position entropy for generated sets ──────────────────────────
function compute_entropy(seqs, L)
    entropy = zeros(L)
    for j in 1:L
        col = [s[j] for (_, s) in seqs if j <= length(s) && s[j] ∉ ('-', '.', '~')]
        if isempty(col)
            continue
        end
        counts = countmap(col)
        n = length(col)
        entropy[j] = -sum(c/n * log2(c/n) for (_, c) in counts)
    end
    return entropy
end

strong_entropy = compute_entropy(strong_seqs_raw, L)
full_entropy = compute_entropy(full_seqs_raw, L)

# Correlation
r_full = cor(stored_entropy, full_entropy)
r_strong = cor(stored_entropy, strong_entropy)
@info "  Entropy correlation (stored vs full): $(round(r_full, digits=3))"
@info "  Entropy correlation (stored vs strong): $(round(r_strong, digits=3))"

# ── Count substitutions and conservation matches for top 5 ──────────────────
@info "\nSequence-level statistics for top 5:"
n_highly_conserved = count(c -> c >= 0.90, conservation)
n_cys = length(cys_positions)
for t in top5_strong
    seq = t.seq
    n_match_hc = count(j -> conservation[j] >= 0.90 && j <= length(seq) && seq[j] == consensus[j], 1:L)
    n_cys_match = count(j -> j <= length(seq) && seq[j] == 'C', cys_positions)
    n_sub = count(j -> j <= length(seq) && seq[j] != consensus[j], 1:L)
    p1_res = length(seq) >= p1_pos ? seq[p1_pos] : '?'
    @info "  $(t.name): pLDDT=$(round(t.plddt,digits=1)), P1=$(p1_res), conserved=$(n_match_hc)/$(n_highly_conserved), Cys=$(n_cys_match)/$(n_cys), subs=$(n_sub)/$(L)"
end

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
@info "\nGenerating figure..."

# ── Color definitions ────────────────────────────────────────────────────────
color_highly_conserved = RGB(0.776, 0.157, 0.157)  # dark red
color_conserved = RGB(0.937, 0.424, 0.0)            # orange
color_moderate = RGB(1.0, 0.976, 0.769)              # light yellow
color_variable = RGB(0.890, 0.949, 0.992)            # light blue
color_cysteine = RGB(1.0, 0.843, 0.0)                # gold
color_p1_highlight = RGB(0.6, 0.0, 0.6)              # purple for P1
color_sub_variable = RGB(0.733, 0.871, 0.984)        # blue for substitutions at variable sites
color_sub_conserved = RGB(1.0, 0.804, 0.698)         # light orange for subs at conserved sites
color_match_bg = RGB(0.96, 0.96, 0.96)               # light gray for matches

function get_cell_color(cons, is_cys, is_p1)
    is_p1 && return color_p1_highlight
    is_cys && return color_cysteine
    cons >= 0.90 && return color_highly_conserved
    cons >= 0.70 && return color_conserved
    cons >= 0.50 && return color_moderate
    return color_variable
end

function get_text_color(cons, is_cys, is_p1)
    (is_p1 || cons >= 0.70) && return RGB(1,1,1)
    return RGB(0,0,0)
end

# ── Panel A: Sequence alignment heatmap ──────────────────────────────────────
# Build a matrix for the heatmap visualization
n_rows = 2 + length(top5_strong)  # conservation bar + consensus + top 5
row_labels = ["Conservation", "Consensus"]
for t in top5_strong
    p1_res = length(t.seq) >= p1_pos ? string(t.seq[p1_pos]) : "?"
    push!(row_labels, "SA strong $(split(t.name,'_')[end]) (pLDDT=$(round(Int,t.plddt)), P1=$(p1_res))")
end

# We'll use a manual annotation approach with heatmap
# Create color matrix and text matrix
color_matrix = fill(RGB(1.0, 1.0, 1.0), n_rows, L)
text_matrix = fill("", n_rows, L)

for j in 1:L
    is_cys = consensus[j] == 'C'
    is_p1 = j == p1_pos

    # Row 1: Conservation bar
    color_matrix[1, j] = get_cell_color(conservation[j], is_cys, is_p1)
    if conservation[j] >= 0.90 || is_cys || is_p1
        text_matrix[1, j] = "$(round(Int, conservation[j]*100))%"
    end

    # Row 2: Consensus
    color_matrix[2, j] = get_cell_color(conservation[j], is_cys, is_p1)
    text_matrix[2, j] = string(consensus[j])

    # Rows 3+: SA generated sequences
    for (si, t) in enumerate(top5_strong)
        row = si + 2
        seq = t.seq
        if j > length(seq)
            continue
        end
        aa = seq[j]
        if aa == consensus[j]
            color_matrix[row, j] = color_match_bg
            text_matrix[row, j] = "·"
        else
            if is_p1
                color_matrix[row, j] = color_p1_highlight
            elseif is_cys
                color_matrix[row, j] = color_cysteine
            elseif conservation[j] >= 0.70
                color_matrix[row, j] = color_sub_conserved
            else
                color_matrix[row, j] = color_sub_variable
            end
            text_matrix[row, j] = string(aa)
        end
    end
end

# ── Render Panel A using Plots with annotations ─────────────────────────────
# Use a dummy heatmap for the grid, then overlay text
z_matrix = zeros(n_rows, L)  # dummy values; we color manually

p_aln = plot(; size=(1200, 350), margin=10Plots.mm,
    xlims=(0.5, L+0.5), ylims=(0.5, n_rows+0.5),
    yflip=true, grid=false, framestyle=:none,
    xticks=([i for i in 5:5:L], [string(i) for i in 5:5:L]),
    yticks=([i for i in 1:n_rows], row_labels),
    tickfontsize=6, xmirror=true,
    title="(A) Kunitz domain: consensus vs. top SA strong-binder-conditioned sequences",
    titlefontsize=10, titleloc=:left)

# Draw colored rectangles
for row in 1:n_rows
    for col in 1:L
        c = color_matrix[row, col]
        plot!(p_aln, Shape([col-0.48, col+0.48, col+0.48, col-0.48],
                           [row-0.45, row-0.45, row+0.45, row+0.45]),
              fillcolor=c, linecolor=:white, linewidth=0.3, label=false)
        if !isempty(text_matrix[row, col])
            tc = (row <= 2) ? get_text_color(conservation[col], consensus[col]=='C', col==p1_pos) : RGB(0,0,0)
            # For match dots, use gray
            if text_matrix[row, col] == "·"
                tc = RGB(0.6, 0.6, 0.6)
            end
            # For P1 substitutions in generated seqs, use white
            if col == p1_pos && row > 2 && text_matrix[row, col] != "·"
                tc = RGB(1, 1, 1)
            end
            fs = (row == 1) ? 5 : 6
            annotate!(p_aln, col, row, text(text_matrix[row, col], fs, tc, :center))
        end
    end
end

# P1 marker arrow
annotate!(p_aln, p1_pos, 0.3, text("P1↓", 7, color_p1_highlight, :center))

# ── Panel B: Per-position entropy ────────────────────────────────────────────
positions = 1:L
p_ent = plot(; size=(1200, 300), margin=10Plots.mm,
    xlabel="Alignment position", ylabel="Shannon entropy (bits)",
    title="(B) Per-position entropy: stored MSA vs. SA-generated sequences",
    titlefontsize=10, titleloc=:left,
    xlims=(0, L+1), legend=:topright, legendfontsize=7,
    grid=false, framestyle=:box)

# Bar groups
bw = 0.25
bar!(p_ent, collect(positions) .- bw, stored_entropy, bar_width=bw,
    color=RGB(0.47, 0.56, 0.61), alpha=0.85, label="Stored MSA", linewidth=0)
bar!(p_ent, collect(positions), full_entropy, bar_width=bw,
    color=RGB(0.08, 0.40, 0.75), alpha=0.85, label="SA full-family (r=$(round(r_full,digits=3)))", linewidth=0)
bar!(p_ent, collect(positions) .+ bw, strong_entropy, bar_width=bw,
    color=RGB(0.80, 0.20, 0.20), alpha=0.85, label="SA strong-conditioned (r=$(round(r_strong,digits=3)))", linewidth=0)

# Highlight P1 position
vspan!(p_ent, [p1_pos - 0.5, p1_pos + 0.5], alpha=0.15, color=color_p1_highlight, label="P1 position")

# Highlight cysteine positions
for cp in cys_positions
    vspan!(p_ent, [cp - 0.5, cp + 0.5], alpha=0.1, color=color_cysteine, label=(cp == cys_positions[1] ? "Cys (S-S)" : false))
end

# Position ticks every 5
xticks!(p_ent, [i for i in 5:5:L])

# ── Combined figure ──────────────────────────────────────────────────────────
p_combined = plot(p_aln, p_ent; layout=grid(2, 1, heights=[0.55, 0.45]),
    size=(1200, 700), dpi=300)

savefig(p_combined, joinpath(FIG_DIR, "sequence_analysis_kunitz.pdf"))
savefig(p_combined, joinpath(FIG_DIR, "sequence_analysis_kunitz.png"))
@info "Saved to $(joinpath(FIG_DIR, "sequence_analysis_kunitz.pdf"))"

# Copy to paper figs
cp(joinpath(FIG_DIR, "sequence_analysis_kunitz.pdf"),
   joinpath(PAPER_FIG_DIR, "sequence_analysis_kunitz.pdf"); force=true)
cp(joinpath(FIG_DIR, "sequence_analysis_kunitz.png"),
   joinpath(PAPER_FIG_DIR, "sequence_analysis_kunitz.png"); force=true)
@info "Copied to paper/sections/figs/"

@info "\n======================================================================="
@info "Sequence analysis figure complete!"
@info "======================================================================="
