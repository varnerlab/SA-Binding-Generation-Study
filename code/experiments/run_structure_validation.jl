# ──────────────────────────────────────────────────────────────────────────────
# run_structure_validation.jl — Structural validation of SA-generated sequences
#
# Predicts 3D structures using ESMFold, extracts pLDDT confidence scores,
# and computes TM-scores against the experimental Kunitz domain structure (1BPI).
#
# Compares: SA full family, SA strong binders, SA weak binders, stored sequences
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

using HTTP

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
const PFAM_ID = "PF00014"
const CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "structure_validation")
const STRUCT_DIR = joinpath(_CODE_DIR, "data", "kunitz", "structures")
const TMALIGN_BIN = joinpath(_CODE_DIR, "bin", "TMalign")
const REFERENCE_PDB_ID = "1BPI"
const REFERENCE_CHAIN = "A"
const ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
const MAX_SEQS_PER_SOURCE = 50
const API_SLEEP = 1.5  # seconds between API calls (rate limiting)

mkpath(FIG_DIR)
mkpath(STRUCT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

"""
    esmfold_predict(sequence; timeout=120, max_retries=3) -> String

Predict a protein structure using the ESMFold REST API.
Returns the PDB-format string.
"""
function esmfold_predict(sequence::String; timeout::Int=120, max_retries::Int=3)
    for attempt in 1:max_retries
        try
            response = HTTP.post(ESMFOLD_URL, [], sequence;
                                  readtimeout=timeout, connect_timeout=30, retry=false)
            return String(response.body)
        catch e
            if attempt < max_retries
                wait_time = 2^attempt  # exponential backoff
                @warn "  ESMFold attempt $attempt failed: $e. Retrying in $(wait_time)s..."
                sleep(wait_time)
            else
                @error "  ESMFold failed after $max_retries attempts: $e"
                return ""
            end
        end
    end
    return ""
end

"""
    extract_plddt(pdb_string) -> Float64

Extract mean pLDDT from the B-factor column of PDB ATOM records.
ESMFold stores pLDDT in the B-factor field (columns 61-66).
"""
function extract_plddt(pdb_string::String)
    plddts = Float64[]
    for line in split(pdb_string, "\n")
        if startswith(line, "ATOM") && length(line) >= 66
            try
                val = parse(Float64, strip(line[61:66]))
                # ESMFold may return pLDDT on [0,1] or [0,100] scale
                if val < 1.5
                    val *= 100.0
                end
                push!(plddts, val)
            catch
                continue
            end
        end
    end
    return isempty(plddts) ? 0.0 : mean(plddts)
end

"""
    download_reference_pdb(pdb_id, chain, outdir) -> String

Download a reference PDB structure from RCSB and filter to specified chain.
Returns the path to the cleaned PDB file.
"""
function download_reference_pdb(pdb_id::String, chain::String, outdir::String)
    clean_path = joinpath(outdir, "$(pdb_id)_$(chain).pdb")
    if isfile(clean_path)
        @info "  Using cached reference: $clean_path"
        return clean_path
    end

    url = "https://files.rcsb.org/download/$(pdb_id).pdb"
    raw_path = joinpath(outdir, "$(pdb_id)_raw.pdb")
    @info "  Downloading reference PDB from $url"
    Downloads.download(url, raw_path)

    # Filter to specified chain (ATOM records only)
    open(clean_path, "w") do out
        for line in eachline(raw_path)
            if startswith(line, "ATOM") && length(line) >= 22 && line[22] == chain[1]
                println(out, line)
            elseif startswith(line, "END")
                println(out, line)
            end
        end
    end
    @info "  Saved cleaned reference: $clean_path"
    return clean_path
end

"""
    compute_tmscore(query_pdb, reference_pdb) -> Float64

Run TM-align and extract TM-score (normalized by reference length).
"""
function compute_tmscore(query_pdb::String, reference_pdb::String)
    try
        output = read(`$TMALIGN_BIN $query_pdb $reference_pdb`, String)
        # Look for TM-score normalized by Chain_2 (reference)
        for line in split(output, "\n")
            if contains(line, "TM-score") && contains(line, "Chain_2")
                m = match(r"TM-score=\s*([\d.]+)", line)
                if m !== nothing
                    return parse(Float64, m.captures[1])
                end
            end
        end
        # Fallback: first TM-score line
        for line in split(output, "\n")
            if contains(line, "TM-score")
                m = match(r"TM-score=\s*([\d.]+)", line)
                if m !== nothing
                    return parse(Float64, m.captures[1])
                end
            end
        end
    catch e
        @warn "  TM-align failed: $e"
    end
    return 0.0
end

"""
    predict_and_score(name, sequence, source, reference_pdb, struct_dir) -> NamedTuple

Predict structure, extract pLDDT, compute TM-score.
Uses caching: if PDB already exists, skip prediction.
"""
function predict_and_score(name::String, sequence::String, source::String,
                            reference_pdb::String, struct_dir::String)
    # Sanitize name for filename
    safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
    pdb_path = joinpath(struct_dir, "$(source)_$(safe_name).pdb")

    # Check cache
    if isfile(pdb_path) && filesize(pdb_path) > 100
        pdb_str = read(pdb_path, String)
    else
        pdb_str = esmfold_predict(sequence)
        if isempty(pdb_str)
            return (name=name, source=source, plddt=0.0, tmscore=0.0, success=false)
        end
        write(pdb_path, pdb_str)
        sleep(API_SLEEP)  # rate limit
    end

    plddt = extract_plddt(pdb_str)
    tmscore = compute_tmscore(pdb_path, reference_pdb)

    return (name=name, source=source, plddt=plddt, tmscore=tmscore, success=true)
end

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
@info "======================================================================="
@info "Structure Validation: ESMFold pLDDT + TM-align TM-score"
@info "======================================================================="

# Download reference structure
ref_pdb = download_reference_pdb(REFERENCE_PDB_ID, REFERENCE_CHAIN, STRUCT_DIR)

# Load stored sequences (ungapped)
@info "Loading sequences"
sto_file = download_pfam_seed(PFAM_ID; cache_dir=CACHE_DIR)
raw_seqs = parse_stockholm(sto_file)
char_mat, names = clean_alignment(raw_seqs)
K, L = size(char_mat)
stored_seqs_ungapped = [(names[i], replace(String(char_mat[i, :]), r"[.\-~]" => ""))
                         for i in 1:K]

# Load SA-generated sequences
sources = Dict{String, Vector{Tuple{String,String}}}()

# Stored (natural)
sources["stored"] = stored_seqs_ungapped[1:min(MAX_SEQS_PER_SOURCE, length(stored_seqs_ungapped))]

# SA generated
for (label, filename) in [
    ("SA_full", "generated_full_family_example.fasta"),
    ("SA_strong", "generated_strong_conditioned_example.fasta"),
    ("SA_weak", "generated_weak_conditioned_example.fasta"),
]
    fasta_path = joinpath(CACHE_DIR, filename)
    if isfile(fasta_path)
        raw = parse_fasta(fasta_path)
        # Remove gaps from generated sequences
        cleaned = [(n, replace(s, r"[.\-~]" => "")) for (n, s) in raw]
        sources[label] = cleaned[1:min(MAX_SEQS_PER_SOURCE, length(cleaned))]
        @info "  Loaded $(length(sources[label])) $label sequences"
    else
        @warn "  $fasta_path not found, skipping $label"
    end
end

# HMM baseline
hmm_path = joinpath(CACHE_DIR, "hmm_generated.fasta")
if isfile(hmm_path)
    raw = parse_fasta(hmm_path)
    cleaned = [(n, replace(s, r"[.\-~]" => "")) for (n, s) in raw]
    sources["HMM_emit"] = cleaned[1:min(MAX_SEQS_PER_SOURCE, length(cleaned))]
    @info "  Loaded $(length(sources["HMM_emit"])) HMM sequences"
end

# ══════════════════════════════════════════════════════════════════════════════
# Predict structures and compute scores
# ══════════════════════════════════════════════════════════════════════════════
@info "\nPredicting structures and computing scores..."
@info "  This may take a while ($(sum(length(v) for v in values(sources))) sequences × ~2s each)"

all_results = []
for (source_name, seqs) in sources
    @info "\n  Processing $source_name ($(length(seqs)) sequences)..."
    for (i, (name, seq)) in enumerate(seqs)
        if i % 10 == 1
            @info "    $i/$(length(seqs))"
        end
        result = predict_and_score(name, seq, source_name, ref_pdb, STRUCT_DIR)
        push!(all_results, result)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Aggregate results
# ══════════════════════════════════════════════════════════════════════════════
@info "\nAggregating results..."

results_df = DataFrame(
    name = [r.name for r in all_results],
    source = [r.source for r in all_results],
    plddt = [r.plddt for r in all_results],
    tmscore = [r.tmscore for r in all_results],
    success = [r.success for r in all_results],
)

# Filter to successful predictions
success_df = filter(row -> row.success, results_df)

# Save per-sequence results
CSV.write(joinpath(CACHE_DIR, "structure_validation_raw.csv"), results_df)

# Compute summary statistics
summary_df = combine(groupby(success_df, :source),
    :plddt => mean => :plddt_mean,
    :plddt => std => :plddt_std,
    :plddt => (x -> mean(x .> 70)) => :plddt_frac_gt70,
    :tmscore => mean => :tmscore_mean,
    :tmscore => std => :tmscore_std,
    :tmscore => (x -> mean(x .> 0.5)) => :tmscore_frac_gt05,
    nrow => :n_success,
)

@info "\nStructure Validation Summary:"
pretty_table(summary_df)
CSV.write(joinpath(CACHE_DIR, "structure_validation_summary.csv"), summary_df)

# ══════════════════════════════════════════════════════════════════════════════
# Generate figures
# ══════════════════════════════════════════════════════════════════════════════
@info "Generating structure validation figures"

# Color scheme
source_colors = Dict(
    "stored" => RGB(0.50, 0.50, 0.50),
    "SA_full" => RGB(0.20, 0.47, 0.69),
    "SA_strong" => RGB(0.80, 0.20, 0.20),
    "SA_weak" => RGB(0.20, 0.60, 0.20),
    "HMM_emit" => RGB(0.93, 0.68, 0.20),
)

source_order = ["stored", "SA_full", "SA_strong", "SA_weak", "HMM_emit"]
source_labels = ["Stored", "SA (full)", "SA (strong)", "SA (weak)", "HMM emit"]

# Filter to sources that exist in results
available = [s for s in source_order if s in unique(success_df.source)]
avail_labels = [source_labels[findfirst(==(s), source_order)] for s in available]
avail_colors = [source_colors[s] for s in available]

# Get summary values in order
plddt_means = Float64[]
plddt_stds = Float64[]
tm_means = Float64[]
tm_stds = Float64[]
for s in available
    row = filter(r -> r.source == s, summary_df)
    if nrow(row) > 0
        push!(plddt_means, row.plddt_mean[1])
        push!(plddt_stds, row.plddt_std[1])
        push!(tm_means, row.tmscore_mean[1])
        push!(tm_stds, row.tmscore_std[1])
    end
end

# Figure 1: pLDDT comparison
p_plddt = bar(avail_labels, plddt_means; yerror=plddt_stds,
    ylabel="Mean pLDDT", title="Predicted Structure Confidence (ESMFold)",
    color=reshape(avail_colors, 1, :), legend=false, bar_width=0.6,
    linewidth=0, size=(650, 450), dpi=300,
    xrotation=15, margin=5Plots.mm, ylims=(0, 105))
hline!([70], linestyle=:dash, color=:gray, label="Confident threshold")
savefig(p_plddt, joinpath(FIG_DIR, "plddt_comparison.png"))
@info "  Saved plddt_comparison"

# Figure 2: TM-score comparison
p_tm = bar(avail_labels, tm_means; yerror=tm_stds,
    ylabel="Mean TM-score", title="Structural Similarity to 1BPI (TM-align)",
    color=reshape(avail_colors, 1, :), legend=false, bar_width=0.6,
    linewidth=0, size=(650, 450), dpi=300,
    xrotation=15, margin=5Plots.mm, ylims=(0, 1.1))
hline!([0.5], linestyle=:dash, color=:gray, label="Same fold threshold")
savefig(p_tm, joinpath(FIG_DIR, "tmscore_comparison.png"))
@info "  Saved tmscore_comparison"

# Figure 3: Combined 2-panel
p_combined = plot(p_plddt, p_tm; layout=(1, 2), size=(1200, 450), dpi=300,
    plot_title="Structural Validation: SA vs Baselines")
savefig(p_combined, joinpath(FIG_DIR, "structure_validation_combined.png"))
@info "  Saved structure_validation_combined"

# Figure 4: pLDDT vs TM-score scatter
p_scatter = plot(; xlabel="Mean pLDDT", ylabel="TM-score (vs 1BPI)",
    title="Structure Quality: pLDDT vs TM-score",
    size=(600, 500), dpi=300, legend=:bottomright, margin=5Plots.mm)
for s in available
    label = avail_labels[findfirst(==(s), available)]
    color = source_colors[s]
    sdf = filter(r -> r.source == s, success_df)
    scatter!(sdf.plddt, sdf.tmscore; label=label, color=color,
             markersize=5, markerstrokewidth=0, alpha=0.7)
end
hline!([0.5], linestyle=:dash, color=:gray, label="", alpha=0.5)
vline!([70], linestyle=:dash, color=:gray, label="", alpha=0.5)
savefig(p_scatter, joinpath(FIG_DIR, "plddt_vs_tmscore_scatter.png"))
@info "  Saved plddt_vs_tmscore_scatter"

@info "\n======================================================================="
@info "Structure validation complete!"
@info "======================================================================="
