# ──────────────────────────────────────────────────────────────────────────────
# run_conotoxin_structure_validation.jl
#
# Structural validation of SA-generated ω-conotoxin sequences.
# Predicts 3D structures using ESMFold, extracts pLDDT confidence scores,
# and computes TM-scores against the experimental MVIIA structure (1OMG).
#
# Compares: stored (full family), SA full-seeded, SA strong-seeded
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = @__DIR__
_CODE_DIR = joinpath(_SCRIPT_DIR, "code")
include(joinpath(_CODE_DIR, "Include.jl"))

using HTTP

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
const DATA_DIR = joinpath(_CODE_DIR, "data", "omega_conotoxin")
const FIG_DIR = joinpath(_CODE_DIR, "figs", "conotoxin_structure_validation")
const STRUCT_DIR = joinpath(DATA_DIR, "structures")
const TMALIGN_BIN = joinpath(_CODE_DIR, "bin", "TMalign")
const REFERENCE_PDB_ID = "1OMG"   # NMR structure of ω-conotoxin MVIIA
const REFERENCE_CHAIN = "A"
const ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
const MAX_SEQS_PER_SOURCE = 50
const API_SLEEP = 1.5  # seconds between API calls (rate limiting)

mkpath(FIG_DIR)
mkpath(STRUCT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions (same as Kunitz validation)
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
                wait_time = 2^attempt
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
"""
function extract_plddt(pdb_string::String)
    plddts = Float64[]
    for line in split(pdb_string, "\n")
        if startswith(line, "ATOM") && length(line) >= 66
            try
                val = parse(Float64, strip(line[61:66]))
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
"""
function predict_and_score(name::String, sequence::String, source::String,
                            reference_pdb::String, struct_dir::String)
    safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
    pdb_path = joinpath(struct_dir, "$(source)_$(safe_name).pdb")

    if isfile(pdb_path) && filesize(pdb_path) > 100
        pdb_str = read(pdb_path, String)
    else
        pdb_str = esmfold_predict(sequence)
        if isempty(pdb_str)
            return (name=name, source=source, plddt=0.0, tmscore=0.0, success=false)
        end
        write(pdb_path, pdb_str)
        sleep(API_SLEEP)
    end

    plddt = extract_plddt(pdb_str)
    tmscore = compute_tmscore(pdb_path, reference_pdb)

    return (name=name, source=source, plddt=plddt, tmscore=tmscore, success=true)
end

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
@info "======================================================================="
@info "ω-Conotoxin Structure Validation: ESMFold pLDDT + TM-align TM-score"
@info "======================================================================="

# Download reference structure (MVIIA)
ref_pdb = download_reference_pdb(REFERENCE_PDB_ID, REFERENCE_CHAIN, STRUCT_DIR)

# Load stored sequences (ungapped)
@info "Loading sequences"
raw_full = parse_fasta(joinpath(DATA_DIR, "omega_conotoxin_full_family.fasta"))
@info "  Full family: $(length(raw_full)) sequences"

sources = Dict{String, Vector{Tuple{String,String}}}()

# Stored (natural) — use ungapped sequences
sources["stored"] = raw_full[1:min(MAX_SEQS_PER_SOURCE, length(raw_full))]

# SA generated
for (label, filename) in [
    ("SA_full", "generated_full_seeded.fasta"),
    ("SA_strong", "generated_strong_seeded.fasta"),
]
    fasta_path = joinpath(DATA_DIR, filename)
    if isfile(fasta_path)
        raw = parse_fasta(fasta_path)
        cleaned = [(n, replace(s, r"[.\-~]" => "")) for (n, s) in raw]
        sources[label] = cleaned[1:min(MAX_SEQS_PER_SOURCE, length(cleaned))]
        @info "  Loaded $(length(sources[label])) $label sequences"
    else
        @warn "  $fasta_path not found, skipping $label"
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Predict structures and compute scores
# ══════════════════════════════════════════════════════════════════════════════
total_seqs = sum(length(v) for v in values(sources))
@info "\nPredicting structures and computing scores..."
@info "  This may take a while ($total_seqs sequences × ~2s each)"

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

success_df = filter(row -> row.success, results_df)

# Save per-sequence results
CSV.write(joinpath(DATA_DIR, "structure_validation_raw.csv"), results_df)

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
CSV.write(joinpath(DATA_DIR, "structure_validation_summary.csv"), summary_df)

# ══════════════════════════════════════════════════════════════════════════════
# Generate figures
# ══════════════════════════════════════════════════════════════════════════════
@info "Generating structure validation figures"

source_colors = Dict(
    "stored" => RGB(0.50, 0.50, 0.50),
    "SA_full" => RGB(0.20, 0.47, 0.69),
    "SA_strong" => RGB(0.80, 0.20, 0.20),
)

source_order = ["stored", "SA_full", "SA_strong"]
source_labels = ["Stored", "SA (full)", "SA (strong)"]

available = [s for s in source_order if s in unique(success_df.source)]
avail_labels = [source_labels[findfirst(==(s), source_order)] for s in available]
avail_colors = [source_colors[s] for s in available]

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

# pLDDT vs TM-score scatter
p_scatter = plot(; xlabel="Mean pLDDT", ylabel="TM-score (vs 1OMG/MVIIA)",
    title="ω-Conotoxin Structure Quality: pLDDT vs TM-score",
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
savefig(p_scatter, joinpath(FIG_DIR, "conotoxin_plddt_vs_tmscore.png"))
savefig(p_scatter, joinpath(FIG_DIR, "conotoxin_plddt_vs_tmscore.pdf"))
@info "  Saved conotoxin_plddt_vs_tmscore"

@info "\n======================================================================="
@info "ω-Conotoxin structure validation complete!"
@info "======================================================================="
