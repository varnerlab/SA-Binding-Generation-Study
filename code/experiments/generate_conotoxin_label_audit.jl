# Reproduce the accession-designation versus Tyr-marker audit used by the paper.

include(joinpath(@__DIR__, "..", "Include.jl"))
include(joinpath(@__DIR__, "canonical_family_registry.jl"))

const CONOTOXIN_DATA_DIR = joinpath(@__DIR__, "..", "data", "omega_conotoxin")
const OUTPUT_DIR = isempty(ARGS) ? CONOTOXIN_DATA_DIR :
    normpath(replace(only(ARGS), "--output=" => ""))
mkpath(OUTPUT_DIR)

spec = only(filter(s -> s.family == "Conotoxin", CANONICAL_FAMILIES))
char_mat, names, auxiliary = canonical_load_alignment(spec, joinpath(@__DIR__, "..", "data"))
designated, background, marker_col, marker_residues =
    canonical_split(spec, char_mat, names, auxiliary)

designated_set = Set(designated)
marker_positive_set = Set(findall(
    i -> char_mat[i, marker_col] in marker_residues,
    axes(char_mat, 1),
))

binding_data = CSV.read(joinpath(CONOTOXIN_DATA_DIR, "binding_data.csv"), DataFrame)
activity_by_accession = Dict(String(row.uniprot_id) => row for row in eachrow(binding_data))

audit = DataFrame(
    accession=String[],
    name=String[],
    designated_set=Bool[],
    marker_column=Int[],
    marker_residue=String[],
    tyr_marker_positive=Bool[],
    labels_agree=Bool[],
    repository_activity_record=Bool[],
    recorded_target=String[],
    evidence_class=String[],
)

for i in axes(char_mat, 1)
    header_parts = split(names[i], "|"; limit=2)
    accession = header_parts[1]
    name = length(header_parts) == 2 ? header_parts[2] : ""
    is_designated = i in designated_set
    is_marker_positive = i in marker_positive_set
    has_activity = haskey(activity_by_accession, accession)
    target = has_activity ? string(activity_by_accession[accession].target) : ""
    evidence_class =
        has_activity && is_designated ? "activity_recorded_designated" :
        has_activity ? "activity_recorded_background" :
        is_designated ? "designation_only" :
        "unannotated_background"

    push!(audit, (
        accession,
        name,
        is_designated,
        marker_col,
        string(char_mat[i, marker_col]),
        is_marker_positive,
        is_designated == is_marker_positive,
        has_activity,
        target,
        evidence_class,
    ))
end

tp = count(row -> row.designated_set && row.tyr_marker_positive, eachrow(audit))
fn = count(row -> row.designated_set && !row.tyr_marker_positive, eachrow(audit))
fp = count(row -> !row.designated_set && row.tyr_marker_positive, eachrow(audit))
tn = count(row -> !row.designated_set && !row.tyr_marker_positive, eachrow(audit))

summary = DataFrame(
    n_total=nrow(audit),
    n_designated=length(designated),
    n_background=length(background),
    marker_column=marker_col,
    designated_marker_positive=tp,
    designated_marker_negative=fn,
    background_marker_positive=fp,
    background_marker_negative=tn,
    component_marker_agreement=(tp + tn) / nrow(audit),
)

CSV.write(joinpath(OUTPUT_DIR, "conotoxin_label_audit.csv"), audit)
CSV.write(joinpath(OUTPUT_DIR, "conotoxin_label_agreement_summary.csv"), summary)

println(summary)
