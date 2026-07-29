include(joinpath(@__DIR__, "..", "Include.jl"))
include(joinpath(@__DIR__, "..", "experiments", "canonical_family_registry.jl"))
using Test

@testset "conotoxin accession and Tyr13 labels remain distinct" begin
    data_dir = joinpath(@__DIR__, "..", "data")
    spec = only(filter(s -> s.family == "Conotoxin", CANONICAL_FAMILIES))
    char_mat, names, auxiliary = canonical_load_alignment(spec, data_dir)
    designated, background, marker_col, marker_residues =
        canonical_split(spec, char_mat, names, auxiliary)
    marker_positive = Set(findall(
        i -> char_mat[i, marker_col] in marker_residues,
        axes(char_mat, 1),
    ))

    tp = length(intersect(Set(designated), marker_positive))
    fn = length(designated) - tp
    fp = length(intersect(Set(background), marker_positive))
    tn = length(background) - fp

    @test size(char_mat, 1) == 74
    @test marker_col == 13
    @test (length(designated), length(background)) == (23, 51)
    @test (tp, fn, fp, tn) == (19, 4, 6, 45)
    @test (tp + tn) / size(char_mat, 1) == 64 / 74

    summary = CSV.read(
        joinpath(data_dir, "omega_conotoxin", "conotoxin_label_agreement_summary.csv"),
        DataFrame,
    )
    @test nrow(summary) == 1
    @test summary.component_marker_agreement[1] == 64 / 74

    audit = CSV.read(
        joinpath(data_dir, "omega_conotoxin", "conotoxin_label_audit.csv"),
        DataFrame,
    )
    @test nrow(audit) == 74
    @test count(audit.designated_set) == 23
    @test count(audit.repository_activity_record) == 7
    @test count(audit.labels_agree) == 64
    @test Set(audit.accession[audit.designated_set]) == auxiliary
end
