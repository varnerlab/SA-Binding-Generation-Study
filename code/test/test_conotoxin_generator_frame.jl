include(joinpath(@__DIR__, "..", "Include.jl"))
include(joinpath(@__DIR__, "..", "experiments", "canonical_family_registry.jl"))
using Test

@testset "conotoxin generator's designated matrix matches the canonical frame" begin
    data_dir = joinpath(@__DIR__, "..", "data")
    spec = only(filter(s -> s.family == "Conotoxin", CANONICAL_FAMILIES))
    char_full, names_full, strong_ids = canonical_load_alignment(spec, data_dir)
    group_A, _, _, _ = canonical_split(spec, char_full, names_full, strong_ids)

    # This is exactly how run_omega_conotoxin_experiment.jl now builds its
    # designated ("strong binder") memory input.
    generator_designated = char_full[group_A, :]
    @test size(generator_designated) == (23, 26)

    # Guard against the historical bug: independently re-aligning and cleaning the
    # designated FASTA on its own does NOT reproduce the canonical frame. MAFFT
    # places gaps differently for indel-bearing sequences depending on which other
    # sequences are present during alignment, so slicing 23 rows out of the
    # 74-sequence alignment and re-aligning those 23 rows alone land on the same
    # column count (26) but not the same column identity for every row.
    conotoxin_dir = joinpath(data_dir, "omega_conotoxin")
    raw_strong_only = parse_fasta(joinpath(conotoxin_dir, "strong_cav22_binders_aligned.fasta"))
    char_strong_only, names_strong_only =
        clean_alignment(raw_strong_only; max_gap_frac_col=0.5, max_gap_frac_seq=0.4)
    @test size(char_strong_only) == (23, 26)

    by_accession(names, mat) = Dict(split(n, "|")[1] => String(mat[i, :])
                                     for (i, n) in enumerate(names))
    canonical_rows = by_accession(names_full[group_A], generator_designated)
    independent_rows = by_accession(names_strong_only, char_strong_only)
    @test Set(keys(canonical_rows)) == Set(keys(independent_rows))

    disagreeing = sort(collect(acc for acc in keys(canonical_rows)
                                if canonical_rows[acc] != independent_rows[acc]))
    @test disagreeing == sort([
        "A0A384E129", "P01522", "P28880", "P28881", "P58919", "Q9XZL4", "Q9XZL5",
    ])
end
