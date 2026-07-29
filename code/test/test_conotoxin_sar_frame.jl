include(joinpath(@__DIR__, "..", "Include.jl"))
include(joinpath(@__DIR__, "..", "experiments", "canonical_family_registry.jl"))
using Test

# (MVIIA position, accepted residues). Basic positions accept the K/R class,
# matching the permissive counting rule used in the SAR table.
const SAR_POSITIONS = [
    (13, ('Y',)), (2, ('K', 'R')), (10, ('K', 'R')), (11, ('L',)),
    (1, ('C',)), (8, ('C',)), (15, ('C',)), (16, ('C',)),
    (20, ('C',)), (25, ('C',)), (21, ('K', 'R')), (4, ('K', 'R')),
]

@testset "conotoxin SAR input is read in the canonical alignment frame" begin
    data_dir = joinpath(@__DIR__, "..", "data")
    spec = only(filter(s -> s.family == "Conotoxin", CANONICAL_FAMILIES))
    char_mat, names, auxiliary = canonical_load_alignment(spec, data_dir)
    designated, _, _, _ = canonical_split(spec, char_mat, names, auxiliary)

    @test size(char_mat) == (74, 26)
    @test length(designated) == 23

    expected = Dict(pos => count(i -> char_mat[i, pos] in residues, designated) /
                           length(designated)
                    for (pos, residues) in SAR_POSITIONS)

    # All six framework cysteines are invariant across the designated accessions
    # once they are read in the aligned frame. The pre-fix script reported 0.87,
    # 0.70 and 0.52 at positions 16, 20 and 25 because it indexed raw sequences.
    for pos in (1, 8, 15, 16, 20, 25)
        @test expected[pos] == 1.0
    end

    table = CSV.read(joinpath(data_dir, "omega_conotoxin", "sar_agreement.csv"),
                     DataFrame)
    @test nrow(table) == 12
    for row in eachrow(table)
        @test isapprox(row.Input_strong, expected[row.Position]; atol = 5e-4)
    end
end
