using Test, CSV, DataFrames

# Every canonical family, Kunitz included, must be produced by the canonical driver.
#
# Earlier versions of run_canonical_family_sweeps.jl carried Kunitz forward as a legacy
# aggregate: the bare command swept only five families and passed Kunitz through
# unchanged, while the provenance record still credited the driver. The onset correction
# therefore missed Kunitz entirely. These assertions pin the invariant that made that
# possible, not just the numbers it produced.

const CANONICAL_SLUGS = ["kunitz", "sh3", "ww", "homeobox", "forkhead", "omega_conotoxin"]

@testset "canonical sweep provenance" begin
    data_dir = normpath(joinpath(@__DIR__, "..", "data"))

    @testset "every family has a driver execution record" begin
        for slug in CANONICAL_SLUGS
            path = joinpath(data_dir, slug, "canonical_sweep_execution.csv")
            @test isfile(path)
            record = CSV.read(path, DataFrame)
            driver = only(filter(:parameter => ==("driver"), record)).value
            @test driver == "run_canonical_family_sweeps.jl"
        end
    end

    @testset "provenance records Kunitz as driver output" begin
        provenance = CSV.read(joinpath(data_dir, "canonical_sweep_provenance.csv"), DataFrame)
        origin = only(filter(:parameter => ==("kunitz_origin"), provenance)).value
        @test occursin("canonical driver", origin)
        @test !occursin("legacy", origin)
    end

    @testset "the driver has no per-family special case" begin
        source = read(joinpath(@__DIR__, "..", "experiments",
                              "run_canonical_family_sweeps.jl"), String)
        @test !occursin("prepare_legacy_kunitz", source)
        # The sweep set must not be built by excluding a named family.
        @test !occursin("s.family != \"Kunitz\"", source)
    end
end
