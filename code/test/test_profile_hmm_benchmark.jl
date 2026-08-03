using Test, CSV, DataFrames, Statistics

@testset "Matched profile-HMM benchmark artifacts" begin
    code_dir = normpath(joinpath(@__DIR__, ".."))
    data_dir = joinpath(code_dir, "data", "profile_hmm_conditioning")

    aggregate = CSV.read(joinpath(data_dir, "aggregated.csv"), DataFrame)
    raw = CSV.read(joinpath(data_dir, "raw_replicates.csv"), DataFrame)
    execution = CSV.read(joinpath(data_dir, "execution.csv"), DataFrame)
    matched_raw = CSV.read(
        joinpath(data_dir, "kunitz_rho500_matched_esm2_raw.csv"), DataFrame)
    matched_summary = CSV.read(
        joinpath(data_dir, "kunitz_rho500_matched_esm2_summary.csv"), DataFrame)

    families = Set(["Kunitz", "SH3", "WW", "Homeobox", "Forkhead", "Conotoxin"])
    rho_grid = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]

    @test nrow(aggregate) == 54
    @test nrow(raw) == 270
    @test Set(aggregate.family) == families
    @test Set(raw.family) == families
    @test Set(raw.condition) == Set(["multiplicity", "designated_subset"])
    @test all(raw.n_emitted .== 620)
    @test all(raw.valid_fraction .== 1.0)

    for family in families
        rows = raw[(raw.family .== family) .& (raw.condition .== "multiplicity"), :]
        @test sort(unique(rows.rho)) == rho_grid
        for rho in rho_grid
            replicate_rows = rows[rows.rho .== rho, :]
            @test sort(replicate_rows.replicate) == collect(1:5)
        end
        subset = raw[(raw.family .== family) .& (raw.condition .== "designated_subset"), :]
        @test sort(subset.replicate) == collect(1:5)
    end

    execution_map = Dict(string(row.parameter) => string(row.value) for row in eachrow(execution))
    @test execution_map["rho_grid"] == "1;2;5;10;20;50;100;500"
    @test execution_map["n_replicates"] == "5"
    @test execution_map["sequences_per_replicate"] == "620"
    @test occursin("--wgiven", execution_map["hmmbuild_options"])
    @test occursin("--enone", execution_map["hmmbuild_options"])

    expected_sources = Set(["SA_multiplicity_rho500", "HMM_weighted_rho500"])
    @test nrow(matched_raw) == 100
    @test Set(matched_raw.source) == expected_sources
    @test Set(matched_summary.source) == expected_sources
    @test all(matched_summary.n .== 50)
    @test all(isfinite, matched_raw.pseudo_perplexity)
    for summary_row in eachrow(matched_summary)
        source_rows = matched_raw[matched_raw.source .== summary_row.source, :]
        @test isapprox(mean(source_rows.pseudo_perplexity), summary_row.ppl_mean; atol=1e-12)
        @test isapprox(std(source_rows.pseudo_perplexity), summary_row.ppl_std; atol=1e-12)
    end

    sa = only(filter(row -> row.source == "SA_multiplicity_rho500",
                     eachrow(matched_summary)))
    hmm = only(filter(row -> row.source == "HMM_weighted_rho500",
                      eachrow(matched_summary)))
    @test sa.ppl_mean < hmm.ppl_mean

    source = read(joinpath(code_dir, "experiments",
                           "run_profile_hmm_conditioning_benchmark.jl"), String)
    for option in ("--symfrac 0.0", "--wgiven", "--enone", "hmmemit -a")
        @test occursin(option, source)
    end
end
