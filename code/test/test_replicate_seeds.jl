include(joinpath(@__DIR__, "..", "Include.jl"))
using Test

# Replicated experiments seed chain `c` with `MersenneTwister(base + c)`, so replicate base
# seeds must be spaced by more than the chain count. When they were spaced by 1, a
# five-replicate canonical condition performed 100 chain runs but drew on only 24 distinct
# streams, adjacent replicates sharing 19 of 20 chains, and the whole six-family sweep used
# 59 streams for 800 runs. Replicate standard deviations measured almost nothing.
#
# These tests enumerate the seeds each paper-facing calculation actually allocates and
# assert that every chain run gets its own stream.

"Chain seeds consumed by a set of replicate base seeds, as a multiset and a set."
function collect_streams(bases, n_chains)
    all_seeds = Int[]
    for b in bases
        append!(all_seeds, chain_seed_range(b, n_chains))
    end
    return all_seeds, Set(all_seeds)
end

@testset "replicate seed allocation" begin

    @testset "block index is a bijection onto 0:prod(extents)-1" begin
        extents = (6, 8, 5)
        seen = [condition_block_index((f, r, p), extents)
                for f in 1:6, r in 1:8, p in 1:5]
        @test sort(vec(seen)) == collect(0:(prod(extents) - 1))
        @test condition_block_index((1, 1, 1), extents) == 0
        @test condition_block_index((6, 8, 5), extents) == prod(extents) - 1
        # the replicate index varies fastest, so replicates of one condition are adjacent
        @test condition_block_index((1, 1, 2), extents) -
              condition_block_index((1, 1, 1), extents) == 1
        @test_throws ArgumentError condition_block_index((7, 1, 1), extents)
        @test_throws DimensionMismatch condition_block_index((1, 1), extents)
    end

    @testset "blocks are wider than any chain count used here" begin
        @test SEED_BLOCK > 30      # largest n_chains in the repository
        @test replicate_base_seed(0, 1) - replicate_base_seed(0, 0) == SEED_BLOCK
        @test_throws ArgumentError replicate_base_seed(0, -1)
    end

    @testset "canonical multiplicity sweep: every chain run has its own stream" begin
        n_fam, n_rho, n_reps, n_chains = 6, 8, 5, 20
        bases = [replicate_base_seed(20_000_000,
                    condition_block_index((f, r, p), (n_fam, n_rho, n_reps)))
                 for f in 1:n_fam, r in 1:n_rho, p in 1:n_reps]
        seeds, unique_seeds = collect_streams(vec(bases), n_chains)
        @test length(seeds) == n_fam * n_rho * n_reps * n_chains   # 4800 runs
        @test length(unique_seeds) == length(seeds)                # all distinct
    end

    @testset "canonical hard curation: every chain run has its own stream" begin
        bases = [replicate_base_seed(10_000_000, condition_block_index((f, p), (6, 5)))
                 for f in 1:6, p in 1:5]
        seeds, unique_seeds = collect_streams(vec(bases), 20)
        @test length(unique_seeds) == length(seeds) == 600
    end

    @testset "Kunitz replicates: every chain run has its own stream" begin
        bases = [replicate_base_seed(30_000_000, condition_block_index((c, p), (3, 5)))
                 for c in 1:3, p in 1:5]
        seeds, unique_seeds = collect_streams(vec(bases), 30)
        @test length(unique_seeds) == length(seeds) == 450
    end

    @testset "binder scaling: every chain run has its own stream" begin
        # n_chains = max(10, n_use) varies with the subset size
        sizes = [3, 5, 8, 10, 15, 20, 25, 32]
        seeds = Int[]
        for (i, n_bind) in enumerate(sizes), rep in 1:3
            base = replicate_base_seed(40_000_000,
                       condition_block_index((i, rep), (length(sizes), 3)))
            append!(seeds, chain_seed_range(base, max(10, min(n_bind, 32))))
        end
        @test length(Set(seeds)) == length(seeds)
    end

    @testset "HMM comparison SA replicates: every chain run has its own stream" begin
        bases = [
            replicate_base_seed(50_000_000, condition_block_index((c, p), (2, 5)))
            for c in 1:2, p in 1:5
        ]
        seeds, unique_seeds = collect_streams(vec(bases), 30)
        @test length(unique_seeds) == length(seeds) == 300
    end

    @testset "the six calculations cannot collide with each other" begin
        blocks = Dict(
            "hard_curation" => (10_000_000, 6 * 5),
            "sweep"         => (20_000_000, 6 * 8 * 5),
            "kunitz"        => (30_000_000, 3 * 5),
            "scaling"       => (40_000_000, 8 * 3),
            "hmm_sa"        => (50_000_000, 2 * 5),
            "hmm_emit"      => (70_000_000, 4),
        )
        spans = Dict(k => (o, o + (n - 1) * SEED_BLOCK + SEED_BLOCK)
                     for (k, (o, n)) in blocks)
        for (a, (lo_a, hi_a)) in spans, (b, (lo_b, hi_b)) in spans
            a < b || continue
            @test hi_a <= lo_b || hi_b <= lo_a   # disjoint seed spaces
        end
    end

    @testset "paper-facing diversity metrics use local RNGs" begin
        root = normpath(joinpath(@__DIR__, ".."))
        kunitz = read(
            joinpath(root, "experiments", "run_kunitz_binding_experiment_with_replicates.jl"),
            String,
        )
        scaling = read(
            joinpath(root, "experiments", "run_augmented_memory_deepdive.jl"),
            String,
        )
        hmm = read(joinpath(root, "experiments", "run_hmm_baseline.jl"), String)

        @test occursin("rand(rng, 1:n)", kunitz)
        @test occursin("rng=MersenneTwister(seed)", kunitz)
        @test !occursin("i, j = rand(1:n), rand(1:n)", kunitz)

        @test occursin("rand(rng, 1:n)", scaling)
        @test occursin("rng=MersenneTwister(replicate_seed)", scaling)
        @test !occursin("i, j = rand(1:n), rand(1:n)", scaling)
        @test occursin("\"--scaling-only\" in ARGS", scaling)

        @test occursin("HMM_SA_SEED_ORIGIN", hmm)
        @test occursin("HMM_EMIT_SEED_ORIGIN", hmm)
        @test occursin("sequence_pair_diversity", hmm)
        @test occursin("rng=MersenneTwister(seed)", hmm)
        @test occursin("Computing HMM baseline metrics across", hmm)
        @test !occursin("sample_diversity(pca_vecs)", hmm)
        @test !occursin("seed = 10000 + rep", hmm)
        @test !occursin("seed = 20000 + rep", hmm)
        @test !occursin("Random.seed!", hmm)
    end

    @testset "replicate rerun preserves downstream-validation FASTAs by default" begin
        script = read(
            joinpath(
                @__DIR__,
                "..",
                "experiments",
                "run_kunitz_binding_experiment_with_replicates.jl",
            ),
            String,
        )
        @test occursin("\"--refresh-example-fastas\" in ARGS", script)
        @test occursin("Preserved example FASTAs", script)
    end
end
