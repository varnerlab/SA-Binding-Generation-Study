include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random

@testset "samplers use caller-local RNGs" begin
    setup_rng = MersenneTwister(700)
    X = randn(setup_rng, 5, 7)
    X ./= sqrt.(sum(abs2, X; dims=1))
    ξ0 = randn(setup_rng, 5)
    weights = collect(1.0:7.0)

    a = sample(X, ξ0, 40; β=2.0, α=0.05, seed=11).Ξ
    b = sample(X, ξ0, 40; β=2.0, α=0.05, seed=11).Ξ
    c = sample(X, ξ0, 40; β=2.0, α=0.05, seed=12).Ξ
    @test a == b
    @test a != c

    weighted_a = weighted_sample(X, ξ0, 40, weights;
                                 β=2.0, α=0.05, rng=MersenneTwister(21)).Ξ
    weighted_b = weighted_sample(X, ξ0, 40, weights;
                                 β=2.0, α=0.05, rng=MersenneTwister(21)).Ξ
    @test weighted_a == weighted_b

    @test_throws ArgumentError sample(
        X, ξ0, 2; seed=1, rng=MersenneTwister(1))
    @test_throws ArgumentError weighted_sample(
        X, ξ0, 2, weights; seed=1, rng=MersenneTwister(1))

    Random.seed!(909)
    expected_after = rand()
    Random.seed!(909)
    sample(X, ξ0, 5; seed=33)
    @test rand() == expected_after
end

@testset "generation is deterministic and leaves global RNG untouched" begin
    aas = collect("ACDEFGHIKLMNPQRSTVWY")
    char_mat = [aas[mod1(5k + 3pos, length(aas))]
                for k in 1:12, pos in 1:16]
    X, pca, L, _ = build_memory_matrix(char_mat; pratio=0.95)

    args = (; β=3.0, n_chains=2, T=30, burnin=10, thin=10, seed=81)
    seqs_a, latent_a = generate_sequences(X, pca, L; args...)
    seqs_b, latent_b = generate_sequences(X, pca, L; args...)
    @test seqs_a == seqs_b
    @test latent_a == latent_b

    Random.seed!(404)
    expected_after = rand()
    Random.seed!(404)
    generate_sequences(X, pca, L; args...)
    @test rand() == expected_after
end

@testset "synthetic data generation is locally reproducible" begin
    a = datagenerate(4, 3, 2; seed=51)
    b = datagenerate(4, 3, 2; seed=51)
    @test a["datasets"] == b["datasets"]

    Random.seed!(505)
    expected_after = rand()
    Random.seed!(505)
    datagenerate(4, 3, 1; seed=52)
    @test rand() == expected_after
end
