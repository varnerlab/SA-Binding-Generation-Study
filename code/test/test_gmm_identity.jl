include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random, LinearAlgebra, Statistics

unitcols(d, K, seed) = (X = randn(MersenneTwister(seed), d, K);
                        for k in 1:K; X[:, k] ./= norm(X[:, k]); end; X)

@testset "exact_gmm_sample: shape, components, determinism" begin
    X = unitcols(8, 5, 10)
    r = [1.0, 2.0, 3.0, 4.0, 5.0]
    out  = exact_gmm_sample(X, r, 4.0, 100; rng=MersenneTwister(1))
    out2 = exact_gmm_sample(X, r, 4.0, 100; rng=MersenneTwister(1))
    @test size(out.Ξ) == (8, 100)
    @test all(c -> 1 <= c <= 5, out.components)
    @test out.Ξ == out2.Ξ                      # determinism under same seed
    @test out.components == out2.components
end

@testset "exact_gmm_sample: empirical mean matches Σ w_k m_k" begin
    X = unitcols(10, 6, 20)
    r = abs.(randn(MersenneTwister(21), 6)) .+ 0.5
    w = r ./ sum(r)
    out = exact_gmm_sample(X, r, 6.0, 200_000; rng=MersenneTwister(7))
    @test norm(vec(mean(out.Ξ; dims=2)) - X * w) < 0.05
end

@testset "exact_gmm_sample: designated mass matches f_eff" begin
    X = unitcols(10, 8, 30)
    designated = [2, 4, 6]
    r = ones(8); r[designated] .= 20.0
    f_eff = sum(r[designated]) / sum(r)
    out = exact_gmm_sample(X, r, 5.0, 200_000; rng=MersenneTwister(3))
    @test abs(mean(c -> c in designated, out.components) - f_eff) < 0.01
end

@testset "exact_gmm_sample: input validation" begin
    X = unitcols(4, 3, 1)
    @test_throws DimensionMismatch exact_gmm_sample(X, ones(2), 1.0, 5)
    @test_throws ArgumentError exact_gmm_sample(X, zeros(3), 1.0, 5)
    @test_throws ArgumentError exact_gmm_sample(X, ones(3), -1.0, 5)
end
