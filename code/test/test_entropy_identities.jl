include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random, LinearAlgebra

unitcols(d, K, seed) = (X = randn(MersenneTwister(seed), d, K);
                        for k in 1:K; X[:, k] ./= norm(X[:, k]); end; X)

@testset "β→0 attention entropy equals Shannon entropy of normalized weights" begin
    X = unitcols(6, 10, 5)
    r = abs.(randn(MersenneTwister(6), 10)) .+ 0.2
    # at β≈0 the weighted attention weights are w = r/Σr, independent of ξ
    Hβ0 = weighted_attention_entropy(zeros(6), X, 1e-9, r)
    @test isapprox(Hβ0, shannon_entropy(r); atol=1e-6)
end

@testset "Shannon (Renyi-1) and log K_eff (Renyi-2) coincide only for equal weights" begin
    r_eq = ones(8)
    @test isapprox(shannon_entropy(r_eq), log_effective_num_patterns(r_eq); atol=1e-10)
    @test isapprox(shannon_entropy(r_eq), log(8); atol=1e-10)
    # non-uniform: Renyi-1 > Renyi-2 strictly, by a material margin
    r = [500.0; ones(31)]
    Hs = shannon_entropy(r)
    Hr = log_effective_num_patterns(r)
    @test Hs > Hr
    @test Hs - Hr > 0.1
end

@testset "shannon_entropy validation" begin
    @test_throws ArgumentError shannon_entropy(zeros(4))
end
