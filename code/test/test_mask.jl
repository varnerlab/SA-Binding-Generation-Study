# Tests for the unified logit-bias sampler and its mask/weight wrappers.
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test

@testset "logit_bias_sample: b=0 reduces to unconditional sample" begin
    Random.seed!(11)
    d, K, T = 4, 6, 30
    X = randn(d, K); ξ0 = randn(d)
    Ξb = logit_bias_sample(X, ξ0, T, zeros(K); β=1.5, α=0.1, seed=5).Ξ
    Ξs = sample(X, ξ0, T; β=1.5, α=0.1, seed=5).Ξ
    @test Ξb ≈ Ξs
end

@testset "weighted_sample delegates to logit_bias_sample" begin
    Random.seed!(3)
    d, K, T = 4, 6, 30
    X = randn(d, K); ξ0 = randn(d)
    # uniform positive weights == unconditional
    @test weighted_sample(X, ξ0, T, ones(K); β=1.2, seed=8).Ξ ≈ sample(X, ξ0, T; β=1.2, seed=8).Ξ
    # weighted == logit-bias of log-weights
    w = abs.(randn(K)) .+ 0.1
    @test weighted_sample(X, ξ0, T, w; β=1.2, seed=8).Ξ ≈
          logit_bias_sample(X, ξ0, T, log.(w); β=1.2, seed=8).Ξ
    # a single zero weight is now allowed and produces a finite trajectory
    w0 = copy(w); w0[2] = 0.0
    @test all(isfinite, weighted_sample(X, ξ0, T, w0; β=1.2, seed=8).Ξ)
    # all-zero weights throw (no reachable memory)
    @test_throws ArgumentError weighted_sample(X, ξ0, T, zeros(K); β=1.2, seed=8)
end
