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

@testset "masked_sample equals unmasked SA on the reduced matrix" begin
    Random.seed!(7)
    d, K, T = 5, 8, 40
    X = randn(d, K); ξ0 = randn(d)
    keep = Bool[1,1,1,0,0,1,0,0]
    keepcols = findall(keep)
    # The paper identity: masking full X == running SA on X_S.
    Ξm = masked_sample(X, ξ0, T, keep; β=2.0, α=0.05, seed=99).Ξ
    Ξr = sample(X[:, keepcols], ξ0, T; β=2.0, α=0.05, seed=99).Ξ
    @test Ξm ≈ Ξr
    # -∞ safety: no NaN/Inf in the trajectory
    @test all(isfinite, Ξm)
    # index method matches boolean method
    @test masked_sample(X, ξ0, T, keepcols; β=2.0, seed=2).Ξ ≈
          masked_sample(X, ξ0, T, keep; β=2.0, seed=2).Ξ
    # empty keep-set throws (no finite logit)
    @test_throws ArgumentError masked_sample(X, ξ0, T, falses(K); β=2.0, seed=1)
    # mask_vector builds 0/1 weights
    @test mask_vector(5, [1, 3]) == [1.0, 0.0, 1.0, 0.0, 0.0]
end
