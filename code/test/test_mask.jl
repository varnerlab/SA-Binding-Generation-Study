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
