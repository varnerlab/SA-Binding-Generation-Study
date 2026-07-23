include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random, LinearAlgebra

unitcols(d, K, seed) = (X = randn(MersenneTwister(seed), d, K);
                        for k in 1:K; X[:, k] ./= norm(X[:, k]); end; X)

@testset "weighted_hopfield_gradient matches central finite differences" begin
    X = unitcols(6, 7, 42)
    r = abs.(randn(MersenneTwister(43), 7)) .+ 0.3
    β = 3.0
    ξ = 0.5 .* randn(MersenneTwister(44), 6)
    g = weighted_hopfield_gradient(ξ, X, β, r)
    h = 1e-6
    gfd = similar(g)
    for i in eachindex(ξ)
        ξp = copy(ξ); ξp[i] += h
        ξm = copy(ξ); ξm[i] -= h
        gfd[i] = (weighted_hopfield_energy(ξp, X, β, r) -
                  weighted_hopfield_energy(ξm, X, β, r)) / (2h)
    end
    @test norm(g - gfd) < 1e-6
end

@testset "weighted_score = -β ∇E_r, and r≡1 reduces to unweighted gradient" begin
    X = unitcols(5, 4, 1)
    β = 2.0
    ξ = randn(MersenneTwister(2), 5)
    r = abs.(randn(MersenneTwister(3), 4)) .+ 0.5
    @test weighted_score(ξ, X, β, r) ≈ -β .* weighted_hopfield_gradient(ξ, X, β, r)
    a = NNlib.softmax(β .* (X' * ξ))
    @test weighted_hopfield_gradient(ξ, X, β, ones(4)) ≈ ξ .- X * a
end
