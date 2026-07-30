include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Random, LinearAlgebra

unitcols(d, K, seed) = (X = randn(MersenneTwister(seed), d, K);
                        for k in 1:K; X[:, k] ./= norm(X[:, k]); end; X)

@testset "transition_statistics locates the steepest drop of an analytic sigmoid" begin
    logβ = collect(range(-3.0, 5.0, length=400))
    c, k, A = 1.7, 6.0, 3.0
    Hs = A ./ (1 .+ exp.(k .* (logβ .- c)))          # high at low β, drops through c
    st = transition_statistics(logβ, Hs)
    @test isapprox(logβ[st.i_steepest], c; atol=0.05) # steepest slope at the midpoint
    @test logβ[st.i_onset] < logβ[st.i_steepest]      # onset precedes the midpoint
end

@testset "transition_statistics validation" begin
    @test_throws DimensionMismatch transition_statistics([1.0, 2.0], [1.0])
    @test_throws ArgumentError transition_statistics([1.0, 2.0], [1.0, 2.0])
end

@testset "find_entropy_transition with all probes is permutation invariant" begin
    X = unitcols(10, 16, 77)
    a = find_entropy_transition(X; n_betas=40, n_probes=16, seed=1)
    b = find_entropy_transition(X[:, 16:-1:1]; n_betas=40, n_probes=16, seed=1)
    @test isapprox(a.β_steepest, b.β_steepest; rtol=1e-8)
    @test isapprox(a.β_onset, b.β_onset; rtol=1e-8)
    @test isapprox(a.Hs, b.Hs; rtol=1e-8)             # curve independent of column order
    @test a.β_steepest > 0
    @test length(a.Hs) == 40
end

@testset "all_memory_onset is the paper-facing operating-point rule" begin
    X = unitcols(10, 24, 91)
    r = abs.(randn(MersenneTwister(3), 24)) .+ 0.2

    # It is exactly find_entropy_transition over every stored memory.
    @test all_memory_onset(X, r; n_betas=40) ==
          find_entropy_transition(X, r; n_betas=40, n_probes=24).β_onset

    # Unweighted default: r defaults to uniform weights.
    @test all_memory_onset(X; n_betas=40) ==
          find_entropy_transition(X; n_betas=40, n_probes=24).β_onset

    # Order independence, which is the entire point. The first-20-column rule
    # fails this; that is the defect being corrected.
    perm = randperm(MersenneTwister(12), 24)
    @test all_memory_onset(X, r; n_betas=40) ==
          all_memory_onset(X[:, perm], r[perm]; n_betas=40)
end
