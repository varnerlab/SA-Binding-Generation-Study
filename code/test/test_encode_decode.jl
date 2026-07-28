# Tests for the complete sequence ↔ one-hot ↔ PCA decode path and its metrics.
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, LinearAlgebra, MultivariateStats

@testset "one-hot encoding and decoding" begin
    char_mat = ['A' 'C' 'D' 'E';
                'W' 'Y' 'K' 'R';
                'G' 'G' 'P' 'S']
    X = onehot_encode(char_mat)
    K, L = size(char_mat)

    @test size(X) == (N_AA * L, K)
    for k in 1:K
        @test decode_onehot(X[:, k], L) == String(char_mat[k, :])
        @test all(pos -> sum(X[(pos - 1) * N_AA + 1:pos * N_AA, k]) == 1.0, 1:L)
    end

    gapped = ['A' '-' 'D';
              'X' 'C' '.';
              '~' 'W' 'Y']
    Xg = onehot_encode(gapped)
    @test decode_onehot(Xg[:, 1], 3) == "A-D"
    @test decode_onehot(Xg[:, 2], 3) == "-C-"
    @test decode_onehot(Xg[:, 3], 3) == "-WY"
    @test decode_onehot(zeros(Float64, 3N_AA), 3) == "---"
end

@testset "PCA memory construction and decode map" begin
    aas = collect("ACDEFGHIKLMNPQRSTVWY")
    char_mat = [aas[mod1((k - 1) * 7 + (pos - 1) * 3, length(aas))]
                for k in 1:12, pos in 1:16]

    X̂, pca, L, d_full = build_memory_matrix(char_mat; pratio=0.95)
    @test size(X̂, 2) == size(char_mat, 1)
    @test all(isfinite, X̂)
    @test all(k -> isapprox(norm(X̂[:, k]), 1.0; atol=1e-10), axes(X̂, 2))
    @test L == size(char_mat, 2)
    @test d_full == N_AA * L

    for k in axes(X̂, 2)
        reconstructed = vec(MultivariateStats.reconstruct(pca, X̂[:, k]))
        @test decode_sample(X̂[:, k], pca, L) == decode_onehot(reconstructed, L)
    end

    mean_decode = decode_sample(zeros(Float64, size(X̂, 1)), pca, L)
    memory_decodes = [decode_sample(X̂[:, k], pca, L) for k in axes(X̂, 2)]
    @test all(s -> length(s) == L, memory_decodes)
    @test all(s -> valid_residue_fraction(s) == 1.0, memory_decodes)
    @test any(!=(mean_decode), memory_decodes)
end

@testset "gap-aware sequence metrics" begin
    @test all(is_alignment_gap, collect(".-~"))
    @test !is_alignment_gap('B')
    @test sequence_identity("ACDE", "ACDE") == 1.0
    @test sequence_identity("ACDE", "AWDE") == 0.75
    @test sequence_identity("A.C~D-", "ATCXDQ") == 1.0
    @test sequence_identity(".-~", "ACD") == 0.0
    @test sequence_identity("A.C~D-", "ATCXDQ") ==
          sequence_identity("ATCXDQ", "A.C~D-")
    @test sequence_identity("ABCD", "ABCD") == 1.0
    @test sequence_identity("ABCD", "AXCD") == 0.75

    stored = ["AWDE", "ACDF", "YYYY"]
    @test nearest_sequence_identity("ACDE", stored) == 0.75
    @test valid_residue_fraction("ABCD") == 0.75
    @test valid_residue_fraction("A.-~CX") ≈ 2 / 3
    @test valid_residue_fraction(".-~") == 0.0
end
