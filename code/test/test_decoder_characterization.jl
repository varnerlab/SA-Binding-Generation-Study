# Offline characterization of the model-defined PCA-to-sequence map on the
# exact, tracked alignments used by the paper. These values document lossiness;
# they are not assertions that normalized memories should invert the encoder.
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test, Statistics, MultivariateStats

const DECODER_FAMILY_FIXTURES = [
    (family="Kunitz", format=:stockholm, file=joinpath("kunitz", "PF00014_seed.sto"),
     max_gap_frac_seq=0.3, unit_identity=0.836366, score_identity=0.999809),
    (family="SH3", format=:stockholm, file=joinpath("sh3", "PF00018_seed.sto"),
     max_gap_frac_seq=0.3, unit_identity=0.789073, score_identity=1.0),
    (family="WW", format=:stockholm, file=joinpath("ww", "PF00397_seed.sto"),
     max_gap_frac_seq=0.3, unit_identity=0.833680, score_identity=1.0),
    (family="Homeobox", format=:stockholm, file=joinpath("homeobox", "PF00046_seed.sto"),
     max_gap_frac_seq=0.3, unit_identity=0.750753, score_identity=1.0),
    (family="Forkhead", format=:stockholm, file=joinpath("forkhead", "PF00250_seed.sto"),
     max_gap_frac_seq=0.3, unit_identity=0.729774, score_identity=1.0),
    (family="Conotoxin", format=:fasta,
     file=joinpath("omega_conotoxin", "omega_conotoxin_full_family_aligned.fasta"),
     max_gap_frac_seq=0.4, unit_identity=0.943108, score_identity=0.995584),
]

@testset "stored-memory decoder characterization (tracked fixtures)" begin
    for fixture in DECODER_FAMILY_FIXTURES
        path = joinpath(@__DIR__, "..", "data", fixture.file)
        @test isfile(path)
        raw = fixture.format == :stockholm ? parse_stockholm(path) : parse_fasta(path)
        char_mat, _ = clean_alignment(raw;
            max_gap_frac_col=0.5,
            max_gap_frac_seq=fixture.max_gap_frac_seq)
        X_onehot = onehot_encode(char_mat)
        X̂, pca, L, _ = build_memory_matrix(char_mat; pratio=0.95)
        scores = MultivariateStats.transform(pca, X_onehot)
        originals = [String(char_mat[k, :]) for k in axes(char_mat, 1)]

        unit_decodes = [decode_sample(X̂[:, k], pca, L) for k in axes(X̂, 2)]
        score_decodes = [decode_sample(scores[:, k], pca, L) for k in axes(scores, 2)]
        unit_identity = mean(sequence_identity(unit_decodes[k], originals[k])
                             for k in eachindex(originals))
        score_identity = mean(sequence_identity(score_decodes[k], originals[k])
                              for k in eachindex(originals))

        @test unit_identity ≈ fixture.unit_identity atol=0.002
        @test score_identity ≈ fixture.score_identity atol=0.002
        @test score_identity >= unit_identity
        @test all(s -> length(s) == L && valid_residue_fraction(s) == 1.0,
                  unit_decodes)
    end
end
