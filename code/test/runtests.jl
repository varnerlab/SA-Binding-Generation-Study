# Full unit suite for the SA Binding Generation Study.
# Each included file self-includes Include.jl (idempotent) and defines its own testsets.
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test

@testset "SA Binding Generation Study" begin
    include("test_score_gradient.jl")
    include("test_gmm_identity.jl")
    include("test_entropy_identities.jl")
    include("test_transition_statistic.jl")
    include("test_operating_point_rule.jl")
    include("test_canonical_provenance.jl")
    include("test_mask.jl")
    include("test_encode_decode.jl")
    include("test_decoder_characterization.jl")
    include("test_generated_paper_tables.jl")
    include("test_manuscript_consistency.jl")
    include("test_rng_reproducibility.jl")
    include("test_conotoxin_labels.jl")
    include("test_conotoxin_sar_frame.jl")
end
