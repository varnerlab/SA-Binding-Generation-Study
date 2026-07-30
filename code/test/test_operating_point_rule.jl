using Test

# Paper-facing producer scripts must select the operating point with the
# order-independent all-memory rule. The first-20-column detectors remain in the
# source tree for the legacy scripts, so this is a per-script assertion.
const PAPER_FACING_SCRIPTS = [
    "run_canonical_family_sweeps.jl",
    "run_kunitz_mask_experiment.jl",
    "run_gmm_baseline.jl",
    "dump_entropy_curves.jl",
    "run_kunitz_binding_experiment_with_replicates.jl",
    "run_kunitz_binding_experiment.jl",
    "run_omega_conotoxin_experiment.jl",
    "run_augmented_memory_deepdive.jl",
    # Produces calibration_beta_sweep.csv, the source of the appendix beta-sweep table.
    "run_calibration_diagnostics.jl",
]

@testset "paper-facing scripts use the all-memory operating point" begin
    exp_dir = normpath(joinpath(@__DIR__, "..", "experiments"))
    for name in PAPER_FACING_SCRIPTS
        path = joinpath(exp_dir, name)
        @test isfile(path)
        source = read(path, String)
        # strip comment lines so a mention in a comment does not fail the test
        code = join(filter(l -> !startswith(strip(l), "#"), split(source, "\n")), "\n")
        @test !occursin("find_entropy_inflection(", code)
        @test !occursin("find_weighted_entropy_inflection(", code)
        # Either the wrapper or the function it wraps. dump_entropy_curves.jl needs the
        # full NamedTuple (βs, Hs) to write the curve, so it calls find_entropy_transition
        # with n_probes = size(X̂,2) directly and reads β_onset. That is the same
        # computation all_memory_onset performs, so the two cannot disagree.
        @test occursin("all_memory_onset(", code) ||
              occursin("find_entropy_transition(", code)
    end
end
