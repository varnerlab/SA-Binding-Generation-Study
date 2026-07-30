using Test

@testset "Canonical CSV manuscript artifacts" begin
    code_dir = normpath(joinpath(@__DIR__, ".."))
    repo_dir = normpath(joinpath(@__DIR__, "..", ".."))
    generator = joinpath(code_dir, "experiments", "generate_paper_tables.jl")
    # Both trees are checked. The JCIM tree is the submission target, so leaving it
    # unverified would let the submitted manuscript drift from the canonical CSVs.
    committed_trees = [
        joinpath(repo_dir, "paper-arxiv", "sections", "generated"),
        joinpath(repo_dir, "paper-jcim", "sections", "generated"),
    ]

    mktempdir() do generated
        command = `$(Base.julia_cmd()) --project=$code_dir $generator --output=$generated`
        run(command)
        expected_files = sort([
            "numbers.tex",
            "tab_cross_family.tex",
            "tab_per_family_rho.tex",
            "tab_kunitz_rho.tex",
            "tab_sh3_rho.tex",
            "tab_ww_rho.tex",
            "tab_homeobox_rho.tex",
            "tab_forkhead_rho.tex",
            "tab_omega_conotoxin_rho.tex",
            "tab_sar_agreement.tex",
            "tab_beta_sweep.tex",
        ])
        @test sort(readdir(generated)) == expected_files
        for committed in committed_trees
            @test sort(readdir(committed)) == expected_files
            for filename in expected_files
                @test read(joinpath(generated, filename)) == read(joinpath(committed, filename))
            end
        end
    end
end
