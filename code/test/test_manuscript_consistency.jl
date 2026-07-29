include(joinpath(@__DIR__, "..", "Include.jl"))
using Test

const SHARED_SECTIONS = [
    "abstract.tex", "appendix.tex", "discussion.tex", "introduction.tex",
    "methods.tex", "results.tex", "si_gmm_derivation.tex",
    "significance_statement.tex", "theory.tex",
]

# The arXiv theory.tex wraps the exact-mixture proposition in a samepage box for
# layout only. That wrapper is the single sanctioned difference between the trees.
const LAYOUT_ONLY_LINES = Set(["\\begin{samepage}", "\\end{samepage}"])

manuscript_body(path) =
    filter(line -> !(strip(line) in LAYOUT_ONLY_LINES), readlines(path))

@testset "JCIM and arXiv section sources stay in sync" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    jcim = joinpath(repo, "paper-jcim", "sections")
    arxiv = joinpath(repo, "paper-arxiv", "sections")
    for name in SHARED_SECTIONS
        jcim_path = joinpath(jcim, name)
        arxiv_path = joinpath(arxiv, name)
        @test isfile(jcim_path)
        @test isfile(arxiv_path)
        @test manuscript_body(jcim_path) == manuscript_body(arxiv_path)
    end
end
