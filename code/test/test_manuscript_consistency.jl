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

# Both claims are false. The beta -> 0 attention entropy is the Shannon entropy of the
# weights, not the Renyi-2 entropy log K_eff; see the shannon_entropy docstring in
# code/src/Binding.jl. And beta* rises while log K_eff falls, so the proportionality in
# the appendix has the wrong direction and contradicts the mechanism argued earlier in
# the same file.
const RETRACTED_CLAIMS = [
    "H_{\\vr}(0) = \\log K_{\\mathrm{eff}}",
    "H_{\\mathbf{r}}(0) = \\log K_{\\mathrm{eff}}",
    "\\beta^{*} \\propto \\log K_{\\mathrm{eff}}",
]

squeeze_whitespace(text) = replace(text, r"\s+" => " ")

@testset "retracted entropy claims stay out of the manuscript" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    for tree in ("paper-jcim", "paper-arxiv"), name in SHARED_SECTIONS
        text = squeeze_whitespace(read(joinpath(repo, tree, "sections", name), String))
        for claim in RETRACTED_CLAIMS
            @test !occursin(claim, text)
        end
    end
end
