include(joinpath(@__DIR__, "..", "Include.jl"))
using Test
using Printf

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

pm2(row, mean_col, std_col) =
    @sprintf("\$%.2f \\pm %.2f\$", row[mean_col], row[std_col])
kl_pm1(row) =
    @sprintf("\$%.1f \\pm %.1f\$", 1000 * row.kl_mean, 1000 * row.kl_std)
div2(row) = @sprintf("%.2f", row.diversity_mean)

function table_row(text, label)
    rows = filter(
        line -> occursin(label, line) && count(==('&'), line) == 6 &&
                occursin("\\\\", line),
        split(text, '\n'),
    )
    @test length(rows) == 1
    return only(rows)
end

@testset "Kunitz sequence-metric table matches its CSV sources" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    baseline = CSV.read(
        joinpath(repo, "code", "data", "kunitz", "baseline_comparison.csv"),
        DataFrame,
    )
    binding = CSV.read(
        joinpath(repo, "code", "data", "kunitz", "binding_experiment_aggregated.csv"),
        DataFrame,
    )

    source_rows = [
        ("SA (full family)", only(filter(r -> r.method == "SA (full family)", eachrow(baseline)))),
        ("SA (K/R-positive)", only(filter(r -> r.method == "SA (strong binders)", eachrow(baseline)))),
        ("HMM emit (HMMER3)", only(filter(r -> r.method == "HMM emit", eachrow(baseline)))),
        ("Bootstrap (resample)", only(filter(r -> r.method == "Bootstrap", eachrow(baseline)))),
        ("SA (K/R-negative)", only(filter(r -> r.condition == "Weak binders", eachrow(binding)))),
    ]

    for (tree, wrapper) in (
        ("paper-jcim", "Paper_JCIM.tex"),
        ("paper-arxiv", "Paper_v1.tex"),
    )
        text = read(joinpath(repo, tree, wrapper), String)
        for (label, source) in source_rows
            row = table_row(text, label)
            @test occursin(pm2(source, :p1_kr_mean, :p1_kr_std), row)
            @test occursin(kl_pm1(source), row)
            @test occursin("& $(div2(source)) \\\\", row)
        end
    end
end
