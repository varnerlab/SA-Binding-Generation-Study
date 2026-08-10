include(joinpath(@__DIR__, "..", "Include.jl"))
using Test
using Printf

const SHARED_SECTIONS = [
    "abstract.tex", "appendix.tex", "discussion.tex", "introduction.tex",
    "methods.tex", "results.tex", "si_gmm_derivation.tex",
    "significance_statement.tex", "theory.tex",
]

# Two sanctioned layout-only differences between the trees. The arXiv theory.tex
# wraps the exact-mixture proposition in a samepage box. The JCIM results.tex
# interleaves \Display... float-placement macros between paragraphs, because JCIM
# sets its floats inline while the arXiv tree places them after the bibliography.
# Neither carries manuscript prose, so both are stripped before comparison.
const LAYOUT_ONLY_LINES = Set(["\\begin{samepage}", "\\end{samepage}"])

is_layout_only(line) =
    strip(line) in LAYOUT_ONLY_LINES || startswith(strip(line), "\\Display")

# Dropping a layout-only line strands the blank line that separated it from the
# surrounding prose, so collapse blank runs and trailing blanks after filtering.
function manuscript_body(path)
    body = String[]
    for line in readlines(path)
        is_layout_only(line) && continue
        isempty(strip(line)) &&
            (isempty(body) || isempty(strip(last(body)))) && continue
        push!(body, line)
    end
    while !isempty(body) && isempty(strip(last(body)))
        pop!(body)
    end
    return body
end

# Pull each labeled table or figure from a wrapper file and remove only its float-placement
# option. The arXiv wrapper uses [p], while the JCIM display macros use [htbp]; everything
# inside the environments should otherwise stay identical.
function labeled_displays(path)
    text = read(path, String)
    displays = Dict{String,String}()
    for label_match in eachmatch(r"\\label\{((?:tab|fig):[^}]+)\}", text)
        label = label_match.captures[1]
        label_pos = label_match.offset
        table_start = findprev("\\begin{table}", text, label_pos)
        figure_start = findprev("\\begin{figure}", text, label_pos)
        candidates = filter(!isnothing, (table_start, figure_start))
        @test !isempty(candidates)
        start_range = candidates[argmax(first.(candidates))]
        kind = start_range == table_start ? "table" : "figure"
        stop_range = findnext("\\end{$kind}", text, label_pos)
        @test stop_range !== nothing
        block = text[first(start_range):last(stop_range)]
        block = replace(
            block,
            Regex("\\\\begin\\{$kind\\}\\[[^]]+\\]") => "\\begin{$kind}",
        )
        @test !haskey(displays, label)
        displays[label] = block
    end
    return displays
end

@testset "JCIM submission sections and benchmark positioning" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    wrapper = read(joinpath(repo, "paper-jcim", "Paper_JCIM.tex"), String)
    results = read(joinpath(repo, "paper-jcim", "sections", "results.tex"), String)
    appendix = read(joinpath(repo, "paper-jcim", "sections", "appendix.tex"), String)

    for required in (
        "keywords=true", "\\keywords{", "\\begin{tocentry}",
        "\\section*{Data and Software Availability}",
        "\\section*{Author Contributions}", "\\section*{Notes}",
    )
        @test occursin(required, wrapper)
    end
    @test isfile(joinpath(repo, "paper-jcim", "sections", "figs", "toc_graphic.png"))
    @test occursin("tab:profile-hmm-benchmark", results)
    @test !occursin("tab:docking-validation", results)
    @test occursin("tab:profile-hmm-rho-si", appendix)
    @test occursin("tab:docking-validation", appendix)
end

# Stripping the \Display macros above hides them from the cross-tree diff, so
# check separately that every one invoked in the JCIM text is defined, and that
# display_items.tex carries no orphan definitions.
@testset "JCIM display-item macros are defined and all used" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    sections = joinpath(repo, "paper-jcim", "sections")
    invoked = Set{String}()
    for name in SHARED_SECTIONS
        text = read(joinpath(sections, name), String)
        for m in eachmatch(r"^\\(Display\w+)"m, text)
            push!(invoked, m.captures[1])
        end
    end
    defined = Set(
        m.captures[1] for m in
        eachmatch(r"\\newcommand\{\\(Display\w+)\}",
                  read(joinpath(sections, "display_items.tex"), String))
    )
    @test !isempty(invoked)
    @test invoked == defined
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

function table_row(text, label; ampersands=6)
    rows = filter(
        line -> occursin(label, line) && count(==('&'), line) == ampersands &&
                occursin("\\\\", line),
        split(text, '\n'),
    )
    @test length(rows) == 1
    return only(rows)
end

function labeled_table(text, table_label)
    label_pos = findfirst(table_label, text)
    @test label_pos !== nothing
    table_start = findprev("\\begin{table}", text, first(label_pos))
    table_stop = findnext("\\end{table}", text, last(label_pos))
    @test table_start !== nothing
    @test table_stop !== nothing
    return text[first(table_start):last(table_stop)]
end

@testset "Kunitz mask-recovery table matches its CSV sources" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    experiment = CSV.read(
        joinpath(repo, "code", "data", "kunitz", "mask_experiment.csv"),
        DataFrame,
    )
    beta_sweep = CSV.read(
        joinpath(repo, "code", "data", "kunitz", "mask_betasweep.csv"),
        DataFrame,
    )
    text = read(joinpath(repo, "paper-arxiv", "Paper_v1.tex"), String)
    table = labeled_table(text, "\\label{tab:mask-recovery}")

    checks = [
        ("Unconditional", only(filter(:condition => ==("unconditional"), experiment))),
        ("Multiplicity", only(filter(r -> r.condition == "multiplicity" &&
                                          r.f_target == maximum(experiment.f_target[experiment.condition .== "multiplicity"]),
                                      experiment))),
        ("at \$\\beta^{*}\$", only(filter(:condition => ==("mask"), experiment))),
        ("Hard curation", only(filter(:condition => ==("curation"), experiment))),
    ]
    for (label, source) in checks
        row = table_row(table, label; ampersands=3)
        @test occursin(@sprintf("%.1f", source.beta), row)
        @test occursin(@sprintf("\$%.2f \\pm %.2f\$", source.p1_kr, source.p1_kr_se), row)
        @test occursin(@sprintf("%.2f", source.novelty), row)
    end

    sharp = only(filter(Symbol("β") => ==(512.0), beta_sweep))
    row = table_row(table, "\\beta = 512"; ampersands=3)
    @test occursin(@sprintf("\$%.2f \\pm %.2f\$", sharp.p1_kr, sharp.p1_kr_se), row)
    @test occursin(@sprintf("%.2f", sharp.novelty), row)
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

    # The arXiv tree keeps its tables in the wrapper document; the JCIM tree
    # defines them in display_items.tex and invokes them from the section text.
    for (tree, wrappers) in (
        ("paper-jcim", ["Paper_JCIM.tex", joinpath("sections", "display_items.tex")]),
        ("paper-arxiv", ["Paper_v1.tex"]),
    )
        text = join((read(joinpath(repo, tree, w), String) for w in wrappers), '\n')
        table = labeled_table(text, "\\label{tab:structure-validation}")
        for (label, source) in source_rows
            row = table_row(table, label)
            @test occursin(pm2(source, :p1_kr_mean, :p1_kr_std), row)
            @test occursin(kl_pm1(source), row)
            @test occursin("& $(div2(source)) \\\\", row)
        end
    end
end

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

@testset "JCIM and arXiv references, display items, and shared figures stay in sync" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    jcim = joinpath(repo, "paper-jcim")
    arxiv = joinpath(repo, "paper-arxiv")

    @test read(joinpath(jcim, "References_v1.bib")) ==
          read(joinpath(arxiv, "References_v1.bib"))

    arxiv_displays = labeled_displays(joinpath(arxiv, "Paper_v1.tex"))
    jcim_displays = labeled_displays(joinpath(jcim, "sections", "display_items.tex"))
    merge!(jcim_displays, labeled_displays(joinpath(jcim, "Paper_JCIM_SI.tex")))
    @test keys(jcim_displays) == keys(arxiv_displays)
    for label in keys(arxiv_displays)
        @test jcim_displays[label] == arxiv_displays[label]
    end

    arxiv_figs = joinpath(arxiv, "sections", "figs")
    jcim_figs = joinpath(jcim, "sections", "figs")
    arxiv_names = Set(readdir(arxiv_figs))
    jcim_names = Set(readdir(jcim_figs))
    @test setdiff(arxiv_names, jcim_names) ==
          Set(["loop_heatmap_with_residuals.png", "sequence_analysis_conotoxin.png"])
    @test setdiff(jcim_names, arxiv_names) == Set(["toc_graphic.png"])
    for name in intersect(arxiv_names, jcim_names)
        @test read(joinpath(arxiv_figs, name)) == read(joinpath(jcim_figs, name))
    end
end
