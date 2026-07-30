#!/usr/bin/env julia

# Generate all manuscript-facing multiplicity tables and prose-number macros
# from the canonical replicated CSVs.

const SCRIPT_DIR = @__DIR__
const CODE_DIR = dirname(SCRIPT_DIR)
const REPO_DIR = dirname(CODE_DIR)
const DATA_DIR = joinpath(CODE_DIR, "data")
const DEFAULT_OUTPUT = joinpath(REPO_DIR, "paper-arxiv", "sections", "generated")
const FAMILY_SLUGS = [
    "Kunitz" => "kunitz",
    "SH3" => "sh3",
    "WW" => "ww",
    "Homeobox" => "homeobox",
    "Forkhead" => "forkhead",
    "Conotoxin" => "omega_conotoxin",
]
const RHO_GRID = Float64[1, 2, 5, 10, 20, 50, 100, 500]

# The appendix beta-sweep table reports a subset of the multipliers in
# calibration_beta_sweep.csv. Both lists are authoritative: the table must contain exactly
# these rows, and the CSV must contain exactly these multipliers.
const BETA_SWEEP_RHO = Float64[10, 50, 200]
const BETA_SWEEP_REPORTED = Float64[0.5, 1.0, 1.5, 2.0, 3.0]
const BETA_SWEEP_ALL_MULTIPLIERS = Float64[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

using CSV, DataFrames, Printf, Statistics

fmt3(x) = @sprintf("%.3f", x)
fmt2(x) = @sprintf("%.2f", x)
fmt1(x) = @sprintf("%.1f", x)

function latex_family(name)
    name == "Conotoxin" ? raw"$\omega$-Conotoxin" : name
end

# Typeset labels for the conotoxin SAR table, keyed by MVIIA position. The frequencies
# come from the CSV so they cannot drift from the analysis. The residue names and the
# abbreviated role and effect strings are typesetting choices and stay here. The key set
# is the authoritative list of rows the table must contain.
const SAR_LABELS = Dict(
    13 => ("Tyr", "Primary reported marker", "Ala: abolishes activity"),
    2  => ("Lys", "Loop 2 stabilization", "Ala: 40\$\\times\$ loss (GVIA)"),
    10 => ("Arg", "Loop 2 binding", "Critical for interaction"),
    11 => ("Leu", "Loop 2 binding", "Critical for interaction"),
    1  => ("Cys", "Disulfide framework", "Required for fold"),
    8  => ("Cys", "Disulfide framework", "Required for fold"),
    15 => ("Cys", "Disulfide framework", "Required for fold"),
    16 => ("Cys", "Disulfide framework", "Required for fold"),
    20 => ("Cys", "Disulfide framework", "Required for fold"),
    25 => ("Cys", "Disulfide framework", "Required for fold"),
    21 => ("Arg", "Electrostatic", "Ala: reduced potency"),
    4  => ("Lys", "P/Q selectivity", "Ala: important for P/Q"),
)

function validate_inputs()
    cross_path = joinpath(DATA_DIR, "multi_family_comparison_6fam_aggregated.csv")
    isfile(cross_path) || error("Missing canonical cross-family CSV: $cross_path")
    cross = CSV.read(cross_path, DataFrame)
    expected_cross = [:family, :source, :pfam_id, :K, :L, :d_pca, :K_A, :K_B,
        :marker, :natural_frac, :separation_index, :hard_curation_mean,
        :hard_curation_std, :cal_gap_mean, :cal_gap_std, :fit_included]
    propertynames(cross) == expected_cross ||
        error("Cross-family schema mismatch: $(propertynames(cross))")
    nrow(cross) == 6 || error("Canonical cross-family CSV must have six rows")
    Set(cross.family) == Set(first.(FAMILY_SLUGS)) ||
        error("Canonical cross-family CSV has unexpected families")

    aggregates = Dict{String,DataFrame}()
    for (family, slug) in FAMILY_SLUGS
        aggregate_path = joinpath(DATA_DIR, slug, "multiplicity_sweep_aggregated.csv")
        raw_path = joinpath(DATA_DIR, slug, "multiplicity_sweep_raw_replicates.csv")
        execution_path = joinpath(DATA_DIR, slug, "canonical_sweep_execution.csv")
        isfile(aggregate_path) || error("Missing canonical aggregate: $aggregate_path")
        isfile(raw_path) || error("Missing canonical raw replicates: $raw_path")
        isfile(execution_path) || error("Missing canonical execution record: $execution_path")
        execution = CSV.read(execution_path, DataFrame)
        execution_map = Dict(string(row.parameter) => string(row.value)
                             for row in eachrow(execution))
        get(execution_map, "driver", "") == "run_canonical_family_sweeps.jl" ||
            error("$family was not recorded as a canonical-driver run")
        get(execution_map, "rho_grid", "") == join(Int.(RHO_GRID), ";") ||
            error("$family execution record has a noncanonical rho grid")
        get(execution_map, "n_replicates", "") == "5" ||
            error("$family execution record has the wrong replicate count")
        aggregate = CSV.read(aggregate_path, DataFrame)
        expected = [:rho, :f_eff, :f_obs_mean, :f_obs_std, :attn_A_mean,
                    :attn_A_std, :diversity_mean, :diversity_std]
        propertynames(aggregate) == expected ||
            error("$family aggregate schema mismatch: $(propertynames(aggregate))")
        nrow(aggregate) == 8 || error("$family aggregate must contain eight rho rows")
        aggregate.rho == RHO_GRID || error("$family has a noncanonical rho grid")
        length(unique(aggregate.rho)) == 8 || error("$family has duplicate rho rows")
        raw = CSV.read(raw_path, DataFrame)
        nrow(raw) == 40 || error("$family raw replicate file must contain 40 rows")
        Set(propertynames(raw)) == Set([:rho, :replicate, :f_eff, :f_obs, :attn_A, :diversity]) ||
            error("$family raw replicate schema mismatch")
        all(combine(groupby(raw, :rho), nrow => :n).n .== 5) ||
            error("$family must contain five replicates per rho")
        for col in expected[2:end]
            all(isfinite, aggregate[!, col]) || error("$family contains non-finite $col")
        end
        for col in [:f_eff, :f_obs_mean, :attn_A_mean, :diversity_mean]
            all(x -> 0 <= x <= 1, aggregate[!, col]) ||
                error("$family contains $col outside [0,1]")
        end
        cross_row = only(eachrow(filter(:family => ==(family), cross)))
        high = only(eachrow(filter(:rho => ==(500.0), aggregate)))
        isapprox(cross_row.cal_gap_mean, high.f_eff - high.f_obs_mean; atol=1e-12) ||
            error("$family cross-family gap disagrees with its aggregate")
        isapprox(cross_row.cal_gap_std, high.f_obs_std; atol=1e-12) ||
            error("$family cross-family gap SD disagrees with its aggregate")
        aggregates[family] = aggregate
    end

    sar_path = joinpath(DATA_DIR, "omega_conotoxin", "sar_agreement.csv")
    isfile(sar_path) || error("Missing conotoxin SAR table: $sar_path")
    sar = CSV.read(sar_path, DataFrame)
    nrow(sar) == length(SAR_LABELS) ||
        error("Conotoxin SAR table must have $(length(SAR_LABELS)) rows, found $(nrow(sar))")
    length(unique(sar.Position)) == nrow(sar) ||
        error("Conotoxin SAR positions must be unique")
    Set(Int.(sar.Position)) == Set(keys(SAR_LABELS)) ||
        error("Conotoxin SAR position set mismatch: $(sort(Int.(sar.Position)))")
    expected_sar = [:Position, :WT_Residue, :Role, :Effect_of_mutation,
                    :Input_strong, :SA_strong, :SA_full, :Citation]
    propertynames(sar) == expected_sar ||
        error("Conotoxin SAR schema mismatch: $(propertynames(sar))")
    for col in [:Input_strong, :SA_strong, :SA_full]
        all(x -> 0 <= x <= 1, sar[!, col]) ||
            error("Conotoxin SAR $col outside [0,1]")
    end

    beta_path = joinpath(DATA_DIR, "kunitz", "calibration_beta_sweep.csv")
    isfile(beta_path) || error("Missing Kunitz beta sweep: $beta_path")
    beta_sweep = CSV.read(beta_path, DataFrame)
    expected_beta = [Symbol("ρ"), Symbol("β_mult"), Symbol("β_used"), Symbol("β_star"),
                     :f_observed, :diversity, :mean_valid]
    propertynames(beta_sweep) == expected_beta ||
        error("Kunitz beta sweep schema mismatch: $(propertynames(beta_sweep))")
    Set(beta_sweep[!, Symbol("ρ")]) == Set(BETA_SWEEP_RHO) ||
        error("Kunitz beta sweep rho set mismatch: $(sort(unique(beta_sweep[!, Symbol("ρ")])))")
    for rho in BETA_SWEEP_RHO
        block = filter(Symbol("ρ") => ==(rho), beta_sweep)
        nrow(block) == length(BETA_SWEEP_ALL_MULTIPLIERS) ||
            error("Kunitz beta sweep rho=$rho must have $(length(BETA_SWEEP_ALL_MULTIPLIERS)) rows")
        block[!, Symbol("β_mult")] == BETA_SWEEP_ALL_MULTIPLIERS ||
            error("Kunitz beta sweep rho=$rho multiplier grid mismatch")
        # One operating point per rho, and beta_used must be the multiplier times it.
        length(unique(block[!, Symbol("β_star")])) == 1 ||
            error("Kunitz beta sweep rho=$rho has more than one operating point")
        for row in eachrow(block)
            isapprox(row[Symbol("β_used")],
                     row[Symbol("β_mult")] * row[Symbol("β_star")]; rtol=1e-9) ||
                error("Kunitz beta sweep rho=$rho beta_used disagrees with mult times beta_star")
        end
    end
    for col in [:f_observed, :diversity]
        all(x -> 0 <= x <= 1, beta_sweep[!, col]) ||
            error("Kunitz beta sweep $col outside [0,1]")
    end

    return cross, aggregates, sar, beta_sweep
end

function linear_fit(x, y)
    slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)).^2)
    intercept = mean(y) - slope * mean(x)
    fitted = intercept .+ slope .* x
    r2 = 1 - sum((y .- fitted).^2) / sum((y .- mean(y)).^2)
    return intercept, slope, r2
end

function write_text(path, content)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, content)
        endswith(content, "\n") || write(io, "\n")
    end
end

function generate(output_dir)
    cross, aggregates, sar, beta_sweep = validate_inputs()
    mkpath(output_dir)

    for (family, slug) in FAMILY_SLUGS
        rows = String[]
        for row in eachrow(aggregates[family])
            push!(rows, @sprintf(
                "%d & %.3f & \$%.3f \\pm %.3f\$ & \$%.3f \\pm %.3f\$ & \$%.3f \\pm %.3f\$ \\\\",
                Int(row.rho), row.f_eff, row.f_obs_mean, row.f_obs_std,
                row.attn_A_mean, row.attn_A_std, row.diversity_mean, row.diversity_std))
        end
        write_text(joinpath(output_dir, "tab_$(slug)_rho.tex"),
                   join(rows, "\n") * "\n" * raw"\bottomrule")
    end

    cross_rows = String[]
    for row in eachrow(sort(cross, :separation_index))
        source = row.source == "Pfam" ? row.pfam_id : row.source
        hard = @sprintf("\$%.3f \\pm %.3f\$", row.hard_curation_mean, row.hard_curation_std)
        gap = @sprintf("\$%.3f \\pm %.3f\$", row.cal_gap_mean, row.cal_gap_std)
        push!(cross_rows, "$(latex_family(row.family)) & $source & $(row.K) & $(row.L) & " *
            "$(row.d_pca) & $(row.K_A) & $(row.marker) & $(fmt2(row.natural_frac)) & " *
            "$(fmt2(row.separation_index)) & $hard & $gap \\\\")
    end
    write_text(joinpath(output_dir, "tab_cross_family.tex"),
               join(cross_rows, "\n") * "\n" * raw"\bottomrule")

    sar_rows = String[]
    for row in eachrow(sar)
        position = Int(row.Position)
        residue, role, effect = SAR_LABELS[position]
        # Display threshold reproducing the emphasis of the hand-typed table, where
        # every designated-seeded value was bold except Leu11 at 0.24.
        designated = row.SA_strong >= 0.5 ?
            @sprintf("\\textbf{%.2f}", row.SA_strong) : @sprintf("%.2f", row.SA_strong)
        push!(sar_rows, @sprintf(
            "%d & %s & %s & %s & %.2f & %s & %.2f \\\\",
            position, residue, role, effect,
            row.Input_strong, designated, row.SA_full))
    end
    write_text(joinpath(output_dir, "tab_sar_agreement.tex"),
               join(sar_rows, "\n") * "\n" * raw"\bottomrule")

    beta_rows = String[]
    for (block_index, rho) in enumerate(BETA_SWEEP_RHO)
        block_index > 1 && push!(beta_rows, raw"\midrule")
        block = filter(Symbol("ρ") => ==(rho), beta_sweep)
        for mult in BETA_SWEEP_REPORTED
            row = only(eachrow(filter(Symbol("β_mult") => ==(mult), block)))
            push!(beta_rows, @sprintf("%-3d & %.2f & %.3f & %.3f \\\\",
                                      Int(rho), mult, row.f_observed, row.diversity))
        end
    end
    write_text(joinpath(output_dir, "tab_beta_sweep.tex"),
               join(beta_rows, "\n") * "\n" * raw"\bottomrule")

    selected_rho = Set([1.0, 10.0, 100.0, 500.0])
    per_family_rows = String[]
    ordered = sort(cross, :separation_index)
    for (family_index, meta) in enumerate(eachrow(ordered))
        aggregate = filter(:rho => x -> x in selected_rho, aggregates[meta.family])
        for (row_index, row) in enumerate(eachrow(aggregate))
            prefix = row_index == 1 ?
                "\\multirow{4}{*}{$(latex_family(meta.family))} & " *
                "\\multirow{4}{*}{($(meta.K), $(meta.K_A), $(fmt2(meta.separation_index)))}" :
                "&"
            gap = row.f_eff - row.f_obs_mean
            push!(per_family_rows, @sprintf(
                "%s & %d & %.3f & \$%.3f \\pm %.3f\$ & \$%.3f \\pm %.3f\$ & %.2f & %.3f \\\\",
                prefix, Int(row.rho), row.f_eff, row.attn_A_mean, row.attn_A_std,
                row.f_obs_mean, row.f_obs_std, gap, row.diversity_mean))
        end
        family_index < nrow(ordered) && push!(per_family_rows, raw"\midrule")
    end
    write_text(joinpath(output_dir, "tab_per_family_rho.tex"),
               join(per_family_rows, "\n") * "\n" * raw"\bottomrule")

    pfam = filter(:fit_included => identity, cross)
    intercept, slope, r2 = linear_fit(pfam.separation_index, pfam.cal_gap_mean)
    loo = [linear_fit(pfam.separation_index[setdiff(1:nrow(pfam), [i])],
                      pfam.cal_gap_mean[setdiff(1:nrow(pfam), [i])])
           for i in 1:nrow(pfam)]
    loo_slopes = [fit[2] for fit in loo]
    loo_r2 = [fit[3] for fit in loo]

    deviations = NamedTuple[]
    for (family, _) in FAMILY_SLUGS, row in eachrow(aggregates[family])
        push!(deviations, (family=family, rho=row.rho,
                           deviation=abs(row.attn_A_mean - row.f_eff)))
    end
    maximum_deviation = deviations[argmax(getfield.(deviations, :deviation))]
    kunitz_500 = only(eachrow(filter(:rho => ==(500.0), aggregates["Kunitz"])))

    macros = [
        raw"% Generated by code/experiments/generate_paper_tables.jl; do not edit.",
        "\\newcommand{\\KunitzFobsFiveHundred}{$(fmt3(kunitz_500.f_obs_mean))}",
        "\\newcommand{\\KunitzFobsFiveHundredSD}{$(fmt3(kunitz_500.f_obs_std))}",
        "\\newcommand{\\AttnMaxDevPts}{$(fmt1(100maximum_deviation.deviation))}",
        "\\newcommand{\\AttnMaxDevFamily}{$(latex_family(maximum_deviation.family))}",
        "\\newcommand{\\AttnMaxDevRho}{$(Int(maximum_deviation.rho))}",
        "\\newcommand{\\RelationIntercept}{$(fmt2(intercept))}",
        "\\newcommand{\\RelationSlope}{$(fmt1(slope))}",
        "\\newcommand{\\RelationRsq}{$(fmt2(r2))}",
        "\\newcommand{\\RelationLooSlopeMin}{$(fmt1(minimum(loo_slopes)))}",
        "\\newcommand{\\RelationLooSlopeMax}{$(fmt1(maximum(loo_slopes)))}",
        "\\newcommand{\\RelationLooRsqMin}{$(fmt2(minimum(loo_r2)))}",
        "\\newcommand{\\RelationLooRsqMax}{$(fmt2(maximum(loo_r2)))}",
    ]
    macro_names = Dict("SH3" => "SHThree")
    for row in eachrow(cross)
        macro_family = get(macro_names, row.family, replace(row.family, r"[^A-Za-z]" => ""))
        push!(macros, "\\newcommand{\\$(macro_family)NaturalFraction}{$(fmt3(row.natural_frac))}")
        push!(macros, "\\newcommand{\\$(macro_family)Separation}{$(fmt2(row.separation_index))}")
    end
    write_text(joinpath(output_dir, "numbers.tex"), join(macros, "\n"))
end

function main(args)
    output_dir = DEFAULT_OUTPUT
    for arg in args
        startswith(arg, "--output=") || error("Unknown argument: $arg")
        output_dir = abspath(split(arg, "=", limit=2)[2])
    end
    generate(output_dir)
    println("Generated manuscript tables and macros in $output_dir")
end

main(ARGS)
