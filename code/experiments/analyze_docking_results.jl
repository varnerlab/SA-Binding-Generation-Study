# analyze_docking_results.jl
#
# Analyze ColabFold multimer docking results and generate figures
# comparing SA strong-seeded vs full-seeded vs control conotoxin binding

using CSV, DataFrames
using StatsPlots, Plots
using Statistics, StatsBase
using HypothesisTests

const DATA_DIR = joinpath(@__DIR__, "..", "data", "omega_conotoxin")
const DOCKING_DIR = joinpath(DATA_DIR, "docking_validation")
const FIG_DIR = joinpath(@__DIR__, "..", "figs", "docking_validation")
mkpath(FIG_DIR)

function load_docking_results()
    """Load docking results from CSV file."""
    results_file = joinpath(DOCKING_DIR, "docking_results.csv")
    if !isfile(results_file)
        error("Docking results not found. Run docking pipeline first.")
    end

    df = CSV.read(results_file, DataFrame)

    # Clean up group names for plotting
    df.group = replace.(df.group, "SA_" => "SA ")
    df.group = replace.(df.group, "controls" => "Known binders")

    @info "Loaded docking results: $(nrow(df)) predictions across $(length(unique(df.group))) groups"
    return df
end

function plot_binding_scores(df)
    """Create violin/box plots comparing binding scores across groups."""

    # ipTM scores (interface quality)
    p1 = @df df violin(:group, :iptm_score, fillalpha=0.6, linewidth=2)
    @df df boxplot!(:group, :iptm_score, fillalpha=0.0, linewidth=2, outliers=false)
    plot!(title="Interface Quality (ipTM)", ylabel="ipTM Score",
          ylims=(0, 1.0), legend=false)
    hline!([0.8], line=(:dash, :red), label="High confidence threshold")

    # Confidence scores (overall prediction quality)
    p2 = @df df violin(:group, :confidence, fillalpha=0.6, linewidth=2)
    @df df boxplot!(:group, :confidence, fillalpha=0.0, linewidth=2, outliers=false)
    plot!(title="Overall Confidence", ylabel="Confidence Score",
          ylims=(0, 1.0), legend=false)

    # Interface pLDDT (local confidence at binding site)
    p3 = @df df violin(:group, :interface_plddt, fillalpha=0.6, linewidth=2)
    @df df boxplot!(:group, :interface_plddt, fillalpha=0.0, linewidth=2, outliers=false)
    plot!(title="Interface pLDDT", ylabel="Interface pLDDT Score",
          ylims=(50, 100), legend=false)
    hline!([70], line=(:dash, :red), label="Confident structure threshold")

    # Combined plot
    combined = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400),
                   plot_title="Conotoxin-Cav2.2 Binding Prediction Quality")

    savefig(combined, joinpath(FIG_DIR, "binding_scores_comparison.pdf"))
    savefig(combined, joinpath(FIG_DIR, "binding_scores_comparison.png"))

    return combined
end

function plot_score_correlations(df)
    """Create scatter plot matrix showing correlations between metrics."""

    p1 = @df df scatter(:iptm_score, :confidence, group=:group, alpha=0.7)
    plot!(xlabel="ipTM Score", ylabel="Confidence", legend=:bottomright)

    p2 = @df df scatter(:iptm_score, :interface_plddt, group=:group, alpha=0.7)
    plot!(xlabel="ipTM Score", ylabel="Interface pLDDT")

    p3 = @df df scatter(:confidence, :interface_plddt, group=:group, alpha=0.7)
    plot!(xlabel="Confidence", ylabel="Interface pLDDT")

    combined = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400),
                   plot_title="Binding Score Correlations")

    savefig(combined, joinpath(FIG_DIR, "score_correlations.pdf"))

    return combined
end

function statistical_analysis(df)
    """Perform statistical tests comparing groups."""

    groups = unique(df.group)
    metrics = [:iptm_score, :confidence, :interface_plddt]

    results = DataFrame(
        metric = String[],
        group1 = String[],
        group2 = String[],
        p_value = Float64[],
        effect_size = Float64[],
        test = String[]
    )

    for metric in metrics
        # Remove missing values
        clean_df = df[.!ismissing.(df[!, metric]), :]

        for i in 1:length(groups)
            for j in (i+1):length(groups)
                group1, group2 = groups[i], groups[j]

                data1 = clean_df[clean_df.group .== group1, metric]
                data2 = clean_df[clean_df.group .== group2, metric]

                if length(data1) > 0 && length(data2) > 0
                    # Mann-Whitney U test (non-parametric)
                    test_result = MannWhitneyUTest(data1, data2)
                    p_val = pvalue(test_result)

                    # Effect size (Cohen's d approximation)
                    pooled_std = sqrt(((length(data1)-1)*var(data1) + (length(data2)-1)*var(data2)) /
                                    (length(data1) + length(data2) - 2))
                    cohens_d = (mean(data1) - mean(data2)) / pooled_std

                    push!(results, (String(metric), group1, group2, p_val, cohens_d, "Mann-Whitney"))
                end
            end
        end
    end

    # Multiple testing correction
    results.p_adjusted = adjust(results.p_value, Bonferroni())

    @info "Statistical analysis completed"
    println(results)

    CSV.write(joinpath(DOCKING_DIR, "statistical_analysis.csv"), results)

    return results
end

function generate_summary_table(df)
    """Generate summary statistics table."""

    summary = combine(groupby(df, :group)) do group_df
        DataFrame(
            n = nrow(group_df),
            iptm_mean = round(mean(skipmissing(group_df.iptm_score)), digits=3),
            iptm_std = round(std(skipmissing(group_df.iptm_score)), digits=3),
            confidence_mean = round(mean(skipmissing(group_df.confidence)), digits=3),
            confidence_std = round(std(skipmissing(group_df.confidence)), digits=3),
            interface_plddt_mean = round(mean(skipmissing(group_df.interface_plddt)), digits=1),
            interface_plddt_std = round(std(skipmissing(group_df.interface_plddt)), digits=1),
            high_quality_fraction = round(mean(group_df.iptm_score .> 0.8), digits=3)
        )
    end

    @info "Summary statistics:"
    println(summary)

    CSV.write(joinpath(DOCKING_DIR, "summary_statistics.csv"), summary)

    return summary
end

function main()
    """Main analysis function."""

    @info "Starting docking results analysis"

    # Load results
    df = load_docking_results()

    # Generate plots
    @info "Creating binding scores comparison plots..."
    plot_binding_scores(df)

    @info "Creating correlation plots..."
    plot_score_correlations(df)

    # Statistical analysis
    @info "Performing statistical analysis..."
    stats = statistical_analysis(df)

    # Summary table
    @info "Generating summary table..."
    summary = generate_summary_table(df)

    @info "Analysis complete. Results saved to $(DOCKING_DIR) and $(FIG_DIR)"

    return df, stats, summary
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end