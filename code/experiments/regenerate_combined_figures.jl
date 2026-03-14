# Regenerate figures from saved CSVs
_SCRIPT_DIR = @__DIR__
_CODE_DIR = dirname(_SCRIPT_DIR)
cd(_CODE_DIR)
include(joinpath(_CODE_DIR, "Include.jl"))

CACHE_DIR = joinpath(_CODE_DIR, "data", "kunitz")
FIG_DIR = joinpath(_CODE_DIR, "figs", "combined")
mkpath(FIG_DIR)

# load results
cal = CSV.read(joinpath(CACHE_DIR, "combined_calibration.csv"), DataFrame)
rho_c = CSV.read(joinpath(CACHE_DIR, "combined_rho_sweep.csv"), DataFrame)
wt = CSV.read(joinpath(CACHE_DIR, "combined_weight_sweep.csv"), DataFrame)

# also load standard PCA rho sweep from previous experiment
rho_s_file = joinpath(CACHE_DIR, "multiplicity_generation.csv")
rho_s = isfile(rho_s_file) ? CSV.read(rho_s_file, DataFrame) : nothing

# natural binder fraction
natural_frac = 32 / 99

# --- Fig 1: weight sweep ---
p1 = plot(layout=(1, 3), size=(1200, 400), margin=10Plots.mm)
plot!(p1[1], wt.weight, wt.f_obs,
    marker=:circle, linewidth=2, color=:steelblue, label="P1 K/R (hard)",
    xlabel="Interface weight w", ylabel="Fraction",
    title="Phenotype Fidelity (ρ=50)")
plot!(p1[1], wt.weight, wt.soft_p1,
    marker=:square, linewidth=2, color=:purple, label="P1 K/R (soft)")
plot!(p1[2], wt.weight, wt.diversity,
    marker=:circle, linewidth=2, color=:coral, label="",
    xlabel="Interface weight w", ylabel="Diversity", title="Sequence Diversity")
plot!(p1[3], wt.weight, wt.d_pca,
    marker=:circle, linewidth=2, color=:forestgreen, label="",
    xlabel="Interface weight w", ylabel="PCA dimensions", title="PCA Dimensionality")
savefig(p1, joinpath(FIG_DIR, "fig1_weight_sweep.png"))
@info "Saved fig1"

# --- Fig 2: calibration comparison ---
p2 = plot(size=(700, 550), margin=12Plots.mm,
    xlabel="Target effective binder fraction (f_target)",
    ylabel="Observed P1 K/R fraction",
    title="Calibration: Standard PCA vs Combined Method",
    legend=:topleft, ylim=(0, 1.05), xlim=(0, 1.05))
plot!(p2, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="ideal (y=x)", linewidth=1.5)
plot!(p2, cal.f_target, cal.f_obs_standard,
    marker=:circle, linewidth=2.5, color=:coral, label="Standard PCA", markersize=6)
plot!(p2, cal.f_target, cal.f_obs_combined,
    marker=:diamond, linewidth=2.5, color=:steelblue, label="Combined (w=1.5)", markersize=6)
savefig(p2, joinpath(FIG_DIR, "fig2_calibration_comparison.png"))
savefig(p2, joinpath(FIG_DIR, "fig2_calibration_comparison.pdf"))
@info "Saved fig2"

# --- Fig 3: ρ sweep comparison ---
if rho_s !== nothing
    p3 = plot(layout=(1, 2), size=(1000, 450), margin=10Plots.mm)
    plot!(p3[1], log10.(rho_s[!, :ρ]), rho_s.p1_kr_frac,
        marker=:circle, linewidth=2, color=:coral, label="Standard PCA",
        xlabel="log₁₀(ρ)", ylabel="P1 K/R fraction", ylim=(0, 1.05),
        title="Phenotype Fidelity")
    plot!(p3[1], log10.(rho_c.ρ), rho_c.f_obs,
        marker=:diamond, linewidth=2, color=:steelblue, label="Combined (w=1.5)")
    hline!(p3[1], [natural_frac], linestyle=:dot, color=:gray, label="natural")

    plot!(p3[2], log10.(rho_s[!, :ρ]), rho_s.diversity,
        marker=:circle, linewidth=2, color=:coral, label="Standard PCA",
        xlabel="log₁₀(ρ)", ylabel="Diversity", title="Sequence Diversity")
    plot!(p3[2], log10.(rho_c.ρ), rho_c.diversity,
        marker=:diamond, linewidth=2, color=:steelblue, label="Combined (w=1.5)")
    savefig(p3, joinpath(FIG_DIR, "fig3_rho_comparison.png"))
    savefig(p3, joinpath(FIG_DIR, "fig3_rho_comparison.pdf"))
    @info "Saved fig3"
end

# --- Fig 4: soft score comparison ---
p4 = plot(size=(700, 450), margin=10Plots.mm,
    xlabel="log₁₀(ρ)", ylabel="Soft P1 K/R probability",
    title="PCA Bottleneck: Standard vs Interface-Weighted",
    legend=:topleft)
plot!(p4, log10.(rho_c.ρ), rho_c.soft_p1,
    marker=:diamond, linewidth=2.5, color=:steelblue, label="Combined (w=1.5)", markersize=5)
if rho_s !== nothing
    # match ρ values that exist in both
    std_soft_at_rho = Float64[]
    for ρ_val in rho_c.ρ
        idx = findfirst(x -> abs(x - ρ_val) < 0.1, rho_s[!, :ρ])
        if idx !== nothing
            # we don't have soft_p1 in the multiplicity_generation.csv
            # so skip this comparison for now
        end
    end
end
hline!(p4, [natural_frac], linestyle=:dot, color=:gray, label="natural binder fraction")
savefig(p4, joinpath(FIG_DIR, "fig4_soft_score_comparison.png"))
savefig(p4, joinpath(FIG_DIR, "fig4_soft_score_comparison.pdf"))
@info "Saved fig4"

# --- Fig 6: Pareto front ---
p6 = plot(size=(700, 500), margin=10Plots.mm,
    xlabel="Pairwise diversity", ylabel="P1 K/R fraction",
    title="Fidelity–Diversity Pareto Front",
    legend=:bottomleft, ylim=(0.3, 1.05))
if rho_s !== nothing
    scatter!(p6, rho_s.diversity, rho_s.p1_kr_frac,
        marker=:circle, markersize=7, color=:coral, label="Multiplicity only (std PCA)")
end
scatter!(p6, rho_c.diversity, rho_c.f_obs,
    marker=:diamond, markersize=7, color=:steelblue, label="Combined (w=1.5)")
scatter!(p6, [0.503], [1.0],
    marker=:star5, markersize=12, color=:red, label="Hard curation")
savefig(p6, joinpath(FIG_DIR, "fig6_pareto.png"))
savefig(p6, joinpath(FIG_DIR, "fig6_pareto.pdf"))
@info "Saved fig6"

@info "All figures regenerated!"
