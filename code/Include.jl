# ──────────────────────────────────────────────────────────────────────────────
# Include.jl — SA Binding Generation Study
# Sets up the environment and loads all source modules
# ──────────────────────────────────────────────────────────────────────────────

# setup paths
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src")
const _PATH_TO_DATA = joinpath(_ROOT, "data")
const _PATH_TO_FIG = joinpath(_ROOT, "figs")

# activate and check environment
using Pkg
Pkg.activate(_ROOT)
if !isfile(joinpath(_ROOT, "Manifest.toml"))
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load required packages
using Distributions
using Plots
using StatsPlots
using Colors
using LinearAlgebra
using Statistics
using DataFrames
using PrettyTables
using Random
using CSV
using FileIO
using JLD2
using Dates
using NNlib
using CategoricalArrays
using StatsBase
using MultivariateStats
using Downloads

# set the random seed for reproducibility
Random.seed!(1234)

# include the source modules (upstream SA core)
include(joinpath(_PATH_TO_SRC, "Data.jl"))
include(joinpath(_PATH_TO_SRC, "Compute.jl"))
include(joinpath(_PATH_TO_SRC, "Utilities.jl"))
include(joinpath(_PATH_TO_SRC, "Protein.jl"))

# include the new binding-specific module
include(joinpath(_PATH_TO_SRC, "Binding.jl"))
