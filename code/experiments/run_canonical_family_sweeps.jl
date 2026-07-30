#!/usr/bin/env julia

# Generate the canonical replicated multiplicity sweeps and six-family summary.

const SCRIPT_DIR = @__DIR__
const CODE_DIR = dirname(SCRIPT_DIR)
cd(CODE_DIR)
include(joinpath(CODE_DIR, "Include.jl"))
include(joinpath(SCRIPT_DIR, "canonical_family_registry.jl"))

using CSV, DataFrames, Random, Statistics

const DATA_DIR = joinpath(CODE_DIR, "data")
const RHO_GRID = Float64[1, 2, 5, 10, 20, 50, 100, 500]
const N_REPS = 5
const N_CHAINS = 20
const N_STEPS = 5000
const BURN_IN = 2000
const THIN = 100

# Seed-space origins for the two replicated calculations in this driver. They are distinct
# so the hard-curation and multiplicity blocks cannot collide with each other, and each
# block is SEED_BLOCK wide so a replicate's chain seeds cannot reach into its neighbour.
const HARD_CURATION_SEED_ORIGIN = 10_000_000
const SWEEP_SEED_ORIGIN = 20_000_000

# Registry position, used as the family coordinate of the seed block. Stable because
# CANONICAL_FAMILIES is a fixed literal.
family_index(spec) = findfirst(s -> s.slug == spec.slug, CANONICAL_FAMILIES)

function atomic_csv_write(path, table)
    mkpath(dirname(path))
    temp = path * ".tmp"
    CSV.write(temp, table)
    mv(temp, path; force=true)
end

function normalize_aggregate(df::DataFrame)
    rename_map = Dict(
        Symbol("ρ") => :rho,
        :f_obs => :f_obs_mean,
        :attn_A => :attn_A_mean,
        :diversity => :diversity_mean,
    )
    for (old, new) in rename_map
        old in propertynames(df) && !(new in propertynames(df)) && rename!(df, old => new)
    end
    return df
end

function validate_aggregate(df::DataFrame, family::String)
    expected = [:rho, :f_eff, :f_obs_mean, :f_obs_std, :attn_A_mean, :attn_A_std,
                :diversity_mean, :diversity_std]
    propertynames(df) == expected ||
        error("$family aggregate schema mismatch: $(propertynames(df)); expected $expected")
    nrow(df) == length(RHO_GRID) || error("$family aggregate must have 8 rows")
    df.rho == RHO_GRID || error("$family aggregate rho grid is $(df.rho), expected $RHO_GRID")
    length(unique(df.rho)) == nrow(df) || error("$family aggregate has duplicate rho rows")
    for col in expected[2:end]
        all(isfinite, df[!, col]) || error("$family aggregate has non-finite $col")
    end
    for col in [:f_eff, :f_obs_mean, :attn_A_mean, :diversity_mean]
        all(x -> 0 <= x <= 1, df[!, col]) || error("$family aggregate $col is outside [0,1]")
    end
    all(x -> x >= 0, df.f_obs_std) || error("$family aggregate has negative SD")
    return df
end

function family_analysis(spec::CanonicalFamilySpec)
    char_mat, names, auxiliary = canonical_load_alignment(spec, DATA_DIR)
    group_A, group_B, marker_pos, marker_residues =
        canonical_split(spec, char_mat, names, auxiliary)
    length(group_A) >= 3 || error("$(spec.family) designated group has fewer than 3 sequences")
    length(group_B) >= 3 || error("$(spec.family) background group has fewer than 3 sequences")
    X, pca_model, _, _ = build_memory_matrix(char_mat; pratio=0.95)
    return (
        spec=spec, char_mat=char_mat, names=names, X=X, pca_model=pca_model,
        group_A=group_A, group_B=group_B, marker_pos=marker_pos,
        marker_residues=marker_residues,
        separation_index=canonical_separation(X, group_A, group_B),
    )
end

function hard_curation_replicates(analysis)
    group_char = analysis.char_mat[analysis.group_A, :]
    X_hard, pca_hard, _, _ = build_memory_matrix(group_char; pratio=0.95)
    beta_hard = all_memory_onset(X_hard)
    values = Float64[]
    L = size(analysis.char_mat, 2)
    fam = family_index(analysis.spec)
    for rep in 1:N_REPS
        seed = replicate_base_seed(HARD_CURATION_SEED_ORIGIN,
            condition_block_index((fam, rep), (length(CANONICAL_FAMILIES), N_REPS)))
        sequences, _ = generate_sequences(X_hard, pca_hard, L;
            β=beta_hard, n_chains=N_CHAINS, T=N_STEPS, seed=seed)
        fraction = count(s -> length(s) >= analysis.marker_pos &&
                             s[analysis.marker_pos] in analysis.marker_residues,
                         sequences) / length(sequences)
        push!(values, fraction)
    end
    return mean(values), std(values)
end

function run_replicated_sweep(analysis)
    K, L = size(analysis.char_mat)
    d = size(analysis.X, 1)
    fam = family_index(analysis.spec)
    raw = DataFrame(
        rho=Float64[], replicate=Int[], f_eff=Float64[], f_obs=Float64[],
        attn_A=Float64[], diversity=Float64[],
    )

    for (rho_index, rho) in enumerate(RHO_GRID)
        @info "$(analysis.spec.family): rho=$rho"
        weights = multiplicity_vector(K, analysis.group_A; ρ=rho)
        f_eff = effective_binder_fraction(weights, analysis.group_A)
        log_weights = log.(weights)
        beta = all_memory_onset(analysis.X, weights; n_betas=50)

        for rep in 1:N_REPS
            seed = replicate_base_seed(SWEEP_SEED_ORIGIN,
                condition_block_index((fam, rho_index, rep),
                                      (length(CANONICAL_FAMILIES), length(RHO_GRID), N_REPS)))
            replicate_rng = MersenneTwister(seed)
            generated = String[]
            attention = Float64[]
            for chain in 1:N_CHAINS
                chain_rng = MersenneTwister(seed + chain)
                memory_index = mod1(chain, K)
                xi0 = analysis.X[:, memory_index] .+ 0.01 .* randn(chain_rng, d)
                sampled = weighted_sample(analysis.X, xi0, N_STEPS, weights;
                                          β=beta, α=0.01, rng=chain_rng)
                for t in BURN_IN:THIN:N_STEPS
                    xi = sampled.Ξ[t + 1, :]
                    push!(generated, decode_sample(xi, analysis.pca_model, L))
                    logits = beta .* (analysis.X' * xi) .+ log_weights
                    attn = NNlib.softmax(logits)
                    push!(attention, sum(attn[i] for i in analysis.group_A))
                end
            end

            f_obs = count(s -> length(s) >= analysis.marker_pos &&
                              s[analysis.marker_pos] in analysis.marker_residues,
                          generated) / length(generated)
            pair_ids = Float64[]
            n = length(generated)
            for _ in 1:min(300, n * (n - 1) ÷ 2)
                i, j = rand(replicate_rng, 1:n), rand(replicate_rng, 1:n)
                while i == j
                    j = rand(replicate_rng, 1:n)
                end
                push!(pair_ids, sequence_identity(generated[i], generated[j]))
            end
            push!(raw, (rho, rep, f_eff, f_obs, mean(attention), 1 - mean(pair_ids)))
        end
    end

    aggregate = combine(groupby(raw, :rho),
        :f_eff => first => :f_eff,
        :f_obs => mean => :f_obs_mean,
        :f_obs => std => :f_obs_std,
        :attn_A => mean => :attn_A_mean,
        :attn_A => std => :attn_A_std,
        :diversity => mean => :diversity_mean,
        :diversity => std => :diversity_std,
    )
    validate_aggregate(aggregate, analysis.spec.family)
    return raw, aggregate
end

function write_family_metadata(analysis, hard_mean, hard_std)
    spec = analysis.spec
    K, L = size(analysis.char_mat)
    metadata = DataFrame(
        family=[spec.family], source=[spec.source], pfam_id=[spec.pfam_id],
        K=[K], L=[L], d_pca=[size(analysis.X, 1)],
        K_A=[length(analysis.group_A)], K_B=[length(analysis.group_B)],
        marker=[spec.marker], natural_frac=[length(analysis.group_A) / K],
        separation_index=[analysis.separation_index],
        hard_curation_mean=[hard_mean], hard_curation_std=[hard_std],
        fit_included=[spec.fit_included],
    )
    atomic_csv_write(joinpath(DATA_DIR, spec.slug, "canonical_family_metadata.csv"), metadata)
end

function write_execution_record(spec::CanonicalFamilySpec)
    record = DataFrame(
        parameter=["family", "rho_grid", "n_replicates", "n_chains", "n_steps",
                   "burn_in", "thin", "driver"],
        value=[spec.family, join(Int.(RHO_GRID), ";"), string(N_REPS),
               string(N_CHAINS), string(N_STEPS), string(BURN_IN), string(THIN),
               "run_canonical_family_sweeps.jl"],
    )
    atomic_csv_write(joinpath(DATA_DIR, spec.slug, "canonical_sweep_execution.csv"), record)
end

function assemble_cross_family()
    rows = DataFrame[]
    for spec in CANONICAL_FAMILIES
        metadata_path = joinpath(DATA_DIR, spec.slug, "canonical_family_metadata.csv")
        aggregate_path = joinpath(DATA_DIR, spec.slug, "multiplicity_sweep_aggregated.csv")
        isfile(metadata_path) || error("Missing metadata for $(spec.family): $metadata_path")
        isfile(aggregate_path) || error("Missing aggregate for $(spec.family): $aggregate_path")
        metadata = CSV.read(metadata_path, DataFrame)
        nrow(metadata) == 1 || error("Expected one metadata row for $(spec.family)")
        # Refresh deterministic metadata from the shared registry without rerunning
        # stochastic sweeps or hard-curation replicates.
        analysis = family_analysis(spec)
        write_family_metadata(analysis, metadata.hard_curation_mean[1],
                              metadata.hard_curation_std[1])
        metadata = CSV.read(metadata_path, DataFrame)
        aggregate = validate_aggregate(normalize_aggregate(CSV.read(aggregate_path, DataFrame)),
                                       spec.family)
        high = only(eachrow(filter(:rho => ==(500.0), aggregate)))
        metadata.cal_gap_mean = [high.f_eff - high.f_obs_mean]
        metadata.cal_gap_std = [high.f_obs_std]
        push!(rows, metadata)
    end
    comparison = vcat(rows...; cols=:union)
    select!(comparison, :family, :source, :pfam_id, :K, :L, :d_pca, :K_A, :K_B,
            :marker, :natural_frac, :separation_index, :hard_curation_mean,
            :hard_curation_std, :cal_gap_mean, :cal_gap_std, :fit_included)
    atomic_csv_write(joinpath(DATA_DIR, "multi_family_comparison_6fam_aggregated.csv"), comparison)

    # Every canonical family, Kunitz included, is swept by this driver. Earlier versions
    # carried Kunitz forward as a legacy aggregate; that path is gone, so the recorded
    # origin is unconditional rather than inferred from a file's existence.
    kunitz_origin = "canonical driver; see data/kunitz/canonical_sweep_execution.csv"
    provenance = DataFrame(
        parameter=["rho_grid", "n_replicates", "n_chains", "n_steps", "burn_in", "thin",
                   "replicate_seed_formula", "kunitz_origin"],
        value=[join(Int.(RHO_GRID), ";"), string(N_REPS), string(N_CHAINS),
               string(N_STEPS), string(BURN_IN), string(THIN),
               "replicate_base_seed(20000000, condition_block_index((family, rho, replicate), " *
               "(6, 8, 5))) with block 1000; chain adds chain_index. Hard curation uses " *
               "origin 10000000 over (family, replicate)",
               kunitz_origin],
    )
    atomic_csv_write(joinpath(DATA_DIR, "canonical_sweep_provenance.csv"), provenance)
end

function parse_requested_families(args)
    requested = String[]
    assemble_only = false
    for arg in args
        if startswith(arg, "--families=")
            requested = split(split(arg, "=", limit=2)[2], ",")
        elseif arg == "--assemble-only"
            assemble_only = true
        else
            error("Unknown argument: $arg")
        end
    end
    return Set(lowercase.(requested)), assemble_only
end

requested, assemble_only = parse_requested_families(ARGS)

if !assemble_only
    rerun_specs = CANONICAL_FAMILIES
    !isempty(requested) && (rerun_specs = filter(s -> lowercase(s.family) in requested ||
                                                  s.slug in requested, rerun_specs))
    for spec in rerun_specs
        @info "Running canonical replicated sweep for $(spec.family)"
        analysis = family_analysis(spec)
        hard_mean, hard_std = hard_curation_replicates(analysis)
        raw, aggregate = run_replicated_sweep(analysis)
        family_dir = joinpath(DATA_DIR, spec.slug)
        atomic_csv_write(joinpath(family_dir, "multiplicity_sweep_raw_replicates.csv"), raw)
        atomic_csv_write(joinpath(family_dir, "multiplicity_sweep_aggregated.csv"), aggregate)
        write_family_metadata(analysis, hard_mean, hard_std)
        write_execution_record(spec)
    end
end

assemble_cross_family()
@info "Canonical family sweeps and cross-family CSV are complete"
