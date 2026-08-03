#!/usr/bin/env julia

# Matched training-free baseline for multiplicity-conditioned stochastic attention.
#
# A profile HMM is fit to the same aligned sequences used by SA. Designated
# sequences receive relative weight rho and background sequences receive weight
# one. The finite-rho benchmark therefore receives the same designation labels
# and multiplicity ratio as SA. A designated-subset HMM supplies the hard
# curation endpoint. HMMER's emitted alignment is reduced to match-state columns
# so every output remains in the canonical alignment coordinate frame.

const SCRIPT_DIR = @__DIR__
const CODE_DIR = dirname(SCRIPT_DIR)
cd(CODE_DIR)
include(joinpath(CODE_DIR, "Include.jl"))
include(joinpath(SCRIPT_DIR, "canonical_family_registry.jl"))

using CSV, DataFrames, Random, Statistics

const PROFILE_HMM_RHO_GRID = Float64[1, 2, 5, 10, 20, 50, 100, 500]
const PROFILE_HMM_N_REPS = 5
const PROFILE_HMM_N_EMIT = 620
const PROFILE_HMM_SEED_ORIGIN = 80_000_000
const PROFILE_HMM_DATA_DIR = joinpath(CODE_DIR, "data", "profile_hmm_conditioning")
const CANONICAL_SA_SEED_ORIGIN = 20_000_000
const CANONICAL_SA_N_CHAINS = 20
const CANONICAL_SA_N_STEPS = 5000
const CANONICAL_SA_BURN_IN = 2000
const CANONICAL_SA_THIN = 100

function require_hmmer()
    for executable in ("hmmbuild", "hmmemit")
        Sys.which(executable) === nothing &&
            error("$executable is required for the profile-HMM benchmark")
    end
end

function atomic_csv_write(path, table)
    mkpath(dirname(path))
    temp = path * ".tmp"
    CSV.write(temp, table)
    mv(temp, path; force=true)
end

function stockholm_residue(c::Char)
    c in AA_ALPHABET && return c
    is_alignment_gap(c) && return '-'
    return 'X'
end

function write_weighted_stockholm(path, char_mat, group_A; rho::Float64=1.0)
    K, _ = size(char_mat)
    designated = Set(group_A)
    open(path, "w") do io
        println(io, "# STOCKHOLM 1.0")
        println(io)
        for i in 1:K
            weight = i in designated ? rho : 1.0
            println(io, "#=GS seq$(i) WT $(weight)")
        end
        println(io)
        for i in 1:K
            sequence = String([stockholm_residue(c) for c in char_mat[i, :]])
            println(io, "seq$(i) $sequence")
        end
        println(io, "//")
    end
    return path
end

function build_profile_hmm(msa_path, hmm_path, model_name)
    command = `hmmbuild --amino --symfrac 0.0 --wgiven --enone -n $model_name $hmm_path $msa_path`
    run(pipeline(command; stdout=devnull, stderr=devnull))
    return hmm_path
end

"""
    parse_emitted_match_alignment(path, expected_length)

Read a Stockholm alignment produced by `hmmemit -a` and retain only columns
marked as profile-HMM match states by the `#=GC RF` annotation. With
`hmmbuild --symfrac 0.0`, these columns correspond one-to-one with the input
alignment columns, including columns containing deletions in individual rows.
"""
function parse_emitted_match_alignment(path, expected_length)
    fragments = Dict{String, String}()
    order = String[]
    rf = ""
    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "//") && continue
        if startswith(stripped, "#=GC RF")
            parts = split(stripped)
            length(parts) >= 3 || error("Malformed RF annotation in $path")
            rf *= parts[3]
            continue
        end
        startswith(stripped, "#") && continue
        parts = split(stripped)
        length(parts) >= 2 || continue
        name, fragment = parts[1], uppercase(parts[2])
        if !haskey(fragments, name)
            fragments[name] = ""
            push!(order, name)
        end
        fragments[name] *= fragment
    end

    isempty(rf) && error("No RF annotation found in emitted HMM alignment: $path")
    match_columns = findall(c -> c in ('x', 'X'), collect(rf))
    length(match_columns) == expected_length || error(
        "Emitted HMM has $(length(match_columns)) match columns; expected $expected_length")

    sequences = String[]
    for name in order
        aligned = fragments[name]
        length(aligned) == length(rf) || error(
            "Alignment/RF length mismatch for $name: $(length(aligned)) versus $(length(rf))")
        push!(sequences, String([aligned[j] for j in match_columns]))
    end
    return sequences
end

function emit_profile_sequences(hmm_path, output_path, n_emit, seed, expected_length)
    command = `hmmemit -a -N $n_emit --seed $seed -o $output_path $hmm_path`
    run(pipeline(command; stdout=devnull, stderr=devnull))
    sequences = parse_emitted_match_alignment(output_path, expected_length)
    length(sequences) == n_emit || error(
        "hmmemit returned $(length(sequences)) sequences; expected $n_emit")
    return sequences
end

function sequence_pair_diversity(sequences, rng; n_pairs=300)
    n = length(sequences)
    n < 2 && return 0.0
    identities = Float64[]
    for _ in 1:min(n_pairs, n * (n - 1) ÷ 2)
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        while i == j
            j = rand(rng, 1:n)
        end
        push!(identities, sequence_identity(sequences[i], sequences[j]))
    end
    return 1 - mean(identities)
end

function profile_hmm_metrics(sequences, analysis, rng)
    full_sequences = [String(analysis.char_mat[i, :])
                      for i in axes(analysis.char_mat, 1)]
    designated_sequences = full_sequences[analysis.group_A]
    marker_fraction = mean(s -> s[analysis.marker_pos] in analysis.marker_residues,
                           sequences)
    nearest_identity = mean(nearest_sequence_identity(s, full_sequences) for s in sequences)
    gap_fraction = mean(s -> count(is_alignment_gap, s) / length(s), sequences)
    return (
        marker_fraction=marker_fraction,
        kl_to_full=aa_composition_kl(sequences, full_sequences),
        kl_to_designated=aa_composition_kl(sequences, designated_sequences),
        novelty=1 - nearest_identity,
        diversity=sequence_pair_diversity(sequences, rng),
        gap_fraction=gap_fraction,
        valid_fraction=mean(valid_residue_fraction, sequences),
    )
end

function profile_hmm_family_analysis(spec)
    char_mat, names, auxiliary = canonical_load_alignment(spec, joinpath(CODE_DIR, "data"))
    group_A, group_B, marker_pos, marker_residues =
        canonical_split(spec, char_mat, names, auxiliary)
    return (
        spec=spec,
        char_mat=char_mat,
        group_A=group_A,
        group_B=group_B,
        marker_pos=marker_pos,
        marker_residues=marker_residues,
    )
end

function write_kunitz_sa_rho500_reference(analysis)
    K, L = size(analysis.char_mat)
    X, pca_model, _, _ = build_memory_matrix(analysis.char_mat; pratio=0.95)
    weights = multiplicity_vector(K, analysis.group_A; ρ=500.0)
    beta = all_memory_onset(X, weights; n_betas=50)
    seed = replicate_base_seed(
        CANONICAL_SA_SEED_ORIGIN,
        condition_block_index(
            (1, length(PROFILE_HMM_RHO_GRID), 1),
            (length(CANONICAL_FAMILIES), length(PROFILE_HMM_RHO_GRID),
             PROFILE_HMM_N_REPS),
        ),
    )

    generated = String[]
    d = size(X, 1)
    for chain in 1:CANONICAL_SA_N_CHAINS
        chain_rng = MersenneTwister(seed + chain)
        memory_index = mod1(chain, K)
        xi0 = X[:, memory_index] .+ 0.01 .* randn(chain_rng, d)
        sampled = weighted_sample(X, xi0, CANONICAL_SA_N_STEPS, weights;
                                  β=beta, α=0.01, rng=chain_rng)
        for t in CANONICAL_SA_BURN_IN:CANONICAL_SA_THIN:CANONICAL_SA_N_STEPS
            push!(generated, decode_sample(sampled.Ξ[t + 1, :], pca_model, L))
        end
    end
    length(generated) == PROFILE_HMM_N_EMIT || error(
        "Canonical SA reference returned $(length(generated)) sequences")

    observed = mean(s -> s[analysis.marker_pos] in analysis.marker_residues, generated)
    canonical_raw = CSV.read(
        joinpath(CODE_DIR, "data", "kunitz", "multiplicity_sweep_raw_replicates.csv"),
        DataFrame,
    )
    expected = only(canonical_raw.f_obs[
        (canonical_raw.rho .== 500.0) .& (canonical_raw.replicate .== 1)])
    isapprox(observed, expected; atol=1e-12) || error(
        "Reproduced Kunitz SA marker fraction $observed does not match canonical $expected")

    fasta_path = joinpath(PROFILE_HMM_DATA_DIR, "kunitz_sa_rho500_rep1.fasta")
    open(fasta_path, "w") do io
        for (i, sequence) in enumerate(generated)
            println(io, ">sa_multiplicity_rho500_$(i)")
            println(io, replace(sequence, "-" => ""))
        end
    end
    return fasta_path
end

function run_profile_hmm_benchmark()
    require_hmmer()
    raw = DataFrame(
        family=String[], condition=String[], rho=Union{Missing, Float64}[],
        replicate=Int[], n_emitted=Int[], marker_fraction=Float64[],
        kl_to_full=Float64[], kl_to_designated=Float64[], novelty=Float64[],
        diversity=Float64[], gap_fraction=Float64[], valid_fraction=Float64[],
    )

    mkpath(PROFILE_HMM_DATA_DIR)
    mktempdir() do temporary_dir
        for (family_index, spec) in enumerate(CANONICAL_FAMILIES)
            analysis = profile_hmm_family_analysis(spec)
            K, L = size(analysis.char_mat)

            conditions = Tuple{String, Union{Missing, Float64}, Matrix{Char}, Vector{Int}}[]
            for rho in PROFILE_HMM_RHO_GRID
                push!(conditions,
                      ("multiplicity", rho, analysis.char_mat, analysis.group_A))
            end
            push!(conditions,
                  ("designated_subset", missing,
                   analysis.char_mat[analysis.group_A, :],
                   collect(1:length(analysis.group_A))))

            for (condition_index, (condition, rho, model_alignment, model_group_A)) in
                    enumerate(conditions)
                rho_for_model = ismissing(rho) ? 1.0 : rho
                stem = "$(spec.slug)_$(condition)_$(ismissing(rho) ? "hard" : Int(rho))"
                msa_path = joinpath(temporary_dir, stem * ".sto")
                hmm_path = joinpath(temporary_dir, stem * ".hmm")
                write_weighted_stockholm(msa_path, model_alignment, model_group_A;
                                         rho=rho_for_model)
                build_profile_hmm(msa_path, hmm_path, stem)

                @info "Profile HMM: $(spec.family), $condition, rho=$rho"
                for replicate in 1:PROFILE_HMM_N_REPS
                    seed = replicate_base_seed(
                        PROFILE_HMM_SEED_ORIGIN,
                        condition_block_index(
                            (family_index, condition_index, replicate),
                            (length(CANONICAL_FAMILIES), length(conditions),
                             PROFILE_HMM_N_REPS),
                        ),
                    )
                    output_path = joinpath(temporary_dir, "$(stem)_rep$(replicate).sto")
                    sequences = emit_profile_sequences(
                        hmm_path, output_path, PROFILE_HMM_N_EMIT, seed, L)
                    metrics = profile_hmm_metrics(
                        sequences, analysis, MersenneTwister(seed))
                    push!(raw, (
                        spec.family, condition, rho, replicate, length(sequences),
                        metrics.marker_fraction, metrics.kl_to_full,
                        metrics.kl_to_designated, metrics.novelty, metrics.diversity,
                        metrics.gap_fraction, metrics.valid_fraction,
                    ))

                    if spec.family == "Kunitz" && condition == "multiplicity" &&
                            rho == 500 && replicate == 1
                        fasta_path = joinpath(PROFILE_HMM_DATA_DIR,
                                              "kunitz_rho500_rep1.fasta")
                        open(fasta_path, "w") do io
                            for (i, sequence) in enumerate(sequences)
                                println(io, ">weighted_profile_hmm_$(i)")
                                println(io, replace(sequence, "-" => ""))
                            end
                        end
                    end
                end
            end

            spec.family == "Kunitz" && write_kunitz_sa_rho500_reference(analysis)
        end
    end

    aggregate = combine(groupby(raw, [:family, :condition, :rho]),
        :n_emitted => first => :n_emitted,
        :marker_fraction => mean => :marker_fraction_mean,
        :marker_fraction => std => :marker_fraction_std,
        :kl_to_full => mean => :kl_to_full_mean,
        :kl_to_full => std => :kl_to_full_std,
        :kl_to_designated => mean => :kl_to_designated_mean,
        :kl_to_designated => std => :kl_to_designated_std,
        :novelty => mean => :novelty_mean,
        :novelty => std => :novelty_std,
        :diversity => mean => :diversity_mean,
        :diversity => std => :diversity_std,
        :gap_fraction => mean => :gap_fraction_mean,
        :gap_fraction => std => :gap_fraction_std,
        :valid_fraction => mean => :valid_fraction_mean,
        :valid_fraction => std => :valid_fraction_std,
    )

    atomic_csv_write(joinpath(PROFILE_HMM_DATA_DIR, "raw_replicates.csv"), raw)
    atomic_csv_write(joinpath(PROFILE_HMM_DATA_DIR, "aggregated.csv"), aggregate)
    execution = DataFrame(
        parameter=["families", "rho_grid", "n_replicates", "sequences_per_replicate",
                   "hmmbuild_options", "hmmemit_options", "seed_origin"],
        value=[join(getfield.(CANONICAL_FAMILIES, :family), ";"),
               join(Int.(PROFILE_HMM_RHO_GRID), ";"), string(PROFILE_HMM_N_REPS),
               string(PROFILE_HMM_N_EMIT),
               "--amino --symfrac 0.0 --wgiven --enone",
               "-a", string(PROFILE_HMM_SEED_ORIGIN)],
    )
    atomic_csv_write(joinpath(PROFILE_HMM_DATA_DIR, "execution.csv"), execution)
    return raw, aggregate
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    raw, aggregate = run_profile_hmm_benchmark()
    @info "Profile-HMM benchmark complete" raw_rows=nrow(raw) aggregate_rows=nrow(aggregate)
end
