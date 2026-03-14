# ──────────────────────────────────────────────────────────────────────────────
# Binding.jl — Extensions for binder-conditioned sequence generation
#
# Implements four approaches from binding.md:
#   1. Curated memory matrix (training-free, immediate)
#   2. Biased energy landscape (E_Hopfield + λ E_bind)
#   3. Post-hoc filtering (generate pool → score → rank)
#   4. Interface-aware PCA encoding (upweight interface positions)
# ──────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# Approach 1: Curated Memory Matrix
# ══════════════════════════════════════════════════════════════════════════════
# This is conceptually simple: build_memory_matrix from a curated subset.
# The function below wraps the standard pipeline with subset selection.

"""
    build_binder_memory(char_mat, names, binder_names; pratio=0.95)

Build a memory matrix using only the sequences in `binder_names`.
This is Approach 1: curate the memory matrix to known binders so SA
generates sequences that inherit binder characteristics.

Returns: (X̂, pca_model, L, d_full, binder_char_mat, binder_indices)
"""
function build_binder_memory(char_mat::Matrix{Char}, names::Vector{String},
                              binder_names::Vector{String}; pratio::Float64=0.95)

    binder_idx = findall(n -> n in binder_names, names)
    if isempty(binder_idx)
        error("No matching binder names found in alignment")
    end
    binder_char_mat = char_mat[binder_idx, :]
    @info "  Curated memory: $(length(binder_idx)) binders from $(size(char_mat, 1)) total sequences"

    X̂, pca_model, L, d_full = build_memory_matrix(binder_char_mat; pratio=pratio)
    return (X̂=X̂, pca_model=pca_model, L=L, d_full=d_full,
            binder_char_mat=binder_char_mat, binder_indices=binder_idx)
end

"""
    build_binder_memory(char_mat, binder_indices; pratio=0.95)

Build a memory matrix using integer indices to select binder sequences.
"""
function build_binder_memory(char_mat::Matrix{Char}, binder_indices::Vector{Int};
                              pratio::Float64=0.95)
    binder_char_mat = char_mat[binder_indices, :]
    @info "  Curated memory: $(length(binder_indices)) binders from $(size(char_mat, 1)) total sequences"

    X̂, pca_model, L, d_full = build_memory_matrix(binder_char_mat; pratio=pratio)
    return (X̂=X̂, pca_model=pca_model, L=L, d_full=d_full,
            binder_char_mat=binder_char_mat, binder_indices=binder_indices)
end

# ══════════════════════════════════════════════════════════════════════════════
# Approach 2: Biased Energy Landscape
# ══════════════════════════════════════════════════════════════════════════════
# E_total(ξ) = E_Hopfield(ξ) + λ · E_bind(ξ)
# where E_bind scores agreement with an interface profile.
#
# The interface profile is a (N_AA × L) matrix of preferred amino acid
# frequencies at binding positions. Non-interface positions have uniform
# distributions, contributing zero bias.

"""
    InterfaceProfile

Holds the binding interface profile for biased energy scoring.

# Fields
- `profile::Matrix{Float64}`: (N_AA × L) log-odds matrix at each position.
  Positive values = preferred at interface, zero = no preference (non-interface).
- `interface_positions::Vector{Int}`: Positions that are part of the binding interface.
- `pca_model`: PCA model for converting between PCA space and one-hot space.
- `L::Int`: Alignment length.
"""
struct InterfaceProfile
    profile::Matrix{Float64}     # N_AA × L log-odds matrix
    interface_positions::Vector{Int}
    pca_model                     # PCA model for coordinate transforms
    L::Int
end

"""
    build_interface_profile(binder_seqs, all_seqs, interface_positions, pca_model, L)

Build an interface profile from known binder sequences. At interface positions,
compute log-odds of amino acid frequencies in binders vs. the full family.
Non-interface positions get zero log-odds (no bias).

# Arguments
- `binder_seqs::Vector{String}`: Known binder sequences
- `all_seqs::Vector{String}`: Full family sequences (background)
- `interface_positions::Vector{Int}`: Which alignment positions contact the target
- `pca_model`: Fitted PCA model
- `L::Int`: Alignment length
"""
function build_interface_profile(binder_seqs::Vector{String}, all_seqs::Vector{String},
                                  interface_positions::Vector{Int}, pca_model, L::Int)

    # compute per-position AA frequencies for binders and background
    freq_binder = aa_freq_matrix(binder_seqs, L)    # N_AA × L
    freq_all = aa_freq_matrix(all_seqs, L)           # N_AA × L

    # pseudo-count to avoid log(0)
    ε = 1.0 / length(all_seqs)

    # log-odds at interface positions, zero elsewhere
    profile = zeros(N_AA, L)
    for pos in interface_positions
        if pos > L
            @warn "Interface position $pos exceeds alignment length $L, skipping"
            continue
        end
        for aa in 1:N_AA
            p_bind = freq_binder[aa, pos] + ε
            p_bg = freq_all[aa, pos] + ε
            profile[aa, pos] = log(p_bind / p_bg)
        end
    end

    @info "  Interface profile: $(length(interface_positions)) positions, L=$L"
    return InterfaceProfile(profile, interface_positions, pca_model, L)
end

"""
    binding_energy(ξ_pca, iface::InterfaceProfile) -> Float64

Compute binding proxy energy in PCA space. Maps ξ back to one-hot space,
then scores agreement with interface profile (lower = better binder).

E_bind(ξ) = -Σ_pos Σ_aa  profile[aa, pos] * x_onehot[aa, pos]

where x_onehot is the soft (continuous) one-hot reconstruction from PCA.
"""
function binding_energy(ξ_pca::Vector{Float64}, iface::InterfaceProfile)::Float64
    # reconstruct to one-hot space
    x_onehot = vec(MultivariateStats.reconstruct(iface.pca_model, ξ_pca))

    # score only at interface positions
    E = 0.0
    for pos in iface.interface_positions
        pos > iface.L && continue
        start_idx = (pos - 1) * N_AA + 1
        for aa in 1:N_AA
            E -= iface.profile[aa, pos] * x_onehot[start_idx + aa - 1]
        end
    end
    return E
end

"""
    binding_energy_gradient(ξ_pca, iface::InterfaceProfile) -> Vector{Float64}

Gradient of binding energy with respect to ξ in PCA space.

Since E_bind = -profile^T · reconstruct(ξ) and reconstruct is linear (PCA),
∇_ξ E_bind = -P^T · profile_vec, where P is the PCA projection matrix.
This gradient is constant — it's a linear bias in PCA space.
"""
function binding_energy_gradient(iface::InterfaceProfile)
    # build the profile as a vector in one-hot space
    d_full = N_AA * iface.L
    profile_vec = zeros(d_full)
    for pos in iface.interface_positions
        pos > iface.L && continue
        start_idx = (pos - 1) * N_AA + 1
        for aa in 1:N_AA
            profile_vec[start_idx + aa - 1] = iface.profile[aa, pos]
        end
    end

    # gradient in PCA space: -P^T * profile_vec
    # P = projection(pca_model) maps d_full → d_pca
    P = MultivariateStats.projection(iface.pca_model)  # d_full × d_pca
    return -(P' * profile_vec)  # d_pca vector
end

"""
    biased_sample(X, ξ₀, T, iface; β=1.0, α=0.1, λ=0.1, seed=nothing)

Modified Langevin sampler with binding energy bias (Approach 2).

E_total(ξ) = E_Hopfield(ξ) + λ · E_bind(ξ)

The update becomes:
  ξ_{t+1} = (1-α)ξ_t + α·X·softmax(β·X^T·ξ_t) - α·λ·∇E_bind(ξ_t) + noise

Since ∇E_bind is constant (linear profile), this pre-computes the bias
once and adds it at each step.

# Arguments
- `X::Matrix{Float64}`: Memory matrix (d × K)
- `ξ₀::Vector{Float64}`: Initial state
- `T::Int`: Number of iterations
- `iface::InterfaceProfile`: Interface profile for binding bias

# Keyword Arguments
- `β::Float64`: Inverse temperature
- `α::Float64`: Step size
- `λ::Float64`: Binding energy weight (higher = stronger bias toward binders)
- `seed`: Optional random seed
"""
function biased_sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int,
                        iface::InterfaceProfile;
                        β::Float64=1.0, α::Float64=0.1, λ::Float64=0.1,
                        seed::Union{Int, Nothing}=nothing)

    d = size(X, 1)
    length(ξ₀) == d || throw(DimensionMismatch(
        "Initial state has length $(length(ξ₀)) but X has $d rows"))
    T > 0   || throw(ArgumentError("T must be positive"))
    β > 0   || throw(ArgumentError("β must be positive"))
    0 < α < 1 || throw(ArgumentError("α must be in (0,1)"))

    if seed !== nothing
        Random.seed!(seed)
    end

    Ξ = Matrix{Float64}(undef, T + 1, d)
    Ξ[1, :] .= ξ₀

    noise_scale = sqrt(2.0 * α / β)

    # pre-compute the constant binding gradient bias
    ∇E_bind = binding_energy_gradient(iface)  # constant since profile is linear
    bias_step = α * λ .* ∇E_bind

    ξ = copy(ξ₀)
    for t in 1:T
        # attention weights
        logits = β .* (X' * ξ)
        w = softmax(logits)

        # noise
        ε = randn(d)

        # biased Langevin update
        ξ .= (1.0 - α) .* ξ .+ α .* (X * w) .- bias_step .+ noise_scale .* ε

        Ξ[t + 1, :] .= ξ
    end

    t_vec = collect(0:T)
    return (t = t_vec, Ξ = Ξ)
end

# ══════════════════════════════════════════════════════════════════════════════
# Approach 3: Post-hoc Filtering
# ══════════════════════════════════════════════════════════════════════════════

"""
    interface_conservation_score(seq, iface::InterfaceProfile) -> Float64

Score a generated sequence by how well it matches the interface profile.
Higher score = better agreement with binder preferences at interface positions.
"""
function interface_conservation_score(seq::String, iface::InterfaceProfile)::Float64
    score = 0.0
    n_scored = 0
    for pos in iface.interface_positions
        pos > min(iface.L, length(seq)) && continue
        aa = seq[pos]
        idx = get(AA_TO_IDX, aa, 0)
        if idx > 0
            score += iface.profile[idx, pos]
            n_scored += 1
        end
    end
    return n_scored > 0 ? score / n_scored : 0.0
end

"""
    charge_complementarity_score(seq, target_charge_positions; positive_target=true) -> Float64

Simple charge complementarity score. For each target contact position,
check if the generated sequence has the complementary charge.

# Arguments
- `seq::String`: Generated sequence
- `target_charge_positions::Vector{Int}`: Positions that contact charged target residues
- `positive_target::Bool`: If true, target presents positive charges (we want negative);
  if false, target presents negative charges (we want positive)
"""
function charge_complementarity_score(seq::String, target_charge_positions::Vector{Int};
                                       positive_target::Bool=true)::Float64
    positive_aa = Set(['R', 'K', 'H'])
    negative_aa = Set(['D', 'E'])
    desired = positive_target ? negative_aa : positive_aa

    n_match = 0
    n_scored = 0
    for pos in target_charge_positions
        pos > length(seq) && continue
        n_scored += 1
        if seq[pos] in desired
            n_match += 1
        end
    end
    return n_scored > 0 ? n_match / n_scored : 0.0
end

"""
    hydrophobic_patch_score(seq, interface_positions) -> Float64

Score the fraction of hydrophobic residues at interface positions.
Many protein-protein interfaces are enriched in hydrophobic contacts.
"""
function hydrophobic_patch_score(seq::String, interface_positions::Vector{Int})::Float64
    hydrophobic = Set(['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'])
    n_hydro = 0
    n_scored = 0
    for pos in interface_positions
        pos > length(seq) && continue
        aa = seq[pos]
        get(AA_TO_IDX, aa, 0) == 0 && continue  # skip gaps
        n_scored += 1
        if aa in hydrophobic
            n_hydro += 1
        end
    end
    return n_scored > 0 ? n_hydro / n_scored : 0.0
end

"""
    filter_and_rank(gen_seqs, gen_pca_vecs, iface;
                    stored_seqs, X̂, β,
                    top_k=50, min_novelty=0.05, min_valid_frac=0.9) -> DataFrame

Post-hoc filtering and ranking of generated sequences (Approach 3).

Generates a comprehensive score table and returns the top-k candidates.
Filters by minimum novelty and valid residue fraction, then ranks by
a composite score combining interface conservation, novelty, and family quality.
"""
function filter_and_rank(gen_seqs::Vector{String}, gen_pca_vecs::Vector{Vector{Float64}},
                          iface::InterfaceProfile;
                          stored_seqs::Vector{String},
                          X̂::Matrix{Float64}, β::Float64,
                          top_k::Int=50, min_novelty::Float64=0.05,
                          min_valid_frac::Float64=0.9)

    n = length(gen_seqs)
    @info "  Scoring $n generated sequences for post-hoc ranking..."

    # compute all scores
    iface_scores = [interface_conservation_score(s, iface) for s in gen_seqs]
    novelties = [sample_novelty(v, X̂) for v in gen_pca_vecs]
    seq_ids = [nearest_sequence_identity(s, stored_seqs) for s in gen_seqs]
    valid_fracs = [valid_residue_fraction(s) for s in gen_seqs]
    energies = [hopfield_energy(v, X̂, β) for v in gen_pca_vecs]

    # build DataFrame
    df = DataFrame(
        rank = 1:n,
        sequence = gen_seqs,
        interface_score = iface_scores,
        novelty = novelties,
        nearest_seqid = seq_ids,
        valid_frac = valid_fracs,
        energy = energies,
    )

    # filter
    df = filter(row -> row.valid_frac >= min_valid_frac && row.novelty >= min_novelty, df)

    # composite score: high interface agreement + moderate novelty + low energy
    # normalize each component to [0,1]
    if nrow(df) > 0
        iface_range = maximum(df.interface_score) - minimum(df.interface_score)
        energy_range = maximum(df.energy) - minimum(df.energy)

        df.composite_score = map(eachrow(df)) do row
            iface_norm = iface_range > 0 ? (row.interface_score - minimum(df.interface_score)) / iface_range : 0.5
            energy_norm = energy_range > 0 ? 1.0 - (row.energy - minimum(df.energy)) / energy_range : 0.5
            0.5 * iface_norm + 0.3 * row.novelty + 0.2 * energy_norm
        end

        sort!(df, :composite_score; rev=true)
        df.rank = 1:nrow(df)
    end

    n_returned = min(top_k, nrow(df))
    @info "  After filtering: $(nrow(df)) candidates, returning top $n_returned"

    return first(df, n_returned)
end

# ══════════════════════════════════════════════════════════════════════════════
# Approach 4: Interface-Aware PCA Encoding
# ══════════════════════════════════════════════════════════════════════════════

"""
    interface_weighted_onehot(char_mat, interface_positions; weight=3.0)

One-hot encode with upweighted interface positions. Positions in
`interface_positions` get their one-hot values multiplied by `weight`,
amplifying their influence on the PCA decomposition.

This causes the principal components to preferentially capture variation
at interface positions, making SA sampling more sensitive to interface
residue identity.
"""
function interface_weighted_onehot(char_mat::Matrix{Char},
                                    interface_positions::Vector{Int};
                                    weight::Float64=3.0)
    K, L = size(char_mat)
    d_full = N_AA * L
    X = zeros(Float64, d_full, K)

    for k in 1:K
        for pos in 1:L
            aa = char_mat[k, pos]
            idx = get(AA_TO_IDX, aa, 0)
            if idx > 0
                w = pos in interface_positions ? weight : 1.0
                X[(pos-1)*N_AA + idx, k] = w
            end
        end
    end

    return X
end

"""
    build_interface_weighted_memory(char_mat, interface_positions;
                                    pratio=0.95, weight=3.0)

Build memory matrix with interface-aware PCA (Approach 4).
Same pipeline as build_memory_matrix, but uses weighted one-hot encoding
that amplifies interface positions.

Returns: (X̂, pca_model, L, d_full, interface_positions, weight)
"""
function build_interface_weighted_memory(char_mat::Matrix{Char},
                                          interface_positions::Vector{Int};
                                          pratio::Float64=0.95,
                                          weight::Float64=3.0)
    K, L = size(char_mat)

    # weighted one-hot encode
    X_onehot = interface_weighted_onehot(char_mat, interface_positions; weight=weight)
    d_full = size(X_onehot, 1)
    @info "  Interface-weighted one-hot: $d_full × $K (weight=$weight at $(length(interface_positions)) positions)"

    # PCA
    pca_model = MultivariateStats.fit(PCA, X_onehot; pratio=pratio)
    d_pca = MultivariateStats.outdim(pca_model)
    Z = MultivariateStats.transform(pca_model, X_onehot)
    var_retained = round(100 * sum(MultivariateStats.principalvars(pca_model)) /
                         MultivariateStats.tvar(pca_model), digits=1)
    @info "  PCA: $d_full → $d_pca dimensions ($var_retained% variance)"

    # normalize to unit norm
    X̂ = copy(Z)
    for k in 1:K
        nk = norm(X̂[:, k])
        X̂[:, k] ./= (nk + 1e-12)
    end
    @info "  Interface-weighted memory matrix X̂: $d_pca × $K (unit-norm columns)"

    return (X̂=X̂, pca_model=pca_model, L=L, d_full=d_full,
            interface_positions=interface_positions, weight=weight)
end

"""
    decode_weighted_sample(ξ_pca, pca_model, L, interface_positions; weight=3.0)

Decode a PCA-space vector back to an amino acid sequence, undoing the
interface weighting before argmax decoding.
"""
function decode_weighted_sample(ξ_pca::Vector{Float64}, pca_model, L::Int,
                                 interface_positions::Vector{Int};
                                 weight::Float64=3.0)
    x_onehot = vec(MultivariateStats.reconstruct(pca_model, ξ_pca))

    # undo the weighting before decoding
    for pos in interface_positions
        pos > L && continue
        start_idx = (pos - 1) * N_AA + 1
        for aa in 1:N_AA
            x_onehot[start_idx + aa - 1] /= weight
        end
    end

    return decode_onehot(x_onehot, L)
end

# ══════════════════════════════════════════════════════════════════════════════
# Multiplicity-Weighted Hopfield Energy
# ══════════════════════════════════════════════════════════════════════════════
#
# Central theoretical result: pattern multiplicity is a principled conditioning
# mechanism for the modern Hopfield energy. Assigning multiplicity r_k > 0 to
# each stored pattern m_k defines a *tilted* energy landscape:
#
#   E_r(ξ) = ½‖ξ‖² - (1/β) log Σ_k r_k exp(β m_k^T ξ)
#
# The corresponding score function ∇_ξ log p_r(ξ) yields the Langevin update:
#
#   ξ_{t+1} = (1-α) ξ_t + α X softmax(β X^T ξ_t + log r) + √(2α/β) ε_t
#
# Key properties:
#   1. The memory matrix X stays compact (d × K) — no column duplication
#   2. Conditioning is achieved through a log-multiplicity bias on the logits
#   3. The stationary distribution p_r(ξ) ∝ exp(-β E_r(ξ)) is known in closed form
#   4. At r_k = 1 for all k, we recover standard SA
#   5. As r_binder / r_nonbinder → ∞, we converge to hard curation
#
# The multiplicity ratio ρ = r_binder / r_nonbinder parameterizes a continuous
# family of distributions that interpolate between "generate any family member"
# (ρ = 1) and "generate only binders" (ρ → ∞). The effective number of patterns:
#
#   K_eff(r) = (Σ_k r_k)² / Σ_k r_k²
#
# controls the phase transition: β*(r) depends on K_eff(r) and d.
# ══════════════════════════════════════════════════════════════════════════════

"""
    weighted_sample(X, ξ₀, T, pattern_weights; β=1.0, α=0.1, seed=nothing)

Langevin sampler with per-pattern weights in the softmax.

Standard SA gives each pattern equal say in the softmax. Here we add a
log-weight bias to each pattern's logit before softmax:

    a_t = softmax(β X^T ξ_t + log(w))

where w is a K-vector of positive weights. Patterns with higher weight
attract the chain more strongly. This is equivalent to the Boltzmann
distribution over a "weighted" Hopfield energy, and it is the exact
score function for p(ξ) ∝ exp(-β E_w(ξ)) where:

    E_w(ξ) = ½‖ξ‖² - (1/β) log Σ_k  w_k exp(β m_k^T ξ)

# Arguments
- `X::Matrix{Float64}`: Memory matrix (d × K)
- `ξ₀::Vector{Float64}`: Initial state
- `T::Int`: Number of iterations
- `pattern_weights::Vector{Float64}`: Per-pattern weights (length K, positive)

# Keyword Arguments
- `β`, `α`, `seed`: Same as `sample()`
"""
function weighted_sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int,
                          pattern_weights::Vector{Float64};
                          β::Float64=1.0, α::Float64=0.1,
                          seed::Union{Int, Nothing}=nothing)

    d, K = size(X)
    length(ξ₀) == d || throw(DimensionMismatch(
        "Initial state has length $(length(ξ₀)) but X has $d rows"))
    length(pattern_weights) == K || throw(DimensionMismatch(
        "pattern_weights has length $(length(pattern_weights)) but X has $K columns"))
    T > 0   || throw(ArgumentError("T must be positive"))
    β > 0   || throw(ArgumentError("β must be positive"))
    0 < α < 1 || throw(ArgumentError("α must be in (0,1)"))
    all(w -> w > 0, pattern_weights) || throw(ArgumentError("All weights must be positive"))

    if seed !== nothing
        Random.seed!(seed)
    end

    Ξ = Matrix{Float64}(undef, T + 1, d)
    Ξ[1, :] .= ξ₀

    noise_scale = sqrt(2.0 * α / β)
    log_weights = log.(pattern_weights)  # pre-compute once

    ξ = copy(ξ₀)
    for t in 1:T
        # weighted attention: softmax(β X^T ξ + log w)
        logits = β .* (X' * ξ) .+ log_weights
        w = softmax(logits)

        ε = randn(d)
        ξ .= (1.0 - α) .* ξ .+ α .* (X * w) .+ noise_scale .* ε
        Ξ[t + 1, :] .= ξ
    end

    return (t = collect(0:T), Ξ = Ξ)
end

"""
    build_weighted_memory(char_mat, binder_indices; pratio=0.95, binder_weight=5.0)

Build a full-family memory matrix with per-pattern weights that upweight binders.
Returns the memory matrix, PCA model, AND a weight vector for use with weighted_sample.

This is "soft curation": every family member stays in the memory, but binders
get higher attention weight. At high binder_weight, this converges to hard curation.
At binder_weight=1.0, this is standard SA.
"""
function build_weighted_memory(char_mat::Matrix{Char}, binder_indices::Vector{Int};
                                pratio::Float64=0.95, binder_weight::Float64=5.0)
    K, L = size(char_mat)
    X̂, pca_model, L_out, d_full = build_memory_matrix(char_mat; pratio=pratio)

    # build weight vector: binders get higher weight
    weights = ones(K)
    for idx in binder_indices
        weights[idx] = binder_weight
    end

    n_binders = length(binder_indices)
    @info "  Weighted memory: $K patterns, $n_binders binders at weight=$binder_weight"
    return (X̂=X̂, pca_model=pca_model, L=L_out, d_full=d_full, weights=weights)
end

# ──────────────────────────────────────────────────────────────────────────────
# Core theoretical functions for multiplicity-weighted Hopfield energy
# ──────────────────────────────────────────────────────────────────────────────

"""
    multiplicity_vector(K, binder_indices; ρ=10.0)

Construct a multiplicity vector r for K patterns where binder patterns get
multiplicity ρ and non-binders get multiplicity 1.

The multiplicity ratio ρ = r_binder / r_nonbinder is the single parameter
that controls the strength of conditioning. This parameterization makes the
theoretical analysis clean:
  - ρ = 1: standard SA (uniform multiplicity)
  - ρ → ∞: hard curation (only binders contribute)
  - intermediate ρ: soft conditioning with tunable strength

The effective binder fraction in the stationary distribution is:
  f_eff = K_b · ρ / (K_b · ρ + K_nb)
where K_b and K_nb are the number of binder and non-binder patterns.
"""
function multiplicity_vector(K::Int, binder_indices::Vector{Int}; ρ::Float64=10.0)
    ρ > 0 || throw(ArgumentError("Multiplicity ratio ρ must be positive"))
    r = ones(K)
    for idx in binder_indices
        r[idx] = ρ
    end
    return r
end

"""
    effective_binder_fraction(r, binder_indices) -> Float64

Compute the effective binder fraction under multiplicity weights r:
  f_eff = Σ_{k ∈ binders} r_k / Σ_k r_k

This is the fraction of the softmax probability mass that binder patterns
would receive if all patterns had equal similarity to the state ξ.
"""
function effective_binder_fraction(r::Vector{Float64}, binder_indices::Vector{Int})
    total = sum(r)
    binder_total = sum(r[idx] for idx in binder_indices)
    return binder_total / total
end

"""
    effective_num_patterns(r) -> Float64

Compute the effective number of patterns under multiplicity weights:
  K_eff(r) = (Σ_k r_k)² / Σ_k r_k²

This is the participation ratio of the multiplicity vector — a measure of
how many patterns effectively contribute to the energy landscape. When all
r_k are equal, K_eff = K. When one r_k dominates, K_eff → 1.

The effective number of patterns controls the phase transition:
β* scales with √d and depends on K_eff(r) rather than K.
"""
function effective_num_patterns(r::Vector{Float64})
    return sum(r)^2 / sum(r .^ 2)
end

"""
    weighted_hopfield_energy(ξ, X, β, r) -> Float64

Compute the multiplicity-weighted Hopfield energy:
  E_r(ξ) = ½‖ξ‖² - (1/β) log Σ_k r_k exp(β m_k^T ξ)

Uses the log-sum-exp trick with log-multiplicity offsets.
"""
function weighted_hopfield_energy(ξ::Vector{Float64}, X::Matrix{Float64},
                                   β::Float64, r::Vector{Float64})::Float64
    logits = β .* (X' * ξ) .+ log.(r)
    logits_max = maximum(logits)
    lse = logits_max + log(sum(exp.(logits .- logits_max)))
    return 0.5 * dot(ξ, ξ) - lse / β
end

"""
    weighted_attention_entropy(ξ, X, β, r) -> Float64

Shannon entropy of the multiplicity-weighted attention distribution:
  p_k = r_k exp(β m_k^T ξ) / Σ_j r_j exp(β m_j^T ξ)

This is the order parameter for the weighted system. At the phase transition,
the entropy drops sharply from H ≈ log(K_eff) to near zero.
"""
function weighted_attention_entropy(ξ::Vector{Float64}, X::Matrix{Float64},
                                     β::Float64, r::Vector{Float64})::Float64
    logits = β .* (X' * ξ) .+ log.(r)
    p = NNlib.softmax(logits)
    H = 0.0
    for pk in p
        pk > 1e-30 && (H -= pk * log(pk))
    end
    return H
end

"""
    find_weighted_entropy_inflection(X̂, r; α=0.01, n_betas=50, β_range=(0.1, 500.0))

Find the phase transition β*(r) for the multiplicity-weighted Hopfield energy.
Same algorithm as find_entropy_inflection but uses weighted attention entropy.
"""
function find_weighted_entropy_inflection(X̂::Matrix{Float64}, r::Vector{Float64};
                                           α::Float64=0.01, n_betas::Int=50,
                                           β_range::Tuple{Float64,Float64}=(0.1, 500.0))
    d, K = size(X̂)
    βs = 10 .^ range(log10(β_range[1]), log10(β_range[2]), length=n_betas)

    n_probes = min(K, 20)
    Hs = zeros(n_betas)
    for (bi, β) in enumerate(βs)
        H_sum = 0.0
        for k in 1:n_probes
            H_sum += weighted_attention_entropy(X̂[:, k], X̂, β, r)
        end
        Hs[bi] = H_sum / n_probes
    end

    log_βs = log.(βs)
    dH = diff(Hs) ./ diff(log_βs)
    d2H = diff(dH) ./ diff(log_βs[1:end-1])

    inflection_idx = 1
    min_d2H = Inf
    for i in eachindex(d2H)
        if d2H[i] < min_d2H
            min_d2H = d2H[i]
            inflection_idx = i + 1
        end
    end

    β_star = βs[inflection_idx]
    K_eff = effective_num_patterns(r)

    return (β_star=β_star, K_eff=K_eff, βs=βs, Hs=Hs)
end

"""
    ρ_for_target_fraction(K_b, K_nb, f_target) -> Float64

Compute the multiplicity ratio ρ needed to achieve a target effective
binder fraction f_target, given K_b binder and K_nb non-binder patterns.

From f_eff = K_b · ρ / (K_b · ρ + K_nb), solving for ρ:
  ρ = f_target · K_nb / (K_b · (1 - f_target))
"""
function ρ_for_target_fraction(K_b::Int, K_nb::Int, f_target::Float64)
    0 < f_target < 1 || throw(ArgumentError("f_target must be in (0, 1)"))
    return f_target * K_nb / (K_b * (1 - f_target))
end

"""
    augment_memory_interpolations(X̂, binder_indices; n_interp=50, seed=42)

Augment the memory matrix with interpolated binder patterns.

Takes pairs of binder columns from X̂, computes convex combinations
at random mixing coefficients, normalizes to unit norm, and appends
them as new columns. This "fills in" the binder subspace, increasing
the diversity of generated sequences while staying in binder territory.

Returns: (X̂_aug, aug_indices) where aug_indices marks the new columns.
"""
function augment_memory_interpolations(X̂::Matrix{Float64}, binder_indices::Vector{Int};
                                        n_interp::Int=50, seed::Int=42)
    Random.seed!(seed)
    d, K = size(X̂)
    n_binders = length(binder_indices)
    n_binders >= 2 || throw(ArgumentError("Need at least 2 binders for interpolation"))

    new_cols = Matrix{Float64}(undef, d, n_interp)
    for i in 1:n_interp
        # pick two random binders
        j1, j2 = rand(binder_indices, 2)
        while j1 == j2
            j2 = rand(binder_indices)
        end

        # random mixing coefficient (avoid extremes)
        t = 0.15 + 0.7 * rand()  # t ∈ [0.15, 0.85]
        ξ_mix = t .* X̂[:, j1] .+ (1 - t) .* X̂[:, j2]

        # normalize to unit norm
        ξ_mix ./= (norm(ξ_mix) + 1e-12)
        new_cols[:, i] = ξ_mix
    end

    X̂_aug = hcat(X̂, new_cols)
    aug_indices = collect((K + 1):(K + n_interp))

    @info "  Augmented memory: $K original + $n_interp interpolated = $(K + n_interp) total patterns"
    return (X̂_aug=X̂_aug, aug_indices=aug_indices)
end

"""
    build_multiplicity_conditioned_memory(char_mat, binder_indices; pratio=0.95,
                                           f_target=0.8)

Build a full-family memory matrix with a multiplicity vector that achieves
a target effective binder fraction f_target.

This is the proper implementation of "mixed memory": ALL patterns stay in
the memory (preserving fold constraints), but the multiplicity ratio ρ is
set so that binder patterns collectively receive f_target of the softmax
probability mass.

Returns the memory matrix, PCA model, multiplicity vector, and diagnostics.
"""
function build_multiplicity_conditioned_memory(char_mat::Matrix{Char},
                                                binder_indices::Vector{Int};
                                                pratio::Float64=0.95,
                                                f_target::Float64=0.8)
    K_total, L = size(char_mat)
    n_binders = length(binder_indices)
    n_nonbinders = K_total - n_binders

    # build full-family memory matrix
    X̂, pca_model, L_out, d_full = build_memory_matrix(char_mat; pratio=pratio)

    # compute the multiplicity ratio needed for target fraction
    ρ = ρ_for_target_fraction(n_binders, n_nonbinders, f_target)
    r = multiplicity_vector(K_total, binder_indices; ρ=ρ)

    # verify
    f_eff = effective_binder_fraction(r, binder_indices)
    K_eff = effective_num_patterns(r)
    natural_frac = n_binders / K_total

    @info "  Multiplicity-conditioned memory:"
    @info "    K=$K_total ($(n_binders) binders, $(n_nonbinders) non-binders)"
    @info "    Natural binder fraction: $(round(natural_frac, digits=3))"
    @info "    Target binder fraction:  $(round(f_target, digits=3))"
    @info "    Multiplicity ratio ρ:    $(round(ρ, digits=2))"
    @info "    Effective binder fraction: $(round(f_eff, digits=3))"
    @info "    Effective K:             $(round(K_eff, digits=1)) (of $K_total)"

    return (X̂=X̂, pca_model=pca_model, L=L_out, d_full=d_full,
            r=r, ρ=ρ, f_eff=f_eff, K_eff=K_eff,
            binder_indices=binder_indices)
end

"""
    build_consensus_seeded_memory(X̂, binder_indices; n_consensus=3)

Add consensus binder pattern(s) to the memory matrix.

Computes the mean of all binder patterns in PCA space, normalizes to
unit norm, and adds it as additional column(s). Optionally adds
slightly perturbed variants of the consensus for robustness.

The consensus acts as an "attractor" that pulls generated sequences
toward the center of the binder distribution.
"""
function build_consensus_seeded_memory(X̂::Matrix{Float64}, binder_indices::Vector{Int};
                                        n_consensus::Int=3, perturbation::Float64=0.05,
                                        seed::Int=42)
    Random.seed!(seed)
    d, K = size(X̂)

    # compute binder centroid
    binder_mean = mean(X̂[:, binder_indices], dims=2) |> vec
    binder_mean ./= (norm(binder_mean) + 1e-12)

    # create consensus columns (centroid + small perturbations)
    consensus_cols = Matrix{Float64}(undef, d, n_consensus)
    consensus_cols[:, 1] = binder_mean  # exact centroid
    for i in 2:n_consensus
        ξ = binder_mean .+ perturbation .* randn(d)
        ξ ./= (norm(ξ) + 1e-12)
        consensus_cols[:, i] = ξ
    end

    X̂_aug = hcat(X̂, consensus_cols)
    @info "  Consensus-seeded memory: $K original + $n_consensus consensus = $(K + n_consensus) total"
    return (X̂_aug=X̂_aug, consensus_indices=collect((K + 1):(K + n_consensus)))
end

"""
    generate_weighted_sequences(X̂, pca_model, L, weights; β, n_chains=30,
                                 T=5000, α=0.01, burnin=2000, thin=100, seed=42)

Generation pipeline using weighted_sample. Same interface as generate_sequences
but takes a pattern weight vector.
"""
function generate_weighted_sequences(X̂::Matrix{Float64}, pca_model, L::Int,
                                      weights::Vector{Float64};
                                      β::Float64, n_chains::Int=30, T::Int=5000,
                                      α::Float64=0.01, burnin::Int=2000, thin::Int=100,
                                      seed::Int=42)
    d, K = size(X̂)
    gen_seqs = String[]
    gen_pca = Vector{Float64}[]

    @info "Generating weighted sequences: $n_chains chains × $T steps (β=$β)"
    for chain in 1:n_chains
        k = mod1(chain, K)
        ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)

        result = weighted_sample(X̂, ξ₀, T, weights; β=β, α=α, seed=seed + chain)

        for t in burnin:thin:T
            ξ = result.Ξ[t + 1, :]
            seq = decode_sample(ξ, pca_model, L)
            push!(gen_seqs, seq)
            push!(gen_pca, ξ)
        end
    end

    @info "  Generated $(length(gen_seqs)) weighted sequences from $n_chains chains"
    return gen_seqs, gen_pca
end

# ══════════════════════════════════════════════════════════════════════════════
# Generation pipeline (wraps all approaches)
# ══════════════════════════════════════════════════════════════════════════════

"""
    generate_sequences(X̂, pca_model, L; β, n_chains=30, T=5000, α=0.01,
                       burnin=2000, thin=100, seed=42) -> (seqs, pca_vecs)

Standard SA generation pipeline. Runs n_chains independent Langevin chains
on memory matrix X̂, collects samples after burn-in with thinning.
"""
function generate_sequences(X̂::Matrix{Float64}, pca_model, L::Int;
                             β::Float64, n_chains::Int=30, T::Int=5000,
                             α::Float64=0.01, burnin::Int=2000, thin::Int=100,
                             seed::Int=42)

    d, K = size(X̂)
    gen_seqs = String[]
    gen_pca = Vector{Float64}[]

    @info "Generating sequences: $n_chains chains × $T steps (β=$β, α=$α)"
    for chain in 1:n_chains
        # initialize near a random stored pattern
        k = mod1(chain, K)
        ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)

        result = sample(X̂, ξ₀, T; β=β, α=α, seed=seed + chain)

        # collect samples after burn-in with thinning
        for t in burnin:thin:T
            ξ = result.Ξ[t + 1, :]
            seq = decode_sample(ξ, pca_model, L)
            push!(gen_seqs, seq)
            push!(gen_pca, ξ)
        end
    end

    @info "  Generated $(length(gen_seqs)) sequences from $n_chains chains"
    return gen_seqs, gen_pca
end

"""
    generate_biased_sequences(X̂, pca_model, L, iface; β, λ=0.1,
                               n_chains=30, T=5000, α=0.01,
                               burnin=2000, thin=100, seed=42) -> (seqs, pca_vecs)

Biased SA generation pipeline (Approach 2). Same as generate_sequences but
uses biased_sample with binding energy term.
"""
function generate_biased_sequences(X̂::Matrix{Float64}, pca_model, L::Int,
                                    iface::InterfaceProfile;
                                    β::Float64, λ::Float64=0.1,
                                    n_chains::Int=30, T::Int=5000,
                                    α::Float64=0.01, burnin::Int=2000,
                                    thin::Int=100, seed::Int=42)

    d, K = size(X̂)
    gen_seqs = String[]
    gen_pca = Vector{Float64}[]

    @info "Generating biased sequences: $n_chains chains × $T steps (β=$β, λ=$λ)"
    for chain in 1:n_chains
        k = mod1(chain, K)
        ξ₀ = X̂[:, k] .+ 0.01 .* randn(d)

        result = biased_sample(X̂, ξ₀, T, iface; β=β, α=α, λ=λ, seed=seed + chain)

        for t in burnin:thin:T
            ξ = result.Ξ[t + 1, :]
            seq = decode_sample(ξ, pca_model, L)
            push!(gen_seqs, seq)
            push!(gen_pca, ξ)
        end
    end

    @info "  Generated $(length(gen_seqs)) biased sequences from $n_chains chains"
    return gen_seqs, gen_pca
end
