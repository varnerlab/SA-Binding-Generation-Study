# ══════════════════════════════════════════════════════════════════════════════
# Seed allocation for replicated experiments
# ══════════════════════════════════════════════════════════════════════════════
#
# Every replicated generator in this repository seeds its chains by offsetting a per
# replicate base seed: chain `c` uses `MersenneTwister(base + c)`. Replicate base seeds
# must therefore be spaced by more than the chain count, or adjacent replicates reuse
# almost all of their random streams and are not independent.
#
# The original allocation spaced replicate bases by 1. With 20 chains and 5 replicates a
# canonical multiplicity condition performed 100 chain runs but drew on only 24 distinct
# streams, adjacent replicates sharing 19 of 20 chains, and the whole six-family sweep used
# 59 distinct streams for 800 chain runs. Replicate standard deviations computed that way
# measure almost nothing. `replicate_base_seed` replaces that with disjoint blocks.

"""
    SEED_BLOCK

Spacing between consecutive replicate base seeds. Must exceed the largest chain count used
by any replicated experiment here, currently 30 in the Kunitz replicate script.
"""
const SEED_BLOCK = 1_000

"""
    replicate_base_seed(origin, block_index; block=SEED_BLOCK) -> Int

Base seed for the replicate identified by the zero-based `block_index`, inside an
experiment whose seed space starts at `origin`. Chains then use `base + 1 ... base +
n_chains`, so as long as `n_chains < block` no two replicates can share a stream.

`block_index` must be a stable function of the condition and the replicate, so that the
same run reproduces. Callers build it by flattening their loop indices; see
`condition_block_index`.

Blocks are spaced `block` apart rather than exactly `n_chains` apart so that raising a
chain count later cannot silently reintroduce overlap.
"""
function replicate_base_seed(origin::Integer, block_index::Integer;
                             block::Integer=SEED_BLOCK)
    block_index >= 0 || throw(ArgumentError("block_index must be non-negative"))
    block > 0 || throw(ArgumentError("block must be positive"))
    return Int(origin) + Int(block_index) * Int(block)
end

"""
    condition_block_index(indices, extents) -> Int

Flatten one-based loop `indices` into a single zero-based block index, given the `extents`
of each loop. Row-major, last index varying fastest, so the replicate index belongs last.

Use it to turn nested experiment loops into distinct seed blocks. For the canonical sweep,
`condition_block_index((family, rho, rep), (6, 8, 5))`.
"""
function condition_block_index(indices::Tuple, extents::Tuple)
    length(indices) == length(extents) ||
        throw(DimensionMismatch("indices and extents must have equal length"))
    flat = 0
    for (i, n) in zip(indices, extents)
        (1 <= i <= n) || throw(ArgumentError("index $i out of range 1:$n"))
        flat = flat * n + (i - 1)
    end
    return flat
end

"""
    chain_seed_range(base, n_chains) -> UnitRange{Int}

The chain seeds a replicate with base seed `base` will consume. Provided so tests can
assert that the seed blocks of a whole experiment are pairwise disjoint rather than
re-deriving the offset convention.
"""
chain_seed_range(base::Integer, n_chains::Integer) = (Int(base) + 1):(Int(base) + Int(n_chains))
