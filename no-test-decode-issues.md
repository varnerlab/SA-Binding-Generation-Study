# Test coverage gap: the encode/decode pipeline is entirely unasserted

> **Status: RESOLVED.** Closed by commits `094a66e` (decoder pipeline tests plus the
> `sequence_identity` gap-handling fix) and `29c3f78` (contract documentation and
> additional guard assertions). The suite went from 52 to 126 tests. The final corrected
> account of the review that followed is in `remaining-issues-list.md`; the per-file
> provenance of affected legacy artifacts is in `code/data/kunitz/README.md`.
> Retained as the historical record that opened the work. Its proposed test code was not
> adopted verbatim, and its pre-correction fidelity values are superseded.

Written 2026-07-28 by Claude (Opus 5), for independent review.

Original report state: `SA-Binding-Generation-Study`, `main` at `4fe9ad9`.
Repair under review: commit `094a66e` (`fix: test decoder pipeline and correct
gap metrics`) on `main`.
Related: `docs/2026-07-28-decoder-postmortem.md` (why this matters, and a reasoning
error of mine that this gap contributed to).

## Handoff to Claude for second review

Please review commit `094a66e0f9bc360d869588e342df764bf8292ed6`, not just the
original proposal preserved below. The purpose of this review is to look for a
specific technical mistake in the repair, not to generate another plan or branch.
The implementation is already on `main`.

Please verify these claims independently:

1. The original coverage-gap diagnosis was correct, but the proposed test code was
   not implemented verbatim.
2. Treating `.`, `-`, and `~` consistently as alignment gaps is correct for the
   tracked Pfam and conotoxin alignments.
3. The expected characterization values in
   `code/test/test_decoder_characterization.jl` are independently reproducible from
   the tracked fixtures and are not circularly derived from constants in production
   code.
4. The `atol=0.002` characterization tolerance is tight enough to detect a material
   encoder/decoder change without being fragile to harmless numerical variation.
5. The seeded Kunitz rerun changes only gap-sensitive identity/novelty metrics:
   phenotype fractions, diversity, beta values, and generated sequences are
   unchanged.
6. No other manuscript number computed with nearest stored-sequence identity needs
   regeneration or correction.
7. The paper now consistently reports hard-curation novelty as 0.32 and
   stored-memory fidelity ranges as 0.730--0.943 and 0.996--1.000.

Reproduction commands:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git diff 4fe9ad9..094a66e

cd code
julia --project=. test/runtests.jl
# Expected: 121/121 tests pass.

cd ..
git diff --check
```

An optional independent Kunitz rerun is:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code
julia --project=. experiments/run_kunitz_mask_experiment.jl
```

That command overwrites tracked result artifacts. The CSVs and PNGs should reproduce
the current commit byte-for-byte. Plot PDFs embed build timestamps and can therefore
show metadata-only diffs even when the plotted data are unchanged; do not interpret
such a binary PDF diff as a scientific result change.

Files that deserve direct inspection:

- `code/src/Protein.jl`
- `code/test/test_encode_decode.jl`
- `code/test/test_decoder_characterization.jl`
- `code/data/kunitz/mask_experiment.csv`
- `code/data/kunitz/mask_betasweep.csv`
- `paper-arxiv/Paper_v1.tex`
- `paper-arxiv/sections/results.tex`
- `paper-arxiv/sections/appendix.tex`
- `docs/2026-07-28-decoder-postmortem.md`

If you find a problem, please report the exact file and line, the failing
assumption, and a minimal reproduction. In particular, distinguish:

- a defect in the implementation;
- a test-design concern;
- a stale derived artifact; and
- a disagreement about the scientific interpretation.

Do not silently rewrite results, regenerate unrelated experiments, create a new
branch, or propose a decoder change unless a concrete failing check supports it.

## Resolution and independent findings (Codex, 2026-07-28)

The central coverage diagnosis was true: the original 52-test suite did not directly
assert the encoder, decoder, or sequence-identity behavior. The proposed patch below
was not accepted verbatim. Its random PCA fixture was replaced with a deterministic
one, and the real-family characterization reads only the six alignments tracked in
this repository, so the test suite cannot silently download a newer Pfam release.

The review also found a real defect that this report missed. `clean_alignment` treated
`.`, `-`, and `~` as gaps, while `sequence_identity` and `valid_residue_fraction`
recognized only `-`. Consequently, Pfam `.` gaps were counted as residue mismatches
despite the metric docstrings saying gaps were excluded. The implementation now uses
one shared definition for all three alignment-gap characters.

Corrected, gap-aware stored-memory characterization:

| Family | Unit-normalized memory | Original PCA score |
|---|---:|---:|
| Kunitz | 0.836366 | 0.999809 |
| SH3 | 0.789073 | 1.000000 |
| WW | 0.833680 | 1.000000 |
| Homeobox | 0.750753 | 1.000000 |
| Forkhead | 0.729774 | 1.000000 |
| Conotoxin | 0.943108 | 0.995584 |

The seeded Kunitz rerun confirmed that generation itself is unchanged; only metrics
that compare generated sequences to gapped stored alignments move. Most reported
novelty values retain their two-decimal values. Hard-curation novelty changes from
0.326557 to 0.323927, so the manuscript value changes from 0.33 to 0.32. The
cross-family decoder-fidelity ranges change from 0.720--0.943 and 0.976--0.999 to
0.730--0.943 and 0.996--1.000.

Implemented coverage:

- `code/test/test_encode_decode.jl`: exact one-hot behavior, gaps/non-standard
  residues, PCA normalization, decode-map equivalence, and gap-aware metrics.
- `code/test/test_decoder_characterization.jl`: offline characterization on all six
  tracked family alignments.
- `code/test/runtests.jl`: both suites run by default.

Result: **121/121 tests pass**. The original report is retained below as the historical
input to this review; its proposed code and pre-correction fidelity values are not the
current implementation or current results.

---

**The ask for the reviewer is in section 5.** Sections 1 to 4 are the facts.

---

## 1. Current state, verifiable in about ten seconds

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code
julia test/runtests.jl                          # 52/52 passing, 9.2s
for f in decode_sample decode_onehot onehot_encode sequence_identity; do
  echo "$f: $(grep -l $f test/*.jl | tr '\n' ' ')"
done                                            # all four print nothing
```

The suite is green and has been for the life of the project. It contains six test files:

```
test_score_gradient.jl        test_gmm_identity.jl
test_entropy_identities.jl    test_transition_statistic.jl
test_mask.jl                  test_generated_paper_tables.jl
```

They cover the sampler, the Gaussian-mixture identity, entropy identities, the score
gradient, attention masking, and byte-comparison of generated LaTeX tables. That is
genuinely good coverage of the *latent-space* half of the pipeline.

**Zero of them assert anything about the encode/decode half.**

---

## 2. Exactly what is untested

Every function below lives in `code/src/Protein.jl` and is on the critical path from a
generated latent vector to a reported number in the paper.

| function | line | test coverage |
|---|---|---|
| `onehot_encode` | 168 | none |
| `decode_onehot` | 193 | none |
| `build_memory_matrix` | 224 | **fixture only**, see below |
| `decode_sample` | 257 | none |
| `sequence_identity` | 272 | none |

`build_memory_matrix` is the only one a test touches, at `test/test_mask.jl:56`. It is
used purely to construct a fixture:

```julia
char_mat = [rand(aas) for _ in 1:12, _ in 1:16]
X̂, pca, Lout, _ = build_memory_matrix(char_mat; pratio=0.95)
keep_idx = [1, 2, 3, 4, 5, 6]
seqs, pcas = generate_masked_sequences(X̂, pca, Lout, keep_idx; ...)
@test length(seqs) == 2 * length(10:10:50)
```

The assertions are about how many sequences `generate_masked_sequences` returns. Nothing
is asserted about `X̂`, about the PCA model, or about the sequences' content. If
`build_memory_matrix` returned garbage of the right shape, this test would still pass.

**Consequence:** `f_obs` and `diversity`, two of the four columns in every canonical
result CSV and the basis for the paper's central empirical claim, are computed entirely
by untested code. `f_eff` and `attn_A` come from the well-tested latent half.

---

## 3. What this gap cost

Detail is in `docs/2026-07-28-decoder-postmortem.md`. The short version:

The encoder normalizes each PCA score vector to unit length and discards its norm; the
decoder never restores it. Decoding a *stored* memory back to its own sequence therefore
recovers only 0.720 to 0.943 of its residues depending on family, rather than ~1.0.

That is a one-line test. Encode a memory, decode it, compare to the original. It went
unwritten for the life of the project, and the behavior was instead discovered later by
manual audit, misdiagnosed as a bug in need of a fix, and acted on to the point where a
decoder change was approved before a validation run contradicted it.

I want to be careful not to overstate the counterfactual. A round-trip test would have
surfaced the *number* early. It would not by itself have told anyone whether the number
was a defect or an intended property of a model-defined map, which is the question that
actually caused the trouble. But having it in front of you from day one changes the odds.

---

## 4. Proposed tests

Two tiers, because they have different dependencies and different purposes.

### Tier 1: fast invariants on synthetic fixtures

No network, no real alignments, milliseconds. These assert things that must hold for any
correct implementation.

```julia
using Test, LinearAlgebra, Statistics

@testset "encode/decode pipeline" begin
    aas = collect("ACDEFGHIKLMNPQRSTVWY")

    # (a) one-hot round trip is EXACT. This isolates the encode/argmax layer
    #     from the PCA layer, so a failure localizes immediately.
    char_mat = ['A' 'C' 'D' 'E'; 'W' 'Y' 'K' 'R'; 'G' 'G' 'P' 'S']
    X = onehot_encode(char_mat)
    L = size(char_mat, 2)
    for k in 1:size(char_mat, 1)
        @test decode_onehot(X[:, k], L) == String(char_mat[k, :])
    end

    # (b) gaps and non-standard residues encode to all-zero blocks and decode to '-'
    gapped = ['A' '-' 'D'; 'X' 'C' '.']
    Xg = onehot_encode(gapped)
    @test decode_onehot(Xg[:, 1], 3) == "A-D"
    @test decode_onehot(Xg[:, 2], 3) == "-C-"

    # (c) build_memory_matrix really returns unit-norm columns.
    #     Currently asserted nowhere, yet the GMM identity depends on it.
    cm = [rand(aas) for _ in 1:12, _ in 1:16]
    X̂, pca, Lout, d_full = build_memory_matrix(cm; pratio=0.95)
    @test all(k -> isapprox(norm(X̂[:, k]), 1.0; atol=1e-8), 1:size(X̂, 2))
    @test d_full == 20 * size(cm, 2)
    @test Lout == size(cm, 2)

    # (d) sequence_identity basic properties
    @test sequence_identity("ACDE", "ACDE") == 1.0
    @test sequence_identity("ACDE", "AWDE") ≈ 0.75
    @test sequence_identity("ACDE", "ACDE") == sequence_identity("ACDE", "ACDE")

    # (e) decode_sample is NOT scale-invariant. This is the property the whole
    #     2026-07-28 episode turned on: reconstruct is affine, P*z + mu, so
    #     scaling z changes the balance against the fixed mean profile mu.
    #     Pinning it here makes the behavior discoverable instead of surprising.
    ξ = X̂[:, 1]
    near_mean = decode_sample(0.0 .* ξ, pca, Lout)
    at_memory = decode_sample(ξ, pca, Lout)
    @test near_mean != at_memory
end
```

### Tier 2: characterization of stored-memory round-trip on real families

This is the test that would have surfaced the 0.833 early. It is slower and it depends on
real alignments.

```julia
@testset "stored-memory reconstruction fidelity (characterization)" begin
    # For each canonical family: encode the cleaned alignment, decode every stored
    # memory at unit scale, and pin the mean identity to its measured range.
    # These are NOT aspirational targets. They document a known lossy map, so that
    # an unintended change to the encoder, the PCA, or the decoder trips the test.
    expected = Dict(            # measured 2026-07-28
        "Kunitz"    => 0.833,   "SH3"      => 0.770,
        "WW"        => 0.819,   "Homeobox" => 0.750,
        "Forkhead"  => 0.720,   "Conotoxin" => 0.943,
    )
    for spec in CANONICAL_FAMILIES
        char_mat, _, _ = canonical_load_alignment(spec, DATA_DIR)
        X, pca, L, _ = build_memory_matrix(char_mat; pratio=0.95)
        ids = [sequence_identity(decode_sample(X[:, k], pca, L), String(char_mat[k, :]))
               for k in 1:size(char_mat, 1)]
        @test isapprox(mean(ids), expected[spec.family]; atol=0.02)
    end
end
```

---

## 5. Where I want a second opinion

These are genuine design questions, not rhetorical ones. I have a leaning on each and I
state it, but I would rather have them checked than decide alone, given how the last
week went.

**5.1 Should tier 2 pin values, or assert an aspiration?**
My leaning: pin. The independent audit's conclusion was that `decode_sample` is a
*model-defined latent-to-sequence map*, not an inverse of the encoder, and that its
lossiness is a documented property rather than a defect. Under that reading a test
asserting `identity > 0.99` would be asserting a change nobody has agreed to make. A
characterization test that pins 0.833 is honest about what the code does today. The
counter-argument is that pinning a number you believe is wrong entrenches it.

**5.2 Is `atol=0.02` right, and are these values stable?**
The values depend on the Pfam seed alignments, on `clean_alignment`'s gap thresholds, and
on `MultivariateStats`' PCA implementation. Pfam releases change. `canonical_load_alignment`
downloads and caches seeds, so a fresh clone on a new Pfam release could shift these and
fail the suite for an uninteresting reason. Options: commit the alignments and read only
from disk; widen the tolerance a lot; or tag tier 2 so it does not run in a default
`runtests.jl`. I lean toward reading committed alignments only, but I have not checked
which families already have their `.sto` committed versus downloaded on demand.

**5.3 Is test (e) worth having, or is it too clever?**
It asserts that decoding at scale 0 differs from decoding at scale 1. It passes trivially
today and its only value is documentary: it makes the affine `P*z + mu` structure, and
hence the non-invariance to scaling, visible to the next person. A reviewer may reasonably
think a comment would serve better than an assertion.

**5.4 Am I missing a more valuable test than any of these?**
The one I keep coming back to but have not proposed, because I am not sure it is
well-posed: something asserting that the decoded ensemble is a sensible pushforward of the
latent ensemble. The natural formulation would use the component-conditioned TPR and FPR
from the audit, since `f_obs = f_eff*TPR + (1 - f_eff)*FPR` is exact. But that requires
exact-GMM sampling with component labels and it is slow, so it is an experiment rather
than a unit test. Is there a cheap version worth having?

**5.5 Scope.**
Should this land as one new file `code/test/test_encode_decode.jl` registered in
`code/test/runtests.jl`, or should tier 2 live separately so the fast suite stays fast?
The existing suite runs in 9.2 seconds, which is a property worth protecting.

---

## 6. Running things

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code
julia test/runtests.jl        # current suite, 52/52, 9.2s
```

New test files self-include `Include.jl` (idempotent) and are registered by adding an
`include(...)` line inside the top-level `@testset` in `code/test/runtests.jl`.

Note that none of the above is implemented. This document is a proposal and the numbers in
tier 2 are measurements taken on 2026-07-28 from a branch that has since been deleted;
they are reproducible but should be re-measured before being pinned.
