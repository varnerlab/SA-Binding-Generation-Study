# Last Issue to Fix Before Submission

Date: 2026-07-29

## Status

Resolved on 2026-07-29. The seed, FASTA, metric-provenance, manuscript, and PDF checks
described below are complete. The paper is ready for submission after the author's normal
final metadata check.

## Submission blocker: overlapping replicate seeds

The canonical family sweep assigns each replicate a seed that differs by one:

```julia
seed = 20_000 + (rho_index - 1) * N_REPS + rep
```

Each chain then receives:

```julia
chain_rng = MersenneTwister(seed + chain)
```

This causes adjacent replicates to reuse almost all random streams. The 100 canonical
chain runs at a given multiplicity condition use only 24 unique seeds. Adjacent
replicates reuse 19 of 20 chain seeds. They therefore do not support the manuscript's
claim of five independent replicates.

The same pattern affects:

- `code/experiments/run_canonical_family_sweeps.jl`
- `code/experiments/run_kunitz_binding_experiment_with_replicates.jl`
- The binder-scaling calculation in
  `code/experiments/run_augmented_memory_deepdive.jl`

For the Kunitz calculation, 150 chain runs per condition use only 34 unique seeds.
This makes the replicate standard deviations unreliable. The means may also change
after the rerun.

## Required correction

1. Use a collision-free, stable seed mapping over family, multiplicity condition,
   replicate, and chain. At minimum, separate replicate base seeds by more than the
   number of chains.
2. Add a regression test that confirms seed uniqueness within every replicated
   paper-facing calculation.
3. Rerun the complete canonical sweep for all six families.
4. Rerun the Kunitz replicate sequence calculations.
5. Rerun the binder-scaling analysis.
6. Regenerate every table, macro, fit, and figure derived from those calculations.
7. Confirm that the manuscript values match the regenerated outputs.
8. Run the full test suite and rebuild the JCIM article, Supporting Information, and
   arXiv paper.

The AlphaFold, ESM2, docking, and structural calculations do not need to be rerun.
Their inputs were not changed by the all-memory correction or by the replicate seed
allocation.

## Smaller corrections

After the reruns, complete the requested concise and neutral prose pass. Wording that
still needs attention includes:

- “small-set amplifier”
- “central finding”
- “principled”
- “natural axis”
- “near-perfect”
- “faithful recapitulation”

Long, comma-connected sentences remain in the Results and Methods sections. Split them
so that each sentence presents one method, result, or interpretation.

Also correct the `find_entropy_transition` docstring in `code/src/Protein.jl`. A seeded
random subset of memory columns is still sensitive to column order. Only the
all-memory calculation is order-independent.

## Items already verified

- The all-memory operating-point rule is implemented correctly.
- All paper-facing operating-point scripts use the all-memory rule.
- Generated canonical raw and aggregate data are internally consistent.
- The JCIM and arXiv generated sections agree.
- The full Julia test suite passed: 321/321 tests.
- The JCIM article, Supporting Information, and arXiv paper build successfully.
- There are no unresolved references, citations, overfull boxes, or fatal build errors.
- All three PDFs were rendered and visually inspected. No clipping, broken tables,
  overlapping elements, or unreadable figures were found.

## Final recommendation

Make this one focused computational correction and rerun only the affected stochastic
analyses. Then complete the prose cleanup and perform one final preflight review.

---

## Final resolution

Codex, 2026-07-29.

The submission blocker is resolved.

### RNG and deterministic reruns

- Replicate base seeds now occupy disjoint blocks. The allocation covers canonical family
  sweeps, the Kunitz condition experiment, the designated-input scaling experiment, the
  HMM comparison, and HMM emissions.
- Random pair selection for diversity metrics now uses replicate-local RNGs.
- Seed-allocation and source-regression tests pass, including uniqueness of all paper-facing
  chain streams.
- The complete six-family canonical calculation was repeated and produced byte-identical
  CSVs on the verification rerun.
- The Kunitz condition calculation, designated-input scaling calculation, and corrected HMM
  comparison were each repeated. Their CSVs were byte-identical on the verification reruns.

### FASTA disposition

The Kunitz replicate script had been rewriting three example FASTAs as an unrelated side
effect. Those files feed ESM2 and structure-validation calculations. The script now preserves
them unless the caller explicitly passes `--refresh-example-fastas`.

The restored and verified SHA-256 hashes are:

- full-family example: `a349f4daac07a436d878f5ca86157b5d4d62a9f99d0bf6ab140ca8c332032abd`
- K/R-positive example: `9ab887456f56340eabce10520645a5a9be1197f85ca9a080cd1e7933c43848cf`
- K/R-negative example: `7016486c895953708af71e73abb1dd832699e9303e0dfb175733b95cd3bbb64b`
- HMM seed-42 example: `973dbb085456157698ae7bd96c3dc3a66b825eb89df032dcd51896136d72a752`

These hashes remained unchanged through all final reruns. The existing ESM2 and structure
calculations therefore remain paired with the same sequence inputs.

### Table 4 correction

The HMM driver previously stored PCA-space cosine diversity while Table 4 defined `D` as
pairwise sequence diversity. The HMM and bootstrap diversity entries were also hand-entered
as 0.56. The driver now uses the same 300-pair sequence-diversity definition for all methods.
It evaluates five independent 150-sequence HMM emissions, so the reported standard deviations
have the same replicate-level meaning as the other sequence metrics.

The corrected entries are:

- HMM: P1 K/R `0.17 ± 0.04`, KL × 1000 `27.1 ± 2.1`, diversity `0.90`
- Bootstrap: P1 K/R `0.30 ± 0.04`, KL × 1000 `1.0 ± 0.3`, diversity `0.62`

A manuscript consistency test now checks every P1, KL, and diversity entry in Table 4
against `baseline_comparison.csv` or `binding_experiment_aggregated.csv` in both paper trees.

### Manuscript and figure review

- The JCIM and arXiv Introduction, Results, Methods, Discussion, and substantive Theory text
  remain synchronized. The arXiv-only `samepage` wrapper is still the sole sanctioned Theory
  layout difference.
- The prose was revised to use shorter, direct sentences and neutral wording.
- The exact Gaussian-mixture stationary distribution is stated in the body and used as an
  independent equilibrium benchmark.
- Generated numbers, tables, the separation-index fit, and the designated-input scaling figure
  were regenerated from the corrected data.
- The separation-index figure no longer clips negative fitted values to zero. It displays the
  linear fit only across the observed Pfam range.

### Final verification

- Julia suite: **434/434 tests pass**.
- `git diff --check`: clean.
- JCIM main article: **37 pages**.
- JCIM Supporting Information: **12 pages**.
- arXiv article: **29 pages**.
- All three build with zero undefined references, zero undefined citations, zero overfull
  boxes, and no fatal LaTeX errors.
- All **78 pages** were rendered and visually inspected. No clipping, overlap, broken tables,
  missing figure elements, or unreadable pages were found.

### Final disposition

**Approved for submission and arXiv update.** No computational or manuscript blocker remains
from this audit. Before uploading, the author should only perform the normal submission-form
checks for title, author names, affiliations, corresponding-author details, files, and
supplementary-material designation.
