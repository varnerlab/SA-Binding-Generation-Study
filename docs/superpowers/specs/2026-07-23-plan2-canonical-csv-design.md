# Plan 2 Design: W2 — Reconcile to One Canonical CSV

Date: 2026-07-23
Status: corrected design, implemented and verified
Parent spec: `docs/superpowers/specs/2026-07-22-arxiv-revision-design.md` (workstream W2)
Predecessor: Plan 1 (math foundations) complete and merged to main at 9c974fd.

## 1. Context and locked decisions

The arXiv v2 revision requires that every family number in the manuscript trace to one
canonical, replicated data source, with no cross-source conflicts. A reconnaissance pass
found the problem is larger than the parent spec assumed and is driven by hand-typed numbers
drifting from their CSVs.

Key findings from recon (all paths under repo root):

- There is no file literally named `*with_replicates.csv`. The replicate scripts
  (`run_*_with_replicates.jl`) emit `*_aggregated.csv` (mean and std over replicates) and
  `*_raw_replicates.csv` (per-replicate rows).
- Replicated per-rho sweeps with error bars exist for only two families: Kunitz
  (`data/kunitz/multiplicity_sweep_aggregated.csv`) and WW
  (`data/ww/multiplicity_sweep_aggregated.csv`). SH3, Homeobox, Forkhead, and omega-conotoxin
  have single-run sweeps only (`data/<fam>/multiplicity_sweep.csv`).
- The Kunitz rho=500 observed marker-positive fraction appears as five different numbers:
  0.581 plus or minus 0.006 (replicated aggregate), 0.608 (single-run sweep, which the
  separation-gap figure reads), 0.638 (a different experiment: combined multiplicity plus
  interface-PCA), and 0.631 / 0.634 in the appendix table `tab:kunitz-rho`, which matches no
  CSV at all (stale run).
- Two paper per-family tables disagree: `Paper_v1.tex` `tab:per-family-rho` (Kunitz
  0.389 to 0.609, matches the single-run sweep) versus `sections/appendix.tex`
  `tab:kunitz-rho` and siblings (Kunitz 0.406 to 0.631, matches nothing).
- The attention "below 0.3%" tracking claim is contradicted by the repo's own data: the true
  maximum absolute deviation is SH3 at rho=1, 0.621 versus 0.600, i.e. 2.09 percentage points,
  about seven times the claimed bound. It is even printed as 0.621 vs 0.600 in the paper's own
  `Paper_v1.tex` table.
- The two existing cross-family CSVs disagree on the WW A/B split (natural_frac 0.164 in
  `multi_family_comparison_5fam.csv` versus 0.793 in
  `multi_family_comparison_with_uncertainty.csv`).

Decisions locked with the author:

- **Scope:** Plan 2 executes workstream W2 (reconcile to one canonical CSV). Closes audit
  empirical items 1, 2, 3, 8.
- **Depth:** uniform rigor. Re-run SH3, WW, Homeobox, Forkhead, and omega-conotoxin through
  one canonical replicate pipeline so all six families have replicated aggregates with error
  bars. Kunitz alone can reuse its existing replicated sweep. The existing replicated WW run
  is not reusable: it used a different marker-selection algorithm (333 designated sequences,
  rather than the manuscript analysis's 69) and therefore changed both the separation index and
  calibration gap. This is more than an A/B orientation inversion.
- **Anti-drift:** generate LaTeX table bodies and a prose-number macro file from the canonical
  CSVs, committed to the tree. The manuscript then cannot disagree with the CSVs.
- **rho grid caps at 500.** The re-runs use the existing replicated grid
  {1, 2, 5, 10, 20, 50, 100, 500}. The single-run-only rho=1000 rows are dropped from the paper
  (rho=500 already shows saturation). Kunitz and WW are not re-run.
- **Generated files are committed**, not built on the fly. Wiring the generator into
  `Build.sh` is deferred to W6 / Plan 6.
- **The 0.638 combined-approach number is a different experiment** (multiplicity plus
  interface-PCA) and stays only where that approach is discussed, correctly labeled. It is not
  reconciled into the main Kunitz multiplicity number.
- **Deferred, unchanged:** W0 narrative and W1 math prose corrections go to Plan 5; build
  hygiene, artifact gitignore, and JCIM build repair go to Plan 6.

## 2. Data layer: uniform replicated sweeps

Create one canonical family registry and replicate driver, based on
`run_second_family_validation_with_replicates.jl`, to cover the five families requiring a
canonical rerun. Family split functions must live in that registry and be shared by the sweep
and deterministic cross-family analysis; duplicated split definitions are not acceptable.
Use the same parameters as the existing Kunitz replicated run:

- 5 replicates, 20 chains per replicate, T=5000, rho grid {1, 2, 5, 10, 20, 50, 100, 500}.
- Fixed deterministic seeds following the existing scheme.
- The five Pfam families (Kunitz, WW, SH3, Homeobox, Forkhead) use the `download_pfam_seed`
  path. Omega-conotoxin uses its own O-superfamily data under `data/omega_conotoxin/` and needs
  a replicate version of `run_conotoxin_multiplicity_sweep.jl` (single-run today).

Outputs per rerun family: `data/<fam>/multiplicity_sweep_aggregated.csv` and
`_raw_replicates.csv`, schema
`rho,f_eff,f_obs_mean,f_obs_std,attn_A_mean,attn_A_std,diversity_mean,diversity_std`
(matching the existing Kunitz and WW aggregates).

Then produce one canonical cross-family CSV `data/multi_family_comparison_6fam_aggregated.csv`.
Its schema must contain everything needed by the manuscript table and fit:
`family,source,pfam_id,K,L,d_pca,K_A,K_B,marker,natural_frac,separation_index,`
`hard_curation_mean,hard_curation_std,cal_gap_mean,cal_gap_std,fit_included`.
The deterministic family metadata and separation values are computed using the same family
registry used by the sweeps. The rho=500 gap is computed as `f_eff - f_obs_mean`; its standard
deviation is `f_obs_std` because `f_eff` is deterministic. The canonical WW definition is the
manuscript definition (highest-entropy eligible position in the middle third, most common residue
designated), yielding 69 designated sequences before rerun. The old conflicting cross-family CSVs
(`multi_family_comparison_5fam.csv`, `multi_family_comparison.csv`,
`multi_family_comparison_with_uncertainty.csv`) are retired as manuscript sources; they may
remain on disk but nothing in the paper reads them.

Note: the existing pipeline uses global `Random.seed!` (deterministic but not a per-chain local
RNG). The parent spec's "one local RNG per chain" purity fix is a W5 item. Plan 2 reuses the
pipeline as-is and flags this; it does not refactor RNG handling.

Before writing canonical outputs, the driver/generator must fail on schema mismatch, duplicate
rho rows, an incomplete rho grid, non-finite values, fractions outside `[0,1]`, unexpected
replicate counts, or disagreement between cross-family and sweep `f_eff` at rho=500. Raw files
must contain exactly `5 * 8` rows per family. A committed provenance CSV records the common
parameters and seeds; this plan does not claim bitwise reproducibility across a future RNG
refactor.

## 3. Reconciliation targets

- **Kunitz rho=500 f_obs to one number: 0.581 plus or minus 0.006** (replicated aggregate).
  Every 0.608 / 0.631 / 0.634 / 63% occurrence in results.tex, discussion.tex, and the appendix
  table is regenerated or replaced. The 0.638 combined-approach number is left only in its own
  context, labeled as the combined approach.
- **Attention "below 0.3%" to the true maximum deviation.** Compute the maximum
  `abs(attn_A_mean - f_eff)` over all rows of all six canonical aggregates after the reruns,
  report it in percentage points, and identify its family and rho. The previous 2.1-point SH3
  value came from a retired single run and is not locked as the post-rerun result.
- **Five-family relation recast as exploratory.** Drop the `S > 0.3` decision threshold and all
  "predict / prediction / predicted a priori / predictor" language across abstract.tex,
  introduction.tex, results.tex, discussion.tex, theory.tex, significance_statement.tex, and
  `Paper_v1.tex`. Recompute the unweighted linear fit from the five canonical Pfam points rather
  than locking the retired coefficients. Present it as an exploratory association with
  per-point uncertainty, and keep omega-conotoxin as an external displayed point not included
  in the fit. Define leave-one-family-out sensitivity as the range of the five omitted-family
  slopes and R-squared values; generate those ranges as prose macros. Because `n=5`, do not
  present a confidence band or inferential p-value for this descriptive fit.

## 4. Anti-drift tooling

A generator `code/experiments/generate_paper_tables.jl` reads the canonical CSVs and writes,
into a new committed directory `paper-arxiv/sections/generated/`:

- `tab_<fam>_rho.tex`: the tabular body rows for each per-family rho table.
- `tab_cross_family.tex`: the cross-family comparison body.
- `numbers.tex`: `\newcommand` macros for the reconciled key prose numbers, for example
  `\KunitzFobsFiveHundred` (0.581), `\KunitzFobsFiveHundredSD` (0.006), dynamically derived
  attention-maximum macros, dynamically derived fit coefficient/R-squared macros, leave-one-out
  range macros, plus per-family natural fractions.

The generator accepts an output directory so tests can regenerate into a temporary directory.
It validates all inputs before writing and produces deterministic UTF-8 output. The regression
test regenerates the complete directory into a temporary location and byte-compares every file
with the committed generated files; it does not independently duplicate the generator's
rounding logic.

The manuscript `\input`s the generated table bodies in place of hand-typed rows and uses the
macros in place of literal numbers. Rounding and significant figures are decided once, in the
generator, so the printed values are a deterministic function of the CSVs.

## 5. Paper edits

- `paper-arxiv/sections/appendix.tex`: all six per-family rho tables switch to
  `\input{sections/generated/tab_<fam>_rho}`. The stale `tab:kunitz-rho` numbers disappear
  because the body is now generated.
- `paper-arxiv/Paper_v1.tex`: `tab:per-family-rho` and `tab:cross-family` switch to generated
  bodies. Add `\input{sections/generated/numbers}` in the preamble so macros are available.
- Prose files (results, discussion, introduction, theory, significance_statement, abstract):
  attention-claim correction, relation-recast language, and macro-ized key numbers.
- Separation-gap figure: update the figure script (`render_separation_gap_figure_5fam.py`) to
  read the canonical six-family CSV instead of the retired ones, add per-point y error bars from
  cal_gap_std, remove any `S > 0.3` threshold marker, and keep omega-conotoxin as a point not
  included in the Pfam linear fit. Write the output to the `paper-arxiv` figs path the arXiv
  build consumes. The general redirect of all figure scripts to `paper-arxiv` and Python
  dependency pinning stay in W5; Plan 2 touches only this one figure because the relation recast
  requires it.
- The build must produce a clean `Paper_v1.pdf` with the `app:gmm` label still present and all
  cross-references resolved.

Known but deferred: `tab:per-family-rho` and `tab:cross-family` are defined only in
`Paper_v1.tex`, yet `sections/results.tex` references them, so the JCIM build (which inputs
`sections/` but not `Paper_v1.tex`) has dangling refs. Moving those definitions into
`sections/` is a JCIM-build concern, deferred to Plan 6. Plan 2 targets the arXiv `Paper_v1`
build only.

## 6. Verification

- Re-run the generator, build `Paper_v1`, and confirm the PDF builds clean with resolved refs
  and the `app:gmm` label intact.
- Because the tables and macros are generated, the printed numbers equal the CSV values by
  construction. Add one Julia regression test to Plan 1's `code/test/runtests.jl` that reads the
  canonical CSVs and asserts the committed `numbers.tex` macro values and generated table cells
  match, so any future hand-edit or CSV change that desynchronizes them fails the suite.
- Spot-check the five rerun families' aggregates against their retired single-run sweeps. Report
  differences; do not impose an arbitrary closeness assertion. WW is expected to differ from
  the existing replicated file because that file used the wrong split definition.

## 7. Sequencing and dependencies

1. Re-run the five families and build the canonical cross-family CSV (Section 2). Everything
   downstream reads these.
2. Build the generator and emit the generated tables and macros (Section 4).
3. Convert the paper tables and prose to the generated bodies and macros, and apply the three
   reconciliation targets (Sections 3, 5).
4. Add the regression test and verify the build (Section 6).

Plan 2 precedes Plan 5 (prose). Shared files (results.tex, discussion.tex) are touched by Plan 2
for numbers and relation framing, then by Plan 5 for narrative and math prose. Plan 2 goes first
so Plan 5 writes against settled numbers.

## 8. Success criteria

- All six families have replicated aggregate CSVs on the common rho grid.
- One canonical cross-family CSV exists; the WW A/B split is consistent; old cross-family CSVs
  are no longer read by the paper.
- Every multiplicity-sweep and cross-family-fit number in the arXiv manuscript is either an
  `\input` generated table cell or a `numbers.tex` macro; unrelated family descriptors and
  results from other experiments remain outside this workstream. The Kunitz rho=500 fraction
  reads 0.581 plus or minus 0.006 everywhere it appears.
- The attention claim reports the maximum derived from the completed canonical aggregates, with
  points-versus-percent language corrected.
- The five-family relation reads as an exploratory association: no `S > 0.3` threshold, no
  "prediction" language, per-point uncertainty shown, leave-one-family-out noted.
- `Paper_v1.pdf` builds clean with resolved refs; the regression test passes in `runtests.jl`.

## 9. Risks and open questions

- **Re-run cost and reproducibility.** Five families times five replicates times eight rho
  values times twenty chains at T=5000 is the bulk of the compute. Deterministic seeds make it
  reproducible; the run is one-time and cached to CSV.
- **Omega-conotoxin path.** It does not use `download_pfam_seed`; its replicate driver must
  mirror the existing single-run conotoxin script. Treat as a distinct task.
- **Table conversion fragility.** Generated LaTeX bodies must match the surrounding table
  preamble (column count, alignment, booktabs rules). The generator owns the body rows and final
  `\bottomrule`; the `\begin{tabular}` headers stay in the manuscript to minimize surface area.
  Keeping `\bottomrule` inside the input file avoids TeX alignment errors caused by placing a
  no-align rule immediately after an `\input` that ends with a row terminator.
- **Numbers moving.** If a replicated mean lands far from the retired single-run value for any
  family, that is a real result change to surface, not silently absorb. The spot-check in
  Section 6 is the guard.
- **RNG purity.** Reusing the global-seed pipeline is deliberate scope control; the local-RNG
  refactor stays in W5. If W5 later changes seeds, these CSVs regenerate.
- **Kunitz comparability (resolved).** Kunitz was rerun through the canonical driver on
  2026-07-23. Its execution parameters are recorded in
  `data/kunitz/canonical_sweep_execution.csv`; the regenerated aggregate reproduced the previous
  replicated values exactly.

## 10. Implementation outcome

Implemented on 2026-07-23. The completed canonical aggregates changed several values that could
not safely be locked before execution:

- Maximum absolute mean-attention deviation: 3.0 percentage points, omega-conotoxin at rho=1.
- Five-Pfam descriptive fit: Delta approximately 0.77 - 2.0 S, R-squared 0.80.
- Leave-one-family-out slopes: -2.7 to -1.4; R-squared: 0.76 to 0.91.
- Canonical WW rho=500 gap: 0.688 plus or minus 0.004 using the intended 69-sequence subset.
- Canonical Kunitz rho=500 observed marker fraction: 0.581 plus or minus 0.006.

All six families now have canonical-driver provenance. The final Kunitz rerun reproduced its
earlier replicated aggregate exactly, so no downstream numerical result changed.

Verification completed:

- All 52 Julia tests pass, including byte-for-byte regeneration of the committed LaTeX files.
- `Paper_v1.pdf` builds to 27 pages with no LaTeX errors, undefined references, or undefined
  citations; `app:gmm` remains present.
- The changed cross-family table, replicated sweep tables, and separation-gap figure were
  visually inspected in the rendered PDF and are legible without clipping or overlap.
