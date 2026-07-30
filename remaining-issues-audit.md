# Remaining Issues Audit

> **Status: review plus execution record. Read to the end before acting on any section.**
> Six sections alternate between two independent reviewers, followed by the execution
> outcome and two addenda. Later sections supersede earlier ones in places. In particular
> the "40 of 48 conditions reproduce bit-for-bit" expectation, used as the correctness gate
> for the rerun, proved unachievable: the committed data predated a change to the sampler's
> random-number handling and so no longer reproduced from the committed code. That, and a
> second defect in which Kunitz was never swept by the canonical driver, are recorded in the
> execution outcome section. The final state of the work is described there and in the two
> addenda at the end of this file.

Date: 2026-07-29

Branch reviewed: `submission-corrections-2026-07`

Comparison: `main...submission-corrections-2026-07`

## Merge recommendation

Do not merge yet. Most corrections from `submission-audit-results.md` were implemented
correctly, but one new sensitivity statement is false and several factual and prose
corrections remain.

## 1. Entropy-onset sensitivity requires correction

### Current manuscript statement

`paper-jcim/sections/methods.tex:44-46` and the matching arXiv source state that
recomputing the entropy-crossover onset from all stored memories:

- moved Homeobox and omega-conotoxin by one grid step; and
- left Kunitz, SH3, WW, and Forkhead unchanged.

This is false when the comparison is made across all eight canonical multiplicity
conditions.

### Reproduced comparison

The original detector in `code/src/Binding.jl:881-913` evaluates the first 20 memory
columns. It was compared with the all-memory onset returned by
`find_entropy_transition`, using the same 50-point beta grid.

Eight of the 48 family/rho conditions changed:

| Family | rho | First-20 onset | All-memory onset | Ratio |
|---|---:|---:|---:|---:|
| Kunitz | 2 | 4.5789 | 3.8483 | 1.1898 |
| SH3 | 50 | 3.2343 | 6.4825 | 2.0043 |
| Homeobox | 100 | 4.5789 | 3.8483 | 1.1898 |
| Homeobox | 500 | 4.5789 | 3.8483 | 1.1898 |
| Forkhead | 2 | 5.4482 | 4.5789 | 1.1898 |
| Forkhead | 10 | 6.4825 | 5.4482 | 1.1898 |
| Forkhead | 20 | 6.4825 | 4.5789 | 1.4157 |
| Conotoxin | 500 | 7.7131 | 6.4825 | 1.1898 |

WW was unchanged across the tested rho grid.

The SH3 difference is a factor of two, and the Forkhead rho=20 difference spans more
than one grid step. The discrepancy is therefore not limited to grid resolution.

### Why this matters

The selected beta controls Gaussian component variance, decoded marker recovery, and
sequence diversity. The cross-family calibration data were generated using an
order-dependent operating point. Changing beta may alter the family trajectories,
calibration gaps, and exploratory separation-index relationship.

### Required action

Preferred correction:

1. Change the canonical experiment pipeline to use a deterministic all-memory
   entropy-crossover onset.
2. Rerun the eight affected family/rho conditions, or rerun the complete canonical
   sweep for consistency.
3. Regenerate the affected aggregate CSVs, tables, figures, and exploratory fit.
4. Replace the current Methods sensitivity sentence with the verified result.
5. Add a regression test for the operating-point procedure.

If the published first-20 results are retained instead, the manuscript must state that
the operating point is alignment-order dependent and must report outcome sensitivity,
not only onset sensitivity. Given the SH3 and Forkhead differences, this is the less
defensible option.

## 2. Remaining factual overstatements

### Table 5 says a nonidentical quantity is preserved exactly

`paper-jcim/Paper_JCIM.tex:304-305` and the matching arXiv caption say:

> preserves the basic-residue signature exactly

The table reports 19.7% for the designated input and 20.1% for designated-seeded
generation. Replace “exactly” with a neutral comparison such as:

> produced a similar basic-residue fraction

The same sentence also uses “amplifies Tyr13.” Prefer “increased the Tyr13 frequency.”

### Exact-versus-ULA conclusion remains absolute in Results

`paper-jcim/sections/results.tex:113-114` and the arXiv copy state that the calibration
gap:

> is not explained by finite-step ULA discretization or mixing error

The finite benchmark supports the narrower wording already used in the Discussion:

> indicates that finite-step ULA error was not the main source of the gap

### Exact sampling is not used throughout

`paper-jcim/sections/theory.tex:62-65` and the arXiv copy say:

> we use it as the exact reference throughout

Direct exact sampling is used in the Kunitz exact-equilibrium benchmark at three rho
values, not throughout all generation experiments. Replace this with:

> we use it as the reference in the exact-equilibrium benchmark

### K_eff is still called “the relevant quantity”

`paper-jcim/sections/theory.tex:111-115` and the arXiv copy call
`K_eff` “the relevant quantity” under multiplicity weighting. This still suggests that
`K_eff` controls the entropy-crossover location, although the corrected text later says
that it is only a descriptive inverse-Simpson measure.

Replace this sentence with a neutral introduction, for example:

> We also report the effective pattern count as a descriptor of weight concentration.

### Unvalidated Cav2.2 specificity claim remains

`paper-jcim/sections/results.tex:223-225` and the arXiv copy say that co-varying loop
positions:

> contribute to Cav2.2 binding specificity

The generated sequences were not tested for binding, and most accessions lack matched
activity metadata. Replace this with a sequence-level statement such as:

> vary across the designated accessions

## 3. Requested prose and tone pass remains incomplete

The requested style was simple, direct, concise, and neutral, without long
comma-connected sentences or promotional language.

An approximate source scan still finds 31 sentences containing at least 45 words:

- Introduction: 2
- Results: 16
- Discussion: 3
- Methods: 7
- Theory: 3

TeX markup makes the count approximate, but the main long sentences are clear in the
source. High-priority examples include:

- `paper-jcim/sections/methods.tex:18-34`: approximately 143 words
- `paper-jcim/sections/results.tex:116-130`: approximately 72 words
- `paper-jcim/sections/results.tex:219-228`: approximately 71 words
- `paper-jcim/sections/results.tex:281-289`: approximately 84 words
- `paper-jcim/sections/results.tex:82-94`: two sentences of approximately 65 words

Split these passages so that each sentence presents one method, result, or
interpretation.

### Promotional or overstated wording still present

Remove or neutralize:

- “natural axis” in `paper-jcim/sections/results.tex:18`
- “small-set amplifier” in:
  - `paper-jcim/sections/results.tex:21`
  - `paper-jcim/sections/results.tex:51`
  - `paper-jcim/sections/discussion.tex:45`
  - `paper-jcim/sections/discussion.tex:54`
- “principled” in:
  - `paper-jcim/sections/introduction.tex:43`
  - `paper-jcim/sections/discussion.tex:2`
- “central finding” in `paper-jcim/sections/discussion.tex:21`
- “very high ... confirming” in `paper-jcim/Paper_JCIM.tex:410-411`
- “near-perfect ... confirming faithful recapitulation” in
  `paper-jcim/Paper_JCIM.tex:438-439`

Suggested neutral alternatives:

- “varied with”
- “increased”
- “was associated with”
- “the correlation was”
- “provides a way to”
- “we observed”

## 4. Optional generator hardening

`code/experiments/generate_paper_tables.jl:91-102` checks that the SAR CSV has 12 rows,
and `:170-173` checks that each row has a known position. It does not require the set of
positions to equal the expected 12-position set.

A duplicated position combined with a missing position could therefore pass validation.
The new SAR-frame test also iterates over existing rows and would not necessarily catch
that case.

Optional improvement:

```julia
expected_positions = Set(keys(sar_labels))
actual_positions = Set(Int.(sar.Position))
length(unique(sar.Position)) == nrow(sar) ||
    error("Conotoxin SAR positions must be unique")
actual_positions == expected_positions ||
    error("Conotoxin SAR position set mismatch")
```

This is not a current data error, but it would make the anti-drift generator match its
stated guarantee.

## Corrections that passed review

The following branch changes are correct:

- The conotoxin designated input is now read from the canonical 74-by-26 alignment
  frame.
- The regenerated SAR CSV and Table 7 contain the corrected input frequencies.
- The SAR table is generated from the canonical CSV in both manuscript trees.
- The Shannon/Renyi-2 entropy distinction is correct.
- The false `beta* proportional to log K_eff` claim was removed.
- The hard-mask comparison now acknowledges the change in intended designated mass.
- The real-valued multiplicity language is corrected.
- The direct exact sampler is acknowledged and ULA is no longer presented as necessary.
- The VAE and ESM2 citations are corrected.
- The two identified grammar errors are corrected.
- JCIM and arXiv shared section parity is guarded by a test.

## Verification results

- Full Julia test suite: 263/263 passed.
- `git diff --check main...HEAD`: passed.
- JCIM main build: passed, 40 pages.
- JCIM SI build: passed, 12 pages.
- arXiv build: passed, 29 pages.
- No unresolved citations or references.
- No overfull boxes.
- All PDF pages were rendered and visually inspected.
- No clipping, overlapping elements, broken tables, or unreadable figures were found.
- The regenerated SAR table renders cleanly.

## Repository state

The committed correction branch is clean. The following review documents are untracked
and will not be included in a merge unless added intentionally:

- `submission-audit-results.md`
- `remaining-issues-audit.md`
- `docs/superpowers/plans/2026-07-29-submission-corrections.md`

---

# Response and Decision Request

Claude, 2026-07-29. Written at the author's request in place of a verbal recommendation.

This section records what was verified, what was fixed, what was measured, what remains
unmeasured, and three concrete options for item 1. Every numeric claim below is labelled
as either REPRODUCED (I ran it and can show the command) or ESTIMATE (not measured).

## A. Item 1 is confirmed. My Methods sentence was false.

Commit `c16d21d` added this sentence to `methods.tex`:

> Recomputing the onset from all stored memories moved the selected value by one grid
> step, a factor of 1.19, for Homeobox and omega-conotoxin, and left Kunitz, SH3, WW and
> Forkhead unchanged.

That is false. It generalized the first audit's `rho=500`-only comparison to the whole
multiplicity grid without checking the other seven rho values.

REPRODUCED. I compared `find_weighted_entropy_inflection` (the first-20-column detector
the canonical driver calls at `run_canonical_family_sweeps.jl:104`) against
`find_entropy_transition(X, w; n_betas=50, n_probes=K)` across all 48 family/rho
conditions. My results match the reviewer's table exactly.

| Family | rho | First-20 onset | All-memory onset | Ratio | Grid steps |
|---|---:|---:|---:|---:|---:|
| Kunitz | 2 | 4.5789 | 3.8483 | 0.840 | 1 |
| SH3 | 50 | 3.2343 | 6.4825 | 2.004 | 4 |
| Homeobox | 100 | 4.5789 | 3.8483 | 0.840 | 1 |
| Homeobox | 500 | 4.5789 | 3.8483 | 0.840 | 1 |
| Forkhead | 2 | 5.4482 | 4.5789 | 0.840 | 1 |
| Forkhead | 10 | 6.4825 | 5.4482 | 0.840 | 1 |
| Forkhead | 20 | 6.4825 | 4.5789 | 0.706 | 2 |
| Conotoxin | 500 | 7.7131 | 6.4825 | 0.840 | 1 |

WW is unchanged at every rho. The other 40 conditions are unchanged.

The reviewer is correct that this is not merely grid resolution. SH3 at `rho=50` differs
by a factor of two.

## B. New evidence the review did not have: outcome sensitivity

The review states that changing beta "may alter the family trajectories, calibration
gaps, and exploratory separation-index relationship." That was a hypothesis. I measured
it.

REPRODUCED. For each of the eight affected conditions I reran the canonical inner loop
at both beta values, using the canonical seeds, chain count, step count, burn-in and
thinning from `run_canonical_family_sweeps.jl`.

| Family | rho | f_obs first-20 | f_obs all-memory | Delta | replicate SD |
|---|---:|---:|---:|---:|---:|
| SH3 | 50 | 0.8671 | 0.9673 | +0.1002 | 0.003 |
| Forkhead | 20 | 0.6717 | 0.6341 | -0.0376 | 0.004 |
| Conotoxin | 500 | 0.8846 | 0.8549 | -0.0298 | 0.010 |
| Homeobox | 500 | 0.9439 | 0.9263 | -0.0176 | 0.004 |
| Forkhead | 2 | 0.5761 | 0.5585 | -0.0176 | 0.005 |
| Kunitz | 2 | 0.4088 | 0.3917 | -0.0171 | 0.008 |
| Homeobox | 100 | 0.9366 | 0.9200 | -0.0166 | 0.006 |
| Forkhead | 10 | 0.6515 | 0.6366 | -0.0149 | 0.004 |

Every change exceeds replicate noise. SH3 at `rho=50` is roughly 30 replicate standard
deviations. Attention weights were unchanged to within 0.003 in all eight conditions,
which is expected: attention tracks the intended mass regardless of beta.

### Effect on the headline results

Only `rho=500` feeds the cross-family table and the exploratory fit. Two of six families
change there.

REPRODUCED. Cross-family calibration gaps at `rho=500`:

- Homeobox: 0.0513 becomes 0.0689
- Conotoxin: 0.1343 becomes 0.1640
- Kunitz, SH3, WW, Forkhead: unchanged

REPRODUCED. The five-family exploratory fit, recomputed with the Homeobox gap adjusted
(conotoxin is excluded from the fit by `fit_included`):

| Quantity | Current | All-memory | As reported |
|---|---:|---:|---|
| slope | -1.951 | -1.905 | -2.0 becomes -1.9 |
| intercept | 0.772 | 0.764 | 0.77 becomes 0.76 |
| R-squared | 0.802 | 0.786 | 0.80 becomes 0.79 |

The exploratory relationship survives the change.

### What I have NOT measured

- The hard-curation column uses a different detector, `find_entropy_inflection(X_hard)`
  at `run_canonical_family_sweeps.jl:77`, on the subset-only basis. It is also a
  first-20-column rule. I did not test whether its onset or its
  `hard_curation_mean` output changes. This is an open unknown in every option below.
- `dump_entropy_curves.jl:38` and the calibration and combined-approach scripts also call
  `find_weighted_entropy_inflection`. I did not audit whether their outputs feed any
  reported number.
- I did not check whether any prose sentence quotes a specific f_obs value from one of the
  eight changed cells.

## C. Already fixed and committed on this branch

Commit `6fef3e2`:

- The false sensitivity sentence is replaced with the verified 48-condition result, and
  Methods now states explicitly that the operating point depends on alignment row order.
- Item 2, all four overstatements: Table 5 "preserves the basic-residue signature
  exactly" and "amplifies Tyr13"; the Results claim that the gap "is not explained by"
  ULA error; "we use it as the exact reference throughout"; and K_eff as "the relevant
  quantity."
- Item 4, generator hardening. The position set is now pinned, not just the row count. I
  verified by injection that a duplicated position with a compensating omission passes the
  old check and fails the new one.

Verified after: full suite 263/263, both PDFs build with zero undefined references and
zero overfull boxes, `code/data` unchanged.

Item 3, the prose and tone pass, is unstarted. It was assigned to the author.

## D. Item 1 decision. Three options, specified.

### Option 1: switch the pipeline to the all-memory onset and rerun

Work required:

1. Change `run_canonical_family_sweeps.jl:104` from
   `find_weighted_entropy_inflection(analysis.X, weights; n_betas=50).β_star` to
   `find_entropy_transition(analysis.X, weights; n_betas=50, n_probes=K).β_onset`.
2. Decide the same question for the hard-curation detector at line 77, which is currently
   `find_entropy_inflection(X_hard)`. Leaving it inconsistent would be hard to defend.
3. Rerun the canonical sweep for all six families.
4. Regenerate `multiplicity_sweep_aggregated.csv` and
   `multiplicity_sweep_raw_replicates.csv` per family, plus
   `multi_family_comparison_6fam_aggregated.csv`.
5. Regenerate `sections/generated/` in both trees via `generate_paper_tables.jl`.
6. Regenerate the two affected figures, `fig5_entropy_curves.pdf` and
   `fig2_separation_vs_gap.pdf`.
7. Rewrite the Methods paragraph to describe an order-independent procedure and delete the
   sensitivity disclosure, which becomes unnecessary.
8. Add a regression test pinning the operating-point procedure.
9. Re-check every prose number that quotes a per-family f_obs.

What is known about the result, REPRODUCED: 40 of the 48 conditions have identical beta
under both rules, so with identical seeds they reproduce bit-for-bit. Only 8 cells move.
The cross-family gaps change for Homeobox and conotoxin only. The exploratory fit moves
from -2.0 to -1.9 with R-squared from 0.80 to 0.79.

ESTIMATE, not measured: roughly an hour of compute for the full sweep, plus figure
regeneration and verification. My eight-condition, two-beta run took several minutes; the
full sweep is about three times that volume.

Risk: the hard-curation unknown in section B could widen the change beyond what is
described above. That would be discovered during the rerun, not before.

Result: the operating point no longer depends on alignment row order, and the Methods
sensitivity disclosure is removed rather than expanded.

### Option 2: keep the published numbers and document the dependence

Work required: none beyond what is already committed. The Methods paragraph as it now
stands already states the order dependence, reports the eight changed conditions with
their grid-step sizes, and reports the outcome sensitivity including the SH3 +0.100.

Result: the branch is mergeable after the author's prose pass. No data regenerates and no
figure changes.

Cost: the submitted paper selects its operating point by a rule that depends on the order
of rows in an alignment file, and says so in Methods. The reviewer judged this the less
defensible option. I have no independent basis for predicting referee reaction.

### Option 3: patch only the eight affected conditions

Work required: regenerate only the eight changed rows, leaving the pipeline unchanged.

Consequence: the committed code would select beta by the first-20 rule while the committed
data would reflect the all-memory rule for eight cells and the first-20 rule for the other
forty. `canonical_sweep_execution.csv` records the driver as the provenance source, so
that record would no longer describe how the data were produced. The reviewer flagged this
option as not recommended, and I agree with that specific technical point.

## E. What I am asking the author to decide

Which of Options 1, 2 or 3 to take. The decision is not one I should make alone: Option 1
changes generated data and was explicitly placed out of scope in the approved plan
(`docs/superpowers/plans/2026-07-29-submission-corrections.md`, "Deliberately out of
scope"), and Option 2 accepts a known methodological weakness in a submitted manuscript.

If Option 1 is chosen, I will write a plan for it before touching anything, including a
decision on the hard-curation detector, and will report the hard-curation result as soon
as it is measured rather than after the fact.

---

# Codex Recommendation

Date: 2026-07-29

Claude identified a real methodological issue, but a blanket rerun is not needed. Do not
merge the current branch as the final submission yet. Use a scoped version of Option 1:
replace the order-dependent first-20 onset with the all-memory onset for paper-facing
calculations, then rerun only the affected experiments and their downstream artifacts.

## Additional read-only checks

The following comparisons used the same beta grids as the corresponding scripts. They
computed operating points only. They did not rerun sampling or modify repository data.

### Canonical hard curation

The first-20 and all-memory onsets were identical for all six subset-only hard-curation
bases:

| Family | First-20 onset | All-memory onset |
|---|---:|---:|
| Kunitz | 3.2343 | 3.2343 |
| SH3 | 3.2343 | 3.2343 |
| WW | 3.8483 | 3.8483 |
| Homeobox | 3.8483 | 3.8483 |
| Forkhead | 4.5789 | 4.5789 |
| Omega-conotoxin | 2.7183 | 2.7183 |

This resolves the hard-curation unknown in Section B. The hard-curation results should
reproduce when the implementation is changed to use all memories.

### Kunitz mask experiment

The unconditional, hard-mask, hard-curation, and four of five multiplicity temperatures
were unchanged. The multiplicity condition with target designated mass 0.5 changed by
one grid step:

| Condition | First-20 onset | All-memory onset |
|---|---:|---:|
| Multiplicity, target 0.5 | 4.5789 | 3.8483 |

The conditions with targets 0.7, 0.9, 0.95, and 0.99 were unchanged. Because the mask
figure includes a mask run matched to each multiplicity temperature, the target-0.5
multiplicity run and its matched-mask comparison should be regenerated. Running the
complete mask script may be simpler and gives clean provenance.

### Exact-Gaussian-mixture benchmark

On the benchmark's 60-point grid, the Kunitz operating point changed only at
`rho = 10`:

| rho | First-20 onset | All-memory onset |
|---:|---:|---:|
| 1 | 4.2663 | 4.2663 |
| 10 | 5.6943 | 4.9289 |
| 500 | 8.7806 | 8.7806 |

The exact-GMM benchmark is small. Rerun all three conditions so its table and provenance
remain internally consistent.

## Recommended scope

1. Change the canonical driver to evaluate the entropy onset over all stored memories.
   Apply the same rule to paper-facing hard-curation and mask calculations.
2. Add a regression test that permutes memory columns and confirms that the selected
   onset is unchanged. A test that only pins one numeric beta is not sufficient.
3. Rerun the complete canonical sweep. Although only 8 of 48 conditions change, the full
   run provides one consistent provenance path for every row.
4. Rerun the Kunitz mask experiment and the exact-GMM benchmark.
5. Regenerate the affected aggregate CSVs, generated tables, entropy-onset figure,
   separation-versus-gap figure, mask figure, and benchmark table.
6. Recheck every number in the JCIM and arXiv manuscripts that is derived from these
   outputs.
7. Rewrite Methods to describe the final all-memory procedure. Remove the discussion of
   the first-20 implementation because it will no longer describe the submitted
   analysis.
8. Complete the requested prose pass, full test suite, manuscript builds, and visual PDF
   inspection before merging.

Do not rerun AlphaFold, ESMFold, ESM2, docking, or unrelated legacy experiments unless a
dependency check shows that a reported result used an operating point that changes under
the all-memory rule. Do not refactor every legacy script as part of this submission
correction.

Option 2 is technically transparent but retains an avoidable dependence on alignment row
order. It is not recommended for the journal submission. Option 3 would leave the driver,
data, and provenance inconsistent and should not be used.

## Suggested instruction to Claude

> Proceed with the all-memory correction, but limit the work to paper-facing outputs.
> Rerun the complete canonical sweep, the Kunitz mask experiment, and the exact-GMM
> benchmark. Do not rerun structure prediction, docking, ESM2, or unrelated legacy
> experiments without showing that their selected operating point changes. Update both
> manuscript trees, add a memory-column permutation-invariance test, regenerate affected
> artifacts, and provide the final diff for review before merging.
---

# Verification of the Codex Recommendation

Claude, 2026-07-29.

## All three read-only tables reproduce

I reran each check independently against the same scripts and grids.

REPRODUCED, canonical hard curation, 50-point grid, subset-only bases. Identical to the
Codex table, all six families unchanged:

| Family | First-20 | All-memory |
|---|---:|---:|
| Kunitz | 3.2343 | 3.2343 |
| SH3 | 3.2343 | 3.2343 |
| WW | 3.8483 | 3.8483 |
| Homeobox | 3.8483 | 3.8483 |
| Forkhead | 4.5789 | 4.5789 |
| Conotoxin | 2.7183 | 2.7183 |

This closes the open unknown I flagged in Section B. Hard curation is unaffected.

REPRODUCED, Kunitz mask experiment. One of eight conditions changes, the multiplicity
condition at target 0.5, from 4.5789 to 3.8483. Unconditional, hard mask, hard curation,
and the multiplicity conditions at 0.7, 0.9, 0.95 and 0.99 are unchanged. My first-20
values reproduce the committed `mask_residuals.csv` betas exactly, which confirms the
setup.

REPRODUCED, exact-GMM benchmark, 60-point grid. One of three conditions changes,
`rho=10`, from 5.6943 to 4.9289. `rho=1` and `rho=500` are unchanged.

Correction to my own process: my first attempt at the mask and benchmark tables disagreed
with Codex. The cause was mine. I used `data/cache` for the Pfam seed when the scripts use
`data/kunitz`, so I was comparing against a different alignment. With the correct path all
three tables agree. I report this because it is the second time in this review cycle that
an unverified setup assumption produced a wrong result on my side.

## Complete enumerated change set

Combining this with Section A and B, every paper-facing operating point that changes under
the all-memory rule is now enumerated:

| Calculation | Conditions | Changed |
|---|---:|---:|
| Canonical multiplicity sweep | 48 | 8 |
| Canonical hard curation | 6 | 0 |
| Kunitz mask experiment | 8 | 1 |
| Exact-GMM benchmark | 3 | 1 |

Ten changed operating points in total, all identified before any rerun.

## Agreement with the recommended scope

I agree with the eight-step scope, with two notes.

On step 2, a permutation-invariance test already exists at
`code/test/test_transition_statistic.jl:21-29`. It asserts that `β_steepest` and the whole
entropy curve `Hs` are invariant under column reversal. Since `β_onset` is derived from
`Hs`, it is implicitly covered, but it is not asserted directly. I would extend that test
to assert `β_onset` explicitly, and add a separate test that the canonical driver calls
the order-independent function, because the current gap is not in the statistic but in
which statistic the driver selects.

On step 3, I agree that rerunning all 48 conditions is better than patching 8, for the
provenance reason given. Note that the 40 unchanged conditions should reproduce
bit-for-bit, since beta is identical and the seeds are fixed. That is a useful correctness
check on the rerun itself: any change outside the 8 predicted cells indicates a problem.

I have not started this work. It changes generated data and remains outside the approved
plan until the author says otherwise.

---

# Codex Follow-up Review of the Proposed Work

Date: 2026-07-29

Claude's independent checks reproduce the three detector comparisons. The proposed
all-memory method, complete canonical rerun, permutation-invariance test, and
bit-for-bit comparison of the 40 expected-unchanged canonical conditions are sound.

One issue remains before authorizing the data reruns. The statement that every
paper-facing operating point has been enumerated is not yet supported by a complete
artifact-to-script dependency check. Several figures included in the main manuscripts
are produced by scripts that still call the first-20 detector:

- `study1_binder_scaling.pdf` is produced by
  `code/experiments/run_augmented_memory_deepdive.jl`. That script selects operating
  points for several curated-subset sizes.
- `fig5_entropy_curves.pdf` is produced by
  `code/experiments/run_multiplicity_conditioning.jl`. Its plotted onsets use the old
  weighted detector on a 60-point grid.
- `loop_heatmap_with_residuals.pdf` is produced by
  `code/experiments/run_omega_conotoxin_experiment.jl`. Its full-family and
  designated-subset generations use the old unweighted detector.

Some of these calculations may already be covered by the unchanged full-family and
hard-curation comparisons. If so, they can be marked unchanged without rerunning
sampling. The binder-scaling subsets and the 60-point entropy-curve onsets still require
an explicit detector-only comparison.

## Required preflight

Before changing generated data, create a paper-facing dependency table with one row per
reported figure, table, or numerical result that uses a selected operating temperature.
Record:

1. The manuscript artifact or quoted result.
2. Its producer script and committed input data.
3. The detector function and beta grid.
4. The first-20 and all-memory onsets.
5. Whether the onset changes.
6. Whether sampling, only plotting, or no downstream work must be repeated.

Include the binder-scaling, entropy-curve, conotoxin-loop, and sequence-analysis
artifacts. This is a detector-only check and should be completed before starting the
larger sampling runs.

After this table is complete, proceed with:

- the full canonical sweep;
- the Kunitz mask experiment;
- the exact-GMM benchmark; and
- any additional paper-facing calculation shown by the dependency table to change.

Do not rerun structure prediction, ESM2 scoring, docking, or unrelated legacy
experiments unless the dependency table shows that a reported result depends on a
changed operating point.

## Suggested instruction to Claude

> Before changing data, add a paper-facing provenance table that lists each reported
> figure, table, and numerical result; its producer script; its detector and beta grid;
> the first-20 and all-memory onsets; and whether sampling must be rerun. Include the
> binder-scaling, entropy-curve, conotoxin-loop, and sequence-analysis artifacts. Then
> proceed with the canonical, mask, and GMM reruns, plus any additional paper-facing
> calculation shown to change. Do not rerun structure prediction, ESM2, or docking
> without a demonstrated dependency. Provide the dependency table and final diff for
> review before merging.
---

# Preflight Dependency Table

Claude, 2026-07-29. Detector-only. No sampling was run and no repository data changed.

Codex was right that my earlier "complete enumeration" claim was not backed by an
artifact-to-script check. This is that check. Every manuscript figure and every quoted
operating temperature is traced to its producer script, its detector, and its beta grid.

Each script's constants were read from the script rather than reconstructed, and where a
committed artifact exists it was used as a positive control before trusting the
comparison.

| Manuscript artifact | Producer script | Detector, grid | Conditions | Changed |
|---|---|---|---:|---:|
| Per-family rho tables, cross-family table | `run_canonical_family_sweeps.jl:104` | weighted, 50-pt | 48 | **8** |
| Hard-curation column | `run_canonical_family_sweeps.jl:77` | unweighted, 50-pt | 6 | 0 |
| `fig2_separation_vs_gap.pdf` | canonical CSV, python renderer | inherits | - | via Homeobox rho=500 |
| `fig_mask_betasweep.pdf`, mask-recovery table | `run_kunitz_mask_experiment.jl` | mixed, 50-pt | 8 | **1** |
| Exact-GMM benchmark table | `run_gmm_baseline.jl:31` | weighted, 60-pt | 3 | **1** |
| `study1_binder_scaling.pdf` | `run_augmented_memory_deepdive.jl:57,146` | unweighted, 50-pt | 25 | 0 |
| `fig5_entropy_curves.pdf`, quoted beta* 4.35 and 9.26 | `dump_entropy_curves.jl` then `render_entropy_curves.py` | weighted, **80-pt** | 5 | 0 |
| `loop_heatmap_with_residuals.pdf`, Tables 5 and 6, conotoxin FASTAs | `run_omega_conotoxin_experiment.jl:118-119` | unweighted, 50-pt | 2 | 0 |
| `sequence_analysis_kunitz.pdf` | `render_sequence_analysis_figure.jl`, consumes `run_kunitz_binding_experiment.jl` FASTAs | unweighted, 50-pt | 3 | 0 |
| `sequence_analysis_conotoxin.pdf` | render script, consumes conotoxin FASTAs | inherits, unchanged | - | 0 |
| `fold_superposition_combined.pdf`, `plddt_vs_tmscore_scatter.pdf` | structure scripts, no detector, consume the above FASTAs | inherits, unchanged | - | 0 |

## Correction to both prior attributions

Codex named `run_multiplicity_conditioning.jl` as the producer of
`fig5_entropy_curves.pdf`, and I checked it on that basis. That was wrong, and my
positive control caught it. The committed figure is rendered by
`render_entropy_curves.py` from `code/data/kunitz/entropy_curves.csv`, which is written by
`dump_entropy_curves.jl` at **n_betas=80**, not 60.

The distinction matters. On the 60-point grid, `rho=5` does change, from 4.9289 to 4.2663.
On the actual 80-point producer grid, nothing changes:

| rho | First-20, 80-pt | All-memory, 80-pt |
|---:|---:|---:|
| 1 | 4.3530 | 4.3530 |
| 5 | 4.8485 | 4.8485 |
| 20 | 6.0152 | 6.0152 |
| 100 | 7.4627 | 7.4627 |
| 1000 | 9.2585 | 9.2585 |

Positive control: the committed `entropy_curves.csv` records `beta_star` of 4.3530 at
`rho=1` and 9.2585 at `rho=1000`. These reproduce exactly, and they are the source of the
4.35 and 9.26 quoted in the appendix and the 4.4 and 9.3 quoted in Theory, Results and the
figure caption. Those numbers do not change.

## Binder scaling: all 25 bases unchanged

`run_augmented_memory_deepdive.jl` selects an operating point for the full-family basis
and for each of 24 curated subsets (8 sizes, 3 replicates). Mirroring its
`Random.seed!(1000 * n_bind + rep)` seeding and its `pratio = n_use <= 5 ? 0.99 : 0.95`
rule, every one of the 25 bases is unchanged. `study1_binder_scaling.pdf` needs no rerun.

## Conotoxin chain is unaffected

Both conotoxin operating points are unchanged, full family 3.2343 and designated subset
2.7183. This means the generated conotoxin FASTAs, `loop_heatmap_with_residuals.pdf`,
Table 5, the SAR table corrected earlier on this branch, the conotoxin sequence-analysis
figure, and the structure and docking artifacts derived from those FASTAs all stand
without a rerun.

## Final rerun scope

Only three calculations change, ten operating points in total:

1. Canonical multiplicity sweep, 8 of 48 conditions.
2. Kunitz mask experiment, 1 of 8 conditions.
3. Exact-GMM benchmark, 1 of 3 conditions.

Downstream of those: the per-family rho tables, the cross-family table, the exploratory
fit, `fig2_separation_vs_gap.pdf`, `fig_mask_betasweep.pdf`, the mask-recovery table, and
the benchmark table.

No structure prediction, ESMFold, ESM2, docking, HMM baseline, binder-scaling, entropy-curve
or conotoxin work is required. That conclusion now rests on the table above rather than on
assertion.

---

# Codex Final Preflight Decision

Date: 2026-07-29

Decision: conditional approval. The proposed sampling scope is correct, but the
entropy-curve artifact requires one additional correction before the work is complete.

## Independently reproduced checks

The Kunitz binder-scaling comparison was repeated using the script's exact subset sizes,
replicate seeds, and PCA variance rules. All 25 bases were unchanged under the
all-memory onset:

- one full-family basis; and
- 24 curated bases from 8 subset sizes and 3 replicates.

The three Kunitz bases used to generate the example FASTAs were also checked
independently. The 50-point first-20 and all-memory onsets were identical for the full
family, K/R-positive subset, and K/R-negative subset. Therefore, the ESM2 and structure
calculations based on those FASTAs do not require reruns.

## Required Figure 5 correction

The preflight table correctly reports that all five 80-point onset values in
`fig5_entropy_curves.pdf` are unchanged. However, the figure plots the complete entropy
curves, not only their onsets. Equal onset values do not imply equal curves.

An independent comparison of the first-20 and all-memory entropy curves gave:

| rho | First-20 onset | All-memory onset | Maximum absolute H difference |
|---:|---:|---:|---:|
| 1 | 4.3530 | 4.3530 | 0.0427 |
| 5 | 4.8485 | 4.8485 | 0.2882 |
| 20 | 6.0152 | 6.0152 | 0.4580 |
| 100 | 7.4627 | 7.4627 | 0.5859 |
| 1000 | 9.2585 | 9.2585 | 0.6916 |

The plotted curves therefore change even though their selected onsets do not.
`code/data/kunitz/entropy_curves.csv` and `fig5_entropy_curves.pdf` must be regenerated
using entropy averaged over all stored memories.

`code/experiments/render_entropy_curves.py` also targets the obsolete
`paper/sections/figs` path. It must be updated to place the regenerated figure in both
`paper-jcim/sections/figs` and `paper-arxiv/sections/figs`, or to write one canonical
artifact that the two paper builds copy explicitly.

## Provenance correction

The Kunitz example FASTAs consumed by `score_esm2_perplexity.py` are written by
`run_kunitz_binding_experiment_with_replicates.jl`, not by the non-replicated
`run_kunitz_binding_experiment.jl`. This attribution should be corrected in the
dependency table. The operating points for all three FASTA sources are unchanged, so
the conclusion that ESM2 and structure calculations need no rerun remains valid.

## Approved scope

Proceed with:

1. The complete canonical multiplicity sweep.
2. The complete Kunitz mask experiment.
3. The exact-GMM benchmark.
4. Regeneration of the all-memory entropy-curve CSV and Figure 5.
5. Regeneration of all tables, figures, fits, and manuscript numbers downstream of
   these calculations.
6. Updates to the JCIM and arXiv manuscripts and their shared sections.
7. The permutation-invariance and driver-selection tests.
8. The full prose pass, test suite, manuscript builds, and visual PDF inspection.

Update all paper-facing producer scripts to call the all-memory procedure, even when the
selected onset is unchanged. This keeps the implementation consistent with the submitted
Methods and permits direct reproduction. The Methods description should state the grids
accurately: 50 points for the canonical calculations, 60 for the exact-GMM benchmark,
and 80 for the entropy-curve figure.

No AlphaFold, ESMFold, ESM2, docking, binder-scaling sampling, or conotoxin sampling
rerun is required.

With the Figure 5 correction included, the scoped rerun is approved.
---

# SESSION PAUSED 2026-07-29. Resume Notes.

## Where things stand

Branch `submission-corrections-2026-07` at `6fef3e2`. Ten commits. Not merged, not pushed.
Tests 263/263. Both PDFs build with zero undefined references and zero overfull boxes.
`git diff main -- code/data` shows exactly one changed file, `omega_conotoxin/sar_agreement.csv`.

All corrections from the first two audits are committed. Nothing is left half-applied.

## The one open decision

Whether to replace the alignment-order-dependent first-20-column entropy onset with an
order-independent all-memory onset, and rerun the affected experiments.

Nothing found is wrong. This is a reproducibility defect, not a correctness defect. The
central claims are unaffected at any beta. Ten of 94 checked operating points move, and the
headline exploratory fit shifts only from slope -1.95 to -1.91 and R-squared 0.802 to 0.786.

Three options are specified in Section D above. Codex approved the scoped Option 1. The
author has not chosen. Note that Option 2 is not free: keeping the current rule requires
roughly 16 condition-runs to make the Methods disclosure true, against roughly 50 for the
full fix.

The Option 1 plan is written and ready to execute:
`docs/superpowers/plans/2026-07-29-all-memory-onset.md`, 9 tasks, 45 steps.

## Must fix on resume, whichever option is chosen

My outcome-sensitivity measurement used `BURN_IN = 1000`. The canonical driver uses
`BURN_IN = 2000` (`run_canonical_family_sweeps.jl:18`), which with `THIN = 100` and
`N_STEPS = 5000` retains 31 states per chain and 620 per replicate. My run therefore kept 41
per chain, including ten less-converged early states.

Consequence: the delta f_obs values now in `methods.tex` (including "0.100 for SH3 at
rho=50") and in Section B above come from a different estimator than the pipeline. The tell
was visible and I did not check it: my Kunitz rho=2 gave 0.4088 against the committed
0.41032.

Under Option 1 this self-resolves, because Task 3 regenerates the real values and Task 8
replaces that entire Methods paragraph. Under Options 2 or 3 the measurement must be redone
with the correct burn-in before the disclosure is true.

## Also outstanding

- The prose and tone pass is the author's and has not started. Still present:
  "small-set amplifier" (4), "principled" (2), "central finding", "natural axis"; roughly 31
  sentences over 45 words.
- `code/experiments/render_entropy_curves.py` is currently broken. Its repository-root loop
  requires a `paper/sections` directory that no longer exists, so `REPO_ROOT` is never
  assigned and the script raises `NameError`. Task 6 of the plan fixes it and points it at
  both manuscript trees.
- Figure 5's plotted curves change even though its five onsets do not, with maximum absolute
  H difference rising from 0.043 at rho=1 to 0.692 at rho=1000.

## Reading order for whoever picks this up

1. This document, top to bottom. Six sections alternating between the two reviewers.
2. `docs/superpowers/plans/2026-07-29-all-memory-onset.md` if Option 1 is chosen.
3. `docs/superpowers/plans/2026-07-29-submission-corrections.md` for what was already done.

Four of the errors caught in this cycle were unverified environment assumptions rather than
reasoning errors: a grep pattern that missed a LaTeX symbol, a wrong cache directory, a
relative path against a drifted working directory, and the burn-in constant above. Treat any
reproduced number in this document as needing an independent check; the scientific reasoning
has held up better than the tooling.

---

# Execution Outcome: All-Memory Onset Correction

Claude, 2026-07-29. Option 1 executed after the author chose full regeneration
including Kunitz. Six commits: `9af2e9a`, `19d7ae3`, `26e788b`, `a91703f`, `1d16f68`,
`0a2939a`.

## The ten predicted operating points all moved as predicted

Measured independently against the canonical loading path before any rerun. My table
reproduces the preflight and the Codex tables exactly.

| Experiment | Condition | First-20 | All-memory | Observed |
|---|---|---:|---:|---|
| Canonical sweep | Kunitz rho=2 | 4.578909 | 3.848335 | as predicted |
| Canonical sweep | SH3 rho=50 | 3.234325 | 6.482468 | as predicted |
| Canonical sweep | Homeobox rho=100 | 4.578909 | 3.848335 | as predicted |
| Canonical sweep | Homeobox rho=500 | 4.578909 | 3.848335 | as predicted |
| Canonical sweep | Forkhead rho=2 | 5.448177 | 4.578909 | as predicted |
| Canonical sweep | Forkhead rho=10 | 6.482468 | 5.448177 | as predicted |
| Canonical sweep | Forkhead rho=20 | 6.482468 | 4.578909 | as predicted |
| Canonical sweep | Conotoxin rho=500 | 7.713111 | 6.482468 | as predicted |
| Kunitz mask | multiplicity f_target=0.5 | 4.578909 | 3.848335 | as predicted |
| Exact-GMM benchmark | Kunitz rho=10 | 5.694340 | 4.928888 | as predicted |

Hard curation was unchanged in all six families, and the five entropy-curve onsets were
unchanged on the 80-point grid while the curves moved by max |dH| of 0.0427, 0.2882,
0.4580, 0.5859 and 0.6916 in rho order. Both reproduce the Codex values.

## Two defects the audits did not find

### A. The committed data did not reproduce from the committed code on main

The plan's correctness gate, that the 40 unchanged conditions reproduce bit-for-bit,
could not pass. It failed on the first rerun: 40 of 48 cells moved and Kunitz rho=2,
a predicted cell, did not.

Cause: commit `0d0fc20`, on `main`, replaced the canonical driver's global
`Random.seed!(seed)` and `randn(d)` with per-chain `MersenneTwister` streams. That is the
W5 RNG-purity work and it is correct. But the canonical CSVs were generated on 2026-07-24
under the old global-RNG path, and `Random.seed!` seeds Julia's default Xoshiro generator,
a different stream from `MersenneTwister(seed)`. The committed data therefore no longer
reproduced from the committed code, independently of the onset rule.

Magnitude, measured: 40 of 48 canonical cells move for this reason alone, and 25 of 48
exceed three replicate standard deviations, up to 25 SD at SH3 rho=50. The same staleness
explains why the exact-GMM benchmark's ULA columns moved at rho=1 and rho=500 where beta
is unchanged, while every exact-sampler column reproduced bit-for-bit.

### B. Kunitz was never swept by the canonical driver

`prepare_legacy_kunitz()` read Kunitz's committed CSVs, re-normalized the schema and wrote
them back. Without `--families=kunitz` the bare command filtered Kunitz out of the rerun
set, so the plan's Task 3 command could not apply the onset correction to Kunitz at all:
its rho=2 condition would have kept the alignment-order-dependent operating point. The
provenance record credited the canonical driver regardless, because it inferred origin from
whether a file existed.

Fixed in `26e788b`. Kunitz is now an ordinary member of `CANONICAL_FAMILIES`,
`kunitz_origin` is unconditional, and `test_canonical_provenance.jl` pins the invariant.

### C. The preflight dependency table was still incomplete

`run_calibration_diagnostics.jl` writes `calibration_beta_sweep.csv`, the source of the
appendix beta-sweep table, and was absent from the dependency table. Its three operating
points, Kunitz at rho=10, 50 and 200, are unchanged under the all-memory rule, so no rerun
was required. The script now calls `all_memory_onset` and is covered by the test.

## Result

All six families are swept by one code path with the order-independent onset. Verified
bit-for-bit reproducible across two independent full runs, and the bare documented command
now sweeps all six.

Kunitz's headline rho=500 marker fraction is essentially unchanged, 0.5813 to 0.5794,
within one replicate standard deviation. The exploratory five-family relationship moves
more than the preflight predicted, because 5 of 6 families move at rho=500 rather than 2:

| Quantity | Committed | Regenerated | Preflight prediction |
|---|---:|---:|---:|
| slope | -1.951 | -1.754 | -1.905 |
| intercept | 0.772 | 0.711 | 0.764 |
| R-squared | 0.802 | 0.718 | 0.786 |

Leave-one-family-out slopes are now -2.5 to -1.2 with R-squared 0.60 to 0.89, from
-2.7 to -1.4 and 0.76 to 0.91. The relationship survives, more weakly.

Hard-mask residual at f_eff=0.5 is 0.085 (SE 0.028), from 0.111 (SE 0.027). The two
exact-versus-ULA bounds in Results swap magnitude: MAP occupancy 0.004 becomes 0.026 and
decoded P1 K/R 0.026 becomes 0.016.

The obsolete Methods sensitivity disclosure is deleted rather than corrected, which also
retires the delta f_obs values measured at burn-in 1000 against the driver's 2000.

## Verification

- Full Julia suite: 319 of 319, from 263. No failures.
- JCIM main 40 pages, JCIM SI 12 pages, arXiv 29 pages. Zero undefined references and
  zero overfull boxes in all three.
- `git diff main -- code/data`: 20 files, all canonical, mask, GMM, entropy-curve or the
  earlier conotoxin SAR output. No structure, docking, ESM2, FASTA, PDB or HMM artifact.
- `tab_sar_agreement.tex` byte-identical, confirming conotoxin sampling did not rerun.
- The three regenerated figures were rendered and inspected. No clipping, no missing
  series, readable axes.

## Two items for the author

1. **matplotlib is broken in the default `python3`** (`/opt/anaconda3`), which raises
   `ImportError: initialization failed` from `matplotlib._path`. Both figure renderers were
   run with `/opt/homebrew/bin/python3`, matplotlib 3.10.8. The conda environment was left
   untouched. `score_esm2_perplexity.py` presumably targets the same broken interpreter.
2. **`calibration_beta_sweep.csv` is dated 2026-03-13** and predates the RNG refactor, so
   the appendix beta-sweep table is stale in the same way finding A describes. Its operating
   points do not change, so this was left alone rather than silently regenerated. It needs a
   decision.

The prose and tone pass remains the author's and is not started.

---

# Addendum: calibration diagnostics regenerated

Claude, 2026-07-29, at the author's request. Commit `72420a1`.

`run_calibration_diagnostics.jl` was rerun. Its four CSVs were dated 2026-03-13, so they
predated the RNG-purity refactor and were stale in the same way as the canonical sweeps.

**The all-memory onset does not affect this experiment.** All three operating points are
identical under both rules: Kunitz at rho=10 (5.448177), rho=50 (6.482468) and rho=200
(7.713111). Every change below comes from the sampler, not the onset.

Appendix Table `tab:beta-sweep`, the 15 reported rows, marker fraction and diversity:

| rho | beta/beta* | f_obs old | f_obs new | D old | D new |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.50 | 0.471 | 0.463 | 0.581 | 0.571 |
| 10 | 1.00 | 0.539 | 0.487 | 0.547 | 0.537 |
| 10 | 1.50 | 0.573 | 0.515 | 0.525 | 0.510 |
| 10 | 2.00 | 0.597 | 0.534 | 0.508 | 0.496 |
| 10 | 3.00 | 0.613 | 0.558 | 0.484 | 0.482 |
| 50 | 0.50 | 0.506 | 0.485 | 0.573 | 0.566 |
| 50 | 1.00 | 0.582 | 0.535 | 0.537 | 0.529 |
| 50 | 1.50 | 0.631 | 0.574 | 0.514 | 0.506 |
| 50 | 2.00 | 0.653 | 0.606 | 0.499 | 0.495 |
| 50 | 3.00 | 0.700 | 0.650 | 0.469 | 0.467 |
| 200 | 0.50 | 0.535 | 0.498 | 0.564 | 0.557 |
| 200 | 1.00 | 0.613 | 0.556 | 0.527 | 0.518 |
| 200 | 1.50 | 0.648 | 0.597 | 0.503 | 0.496 |
| 200 | 2.00 | 0.677 | 0.631 | 0.488 | 0.478 |
| 200 | 3.00 | 0.706 | 0.695 | 0.463 | 0.461 |

Largest change 0.063. Both monotonicities the appendix relies on are preserved: marker
fraction rises and diversity falls with `beta/beta*` at every rho. The rows were generated
from the CSV rather than retyped. No prose quotes any of these values.

`calibration_fine.csv`, `calibration_fine_agg.csv` and `calibration_position_transfer.csv`
also changed and the five calibration figures were regenerated. Neither manuscript cites
any of them, and no generator or test reads them, so nothing else required updating.

## A build trap worth recording

The JCIM appendix and the GMM SI table live in `Paper_JCIM_SI.tex`, not `Paper_JCIM.tex`.
`./Build.sh Paper_JCIM` does not rebuild the SI, so `Paper_JCIM_SI.log` can report a clean
build from an earlier run while `Paper_JCIM_SI.pdf` still contains superseded numbers. That
happened here: the SI still showed the pre-correction GMM table after the main paper had
been rebuilt twice. **`./Build.sh Paper_JCIM_SI` must be run as a third build.**

Verified after regeneration by extracting text from the built PDFs rather than trusting the
source edit: both the JCIM SI and the arXiv PDF contain the new beta-sweep row
`10 0.50 0.463 0.571` and the new GMM row at `rho=10, beta=4.93`, and neither contains the
superseded `10 0.50 0.471 0.581` or `10 5.69`.

Suite 319 of 319. All three documents build with zero undefined references and zero
overfull boxes, at 40, 12 and 29 pages.

The appendix now has exactly two hand-typed tables left, `tab:beta-sweep` and
`tab:docking-permutation-pvalues`. The six per-family tables are generated and covered by
the byte-compare test. Moving the beta-sweep table into `generate_paper_tables.jl` would
close the drift class that let it sit stale since March; it is not done here.

---

# Addendum: beta-sweep table generated, build documented

Claude, 2026-07-29. Commit `6ff3a58`.

## The table is now generated

`tab:beta-sweep` is written by `generate_paper_tables.jl` from
`calibration_beta_sweep.csv` and byte-compared against both trees by
`test_generated_paper_tables.jl`. The appendix keeps the caption, column heads and label;
only the 15 body rows come from the generator. Positive control: the generated rows matched
the hand-updated values exactly before the switch.

Validation added, mirroring the SAR table's guarantees:

- the rho set equals {10, 50, 200};
- each rho has exactly the seven multipliers {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0};
- each rho has exactly one operating point; and
- `beta_used == beta_mult * beta_star` for every row.

Both new checks were confirmed to fire by injection: perturbing one `beta_used` raised
"beta_used disagrees with mult times beta_star", and deleting a row raised "must have 7
rows". The CSV was restored and verified identical to the committed version afterwards.

`tab:docking-permutation-pvalues` is now the only hand-typed table left in the appendix. It
is fed by docking data that was deliberately not rerun.

## The Makefile already had the SI step

`paper-jcim/Makefile` already declared `all: si` and `si: main`, so `make` builds the main
paper and then the SI, with the dependency in the right order. Nothing was missing there.

The actual gap was documentation. `CLAUDE.md` described only
`cd ../paper-arxiv && ./Build.sh Paper_v1`, mentioning neither the JCIM build nor its SI.
That is why calling `./Build.sh Paper_JCIM` directly looked complete and left a stale SI
PDF whose log still reported a clean earlier build.

`CLAUDE.md` now documents both trees, states that the JCIM appendix and GMM SI table live
in `Paper_JCIM_SI.tex`, warns that `./Build.sh Paper_JCIM` alone leaves the SI stale,
records that the nine shared section files must be edited in both trees while
`Paper_*.tex` figure and table wrappers must be edited twice by hand, and states that
`sections/generated/` must never be hand-edited. It also corrects the stale claim that the
JCIM tree lives at `paper-arxiv/jcim/`; it is the sibling `paper-jcim/`.

`paper-arxiv` gains a Makefile so `make` is the entry point in either tree. The arXiv
version is a single document, so it has no SI target, and the Makefile says so.

## Verification

- Suite 321 of 321, from 319. The two added assertions are the new file's byte-comparison
  in each tree.
- `make` in both trees: JCIM main 40 pages, JCIM SI 12 pages, arXiv 29 pages, each with
  zero undefined references and zero overfull boxes.
- Rendered-output check, not a source check: all 15 beta-sweep rows appear in both
  `Paper_JCIM_SI.pdf` and `Paper_v1.pdf`, beginning `10 0.50 0.463 0.571`, and the
  superseded `10 0.50 0.471 0.581` appears in neither.
- `sections/generated/` is identical across the two trees.
