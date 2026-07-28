# Remaining issues after the gap-handling repair

Written 2026-07-28 by Claude (Opus 5), after independently verifying commit `094a66e`
(`fix: test decoder pipeline and correct gap metrics`).

**None of the three items below is an implementation defect, and none affects any number
in the manuscript.** They are recorded so the next person does not rediscover them the
hard way.

Verification context: all seven claims in the handoff were checked and hold. Suite is
121/121, working tree clean, `git diff --check` clean. The characterization values were
reproduced by an independent script that never reads the test file, matching all twelve
literals exactly to six decimals. Full verification summary in section 4.

---

## Codex audit and resolution (2026-07-28)

### Issue 1: confirmed, but the original file/column description was inaccurate

There are exactly 23 legacy tracked CSVs with a gap-sensitive field, but they do not
all contain gap-sensitive `novelty` values:

- The binding and HMM-baseline pipelines define `novelty` as PCA cosine distance,
  which is unaffected. Their `seqid` fields are the stale fields.
- `approach_comparison.csv` has no gap-sensitive metric.
- No tracked `binding_experiment.csv` exists; only its raw-replicate and aggregated
  variants exist.

These artifacts date to March 2026 (`ebe5449` or `4b0064e`), are not direct manuscript
inputs, and may differ in more than a single metric if regenerated with the current
experiment code. The chosen resolution is therefore documentation rather than
silently combining current metrics with historical outputs. The exact 23-file and
column inventory, provenance, and regeneration policy now live in
`code/data/kunitz/README.md`.

### Issue 2: not a consistency defect; the functions have different contracts

Mapping an ambiguity code such as `B` to a zero one-hot block means the fixed
20-channel representation cannot encode it. It does not imply that `B` is an
alignment gap for every downstream metric.

- `sequence_identity` excludes alignment gaps and conservatively compares other
  symbols literally. Contrary to the report below, `B` *can* match another `B`.
- `valid_residue_fraction` must count a non-standard, non-gap symbol in its
  denominator; otherwise a sequence containing invalid symbols could still score
  100% valid.

No numerical behavior was changed. The three contracts are now explicit in
`Protein.jl`, and synthetic tests pin literal ambiguity-code comparison and the
validity penalty. The Kunitz characterization remains `0.836366`.

### Issue 3: false as written

The `~` branch was already exercised before this report was written:

- `decode_onehot` coverage used a synthetic `~` input;
- `sequence_identity("A.C~D-", "ATCXDQ")` required `~` to be skipped; and
- `valid_residue_fraction(".-~")` required all three gap symbols to be recognized.

A direct `is_alignment_gap` assertion has now been added as an additional guard, but
there was no uncovered branch.

After these documentation and contract tests, the full suite passes 126/126.

The two pre-existing working-tree deletions (`no-test-decode-issues.md` and
`paper-submission-audit.md`) were not part of this audit and were left untouched.

---

## Issue 1: stale derived artifacts (23 tracked Kunitz CSVs)

**Class:** stale derived artifact. Not a defect, not a manuscript problem.

`094a66e` correctly regenerated the two CSVs whose numbers the manuscript cites
(`data/kunitz/mask_experiment.csv`, `data/kunitz/mask_betasweep.csv`). It did not
regenerate the other tracked Kunitz CSVs that carry a `novelty` column computed through
`nearest_sequence_identity`.

Those columns compare generated sequences against the **stored Kunitz alignment**, which
uses `.` as its gap character. Before `094a66e`, `sequence_identity` excluded only `-`, so
every `.` position was counted as a residue mismatch. Those committed values therefore no
longer reproduce under current code.

### Affected files

All under `code/data/kunitz/`:

| CSV family | produced by |
|---|---|
| `approach_comparison{,_aggregated,_raw_replicates}.csv` | `run_all_approaches_with_replicates.jl` |
| `binding_experiment{,_aggregated,_raw_replicates}.csv` | `run_kunitz_binding_experiment{,_with_replicates}.jl` |
| `deepdive_consensus{,_aggregated,_raw_replicates}.csv` | `run_augmented_memory_deepdive{,_with_replicates}.jl` |
| `deepdive_interpolation{,_aggregated,_raw_replicates}.csv` | same |
| `deepdive_mixed{,_aggregated,_raw_replicates}.csv` | same |
| `deepdive_scaling{,_aggregated,_raw_replicates}.csv` | same |
| `deepdive_weighted{,_aggregated,_raw_replicates}.csv` | same |
| `multiplicity_generation{,_aggregated,_raw_replicates}.csv` | `run_multiplicity_conditioning{,_with_replicates}.jl` |
| `baseline_comparison.csv` | `run_hmm_baseline.jl` |

Each of those scripts calls `nearest_sequence_identity` (verified by grep; counts are
1 to 2 call sites each).

### Why this is not a manuscript problem

Every novelty number in the paper was traced to its source:

- `tab:mask-recovery` (`Paper_v1.tex:237-243`) and `results.tex:148-151`
  (`0.42`, `0.16`, `0.32`) come from the two regenerated CSVs. Current values 0.42344,
  0.15679 and 0.32393 still round correctly.
- `discussion.tex:61` (`0.46 vs. 0.60`) is conotoxin. Its alignment uses only `-`, which
  was already excluded before the fix, so it is unaffected.
- `Paper_v1.tex:303` and `Paper_v1.tex:363` use `sample_novelty`
  (`code/src/Utilities.jl:100`), a **PCA cosine distance**, not sequence identity. Not
  gap-sensitive at all. The caption at `Paper_v1.tex:356` states this explicitly.

No manuscript number is drawn from any file in the table above.

### Why it still matters

Anyone who re-runs one of those experiments will get a diff in the `novelty` column that
looks like a scientific result change and is not one. That is precisely the failure mode
that cost this project a week. Two acceptable resolutions:

1. Regenerate them, so committed artifacts reproduce under current code. Cheap for the
   non-replicate variants; the `_with_replicates` ones cost sampling time.
2. Leave them and add a note (a line in `code/data/kunitz/README` or the postmortem)
   recording that these `novelty` columns predate the `094a66e` gap fix, are not used by
   the manuscript, and will shift on regeneration.

Option 2 is defensible and cheap. What is not defensible is leaving them undocumented.

### Reproduction

```bash
cd code
# confirm which scripts use the gap-sensitive metric
grep -c nearest_sequence_identity experiments/run_augmented_memory_deepdive.jl   # 1

# confirm the mechanism on an already-regenerated file
cd .. && git diff 4fe9ad9..094a66e -- code/data/kunitz/mask_experiment.csv
# only the novelty and novelty_se columns move; beta, p1_kr, p1_kr_se,
# diversity, kl and n are byte-identical across all 8 rows
```

---

## Issue 2: gap unification did not cover non-standard residues

**Class:** test-design / consistency concern. Measured impact +0.000172. Immaterial today.

`094a66e` introduced a single shared definition
(`code/src/Protein.jl:15`, `ALIGNMENT_GAP_CHARACTERS = ('.', '-', '~')` and
`is_alignment_gap`) and applied it to `clean_alignment`, `sequence_identity` and
`valid_residue_fraction`. That is correct and it fixed a real bug.

It did not extend to **non-standard residue codes** (`B`, `X`, `Z`, `J`, `U`, `O`), which
are handled inconsistently across the same three layers:

| function | line | treatment of `B` |
|---|---|---|
| `onehot_encode` | `Protein.jl:171` | all-zero block, **identical to a gap** |
| `decode_onehot` | `Protein.jl:196` | can never emit `B` |
| `sequence_identity` | `Protein.jl:275` | compared as a residue, so **can never match** |
| `valid_residue_fraction` | `Protein.jl:299` | counted in the denominator, not the numerator |

`onehot_encode`'s own docstring says "Gap positions and non-standard amino acids map to
all-zeros," so the encoder deliberately lumps the two classes. The metrics do not.

### Measured impact

The Kunitz seed alignment is the only tracked fixture containing a non-standard residue:
exactly one `B`, in a cleaned matrix of 99 x 53 = 5247 cells (0.019%). Its all-zero
one-hot block was confirmed directly (`max=0.000, sum=0.000`, identical to a gap).

```
Kunitz unit_identity as shipped (gaps excluded, B compared) : 0.836366
Kunitz unit_identity if B were also excluded                : 0.836538
delta                                                       : +0.000172
```

That is 12x inside the characterization test's `atol=0.002`, so nothing fails and no
reported value changes.

### Recommendation

Low priority, but it is the same class of inconsistency that was just fixed, and it will
bite harder on any future alignment with more ambiguity codes. Either fold the
non-standard codes into a shared `is_uninformative(c)` predicate used by the metrics, or
add a comment at `Protein.jl:275` recording that non-standard residues are deliberately
compared rather than skipped.

If changed, the Kunitz characterization literal in
`code/test/test_decoder_characterization.jl` moves from `0.836366` to `0.836538`, which is
inside `atol` and so would **not** trip the test. Update the literal anyway if the change
is made, or the test silently stops characterizing what it claims to.

---

## Issue 3 (minor): the `~` branch is unexercised

`~` is included in `ALIGNMENT_GAP_CHARACTERS` but appears in **no** tracked alignment. A
character inventory of the raw parsed sequences found:

| fixture | non-amino-acid characters present |
|---|---|
| Kunitz `PF00014_seed.sto` | `.` `B` |
| SH3 `PF00018_seed.sto` | `.` |
| WW `PF00397_seed.sto` | `.` |
| Homeobox `PF00046_seed.sto` | `.` |
| Forkhead `PF00250_seed.sto` | `.` |
| conotoxin FASTA | `-` |

Including `~` is correct for Stockholm generally and costs nothing. Note only that no
fixture covers it, so a regression in that branch would not be caught. A one-line
synthetic assertion in `code/test/test_encode_decode.jl` would close it.

This table also explains the repair's observed magnitudes: the five Pfam families use only
`.`, so before the fix `sequence_identity` excluded nothing at all in them, which is why
they moved most (SH3 +0.019, WW +0.015) while conotoxin, which uses only `-`, did not move.

---

## 4. Verification summary (all seven handoff claims hold)

| # | claim | verdict | how checked |
|---|---|---|---|
| 1 | diagnosis right, code not verbatim | ✓ | diff; deterministic fixture replaces random one; characterization reads tracked files, never downloads |
| 2 | `.`/`-`/`~` gap unification correct | ✓ | character inventory above; `parse_stockholm:79` uppercases so lowercase insertions stay residues |
| 3 | values reproducible, not circular | ✓ | independent script, does not read the test; all 12 values match **exactly to 6 decimals** |
| 4 | `atol=0.002` well-calibrated | ✓ | guarded bug moved values up to +0.019 (and +0.024 on score identity), ~10x tolerance; one residue flip moves the mean ~0.0002 |
| 5 | rerun touched only gap-sensitive metrics | ✓ | column-by-column over all 8 rows; only `novelty`/`novelty_se` move |
| 6 | no other manuscript number affected | ✓ | every paper novelty number traced to source; see Issue 1 |
| 7 | paper consistent at 0.32, 0.730--0.943, 0.996--1.000 | ✓ | checked against independently computed ranges 0.729774--0.943108 and 0.995584--1.000000; no pre-fix values survive |

Also confirmed: `download_pfam_seed` caches to `<dir>/<pfam_id>_seed.sto`, exactly the
paths the characterization test reads, so the test fixtures are the same files the
production pipeline uses and all six are tracked in git.

No implementation defect was found. No disagreement about scientific interpretation.

### Commands

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git diff 4fe9ad9..094a66e
git diff --check
cd code && julia --project=. test/runtests.jl    # 121/121
```
