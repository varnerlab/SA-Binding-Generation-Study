# Decoder audit: final resolution

Consolidated 2026-07-28. This is the authoritative account. Earlier drafts of this file
contained superseded claims and an incorrect file/column table; git history preserves them.

---

## Current verified state

- Branch `main` at `29c3f78`.
- Full test suite: **126/126 passing**.
- **No manuscript number is affected by anything in this document.**
- No open implementation defect. No disagreement about scientific interpretation.

The chain of work: `094a66e` added encode/decode test coverage and fixed a real
`sequence_identity` gap-handling bug; `02df680` and `29c3f78` recorded the review of that
repair and made three function contracts explicit.

---

## Issue 1: legacy artifacts with stale gap-sensitive fields

**Class:** stale derived artifact. Documented, not regenerated. Not a manuscript problem.

`094a66e` corrected `sequence_identity` to exclude all three alignment gap symbols
(`.`, `-`, `~`). Previously only `-` was excluded, so in the five Pfam families, whose
seed alignments use `.` exclusively, every gap position was counted as a residue mismatch.

It regenerated the two CSVs the manuscript actually cites, `mask_experiment.csv` and
`mask_betasweep.csv`. Exactly **23 other tracked Kunitz CSVs** carry a gap-sensitive field
that predates the fix and will move if their producers are re-run.

### Corrections to the original report

The first version of this document named the wrong files and columns. Corrected:

- **`novelty` is not always the gap-sensitive field.** In the binding and HMM-baseline
  pipelines, `novelty_*` is `sample_novelty` (`code/src/Utilities.jl:100`), a **PCA cosine
  distance**, which is not gap-sensitive. Their **`seqid_*`** fields are the stale ones.
- **`approach_comparison.csv` is unaffected.** Its columns are
  `approach,n_generated,p1_kr_frac,mean_valid_frac,kl_aa` — no gap-sensitive metric at all.
- **There is no tracked bare `binding_experiment.csv`.** Only `_aggregated` and
  `_raw_replicates` variants exist.

The count of 23 was right; the attribution was not. The error came from inferring column
semantics from per-script grep counts rather than reading the CSV headers.

### Definitive inventory

`code/data/kunitz/README.md` holds the per-file, per-column inventory, the originating
commits (March 2026, `ebe5449` / `4b0064e`), and the regeneration policy. Use it, not any
table in this file.

### Why documentation rather than regeneration

These artifacts date to a March 2026 experiment state. Re-running their producers with
current code could change more than the one gap-sensitive column, so regenerating would
silently blend historical outputs with current metrics. Recording provenance is the honest
option. The point of documenting it at all is that a future re-run will produce a diff that
looks like a scientific result change and is not one.

### Manuscript numbers, traced

- `tab:mask-recovery` (`Paper_v1.tex:237-243`) and `results.tex:148-151`
  (`0.42`, `0.16`, `0.32`) come from the two regenerated CSVs. Current values 0.42344,
  0.15679 and 0.32393 round correctly.
- `discussion.tex:61` (`0.46 vs. 0.60`) is conotoxin, whose alignment uses only `-`,
  already excluded before the fix.
- `Paper_v1.tex:303` and `:363` use the cosine `sample_novelty`, stated in the caption at
  `Paper_v1.tex:356`.

---

## Issue 2: three deliberate contracts, not an inconsistency

**Class:** closed. No numerical behavior changed; the Kunitz characterization value remains
`0.836366`.

The original report framed the handling of ambiguity codes such as `B` as an inconsistency
across `onehot_encode`, `sequence_identity` and `valid_residue_fraction`. That framing was
wrong. The three functions have different, deliberate contracts:

- **`onehot_encode`** maps `B` to an all-zero block because a fixed 20-channel
  representation *cannot* encode an ambiguity symbol. That is a statement about
  representability, not a claim that `B` is an alignment gap.
- **`sequence_identity`** excludes alignment gaps and compares every other symbol
  literally, so **`B` does match `B`**. The original report's "can never match" was false in
  general. It holds only in the stored-versus-decoded direction, because `decode_onehot`
  cannot emit `B`, and that narrow case is all that was actually measured (+0.000172 on
  Kunitz, one `B` in 5247 cells).
- **`valid_residue_fraction`** must count a non-standard, non-gap symbol in its
  denominator. Otherwise a sequence made entirely of ambiguity codes would score 100% valid.

`29c3f78` made all three contracts explicit in `Protein.jl` and pinned them with synthetic
tests covering literal ambiguity-code comparison and the validity penalty.

---

## Issue 3: withdrawn, the claim was false

The original report asserted that the `~` branch was unexercised because no tracked
alignment contains `~`. That conclusion was false. `~` was already covered at `094a66e` by
synthetic tests in `code/test/test_encode_decode.jl`:

- a `~` in the decoder input char matrix (line 20);
- `sequence_identity("A.C~D-", "ATCXDQ")`, which requires `~` to be skipped (lines 55-58);
- `valid_residue_fraction(".-~")`, which requires all three gap symbols to be recognized
  (lines 62-63).

`29c3f78` added a direct `is_alignment_gap` predicate assertion as an extra guard, but
there was no uncovered branch to begin with.

The underlying character inventory remains useful, and explains the repair's magnitudes.
The five Pfam seed alignments contain only `.` (Kunitz additionally has one `B`), and the
conotoxin FASTA contains only `-`. So before the fix `sequence_identity` excluded nothing
at all in the Pfam families, which is why they moved most (SH3 +0.019, WW +0.015) while
conotoxin did not move.

---

## Review outcome

Verification of `094a66e` confirmed all seven of its claims, including an independent
reproduction of the characterization values by a script that never reads the test file,
matching all twelve literals exactly to six decimals.

The subsequent review by Claude then produced three findings, of which one was substantive
but misattributed, one was wrongly framed, and one was false. The recurring error was
**generalizing from a partial check**:

- an alignment-fixture inventory generalized to a claim about branch coverage;
- per-script grep counts generalized to CSV column semantics;
- and earlier in the same episode, stored-memory measurements generalized to a decoder
  recommendation for generated samples.

The corrective in each case was the same and is cheap: read the artifact you are making a
claim about. Headers before tables, test files before coverage claims, generated samples
before decoder recommendations.

Related records: `docs/2026-07-28-decoder-postmortem.md` (the decoder episode),
`no-test-decode-issues.md` (the report that opened this work),
`code/data/kunitz/README.md` (legacy artifact provenance).
