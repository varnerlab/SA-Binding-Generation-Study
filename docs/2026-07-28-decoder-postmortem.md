# Decoder ablation post-mortem, 2026-07-28

Written by Claude (Opus 5). This is an account of a day's work that produced a wrong
recommendation, was caught by an independent audit, and was then deleted. It is written
for someone who was **not** in the conversation.

Jeffrey's summary of why this file exists: he could not follow what was being done and
wants another person to check it.

---

## 1. Current state of the repository (check this first)

**Nothing is broken. Nothing was lost. The repo is exactly where it was on 2026-07-24.**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git log --oneline -1     # dba7e6e docs: Plan 3 implementation plan
git status --short       # empty
git branch               # only: main
```

`main` is at `dba7e6e`, identical to `origin/main`, working tree clean, only one branch.
All the work described below happened on a branch that has been deleted. **`main` never
contained a single artifact from it.** You can verify that claim directly:

```bash
git ls-tree -r main --name-only | grep -E "pca_decoder|decoder_pushforward"   # no output
```

The only surviving material is outside the repo, in
`/Users/jdv27/Desktop/julia_work/plan3-salvage-2026-07-28/`.

This file itself is untracked and can be deleted freely.

---

## 2. What was attempted, and why

Background. The codebase generates protein sequences by: one-hot encoding an aligned
protein family, running PCA, **normalizing each PCA score vector to unit length**, running
a Langevin sampler over those unit-norm "memories", and then decoding samples back to
sequences by inverse PCA plus a per-position argmax.

A July audit had found that decoding a *stored* memory back to its own sequence only
recovers about 83% of residues on Kunitz, rather than ~100%. The paper had interpreted
that loss as an intrinsic property of the encoder/decoder and made it the central
empirical claim ("the calibration gap").

Plan 3 (`docs/superpowers/plans/2026-07-24-plan3-pca-decoder-ablation.md`) was designed to
test an alternative explanation: that the loss is an artifact, because unit normalization
*discards* each memory's PCA score norm ("radius") and the decoder never puts it back.
Jeffrey asked me to execute that plan.

---

## 3. Timeline

All on 2026-07-28. Branch `arxiv-rev-plan3-decoder`, created from `main` at `dba7e6e`.

**Tasks 1 to 4 executed and reviewed** (each task implemented by one subagent, then
reviewed by a separate subagent):

| commit | what it did |
|---|---|
| `002f08e`, `305f1af` | added `memory_radii`, `nearest_memory_radius`, and an optional `scale` kwarg on `decode_sample`. Default behavior unchanged. |
| `26a5a71` | extracted a shared sample-generation helper so the ablation and the canonical driver produce identical trajectories. Verified byte-identical: regenerating Kunitz produced no diff. |
| `0616db2` | measured stored-memory reconstruction fidelity for all six families. |
| `2e66e7f` | the "gate": decoded the same generated Kunitz samples two ways and compared. |

**Findings that are correct and were later confirmed by the independent audit:**

- The stored PCA radius is discarded by `build_memory_matrix`.
- Stored-memory round-trip identity is 0.720 to 0.943 under the current decoder, and
  0.976 to 0.999 if the memory's own radius is restored first. (Kunitz: 0.833 vs 0.996,
  reproducing the July audit independently.)
- The gate's two arms genuinely used the same generated samples, and its baseline arm
  reproduced the committed canonical numbers exactly.
- Rescaling generated samples changes decoded marker fractions far beyond replicate noise.

**Then I made the error.** I proposed a fix: decode at a global constant per family, the
mean memory radius r̄. Jeffrey approved it on my recommendation, and also approved making
it the canonical pipeline.

**A validation run then contradicted my own recommendation.** I had extended the next task
to check, at ρ=1 (no conditioning applied), whether the decoded marker fraction matched
the family's natural marker fraction. Under r̄ it did not: r̄ was *worse* than the existing
decoder on three of six families. I stopped, wrote up the failure in `claude-fucked-me.md`,
and told Jeffrey the decision was falsified.

**Jeffrey brought in an outside expert (Codex)** who audited the code, data, manuscript and
my claims, and found that my *diagnostic reasoning* was wrong at a deeper level than I had
realized. Details in section 4.

**Jeffrey chose to delete everything**, keeping only a set of manuscript corrections the
audit had made. That was done. Section 6.

---

## 4. The substantive error (the important part)

I invented an acceptance test: at ρ=1 the conditioning weights are uniform, so I claimed
the fraction of generated sequences carrying the marker (`f_obs`) must equal the fraction
of designated memories in the family (`f_eff`).

**That is wrong, and it is the root of everything I got wrong afterward.**

The correct relation is a pushforward through a noisy channel. Let `C` be the latent
mixture component a sample came from:

```
TPR = P(decoded marker-positive | C is designated)
FPR = P(decoded marker-positive | C is background)

f_obs = f_eff * TPR + (1 - f_eff) * FPR
```

`f_eff` is a property of the latent distribution. `f_obs` is a property of the *decoded
output*. They are different random variables. Consequences:

- `f_obs` can legitimately exceed `f_eff` when false positives outweigh false negatives,
  so the "overshoot" I treated as proof of decoder bias proves nothing.
- A decoder that ignores its input entirely and emits marker-positive at the right base
  rate passes my test perfectly while carrying zero information.
- Therefore **tuning a decoder to match that marginal does not improve fidelity**, and may
  degrade it.

The audit demonstrated the last point empirically. Using 20,000 exact draws per family
with known component labels, and scoring by Youden J (= TPR − FPR, which does not reward a
decoder merely for matching an imbalanced base rate), the radius rescaling I recommended
made the Kunitz marginal look better while slightly *reducing* discrimination
(J 0.193 → 0.185).

**Other errors of mine the audit caught:**

- **"Restoring the radius" is not meaningful for generated samples.** A stored memory has a
  known discarded radius because its source sequence exists. A generated sample has no
  source sequence and therefore no discarded radius. Multiplying it by some memory's radius
  scales its signal *and* its noise equally. It is an intervention, not an inverse.
- **The nearest memory is usually not the generating component** (only 21% to 38% of the
  time), so "use the nearest memory's radius" was not recovering anything.
- **A category error on conotoxin.** Its designated set is defined by 23 accession IDs
  (strong binders), but its marker statistic is Tyr at an alignment column. Those two labels
  agree for only 86.5% of stored sequences. I compared one against the other. For the other
  five families the two labels coincide exactly, which is why I did not notice.
- **"No decode scale works for WW" was false.** I searched only scales ≥ 1. It crosses
  below 1. Asserting a universal negative from a bounded search was sloppy.
- **I overstated what I had established** about the sampler's noise level being the unique
  cause. High-dimensional variance ratios do not determine per-position argmax accuracy,
  which I had flagged as a caveat and then talked past anyway.

**What the audit agreed was true** is the list in section 3. The measurements were sound.
The interpretation built on them was not.

---

## 5. One open technical question a reviewer should judge

I checked the audit's work rather than just accepting it, and found one thing its summary
omits. Its own data file (`extra/decoder_pushforward_audit.csv` in the salvage bundle)
scored **four** decoders, but the summary table reports only two.

The fourth, `mean_radius_times_direction` (rescale a sample to r̄ along its own direction),
beats or ties the current decoder on Youden J in **all six families**:

| family | current J | direction-only J | difference |
|---|---:|---:|---:|
| Kunitz | 0.1927 | 0.1927 | +0.0000 |
| SH3 | 0.1818 | 0.2800 | +0.0981 |
| WW | 0.1529 | 0.1581 | +0.0052 |
| Homeobox | 0.1092 | 0.1610 | +0.0518 |
| Forkhead | 0.1328 | 0.1372 | +0.0044 |
| Conotoxin | 0.3023 | 0.3081 | +0.0058 |

The audit's conclusion that "no radius decoder is supported" is correct for the two
variants it tabulated. It does not cover this one.

**Caveats, stated plainly:** four of those six gains are small enough to need a paired
significance test, which cannot be run from the aggregate CSV. SH3 and Homeobox are large
enough not to be in doubt. Youden J is one criterion; the audit's fuller bar (whole-sequence
fidelity, novelty, diversity, stability, biological validation) is the right one, and this
decoder has not been through it. And this does **not** rehabilitate my recommendation:
I recommended r̄·ξ, which is the *worst* of the four on this data.

This is worth a second opinion. It is not worth acting on without one.

---

## 6. What was deleted, and what was saved

Deleted: branch `arxiv-rev-plan3-decoder` (5 commits), all uncommitted work including the
audit's manuscript edits and rebuilt PDF, and all loose files. `main` was not touched.

Saved to `/Users/jdv27/Desktop/julia_work/plan3-salvage-2026-07-28/`:

**`manuscript-corrections.patch`** — the one item with clear standing value. 79 insertions
and 38 deletions across 8 paper files, **no numbers changed**. It fixes errors that are
present on `main` right now and that have nothing to do with the radius argument:

1. the score equations omit the factor β (already fixed in the Julia by an earlier plan,
   never fixed in the paper)
2. the discussion claims finite-step ULA samples the target "with no approximation", which
   is false; discretization and mixing error are real
3. the appendix says SA latent states live on the unit sphere; only the *stored memories* do
4. the calibration gap is attributed to PCA compression alone
5. the six-family dimensionality ranges are stale, left over from when the study had four
   families
6. adds the pushforward identity from section 4 above

Apply with `git apply /Users/jdv27/Desktop/julia_work/plan3-salvage-2026-07-28/manuscript-corrections.patch`
from the repo root, or discard it.

**`extra/`** — the audit's write-up (`decoder-pushforward-audit.md`), its script and data
(`run_decoder_pushforward_audit.jl`, `decoder_pushforward_audit.csv`,
`decoder_ww_subunit_scale.csv`), and my earlier self-report with the audit's correction
prepended (`claude-fucked-me.md`). Delete the folder if unwanted.

---

## 7. A mistake I made while cleaning up

My first cleanup command was malformed. I piped `git checkout main` into `tail`, which
masked its exit code, so when the checkout aborted the rest of the `&&` chain ran anyway
and `git clean -fd` executed on the branch rather than on `main`. It removed more
directories than intended, including paths under
`code/data/omega_conotoxin/docking_validation_full_rerun_.../` and
`code/figs/docking_validation/`.

**No data was lost.** Verified before doing anything else: `git status` is completely
clean and all 147 tracked files under the docking directory are present, which is
conclusive for tracked content. What was removed were *empty* directories, which git does
not track. Had any contained untracked files they would have appeared as `??` in the
status output taken minutes earlier, and none did.

Reviewer can confirm:

```bash
git status --short                                                    # empty
git ls-files code/data/omega_conotoxin/docking_validation_full_rerun_20260319_184154/ | wc -l   # 147
find code/data/omega_conotoxin/docking_validation_full_rerun_20260319_184154 -type f | wc -l    # 147
```

The command should have used `set -e` and no pipe. The rerun did.

---

## 8. What a reviewer should actually check

In priority order:

1. **Confirm `main` is untouched** (section 1). Two commands. If this holds, nothing else
   here can have damaged the project.
2. **Read `manuscript-corrections.patch`** and decide whether those six paper fixes are
   right. They are independent of everything else in this document and are the only thing
   with a pending decision attached.
3. **Judge the open question in section 5.** Does the direction-only decoder deserve a
   proper evaluation, or is the audit's blanket "no radius decoder" the right call?
4. **Sanity-check my account of the error in section 4** against the audit's own write-up
   in `extra/decoder-pushforward-audit.md`. I have tried to state it against my own
   interest, but I am not a neutral party here.

## 9. What is left to do on the project

Independent of all of the above, the pre-existing plan of record was: Plan 5 (prose and
narrative reframe) and Plan 6 (build hygiene: JCIM dangling references, gitignoring build
artifacts, figure-path redirect, Python dependency pinning). Neither was started. Plan 3 is
dead and its plan and spec documents on `main` should be marked superseded so no one
resumes them.

The canonical data from Plan 2 (all six families, replicated sweeps) and the mathematics
from Plan 1 (the exact Gaussian-mixture identity and the corrected score) are unaffected by
any of this and remain committed on `main`.
