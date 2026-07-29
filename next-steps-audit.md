# Remaining work on the paper

Written 2026-07-28. State checked against the repository, not recalled.

Baseline: `main` at `988e9a6`, clean, in sync with `origin/main`. Test suite 126/126.

Claims below are marked **[verified now]** where I checked them against the repo in this
session, and **[from the July audit]** where they come from
`paper-submission-audit.md` and still need re-verification before acting.

---

## Closed workstreams

| | status |
|---|---|
| **W1** theory corrections + SI derivation | closed. Plan 1 at `9c974fd`; manuscript prose corrections at `72fccb7`. |
| **W2** one canonical CSV | closed. Plan 2 at `1b04df4` / `0dd17e9` / `92c9095`. |
| **W3** exact-sampler baseline + decoder | closed, but not the way it was planned. The exact-GMM baseline landed in Plan 1. The decoder half resolved by the correct characterization reaching the manuscript (pushforward identity, "model-defined map, not an encoder inverse") rather than by the ablation succeeding. See `docs/2026-07-28-decoder-postmortem.md` and `remaining-issues-list.md`. |
| **W5** code correctness, tests | substantially closed. 52 -> 126 tests; the encode/decode coverage hole is closed (`094a66e`, `29c3f78`). |

**[verified now]** The arXiv build is healthy: 27 pages, zero undefined references,
undefined citations, or rerun-needed warnings in `paper-arxiv/Paper_v1.log`.

---

## W4. Biology and label rigor — do this first

This is the only remaining workstream containing numbers that are **wrong**, as opposed to
framing that could be better. Everything else is prose or plumbing.

### 4.1 Docking provenance is unresolved, and worse than the audit recorded

**[verified now]** Three competing files exist, not two. Nothing in the repo marks which is
canonical:

```
25 rows   code/data/omega_conotoxin/docking_validation/docking_results.csv
 2 rows   code/data/omega_conotoxin/docking_validation_full_rerun_20260319_184154/docking_results.csv
34 rows   code/data/omega_conotoxin/docking_validation_stale_20260319_172834/docking_results.csv
```

The manuscript table draws on the 25-row file. The newest directory by timestamp
(`_full_rerun_20260319_184154`) contains two rows. The 34-row file is named "stale".

**This needs a decision that is not mine to make:** which run is canonical, and whether the
2-row rerun means the full rerun was abandoned partway. Until that is settled, any docking
number in the paper rests on an unlabeled choice among three files.

Nothing should be regenerated here until the provenance question is answered.

### 4.2 Numbers to correct

**[from the July audit]** Re-verify each before editing:

- Kunitz HMM TM-score standard deviation: about 0.19 in the data, reported as 0.10.
- iPTM about 0.10 and interface pLDDT about 37 to 41. At those values the permutation test
  shows **no detected difference among uniformly low-confidence predictions**, not
  preserved geometry. The current framing overstates it.
- Switch to "predicted structurally plausible" language throughout.

Note `paper-arxiv/sections/appendix.tex:304-318` is a table of permutation-test *p-values*
(iPTM 1.0000, pTM 0.7502, confidence 0.4752), not the score values themselves. Do not
confuse the two when editing.

### 4.3 Label rigor

**[from the July audit]** Reclassify the family markers (Kunitz Lys/Arg at P1, SH3 Trp, WW
mid-position, Homeobox Gln, Forkhead H/N) as **data-selected sequence markers** unless
independently mapped to a canonical structural residue with biological support. Soften
three specific claims: Kunitz K/R at P1 is a specificity proxy, not proof of binding; 100%
marker recovery after filtering on that marker is retention, not validation; the conotoxin
background is unannotated, not demonstrated non-binders.

Build the conotoxin accession-level evidence table (confirmed / predicted / homolog /
unannotated) with explicit classification rules, and soften "all binding determinants" to
enriching several reported determinants while retaining the cysteine scaffold and Tyr13.

**[verified now, connects to the above]** The conotoxin label ambiguity also surfaced
independently during the decoder work. Its designated set is defined by 23 accession-level
strong-binder IDs, while its `f_obs` marker is Tyr at a selected alignment column. The two
binary labels **agree for only 86.5% of stored sequences** (the
`component_marker_agreement` column of the pushforward audit; 1.0 for all five Pfam
families). These are not interchangeable labels and the paper should not treat them as one.

---

## W0. Narrative reframe — the largest piece, now unblocked

The parent spec sequences W0 after numbers and framing settle. Both now have, so nothing
blocks it.

Scope, from `docs/superpowers/specs/2026-07-22-arxiv-revision-design.md` section W0:

- Rewrite abstract, introduction, theory framing and discussion around the
  Gaussian-mixture spine.
- Replace "functional / binding / strong binder" with "marker-positive /
  specificity-associated / putative" wherever the functional evidence is indirect.
- Add a compact theory/method schematic before the empirical claims.
- Treat therapeutic implications as hypotheses, not findings.

One addition the spec predates: the decoder story is now stronger and should be told
deliberately rather than inherited. The pushforward identity
`f_obs = f_eff*TPR + (1-f_eff)*FPR` makes the calibration gap an empirical property of the
complete latent-to-sequence map, and the exact latent control result means multiplicity
conditioning is a theorem while the gap is the measured contribution.

Style constraints (`CLAUDE.md`): no subsection headings in Introduction, Results or
Discussion; flowing prose; no em dashes or en dashes.

---

## W6. Build hygiene — small, mechanical, do last

**[verified now]** Two concrete defects:

1. **The JCIM build has undefined references.** `paper-arxiv/jcim/Paper_JCIM.log` reports
   `tab:mask-recovery`, `fig:mask-betasweep` and `tab:docking-validation` undefined. Cause:
   those labels are defined in `paper-arxiv/Paper_v1.tex` (for example
   `\label{tab:cross-family}` at line 112, `\label{tab:per-family-rho}` at line 266) but
   are `\ref`d from the shared `sections/*.tex` that both builds include. The JCIM wrapper
   never defines them.

2. **Every build artifact is tracked.** `.aux`, `.log`, `.out`, `.bbl`, `.blg` and both
   PDFs, for the arXiv and JCIM builds alike. This is why every rebuild dirties the tree
   and why PDF binary diffs keep appearing in commits that changed no content. The repo
   `.gitignore` covers Julia artifacts only, nothing LaTeX.

Also outstanding: the figure-path redirect and Python dependency pinning.

---

## Suggested sequencing

1. **W4.1 first**, because it needs your decision, not mine, and everything else in W4
   depends on knowing which docking run is real.
2. **W4.2 and W4.3**, correcting numbers and softening claims against the resolved
   provenance.
3. **W0**, written once against fully settled numbers and labels. Writing it before W4
   means writing the biology prose twice.
4. **W6** as pre-submission cleanup. Doing the gitignore change early would make every
   intermediate diff cleaner, so it is the one W6 item worth pulling forward.

The rationale for putting W4 ahead of W0 is that W4 fixes things that are incorrect while
W0 improves things that are merely underpowered, and W0's biology prose depends on W4's
outcome.
