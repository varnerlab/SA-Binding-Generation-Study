# arXiv Revision Design: Conditioning Protein Generation via Hopfield Pattern Multiplicity

Date: 2026-07-22
Status: approved structure, pending expansion into an execution plan
Source audit: `paper-submission-audit.md`

## 1. Context and locked decisions

A pre-submission audit found roughly thirty issues spanning mathematics, empirics,
statistics, biology labels, code reproducibility, and LaTeX builds. Investigation of
the audit's deepest claim (that the sampler's target is an exactly sampleable Gaussian
mixture) confirmed it is true, and reframed it from a threat into the paper's new spine.

Decisions locked with the author:

- **Target:** arXiv v2 first. JCIM manuscript and Supporting Information build repair is
  tracked but deferred to a later pass.
- **Depth:** lean path. Acknowledging the GMM identity requires NO experiment re-runs: it
  costs the SI derivation, one small exact-sampler-vs-ULA baseline table, and the reframe prose.
  The audit's heavier items (regeneration, ablations, docking rerun) are decoupled from the GMM
  and are opt-in add-ons whose depth the author sets per item, not defaults.
- **The GMM does not replace SA.** SA/ULA remains the generative method. The identity is a
  property of its equilibrium (a theorem + a validation baseline), not a substitute sampler.
  ULA discretization bias was measured to be small, so the existing SA results stand as reported.
- **Narrative spine:** exact Gaussian-mixture (GMM) identity, then exact multiplicity
  control stated as a theorem, then the lossy decoder transfer function
  `f_eff -> f_observed` as the genuine empirical contribution, then the Kunitz,
  cross-family, and conotoxin case studies, then explicit limitations.
- **Canonical pipeline:** the `*_with_replicates` experiment scripts are canonical.
  Chains are the inferential unit. Fixed seeds, one local RNG per chain. Every table,
  figure, and prose number regenerates from these outputs.

## 2. Verified technical foundation

### 2.1 The Gaussian-mixture identity

For unit-norm memories `m_k` (enforced in `build_memory_matrix`, `Protein.jl:240`) and
multiplicities `r_k > 0`, the weighted Hopfield energy
(`weighted_hopfield_energy`, `Binding.jl:747`)

    E_r(xi) = 1/2 ||xi||^2 - (1/beta) log sum_k r_k exp(beta m_k^T xi)

produces the Gibbs target

    p_beta(xi) = sum_k w_k N(xi; m_k, beta^-1 I),   w_k = r_k / sum_j r_j.

Derivation (goes into the SI verbatim):
1. `-beta E_r` cancels the `-1/beta` inside the log against the `-beta` outside, and
   `exp(log S) = S`, giving `exp(-beta E_r) = exp(-beta/2 ||xi||^2) sum_k r_k exp(beta m_k^T xi)`.
   This single cancellation is the crux; the energy is constructed so its Boltzmann
   factor is a plain sum.
2. Complete the square per term: exponent `= -beta/2 ||xi - m_k||^2 + beta/2 ||m_k||^2`.
3. Unit norm makes `exp(beta/2 ||m_k||^2) = exp(beta/2)` a shared constant that factors out,
   so mixture weights are exactly `r_k` (non-unit-norm would give `r_k exp(beta/2 ||m_k||^2)`).
4. Each term is an unnormalized isotropic Gaussian `N(m_k, beta^-1 I)`.
5. Normalizing cancels all constants, leaving the mixture above.

Corollaries for the SI:
- **A (exact sampler):** `k ~ Categorical(w)`, then `xi = m_k + beta^{-1/2} z`. No MCMC.
- **B (exact designated control):** latent mass on designated set B is
  `sum_{k in B} r_k / sum_j r_j = f_eff`, and with `r_k = rho` on B,
  `f_eff = rho K_B / (rho K_B + K_NB)`. No equal-similarity assumption.
- **D (score, fixes missing beta):** `grad log p_beta = beta [X softmax(beta X^T xi + log r) - xi]`.
  The Langevin SDE with this drift has stationary law `p_beta`; its Euler step is the code's
  update, so ULA is a biased discretized sampler of the same target the exact sampler hits.

### 2.2 Numerical confirmation (real Kunitz memory, K=99, d=80, 32 designated)

Diagnostic `check_gmm_identity.jl` result:

| rho | beta* | f_eff | designated-mass exact/ULA/analytic | decoded P1 K/R exact/ULA | AA-KL(exact,ULA) | per-pos TV |
|-----|-------|-------|------------------------------------|--------------------------|------------------|-----------|
| 1   | 4.27  | 0.323 | 0.307 / 0.311 / 0.323              | 0.398 / 0.380            | 0.0001           | 0.027     |
| 10  | 5.69  | 0.827 | 0.888 / 0.891 / 0.827              | 0.530 / 0.504            | 0.0001           | 0.028     |
| 500 | 8.78  | 0.996 | 0.997 / 1.000 / 0.996              | 0.604 / 0.593            | 0.0001           | 0.027     |

Exact sampler was about 2000x faster than ULA for equal sample counts.

### 2.3 The decoder transfer function (the reframe's empirical core)

The check exposed the real result: designated **latent mass** (about 1.00 at rho=500)
maps to only about 0.60 decoded marker-positive sequences. That gap is entirely the
PCA plus argmax decoder, is reproduced identically by the exact sampler, and is therefore
a genuine property of the representation rather than a ULA artifact. Quantitatively, at the
operating temperature the per-sample noise magnitude is about `sqrt(d/beta*) ~ 3.0` while the
memory vectors have norm 1, so components overlap heavily and generation is diffuse. This
gap is the paper's calibration finding, now on rigorous footing and possibly improvable via
the PCA radius fix (W3).

## 3. Workstreams

Each workstream lists the audit items it closes.

### W0. Narrative reframe (the spine)
Rewrite abstract, introduction, theory framing, and discussion around the spine in section 1.
Replace "functional / binding / strong binder" with "marker-positive / specificity-associated
/ putative" wherever the functional evidence is indirect. Add a compact theory/method schematic
before the empirical claims. Treat therapeutic implications as hypotheses.
Closes: narrative and organization section; terminology in empirical items 4 and 5;
discussion overclaims.

### W1. Theory corrections and the SI derivation
- Insert the section 2.1 derivation and corollaries as a new SI section.
- Fix the score proposition to carry beta.
- Replace "no approximation" with finite-step discretization and mixing error language.
- Fix `H_r(0) = log K_eff`: it equates Shannon and Renyi-2 entropy. Either plot the true
  Shannon entropy of the normalized weights or relabel `log K_eff` as Renyi-2. Correct the
  affected figure baseline (up to about 0.20 nats for Kunitz).
- Redefine the transition statistic: the detector finds maximum negative second derivative
  (max curvature / onset), not an inflection (zero crossing). It uses the first 20 columns,
  not random probes, and evaluates entropy at stored memories, not stationary draws. Either
  rename and justify it as a descriptive crossover with uncertainty, or replace with a formal
  estimator. Reframe "phase transition" via the GMM overlap picture (`sqrt(d/beta*)` vs
  inter-mean spacing).
- Fix appendix contradictions: the claim that the separation gap is identical at every ratio,
  and `beta* proportional to log K_eff` while reported `K_eff` falls as `beta*` rises.
- Fix stale appendix text: four families should be six; `d_full/K` range 7.0 to 10.7 versus
  WW at about 620/420 = 1.48.
- State the signal chain as three distinct objects: pointwise attention at a latent state,
  expected posterior attention under the exact target (now exact via the mixture), and
  finite-step ULA averages.
Closes: math items 1 through 7; appendix staleness under the PCA section.

### W2. Reconcile to one canonical CSV (lean; no re-runs by default)
- Designate the existing `*_with_replicates` aggregated CSVs as the single source of truth
  (chains as inferential unit). These already exist on disk; the default is to point every
  table and prose value at them, not to re-run experiments.
- Reconcile the Kunitz rho=500 observed fraction (reported as 0.608, 0.63, 0.631, and about
  0.581 or 0.638) to one canonical number sourced from that CSV.
- Report the true maximum absolute attention deviation and distinguish percentage points from
  relative percentages (the "below 0.3 percent" claim fails, for example SH3 at rho=1 differs
  by about 2.1 points).
- Recast the five-family relation as an exploratory association: add uncertainty, leave-one-
  family-out sensitivity, drop the `S > 0.3` decision threshold and the "prediction" language,
  and note the conotoxin point does not extend a monotonic trend.
- Opt-in only: a full end-to-end regeneration driver, if the author later wants every CSV
  rebuilt from scratch rather than trusting the existing aggregated outputs.
Closes: empirical items 1, 2, 3, 8.

### W3. Exact-sampler baseline (core) and decoder transfer function (opt-in)
Core (lean, in Plan 1): add the exact GMM sampler and productionize the ULA vs exact-GMM
equivalence check into one baseline table. This is the reviewer-insurance the reframe needs.
Opt-in add-ons (only if the author wants them):
- Characterize `f_eff -> f_observed` as the decoder pushforward versus beta and component overlap.
- Run the PCA normalization ablation: normalized vs unnormalized vs restored-radius decoding
  (diagnostic showed reconstruction identity about 0.833 normalized versus 0.996 unnormalized).
  This is a gating experiment: if restoring radius changes generation, some results shift.
- Weighted MALA (the shipped `mala_sample` uses the unweighted energy, so it is not a
  multiplicity baseline as-is).
Closes: math item 2 baseline mandate; the PCA decode item; the baseline recommendations.

### W4. Biology and label rigor
- Reclassify family markers (Kunitz Lys/Arg at P1, SH3 Trp, WW mid-position, Homeobox Gln,
  Forkhead H/N) as data-selected sequence markers unless independently mapped to a canonical
  structural residue with biological support.
- Soften: Kunitz K/R at P1 is a specificity proxy not proof of binding; 100 percent marker
  recovery after filtering on that marker is retention not validation; conotoxin background is
  unannotated not demonstrated nonbinders.
- Build a conotoxin accession-level evidence table (confirmed, predicted, homolog, unannotated)
  with explicit classification rules. Soften the "all binding determinants" claim to enriching
  several reported determinants while retaining the cysteine scaffold and Tyr13 signal.
- Report structure and docking honestly: iPTM about 0.10, interface pLDDT about 37 to 41; a
  nonsignificant permutation test at about 10/10/5 shows no detected difference among uniformly
  low-confidence predictions, not preserved geometry. Resolve the docking CSV provenance (final
  file has two rows, older file has the 25 used by the table) and regenerate the canonical set.
  Fix the Kunitz HMM TM-score standard deviation (about 0.19 in data, reported 0.10). Switch to
  "predicted structurally plausible" language.
Closes: empirical items 4, 5, 6, 7.

### W5. Code correctness, tests, reproducibility
- Add `runtests.jl`: finite-difference verification of the energy gradient and score; exact-GMM
  versus ULA and MALA moments; Shannon and Renyi-2 entropy identities; transition detection on
  known analytic curves; PCA encode/decode behavior; deterministic generation from fixed seeds;
  regression tests tying canonical CSVs to manuscript tables.
- Replace global `randn` initialization and global `Random.seed!` with one local RNG per chain,
  passed to initialization, dynamics, and diversity subsampling.
- Point figure scripts at `paper-arxiv`. Pin Python dependencies. Note the arm64 TMalign
  portability limit. Capture the external Colab and ESMFold steps. Make build scripts fail fast.
Closes: code correctness, RNG reproducibility, release and environment items.

### W6. arXiv build hygiene and git rename
- Commit `paper/` to `paper-arxiv/` as an intentional rename. Gitignore build artifacts
  (`.aux`, `.log`, `.blg`, `.out`, `.bbl`).
- Fix the `neurips_2026.sty` self-identifying as 2025, duplicate appendix hyperlink
  destinations, overfull Table 4, small figure labels and axes, large unused figure areas,
  and interleave figures with the text rather than after the references.
Closes: arXiv build and visual audit. JCIM and SI build repair tracked but deferred.

## 4. Sequencing and dependencies

1. W1 and W2 first: correct the mathematics and establish one source of truth.
2. W3 and W4 build on the corrected theory and canonical outputs.
3. W0 rewrite follows once numbers and framing are settled.
4. W5 and W6 proceed in parallel throughout.

W3's PCA ablation feeds W1's phase-transition and decoder framing. W2's regenerated outputs
feed W0's prose and every table.

## 5. Success criteria

- SI contains the full GMM derivation and corrections; every flagged equation is corrected.
- Every manuscript number traces to a canonical CSV produced by the `*_with_replicates`
  pipeline; no cross-source conflicts remain.
- Exact GMM and MALA baselines and the PCA ablation are reported.
- All functional claims are downgraded to the evidence level; structure and docking claims are
  stated as low-confidence.
- `runtests.jl` passes and covers the items in W5.
- The arXiv manuscript builds clean with zero undefined references and no build artifacts in
  the tree; the `paper/` to `paper-arxiv/` rename is committed.

## 6. Risks and open questions

- **ULA metastability:** at high beta the chain may not mix across components, so realized
  occupancy could depend on warm starts rather than `f_eff`. The check suggests ULA tracks the
  target at the operating beta*, but W3 should confirm mixing and report effective sample size.
- **PCA radius fix scope:** if restoring radius materially changes generation behavior, several
  downstream results may shift and need regeneration. Treat as a gating experiment early in W3.
- **Docking provenance:** the canonical docking dataset must be reconstructed before any
  structure claims can be finalized; if the 25-row set cannot be reproduced, the table must be
  rebuilt from whatever is reproducible.
- **JCIM and SI:** deferred, but the reframe and corrections must be portable to that build
  later without a second science pass.
