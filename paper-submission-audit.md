# Pre-Submission Audit: SA Binding Generation Study

## Overall assessment

The paper is not yet ready for arXiv or journal resubmission. The arXiv version builds and is visually readable, and the existing 14 unit tests pass, but there are substantive mathematical, empirical, statistical, and reproducibility issues. The journal and supporting-information builds also contain unresolved references and compilation errors.

The most important issues are:

1. A missing factor of `beta` in the score-function derivation.
2. An exact Gaussian-mixture identity that substantially changes the theoretical framing and makes exact independent sampling possible.
3. An incorrect equality between Shannon entropy and the logarithm of the inverse-Simpson effective memory count.
4. A transition-point detector that does not calculate an inflection point as claimed.
5. Conflicting numerical results across the main text, appendix, and CSV artifacts.
6. Data-selected sequence markers described too strongly as externally established functional phenotypes.
7. Structural and docking conclusions that exceed the confidence of the underlying predictions.
8. Broken citations and cross-references in the journal and supporting-information builds.

## Submission-blocking mathematical issues

### 1. The score function is missing a factor of beta

The manuscript defines

```math
p_\beta(\xi) \propto \exp[-\beta E(\xi)].
```

Therefore, the score should be

```math
\nabla_\xi \log p_\beta(\xi)
=
\beta\left[
X\operatorname{softmax}(\beta X^\top\xi + \log r)-\xi
\right].
```

The proposition currently omits the leading `beta` in [`paper-arxiv/sections/theory.tex`](paper-arxiv/sections/theory.tex).

The implemented update appears consistent with temperature-scaled Langevin dynamics expressed using `-grad(E)`, so this is principally a derivation and terminology error rather than necessarily an implementation error. Nevertheless, the score proposition and downstream statements based upon it must be corrected.

### 2. The target distribution is an exactly sampleable Gaussian mixture

Because the stored memories are unit normalized,

```math
\begin{aligned}
p_\beta(\xi)
&\propto
e^{-\beta\lVert\xi\rVert^2/2}
\sum_k r_k e^{\beta m_k^\top\xi} \\
&\propto
\sum_k r_k e^{-\beta\lVert\xi-m_k\rVert^2/2}.
\end{aligned}
```

Thus the stated target is exactly a Gaussian mixture:

```math
k\sim\operatorname{Categorical}\left(\frac{r_k}{\sum_j r_j}\right),
\qquad
\xi\mid k\sim\mathcal N(m_k,I/\beta).
```

This has two major consequences:

- Exact independent samples require no MCMC.
- ULA introduces discretization and mixing error even though an exact sampler is available.

This does not make the iterative dynamics scientifically irrelevant: associative-memory trajectories may themselves be the intended object. The paper, however, currently presents ULA as the sampling solution without discussing the closed-form sampler. A reviewer is likely to identify this.

The revision should include the exact sampler as a baseline and explicitly explain why the dynamics, rather than merely sampling the target, matter scientifically. This identity may also provide a cleaner interpretation of multiplicity control.

### 3. “No approximation” is false for ULA

[`paper-arxiv/sections/Discussion.tex`](paper-arxiv/sections/Discussion.tex) states that the Langevin dynamics sample from a known distribution “with no approximation.” Finite-step unadjusted Langevin has discretization bias. Only an accept/reject-corrected procedure such as MALA is asymptotically exact, subject to normal convergence conditions.

A defensible replacement would be:

> The dynamics target an analytically specified distribution, subject to finite-step discretization and mixing error.

### 4. The entropy/effective-memory equality is incorrect

The manuscript states that, at `beta = 0`,

```math
H_r(0)=\log K_{\mathrm{eff}},
\qquad
K_{\mathrm{eff}}=\frac{(\sum_k r_k)^2}{\sum_k r_k^2}.
```

But `H_r(0)` is Shannon entropy, whereas `log(K_eff)` is Rényi-2 entropy. They are not generally equal. This occurs in [`paper-arxiv/sections/theory.tex`](paper-arxiv/sections/theory.tex) and [`paper-arxiv/sections/appendix.tex`](paper-arxiv/sections/appendix.tex).

For the Kunitz weights, numerical differences can be as large as approximately 0.20 nats. The relevant figure baseline and its interpretation therefore need correction. Either plot the true Shannon entropy of the normalized weights or explicitly label `log(K_eff)` as Rényi-2 entropy.

### 5. The reported “inflection point” is not an inflection point

The detector in [`code/src/Protein.jl`](code/src/Protein.jl) and [`code/src/Binding.jl`](code/src/Binding.jl) selects the most negative second derivative.

An inflection point is a zero crossing of the second derivative. For a sigmoid-like curve, the steepest transition would normally be located using an extremum of the first derivative. The present statistic is closer to a maximum-curvature or transition-onset point.

There are two additional concerns:

- The nominally “random probes” are the first 20 memories, making the result order-dependent.
- Entropy is evaluated at stored memories rather than draws from the stationary distribution.

The paper should either rename and justify this descriptive crossover statistic or replace it with a formally defined estimator and uncertainty analysis. “Phase transition” is too strong without finite-size or scaling evidence.

### 6. Appendix claims contradict the reported results

[`paper-arxiv/sections/appendix.tex`](paper-arxiv/sections/appendix.tex) says that the separation gap is identical at every multiplicity ratio, but the reported gap changes substantially with the ratio.

The appendix also states that `beta*` is proportional to `log(K_eff)`, although reported `K_eff` decreases while `beta*` increases. This requires either a derivation with the correct relationship or much weaker empirical wording.

### 7. The signal-chain argument needs clarification

The theory describes the path from effective mixture fraction to mean attention as exact under an equal-similarity assumption. Pointwise equality is not generally exact. A cleaner exact statement is available in expectation under the Gaussian-mixture target, using the law of total expectation. The manuscript should clearly distinguish:

- pointwise attention at a particular latent state;
- expected posterior attention under the exact stationary target; and
- finite-step empirical averages under ULA.

## Empirical inconsistencies and overclaims

### 1. Conflicting multiplicity results

The main table, Results prose, appendix table, and aggregated CSVs do not report the same outcomes for the nominally same experiment. For example, the Kunitz `rho = 500` observed fraction appears as approximately:

- 0.608 in the main table;
- 0.63 in the prose;
- 0.631 in the appendix;
- approximately 0.581 or 0.638 in alternative aggregated outputs.

One canonical pipeline must be designated, and every table, plot, and prose value should be regenerated from its outputs. The analysis should also identify the inferential unit—chains, replicates, or sequences—and report uncertainty consistently.

### 2. Attention agreement is overstated

The paper claims deviations below 0.3% across all families and multiplicities, but several reported examples differ by considerably more. For SH3 at `rho = 1`, `f_eff = 0.600` and attention is approximately 0.621, a difference of 2.1 percentage points.

Report the actual maximum absolute deviation and distinguish percentage points from relative percentages.

### 3. The cross-family relationship is not monotonic

For the five-family regression, the approximate reported values are

```text
S     = (0.11, 0.17, 0.20, 0.34, 0.42)
Delta = (0.64, 0.27, 0.39, 0.01, 0.04)
```

The gap increases from 0.27 to 0.39 while separation increases from 0.17 to 0.20, so the relationship is not strictly monotonic. The conotoxin point, approximately `S = 0.78, Delta = 0.15`, also does not extend the claimed monotonic trend.

With only five in-sample families, “prediction,” decision thresholds, and general rules are premature. This should be presented as an exploratory association. Add uncertainty, leave-one-family-out sensitivity, and avoid describing `S > 0.3` as a validated decision threshold.

### 4. Several phenotype labels are selected from the evaluated data

In [`code/experiments/run_new_family_validation.jl`](code/experiments/run_new_family_validation.jl):

- Kunitz uses an argmax Lys-frequency position.
- SH3 searches for a convenient Trp column.
- WW chooses a high-entropy middle position and its common residue.
- Homeobox searches for high Gln frequency.
- Forkhead deliberately chooses an approximately balanced H/N column.

This conflicts with the narrative that these are externally established functional-site phenotypes and introduces selection bias.

Unless every alignment position is independently mapped to a canonical structural residue and supported by biological evidence, these should be described as synthetic or data-selected sequence markers rather than functional phenotypes.

Related terminology should also be softened:

- Kunitz K/R at P1 is a specificity proxy, not proof that every sequence is a strong binder.
- One hundred percent marker recovery after filtering on that marker is marker retention, not independent functional validation.
- Most of the conotoxin background is unannotated, not demonstrated to consist of nonbinders.

### 5. Conotoxin claims need a clearer evidence hierarchy

The conotoxin set combines experimentally confirmed binders, predicted binders, close homologs, and largely unannotated background sequences. These categories should not be collapsed into a binary binder/nonbinder label without an accession-level evidence table and explicit classification rules.

The abstract claim that the method preserves “all experimentally identified binding determinants” is also too broad. Some reported positions are only partially enriched, and several determinants are peptide- or family-specific. A more accurate statement would be that the outputs enrich several reported determinants while retaining the cysteine scaffold and Tyr13 signal.

### 6. Structural and docking conclusions exceed the evidence

The AlphaFold-Multimer results have very low confidence: iPTM is approximately 0.10 and interface pLDDT is approximately 37–41. A nonsignificant permutation test with sample sizes near 10/10/5 does not establish equivalence or “no degradation of binding geometry.”

The defensible conclusion is that no difference was detected among uniformly low-confidence predictions. Claims of preserved docking geometry require stronger models, confidence thresholds, equivalence tests, interval estimates, or experimental evidence.

The final docking-rerun CSV contains only two data rows despite many output directories, whereas an older CSV contains the 25 rows apparently used by the table. This provenance needs to be resolved and the canonical dataset regenerated.

### 7. Structure table transcription error

The HMM TM-score standard deviation in the Kunitz structure-validation CSV is approximately 0.19, while the manuscript reports 0.10. The table and corresponding prose should be regenerated directly from the canonical data.

Terms such as “structurally valid” and “foldable” also overstate predictions from structure models. Prefer “predicted structurally plausible,” “higher predicted confidence,” or “lower predicted structural fidelity,” as appropriate.

### 8. Statistical dependence and convergence need attention

Samples within a Langevin chain are correlated. Some analyses correctly use chains as the inferential unit, but other metrics appear to treat sequences or replicates inconsistently. The paper should report:

- the inferential unit for every uncertainty estimate;
- autocorrelation or effective sample size;
- burn-in and convergence diagnostics;
- whether warm starts are randomized; and
- whether uncertainty reflects chains, independent runs, or individual sequences.

The implementation cycles through the first stored patterns for some warm starts, which conflicts with language describing random initialization.

## PCA representation and decoding

The appendix describes reconstruction as inverse PCA, but [`code/src/Protein.jl`](code/src/Protein.jl) normalizes PCA coordinates to unit length and later reconstructs without restoring their original radius.

A diagnostic audit found:

- original PCA-coordinate norms of approximately 5.0–6.5;
- gap-aware mean reconstruction identity of approximately 0.836 after unit normalization;
- gap-aware mean reconstruction identity of approximately 1.000 from unnormalized PCA scores; and
- retained P1-marker accuracy in the tested Kunitz diagnostic.

These two identity values were corrected on 2026-07-28 after direct metric tests
showed that the earlier implementation excluded `-` gaps but counted Pfam `.`
gaps as residue mismatches.

The decoder is therefore not an inverse of the encoder. Loss of radial information may contribute materially to the resulting sequence behavior. This should be disclosed and tested with an ablation comparing:

1. normalized PCA coordinates;
2. unnormalized coordinates; and
3. restored-radius decoding.

Several appendix statements are also stale:

- It refers to four families while six are studied.
- It gives a `d_full/K` range of 7.0–10.7, whereas WW is approximately `620/420 = 1.48`.

## Code correctness and reproducibility

### Existing tests

The current mask-related test file passes all 14 assertions. This is useful but covers only a narrow portion of the implementation. There is no comprehensive `runtests.jl` suite.

Important missing tests include:

- finite-difference verification of the energy gradient and score;
- exact-Gaussian-mixture versus ULA moments;
- Shannon and Rényi entropy identities;
- transition-point detection on known analytic curves;
- PCA encode/decode behavior;
- deterministic generation from fixed seeds; and
- regression tests tying canonical CSVs to manuscript tables.

### Random-number reproducibility

Several generation functions initialize states using global `randn` before entering the seeded sampler. Global `Random.seed!` is also unsafe for parallel or compositional use. Pairwise-diversity subsampling similarly uses ambient global randomness.

Create one local RNG per chain or replicate and pass it to initialization, dynamics, and all subsequent randomized analysis.

### Release and environment issues

- Several figure scripts still write to the old `paper/` path rather than `paper-arxiv/`.
- There is no single end-to-end pipeline or code-level README.
- Python dependencies are not pinned.
- `code/bin/TMalign` is an arm64 macOS binary and is not portable.
- External Colab and ESMFold steps are not captured in the automated workflow.
- Build scripts do not fail fast after LaTeX errors.
- The paper directory currently appears in Git as deleted tracked `paper/` files plus an untracked `paper-arxiv/` directory. Commit this as an intentional rename.
- Generated `.aux`, `.log`, `.blg`, and related artifacts should be excluded from the release archive.
- Data and code availability should point to a versioned commit or DOI, with exact environments, seeds, hardware details, and checksums where practical.

## Build and visual audit

### arXiv manuscript

The arXiv manuscript builds to 26 pages without unresolved citations or references. It is generally readable, but the build and rendered pages show:

- a style-version inconsistency in which `neurips_2026.sty` identifies itself as 2025;
- duplicate appendix hyperlink destinations;
- an overfull Table 4;
- small labels and axes in several figures;
- large unused areas on multiple figure-heavy pages;
- Table S2 isolated on one page while Tables S3–S6 resume later; and
- all figures placed after the references, making the argument harder to follow.

### Journal manuscript and supporting information

The journal manuscript is not build-clean. Problems include:

- undefined references to the mask table, beta-sweep figure, and docking table;
- `mciteplus` errors for several citations;
- multiple overfull boxes;
- a journal table that says “all four” families and omits Homeobox and Forkhead while nearby prose discusses five Pfam families or six total families;
- literal `??` references in the supporting information; and
- an unresolved Ramsauer citation in the supporting information.

The journal and supporting-information artifacts should not be submitted until they build with zero undefined-reference and citation errors.

## Narrative and organization

The title and abstract promise functional protein generation, but most of the direct evidence concerns retention of sequence markers and model-predicted structural plausibility. A more defensible narrative would be:

1. Exact probabilistic interpretation and multiplicity control.
2. Kunitz specificity-marker case study.
3. Cross-family marker-recovery stress test.
4. Conotoxin sequence and structure-model predictions.
5. Explicit limitations: marker labels, exact sampling, PCA normalization, finite-step ULA, and absence of experimental validation.

Recommended narrative changes:

- Replace “functional,” “binding,” and “strong binder” with “marker-positive,” “specificity-associated,” or “putative” wherever the functional evidence is indirect.
- Introduce a compact theory/method schematic before the empirical claims.
- Either introduce hard masking as one of three conditioning strategies from the outset or move it fully to the supporting information.
- Remove numerical narration that simply duplicates tables.
- Interleave figures with the arXiv text.
- Treat therapeutic implications as hypotheses rather than demonstrated outcomes.
- Avoid treating an unconditional HMM as a complete conditional-generation baseline.
- Include the exact Gaussian-mixture sampler as a mandatory baseline; curated resampling and hard masking are also natural comparators.

## Recommended revision order

1. Resolve the exact Gaussian-mixture implication and correct the score derivation.
2. Correct the entropy mathematics and transition-point definition.
3. Select one canonical experiment pipeline and regenerate every output.
4. Audit every family label against independent biological evidence.
5. Reframe the statistical, functional, structural, and docking claims.
6. Correct PCA reconstruction or include the necessary ablation.
7. Repair deterministic RNG behavior and add mathematical and regression tests.
8. Make the arXiv, journal, and supporting-information builds completely clean.
9. Revise the narrative around the corrected level of evidence.

## Bottom line

The central observation—multiplicity-weighted associative memories yield controlled mixture weights—may become cleaner and more rigorous when presented through the exact Gaussian-mixture identity. However, the current submission requires major revision. The theoretical identities, numerical provenance, biological label definitions, and strength of empirical claims should all be resolved before either the arXiv update or journal submission.
