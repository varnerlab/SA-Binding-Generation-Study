# Submission Audit Results

> **Status: historical record, partly superseded.** This is the first pre-submission audit
> cycle, three passes across two independent reviewers. Findings 1 through 6 were applied,
> each behind a regression test. Two of its conclusions were later overturned. Section 4
> described the operating-point sensitivity as "grid resolution, not a broken operating
> point," and the closing line states that "a full regeneration of every canonical sweep is
> not required." A follow-on review reached two defects this audit did not, and every
> canonical sweep was in fact regenerated. See `remaining-issues-audit.md` for that review
> and for the execution record.

Date: 2026-07-29

Scope:

- JCIM manuscript: `paper-jcim/Paper_JCIM.pdf`
- JCIM Supporting Information: `paper-jcim/Paper_JCIM_SI.pdf`
- arXiv manuscript: `paper-arxiv/Paper_v1.pdf`
- Shared manuscript sources, analysis code, generated data, references, and LaTeX logs

## Document status

This file contains two passes. The original audit (GPT 5.6, 2026-07-29) is preserved
below. A second independent verification pass (Claude, 2026-07-29) checked each claim
against the code, the generated data, and the TeX sources. Verification results are
recorded inline as `VERIFIED`, `VERIFIED, DEFLATED`, `REFUTED`, or `UPGRADED`, and the
action list at the top is ordered by the verification pass rather than by the original
audit's ordering.

A third pass (GPT 5.6, 2026-07-29) reconciled the two and is recorded at the end of this
file. It overturned one verification verdict: the `beta* \propto log K_eff` claim is real
and must be deleted. The second pass had searched for the word "proportional" and missed
the LaTeX symbol. The verdicts below have been updated accordingly.

One of the original audit's claims did not survive verification and should not be acted
on. One item the audit filed under "interpretive revisions" was upgraded to must-fix.

## Submission recommendation

Do not submit either version yet. The PDFs and LaTeX builds are clean, the exact
Gaussian-mixture result is correct, and the repository tests pass. Two items are
must-fix. The remainder is an afternoon of editing plus optional polish.

## Action list, ordered by priority

| # | Item | Verdict | Effort |
|---|---|---|---|
| 1 | Conotoxin SAR input frequencies use unaligned indices | VERIFIED (data error) | Script fix, regenerate Table 6, adjust caption |
| 2 | Hard-mask "decode effect" inference | UPGRADED (inference error) | Rewrite one paragraph |
| 3 | Entropy identity `H_r(0) = log K_eff` is false | VERIFIED (no numbers change) | Two sentences |
| 4 | "Inflection" / "phase transition" terminology | VERIFIED, DEFLATED | Rename in prose; sensitivity check, not a rerun |
| 5 | VAE citation, two grammar breaks | VERIFIED | Trivial |
| 6 | ULA framing, tone, sentence length, functional language | Agreed, discretionary | Scope as desired |
| - | "SI states beta* proportional to log K_eff, wrong direction" | CONFIRMED (folded into item 3) | Delete `appendix.tex:190` |
| - | Audit's replacement values for Cys15 and position 21 | REFUTED | Recompute, do not paste |

---

## 1. The conotoxin SAR input frequencies use unaligned indices

**VERIFIED. This is the only genuine data error found.**

`code/experiments/compute_conotoxin_sar_agreement.jl:79` loads
`strong_cav22_binders.fasta`, which contains raw variable-length sequences, and then
indexes raw character positions as if they were MVIIA-numbered alignment columns.

Confirmed by direct inspection:

- `strong_cav22_binders.fasta`: 23 sequences of length 24, 25, 26, 27, 28, and 31.
- `generated_strong_seeded.fasta` and `generated_full_seeded.fasta`: 1550 sequences each,
  all length 26, living in the cleaned alignment frame.
- The cleaned full-family alignment is 74 sequences by 26 columns, and MVIIA appears in it
  as `CKGKGAKCSRLMYDCCTGSCRSGKCG`, so alignment column equals MVIIA position in that frame.

The error is visible in the published table itself. `paper-jcim/Paper_JCIM.tex:348-350`
reports framework cysteines at input frequencies of 0.70 and 0.52, which cannot occur at
aligned positions in this family.

### Corrected input frequencies

Recomputed by selecting the 23 designated accessions from the cleaned 26-column
alignment (column gap filter 0.5, sequence gap filter 0.4, matching
`clean_alignment` in `code/src/Protein.jl:126-157`):

| Position | WT | Current input | Corrected input |
|---|---|---:|---:|
| 13 | Tyr | 0.83 | 0.83 |
| 2  | Lys | 0.96 | 0.96 |
| 10 | Arg | 0.70 | 0.70 |
| 11 | Leu | 0.17 | 0.17 |
| 1  | Cys | 1.00 | 1.00 |
| 8  | Cys | 1.00 | 1.00 |
| 15 | Cys | 1.00 | 1.00 |
| 16 | Cys | 0.87 | 1.00 |
| 20 | Cys | 0.70 | 1.00 |
| 25 | Cys | 0.52 | 1.00 |
| 21 | basic | 0.44 | 0.48 |
| 4  | Lys | 0.52 | 0.52 |

The headline result is unaffected: Tyr13 input 0.83 against designated-subset generation
0.98 is unchanged.

The caption claim at `paper-jcim/Paper_JCIM.tex:330-331` that generation "enriches
several reported positions" becomes weaker and must be revised. The apparent enrichment
at Cys20 (0.70 to 1.00) and Cys25 (0.52 to 1.00) disappears entirely, because the
corrected input was already 1.00 at both positions. The enrichment claims that survive
are Tyr13, Arg10, position 21, and Lys4.

### Correction to the original audit

The original audit's replacement values for two rows did not reproduce:

| Position | Audit's value | Verified value |
|---|---:|---:|
| Cys15 | 0.87 | 1.00 |
| Position 21 basic | 0.57 | 0.48 |

The audit appears to have computed in the 33-column `strong_cav22_binders_aligned.fasta`
frame rather than the cleaned 26-column frame that the generated sequences occupy. Fix
the script and regenerate the table from the pipeline. Do not transcribe the audit's
numbers.

Separately, the current position-21 input is double-rounded: `sar_agreement.csv` stores
0.435 and the table shows 0.44, whereas 10/23 = 0.4348 rounds to 0.43 at two decimal
places. Regenerating the table removes this.

### Recommended correction

- Fix `compute_conotoxin_sar_agreement.jl` to read the designated input from the cleaned
  aligned data, in the same coordinate frame as the generated sequences.
- Regenerate `sar_agreement.csv` and Table 6.
- Rewrite the caption and the SAR comparison at `paper-jcim/sections/results.tex:220-228`
  to reflect the reduced set of enriched positions.

## 2. The hard-mask "decode effect" conclusion does not follow

**UPGRADED from the original audit's "interpretive revisions" section. This is the most
defensible objection a referee has, and it belongs with the must-fix items.**

`paper-jcim/sections/results.tex:153-164` states that because attention tracks
`f_eff` to within a fraction of a percent, the hard-masking advantage "is not an
attention effect but a decode effect."

`code/data/kunitz/mask_residuals.csv` shows what is actually being compared:

```
f_target,f_eff,beta_w,  mult_p1kr,mask_p1kr,residual
0.5,     0.5,  4.5789,  0.4280,   0.5387,   -0.1108
0.7,     0.7,  4.5789,  0.4634,   0.5387,   -0.0753
0.99,    0.99, 7.7131,  0.5903,   0.5925,   -0.0022
```

`mask_p1kr` is identical on the first two rows because the mask condition is the same
condition in both: `f_eff = 1` by construction. The comparison is therefore between
conditions with different intended designated mass, 0.5 or 0.7 against 1.0.

`Delta_attn ~ 0` establishes that each condition realizes its own intended attention
mass. It does not establish that the between-condition difference is a decoder effect,
because the intended masses differ. The difference collapsing to 0.002 at
`f_eff = 0.99` is consistent with the difference being driven by the `f_eff` mismatch,
not by a decoder-specific mask advantage.

This is a reframing, not a retraction, and the corrected statement is stronger. A change
of 0.5 in latent designated mass produces only 0.111 in observed marker fraction. That
compression is the decoder transfer function, which is the paper's central contribution.

### Recommended wording

- At matched beta, hard masking produced higher marker recovery because it removed the
  background components, raising the intended designated mass from `f_eff` to 1.
- The advantage contracted monotonically as `f_eff` approached 1, reaching 0.002
  (SE 0.030) at `f_eff = 0.99`.
- A change of 0.5 in intended designated mass produced a change of only 0.111 in observed
  marker fraction, which quantifies the compression imposed by reconstruction and
  decoding.
- Do not claim the between-condition difference is purely a decode effect.

## 3. The entropy identity is incorrect

**VERIFIED as a mathematical error. Numerically negligible at the reported operating
points, so no results change.**

`paper-jcim/sections/theory.tex:123` and `paper-jcim/sections/appendix.tex:154-158`
state that

```
H_r(0) = log K_eff.
```

This is false for unequal weights. At beta = 0 the attention weights are
`w_k = r_k / sum_j r_j`, so the limiting attention entropy is the Shannon entropy
`-sum_k w_k log w_k`, whereas `log K_eff = -log sum_k w_k^2` is the Renyi-2 entropy. The
two agree only for uniform weights.

The repository already states this correctly. `code/src/Binding.jl:848-873` documents in
the `shannon_entropy` docstring that the quantity "is NOT equal to `log K_eff` (the
Renyi-2 entropy) except when all weights are equal." The manuscript currently contradicts
its own source code.

Numerical size of the error for the Kunitz configuration (K = 99, K_des = 32):

| rho | Shannon | log K_eff | Difference |
|---:|---:|---:|---:|
| 10   | 4.0545 | 3.8252 | 0.229 (5.7%) |
| 500  | 3.4958 | 3.4741 | 0.022 (0.6%) |
| 1000 | 3.4823 | 3.4699 | 0.012 (0.35%) |

At the reported operating point of rho = 1000 the discrepancy is 0.35%, so this is a
two-sentence correction with no downstream consequences.

### Recommended correction

- Replace the false identity with the Shannon entropy.
- Present `K_eff` as a separate inverse-Simpson or effective-component descriptor.
- Report the observed displacement of the entropy curve as an observation. The mechanism
  linking lower starting entropy to a higher operating inverse temperature is asserted
  rather than derived, so drop the causal framing unless a derivation is added.

### CONFIRMED sub-claim, and a correction to the second pass

The original audit stated that "the SI statement that beta* is proportional to
log K_eff has the wrong direction: the reported beta* increases while log K_eff
decreases."

The second verification pass marked this REFUTED on the basis of a search for the word
"proportional", which returns nothing. That was wrong. The claim is written with the
LaTeX symbol, at `paper-jcim/sections/appendix.tex:190` and identically at
`paper-arxiv/sections/appendix.tex:190`:

```latex
with $\beta^{*}(\rho)$ increasing from 4.35 ($\rho=1$) to 9.26
($\rho=1000$), consistent with $\beta^{*} \propto \log K_{\mathrm{eff}}$
predicted by the mean-field argument in the main text
```

`Paper_JCIM_SI.tex:57` inputs `sections/appendix`, so the statement is compiled into the
JCIM SI. It is directionally wrong: `beta*` rises from 4.35 to 9.26 while `K_eff` falls
from 99 to 32.1.

The appendix therefore contradicts itself. Line 190 asserts `beta*` proportional to
`log K_eff`, while lines 156 to 160 of the same file argue that a lower `log K_eff` forces
a higher `beta*`. Both cannot hold. Delete the line 190 proportionality claim, and reduce
the lines 156 to 160 mechanism to an observation, since it is asserted rather than derived.

## 4. The reported "inflection point" is not an inflection point

**VERIFIED as a terminology error. DEFLATED on the order-dependence claim: the effect is
one step on the beta grid.**

The detector in `code/src/Binding.jl:881-913` selects the most negative second derivative
of entropy with respect to `log beta`. That is the onset of the entropy drop. A
mathematical inflection point is a zero crossing of the second derivative.

`code/src/Protein.jl:423-474` already distinguishes `beta_onset` (maximum downward
curvature) from `beta_steepest` (steepest drop, the transition midpoint), and already
documents the statistic as a finite-size entropy crossover evaluated at stored memories
rather than a phase-transition estimate. The manuscript uses the older language.

### Recommended correction

- Replace "phase transition" with "entropy crossover."
- Replace "inflection" with "onset," or use the steepest-drop midpoint and identify it
  explicitly.
- Use `beta_onset` rather than `beta*` if retaining the current maximum-curvature
  definition.
- State that entropy is evaluated at stored memory vectors and is used to select a
  descriptive operating point.

### Order dependence: smaller than reported

The detector evaluates the first 20 memory columns, so the result does depend on
alignment order. The original audit reported an all-memory comparison at rho = 500:

- Homeobox: current onset 4.58; all-memory onset 3.85
- Conotoxin: current onset 7.71; all-memory onset 6.48
- Kunitz, SH3, WW, and Forkhead unchanged

All four of these values are exact points on the fixed 50-point logarithmic beta grid,
`10 .^ range(log10(0.1), log10(500), length=50)`, whose successive ratio is 1.1898:

| Value | Grid index (0-based) | Grid value |
|---:|---:|---:|
| 3.85 | 21 | 3.8483 |
| 4.58 | 22 | 4.5789 |
| 6.48 | 24 | 6.4825 |
| 7.71 | 25 | 7.7131 |

Each reported discrepancy is therefore exactly one grid step, and four of six families are
unchanged. This is grid resolution, not a broken operating point.

Recommended action: document the probe procedure explicitly (first 20 alignment columns),
and report a one-grid-step sensitivity check for Homeobox and conotoxin. A full
regeneration of the canonical sweeps is not warranted on this evidence. Note that the
newer `find_entropy_transition` uses 20 randomly chosen probes rather than all memories,
so it is order independent but is still not an all-memory calculation.

## 5. Citation and grammar corrections

**All VERIFIED. All trivial.**

### VAE citation

`paper-jcim/sections/introduction.tex:4` cites `sinaiAdaptiveMLDE2020` as a
variational-autoencoder example. The entry at `paper-jcim/References_v1.bib:190-197` is
Yang, Wu, and Arnold, "Machine-learning-guided directed evolution for protein
engineering," *Nature Methods*, volume 16, pages 687 to 694, 2019. The citation key's implied
author and year are both wrong in addition to the paper being the wrong kind of work.

A suitable primary example:

- Hawkins-Hooker et al., "Generating functional protein variants with variational
  autoencoders," PLOS Computational Biology: <https://doi.org/10.1371/journal.pcbi.1008736>

(This replacement reference was suggested by the original audit and has not been
independently verified here.)

### ESM2 masked-marginal scoring

`paper-jcim/sections/methods.tex:55-57` cites the ESMFold paper for ESM2 masked-marginal
pseudo-perplexity. A more direct primary reference is:

- Meier et al., "Language models enable zero-shot prediction of the effects of mutations
  on protein function," NeurIPS 2021: <https://papers.nips.cc/paper_files/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html>

### Grammar

`paper-jcim/sections/discussion.tex:37-40` contains an unmatched closing parenthesis and a
broken subject and verb construction:

> If the designation comes from an independent experimental campaign, such as phage
> display or yeast surface display, functional selections, or even literature curation)
> produce exactly this kind of small, functionally characterized set.

Rewrite as two or three sentences.

`paper-jcim/sections/methods.tex:22-23` reads:

> for SH3, the selected Trp marker binding groove was the marker

This is incomplete and does not state the marker position.

### Broad introductory claims

Qualify the statement that profile HMMs generally produce low structural confidence and
poor compositional fidelity. The evidence supports this for the reported Kunitz
comparison, not for profile HMMs in general.

Also qualify the statement that most learned models lack sufficient training signal in the
small-family regime. Pretrained models do not rely only on family-specific training data.

## 6. Discretionary items

**Agreed in substance. Scope as desired. None of these block submission on their own.**

### Exact Gaussian-mixture target and the role of ULA

The proposition at `paper-jcim/sections/theory.tex:42-58` is correct and belongs in the
main text. For unit-norm memories the Gibbs density is exactly a Gaussian mixture, so
independent equilibrium samples are available by drawing a component index and adding
Gaussian noise. Since `build_memory_matrix` does project to unit norm, this applies
exactly to every run in the paper, and a referee will ask why Langevin is used at all.
Answer it in the text rather than in a rebuttal.

- Identify direct Gaussian-mixture sampling as the preferred equilibrium implementation
  for the present unit-norm energy.
- Explain that ULA is retained for continuity with the earlier stochastic-attention
  implementation and for studying finite-step dynamics.
- Avoid presenting ULA as necessary for sampling this target.
- Keep the exact-versus-ULA comparison as an implementation benchmark.

### Exact-versus-ULA conclusions

`paper-jcim/sections/discussion.tex:6-10` says the benchmark rules out ULA convergence
error. The finite comparison supports the narrower statement that finite-step ULA error
was not the main source of the observed calibration gap. Also state the direction of the
reported composition KL divergence; the implementation computes reference to generated.

### Real-valued multiplicity

`paper-jcim/sections/theory.tex:38-40` says a real-valued `r_k` is equivalent to storing
`r_k` copies. The literal equivalence holds only for integer multiplicities. For positive
real weights the method is a continuous generalization of integer replication.

### Conotoxin functional language

Only five designated and two background accessions have exact activity metadata matches,
as stated at `paper-jcim/sections/methods.tex:29-34`. No generated sequence was tested for
Cav2.2 binding.

- Use "Tyr13-associated marker" rather than "Cav2.2-specific pharmacophore."
- Use "designated accessions" rather than "binders" where activity is unverified.
- State that generated sequences preserve or enrich reported sequence markers.
- Do not imply preserved binding, specificity, or pharmacological activity.

### Other overstatements

- Replace "ruling out" with "indicating that it was not the main source."
- Replace "preserves exactly" where the reported values differ.
- Avoid "near zero" where deviations approach 0.15.
- Describe the Fisher measure as the study-defined separation index. The definition at
  `paper-jcim/sections/methods.tex:47` is built from pairwise cosine similarities within
  and between groups, which is not the classical Fisher discriminant ratio. Note that the
  canonical CSV column is already named `separation_index`, so only the prose needs
  changing.
- Keep the five-family regression explicitly exploratory.

### Sentence length and tone

The intended tone is simple, direct, concise, and neutral. An automated source scan found
at least 24 sentences longer than 45 words, concentrated in:

- `paper-jcim/sections/results.tex:89-100`
- `paper-jcim/sections/results.tex:153-168`
- `paper-jcim/sections/results.tex:212-228`
- `paper-jcim/sections/results.tex:271-279`
- `paper-jcim/sections/methods.tex:2-17`
- `paper-jcim/sections/methods.tex:18-34`

Editing rule: keep most sentences below 30 to 35 words, put one result or interpretation
in each sentence, and separate methods, results, and interpretation rather than joining
them with commas and semicolons.

Flagged phrases, with verified occurrence counts across `sections/*.tex` and
`Paper_JCIM.tex`:

| Phrase | Occurrences |
|---|---:|
| "small-set amplifier" | 4 |
| "principled" | 2 |
| "central finding" | 1 |
| "negligible computation" | 1 |
| "natural axis" | 1 |
| "near-perfect" | 1 |
| "faithful recapitulation" | 1 |
| "ruling out" | 1 |
| "confirming high fidelity" | 0 (not present) |

Prefer "increased," "was associated with," "matched within," "suggested," "we observed,"
"provides a way to." Figure and table captions should describe the result rather than
announce a conclusion.

### Narrative flow

A clearer main narrative:

1. The energy has an exact Gaussian-mixture equilibrium.
2. Multiplicity directly controls latent component probability.
3. Reconstruction and decoding create a marker-level calibration gap.
4. The six-family study examines how that gap varies with family geometry.
5. The conotoxin analysis is a case study in transferring sequence markers.
6. Structure and language-model calculations provide model-based checks, not functional
   validation.

The hard-mask subsection can be shortened or moved to the SI. The Discussion repeats
several results from the Results section and can be reduced.

## Checks that passed

- All 149 repository tests passed.
- The exact Gaussian-mixture derivation is correct.
- The score expression includes the required factor of beta.
- Chain-level uncertainty is used for the canonical replicated sweeps.
- The five-family relationship is described as exploratory and includes
  leave-one-family-out sensitivity.
- The conotoxin section includes appropriate limitations concerning activity metadata and
  the absence of functional testing.
- The JCIM manuscript, JCIM SI, and arXiv PDFs render cleanly.
- No unresolved citations or references were found.
- No overfull boxes, clipping, or overlapping elements were found.
- Tables, equations, figures, and captions remain inside their page boundaries.
- The JCIM and arXiv section sources are consistent apart from an arXiv layout-only
  wrapper around the exact-mixture proposition.
- The repository was clean after the audit.

The arXiv version uses the same substantive section text as the JCIM version, so all
scientific and prose corrections above apply to both.

---

## Reconciliation of the two audit passes

GPT 5.6 follow-up, 2026-07-29

Claude's verification pass improves the original audit, but it contains one definite
factual error and downplays two reviewer-facing issues.

### Corrections from Claude that should be accepted

#### Conotoxin coordinate frame

The original audit's replacement values for Cys15 and position 21 were wrong. The
original calculation cleaned the designated-only alignment. That operation selected a
different set of alignment columns from the canonical full-family preprocessing step.

The correct calculation selects the 23 designated accessions from the canonical cleaned
74-by-26 full-family alignment. This is the coordinate frame used by the generated
sequences. An independent reproduction using `canonical_load_alignment` and
`canonical_split` gave:

| Position | WT or residue class | Corrected designated-input frequency |
|---|---|---:|
| 13 | Tyr | 19/23 = 0.83 |
| 2 | K/R | 22/23 = 0.96 |
| 10 | K/R | 16/23 = 0.70 |
| 11 | Leu | 4/23 = 0.17 |
| 1 | Cys | 23/23 = 1.00 |
| 8 | Cys | 23/23 = 1.00 |
| 15 | Cys | 23/23 = 1.00 |
| 16 | Cys | 23/23 = 1.00 |
| 20 | Cys | 23/23 = 1.00 |
| 25 | Cys | 23/23 = 1.00 |
| 21 | K/R | 11/23 = 0.48 |
| 4 | K/R | 12/23 = 0.52 |

Claude is therefore correct that:

- Cys15 remains 1.00.
- Position 21 should be 0.48, not 0.57.
- The conotoxin script should be fixed and the table regenerated from the pipeline rather
  than edited by hand.
- The headline Tyr13 comparison is unchanged.
- Apparent enrichment at Cys20 and Cys25 disappears because the aligned input is already
  1.00 at both positions.

#### Hard-mask inference

Claude is also correct to upgrade the hard-mask inference to a must-fix item.
`Delta_attn` near zero shows that each condition realizes its own intended designated
mass. It does not show that the difference between finite multiplicity and hard masking
is purely a decode effect, because hard masking changes the intended designated mass to
one.

The replacement interpretation should state:

- Hard masking removed the background components and raised intended designated mass
  from `f_eff` to 1.
- The recovery difference decreased as `f_eff` approached 1.
- The change in observed marker fraction was smaller than the change in intended
  designated mass.

The last point should be attributed to the complete latent-to-sequence map, including
finite-temperature Gaussian noise, PCA reconstruction, and argmax decoding. It should
not be attributed to reconstruction and decoding alone.

#### Entropy-crossover sensitivity

The one-grid-step differences for Homeobox and conotoxin do not by themselves establish
that every canonical sweep must be regenerated. A defensible minimal correction is:

- rename the statistic;
- disclose how probes were selected;
- report the observed one-grid-step sensitivity; and
- avoid presenting the operating point as a thermodynamic phase transition.

Future calculations should use the order-independent implementation. Describing the
difference as purely “grid resolution” is too dismissive, because the probe selection
caused the result to move to the neighboring grid point. However, the current evidence
does not show a material failure of the generated results.

### Correction to Claude's verification

#### The proportionality statement was not refuted

Claude states that no SI claim of

```
beta* proportional to log K_eff
```

exists. That statement is incorrect.

`paper-jcim/sections/appendix.tex:189-192` explicitly says:

```
with beta*(rho) increasing from 4.35 (rho=1) to 9.26
(rho=1000), consistent with beta* proportional to log K_eff
predicted by the mean-field argument in the main text
```

`paper-jcim/Paper_JCIM_SI.tex:57` includes `sections/appendix`, so the statement is part
of the compiled JCIM SI. It is also present at the same location in
`paper-arxiv/sections/appendix.tex`.

The statement is directionally inconsistent: the reported `beta*` increases while
`log K_eff` decreases. It should be removed. The surrounding observation can instead say
that the entropy crossover shifted to higher inverse temperature as multiplicity
increased, without asserting proportionality or a derived mechanism.

### Items that Claude downplayed

#### Entropy identity

The false identity does not affect the reported simulations because the operating-point
code calculates attention entropy directly. No numerical sweep must be regenerated for
this reason.

However, “numerically negligible” is too broad. The Shannon and Rényi-2 values differ by
about 5.7% at `rho=10`. More importantly, the false identity is used to motivate several
causal statements. The correction should therefore include:

- the correct Shannon-entropy expression;
- a separate definition of `K_eff`;
- removal of the unsupported causal explanation; and
- removal of the false proportionality statement in the appendix.

This is a conceptual correction even though it does not change generated data.

#### Exact sampler and ULA

The exact-sampler/ULA framing should not be treated as optional. For the unit-norm model,
independent equilibrium samples are available directly, so a referee is likely to ask why
an approximate correlated sampler is used.

This does not require replacing all reported results. A short explicit explanation is
enough:

> Direct Gaussian-mixture sampling is the preferred equilibrium implementation for the
> unit-norm model. We retain ULA here to maintain comparability with the original
> stochastic-attention workflow and to evaluate its finite-step behavior.

ULA should not be presented as necessary for sampling the equilibrium target.

### Final agreed priority order

Before submission:

1. Fix the conotoxin SAR pipeline and regenerate Table 6 and its associated prose.
2. Rewrite the hard-mask paragraph without the pure-decode inference.
3. Correct the Shannon/Rényi identity and delete the false
   `beta* proportional to log K_eff` statement.
4. Rename “phase transition” and “inflection” as “entropy crossover” and “onset,” then
   report the operating-point sensitivity.
5. Explain why ULA is retained despite the availability of the exact sampler.
6. Correct the VAE and ESM2 citations and the two grammar errors.
7. Complete the requested prose pass for short, direct, neutral sentences.

On the current evidence, a full regeneration of every canonical sweep is not required.
The conotoxin SAR table must be regenerated after the coordinate-frame correction.
