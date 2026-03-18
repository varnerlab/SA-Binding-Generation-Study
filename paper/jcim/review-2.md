# Simulated PLOS Computational Biology Review

**Paper:** Conditioning Protein Generation via Hopfield Pattern Multiplicity
**Venue:** PLOS Computational Biology (simulated)
**Recommendation:** Major Revision
**Reviewer Expertise:** Computational protein design, generative sequence models, statistical biophysics

---

## Overview

This manuscript describes a training-free method for conditioning protein sequence generation on a user-specified functional subset by assigning multiplicity weights to patterns stored in a modern Hopfield network. The method is theoretically elegant, computationally cheap, and addresses a genuine practical need in protein engineering: expanding a small characterized set into a diverse library without retraining a generative model. The paper is clearly written and the theory section is rigorous. However, several issues related to experimental scope, biological validation, and methodological transparency need to be addressed before this work is suitable for publication.

---

## Major Comments

### M1 — The central claim (Fisher separation predicts calibration gap) is supported by only three data points

The most prominent result — that the calibration gap $\Delta$ is linearly predicted by the Fisher separation index $S$ with $R^2 \approx 1.0$ — is based on three protein families. This is not a validation of a predictive relationship; it is a constraint satisfied by any line through three points. The authors acknowledge this in the Discussion, but continue to present $R^2 \approx 1.0$ in the abstract, results, and figure captions as if it represents strong evidence. The recommendation is to either (a) test at least three additional Pfam families before making this claim, or (b) re-frame the finding as a hypothesis supported by preliminary evidence, removing $R^2$ from the abstract entirely.

### M2 — Functional splits are ad hoc and may not reflect binding specificity

The designated subsets are defined by single marker positions treated as proxies for function (P1 K/R for trypsin inhibition, Trp in SH3 binding groove, specificity-loop residue in WW). While there is biological support for P1 as a specificity determinant in Kunitz domains, the SH3 and WW splits are defined algorithmically rather than from experimental binding data. It is unclear whether generating sequences with Trp at the SH3 binding groove actually produces better binders, or whether the WW specificity-loop split corresponds to any characterized functional difference. The paper should either (a) validate the SH3 and WW splits against available binding data, or (b) be explicit that these are synthetic marker experiments and not bona fide functional conditioning tests.

### M3 — No evaluation of structural plausibility of generated sequences

The paper reports 100% valid residues and low KL divergence as quality metrics. These are necessary but not sufficient for a computational biology audience. The following are expected:

- **Predicted fold quality:** ESMFold or AlphaFold2 pLDDT scores for a random subset of generated sequences would demonstrate that the sequences are structurally plausible.
- **Novelty vs. memorization:** Mean maximum sequence identity to stored patterns should be reported to establish that generated sequences are genuinely novel and not near-duplicates of training patterns.
- **Disulfide framework conservation:** For the $\omega$-conotoxin experiment specifically, do generated sequences preserve the C–C–CC–C–C disulfide pattern that defines the O-superfamily scaffold? This is a hard structural constraint that the method should satisfy.

### M4 — The $\omega$-conotoxin experiment needs stronger biological grounding

The conotoxin case study is presented as demonstrating "pharmacophore transfer," but the analysis is entirely sequence-statistical. Tyr13 frequency is a correlate of Cav2.2 binding, not a direct measure of it. Given that this family has an FDA-approved drug member (ziconotide) and rich structural data (PDB structures exist for MVIIA, GVIA, CVID bound to Cav2.2 mimetics), the paper is missing an obvious validation step: run a subset of generated sequences through a published docking protocol or compute predicted binding free energies using FoldX or Rosetta. Even 10–20 scored sequences would substantially strengthen the "therapeutic peptide design" framing. If wet-lab validation is not feasible, the authors should soften the therapeutic language and frame the conotoxin experiment as a sequence-level pharmacophore recovery test.

### M5 — Comparison to existing training-free methods is absent

The paper positions itself as training-free but does not compare against other training-free baselines for conditional sequence generation. Relevant comparisons include:

- **Direct-coupling analysis (DCA) with sequence reweighting:** A standard approach for generating sequences with functional biases by reweighting the empirical distribution.
- **MSA subsampling:** Simply running standard SA on the curated subset alignment (which the paper does as "hard curation") is itself a baseline; what does multiplicity weighting add over hard curation beyond retaining background fold constraints?
- **EVMutation / EVcouplings conditional sampling:** Established tools for sequence generation that could be applied to the same functional splits.

Without these comparisons, it is impossible to assess the added value of the Hopfield framework.

---

## Minor Comments

### m1 — $R^2 \approx 1.0$ is a misleading statistic for $n = 3$

Every linear fit through three points achieves $R^2 = 1.0$. This value should be removed from the abstract, figure captions, and conclusions. The Pearson correlation coefficient with a confidence interval would be more informative if additional families are added.

### m2 — Phase transition characterization is descriptive, not predictive

The paper identifies $\beta^*(\rho)$ empirically via entropy inflection and states that it depends on $K_\text{eff}(\mathbf{r})$. The Discussion mentions a "mean-field argument" but no derivation is presented. Either derive the $\beta^*$–$K_\text{eff}$ relationship analytically (even approximately) or remove the theoretical claim and present $\beta^*(\rho)$ purely as an empirical observation.

### m3 — Scaling study conclusions are partially circular

The scaling study (Fig. 2) shows that diversity increases with $K_\text{des}$ and saturates near $K_\text{des} = 20$. This is an expected consequence of PCA subspace dimensionality and is not surprising. The paper should quantify the PCA dimension $d$ of each subsampled set to confirm that the diversity saturation tracks the intrinsic dimensionality of the input set.

### m4 — The weak-binder set for $\omega$-conotoxin is only 2 sequences

The weak/non-N-type-selective set (MVIIC and MVIID) contains only 2 sequences and is not used in generation experiments. Its inclusion in the framing implies a contrast that is never tested. Either run a weak-binder-seeded generation experiment (analogous to the Kunitz background-conditioned experiment) or remove references to the weak-binder set.

### m5 — Methods for $\beta^*$ detection are underspecified

The text states that $\beta^*$ is found by "locating the inflection point of the weighted attention entropy." The numerical procedure (derivative method, tolerance, number of $\beta$ points) is not described precisely enough to reproduce. The code is available on GitHub, but the methods section should be self-contained.

### m6 — Figure axis labels are too small in several figures

Figures 3, 5, 6, and 7 have axis labels and tick annotations that are difficult to read at print resolution. Please increase font sizes to a minimum of 10pt for axis labels and 8pt for tick labels.

---

## Recommendations for Revision

**Required for major revision:**
1. Expand cross-family validation to $\geq 5$ families; remove $R^2 \approx 1.0$ from abstract until validated.
2. Add ESMFold pLDDT scores and novelty (max sequence identity to stored patterns) for generated sequences in at least the Kunitz and conotoxin experiments.
3. Verify or remove the claim that SH3/WW splits reflect biologically meaningful functional differences.
4. Add at least one structural or energetic validation for $\omega$-conotoxin generated sequences.
5. Confirm that generated conotoxin sequences preserve the C–C–CC–C–C cysteine framework.

**Recommended:**
6. Add a DCA or EVmutation baseline comparison.
7. Derive or clearly bound the $\beta^*$–$K_\text{eff}$ relationship analytically.
8. Run a weak-binder-seeded generation experiment for $\omega$-conotoxins or remove the weak-binder framing.

---

## Summary

This is an interesting and clearly presented methods paper with a genuine practical contribution: a training-free, logit-bias approach to conditioning Hopfield-based protein generation on user-specified functional subsets. The calibration gap decomposition is the most novel finding. However, the cross-family validation is underpowered (3 points), the functional splits are not fully grounded in experimental biology, and the therapeutic framing of the conotoxin experiment is not supported by any binding-activity evidence. A major revision addressing the cross-family sample size, structural validation, and one binding-activity proxy would make this a strong contribution to the journal.
