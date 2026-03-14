# Conditioning Protein Sequence Generation via Pattern Multiplicity in the Modern Hopfield Energy

## Paper Outline

---

### Abstract (150 words)

Training-free protein sequence generation via stochastic attention (SA) produces plausible family members from small alignments, but cannot condition on functional subsets — e.g., generating binders rather than arbitrary family members. We show that assigning multiplicity weights $r_k$ to stored patterns defines a tilted Hopfield energy $E_r(\xi) = \frac{1}{2}\|\xi\|^2 - \frac{1}{\beta}\log\sum_k r_k \exp(\beta m_k^\top \xi)$ whose Langevin dynamics sample from a known Boltzmann distribution, with no retraining or architectural changes. A single scalar parameter — the multiplicity ratio $\rho = r_{\text{binder}}/r_{\text{nonbinder}}$ — continuously interpolates between unconditioned generation ($\rho=1$) and hard subset curation ($\rho\to\infty$). We characterize the phase transition $\beta^*(\rho)$ as a function of the effective pattern count $K_{\text{eff}}(r)$, and identify a family-dependent calibration gap between the energy-level conditioning and the decoded sequence phenotype, predicted quantitatively by the Fisher separation of functional subsets in PCA space. Experiments on Kunitz, SH3, and WW domains validate the theory.

---

### 1. Introduction

**Opening:** SA generates protein sequences from small family alignments by treating attention-based memory retrieval as an exact score function for Langevin dynamics [cite parent paper]. But the memory matrix encodes fold/family constraints, not functional specificity — all family members contribute equally to the energy landscape. In practice, experimentalists want not just "a family member" but a *binder to a specific target*.

**The gap:** Binding is relational (protein + target). Family alignments encode fold but not target specificity. How do you condition the generative process on a functional subset without retraining?

**Our contribution:** We show that pattern multiplicity in the modern Hopfield energy provides an exact, training-free conditioning mechanism. The key results:

1. **Multiplicity-weighted Hopfield energy** — a one-parameter extension ($\rho$) that tilts the Boltzmann distribution toward any functional subset while preserving the analytic score function
2. **Phase transition theory** — $\beta^*(\rho)$ depends on $K_{\text{eff}}(r) = (\sum r_k)^2 / \sum r_k^2$, not raw $K$
3. **Calibration gap** — the PCA encoding introduces a family-dependent attenuation between energy-level conditioning and decoded phenotype, predicted by the Fisher separation index $S$ of the functional subsets in PCA space
4. **Cross-family validation** — $S$ quantitatively predicts the gap magnitude across three protein families ($R^2 \approx 1.0$)
5. **Practitioner's decision rule** — $S > 0.3$: use multiplicity weighting; $S < 0.15$: use hard curation

---

### 2. Theory

#### 2a. Background: SA and the Modern Hopfield Energy

Briefly recap from parent paper:
- Memory matrix $X = [m_1, \ldots, m_K] \in \mathbb{R}^{d \times K}$ (unit-norm PCA-encoded sequences)
- Hopfield energy: $E(\xi) = \frac{1}{2}\|\xi\|^2 - \frac{1}{\beta}\log\sum_k \exp(\beta m_k^\top \xi)$
- Score function: $\nabla \log p_\beta(\xi) = X \cdot \text{softmax}(\beta X^\top \xi) - \xi$
- ULA update: $\xi_{t+1} = (1-\alpha)\xi_t + \alpha X \text{softmax}(\beta X^\top \xi_t) + \sqrt{2\alpha/\beta}\,\epsilon_t$
- Phase transition at $\beta^*$: entropy inflection separating exploration from retrieval

#### 2b. Multiplicity-Weighted Hopfield Energy (NEW)

**Definition.** Assign multiplicity $r_k > 0$ to each stored pattern. The weighted energy is:

$$E_r(\xi) = \frac{1}{2}\|\xi\|^2 - \frac{1}{\beta}\log\sum_{k=1}^K r_k \exp(\beta m_k^\top \xi)$$

**Proposition 1.** The score function of $p_r(\xi) \propto \exp(-\beta E_r(\xi))$ is:

$$\nabla \log p_r(\xi) = X \cdot \text{softmax}(\beta X^\top \xi + \log r) - \xi$$

yielding the Langevin update:

$$\xi_{t+1} = (1-\alpha)\xi_t + \alpha X \text{softmax}(\beta X^\top \xi_t + \log r) + \sqrt{2\alpha/\beta}\,\epsilon_t$$

*The only change from standard SA is adding $\log r_k$ to each logit before the softmax.* The memory matrix $X$ is unchanged. No retraining, no new parameters to fit.

**Remark.** This is equivalent to a Hopfield network with repeated patterns: storing pattern $m_k$ with multiplicity $r_k$ is equivalent to duplicating column $k$ exactly $r_k$ times in the memory matrix, but without the $O(d \cdot \sum r_k)$ memory cost.

#### 2c. Multiplicity Ratio and Effective Binder Fraction

**Parameterization.** For a binary split (binder / non-binder), define the multiplicity ratio:

$$\rho = \frac{r_{\text{binder}}}{r_{\text{nonbinder}}}$$

The effective binder fraction in the stationary distribution (at uniform similarity) is:

$$f_{\text{eff}}(\rho) = \frac{K_b \cdot \rho}{K_b \cdot \rho + K_{nb}}$$

This gives continuous control: $\rho = 1 \Rightarrow f_{\text{eff}} = K_b/K$ (natural proportion); $\rho \to \infty \Rightarrow f_{\text{eff}} \to 1$ (hard curation).

**Inverse formula.** To achieve a target effective fraction $f$:

$$\rho(f) = \frac{f \cdot K_{nb}}{K_b \cdot (1 - f)}$$

#### 2d. Phase Transition Under Multiplicity

**Effective number of patterns:**

$$K_{\text{eff}}(r) = \frac{(\sum_k r_k)^2}{\sum_k r_k^2}$$

This is the participation ratio of the multiplicity vector. As $\rho \to \infty$, $K_{\text{eff}} \to K_b$.

**Conjecture (supported empirically).** The phase transition of the weighted system satisfies $\beta^*(r) \sim g(K_{\text{eff}}(r), d)$, where $g$ is the same function that governs the unweighted system with $K_{\text{eff}}$ patterns. Empirically, $\beta^*$ increases monotonically as $K_{\text{eff}}$ decreases (the system becomes "harder to order" with fewer effective patterns contributing competing attractors).

#### 2e. The Calibration Gap

The signal chain from multiplicity weights to decoded sequence phenotype:

$$\rho \xrightarrow{\text{exact}} f_{\text{eff}} \xrightarrow{\text{exact}} \text{attention weights} \xrightarrow{\text{lossy (PCA)}} \text{soft residue score} \xrightarrow{\text{nonlinear (argmax)}} \text{decoded phenotype}$$

**Definition.** The calibration gap $\Delta(f_{\text{eff}})$ is:

$$\Delta = f_{\text{eff}} - f_{\text{obs}}$$

where $f_{\text{obs}}$ is the fraction of generated sequences displaying the binder phenotype after decoding.

**Decomposition.** Three independently measurable components:
1. **Attention gap** $\Delta_{\text{attn}} = f_{\text{eff}} - \bar{a}_{\text{binder}}$ — vanishes ($\approx 0$) in all experiments
2. **PCA gap** $\Delta_{\text{PCA}} = \bar{a}_{\text{binder}} - f_{\text{soft}}$ — dominant component, family-dependent
3. **Argmax gap** $\Delta_{\text{argmax}} = f_{\text{soft}} - f_{\text{obs}}$ — negative (argmax partially rescues signal)

**Prediction.** The PCA gap depends on the geometric separation of functional subsets in PCA space. Define the Fisher separation index:

$$S = \frac{\bar{c}_{\text{within}} - \bar{c}_{\text{between}}}{\frac{1}{2}(\sigma_{\text{within}} + \sigma_{\text{between}})}$$

where $\bar{c}$ and $\sigma$ are mean and std of pairwise cosine similarities within/between groups. Families with higher $S$ have smaller calibration gaps.

---

### 3. Results

#### 3a. Curated Memory Validates the Concept (Kunitz Domains)

- Split Kunitz (PF00014) into strong binders (K/R at P1, $K_b=32$) and weak/non-binders ($K_{nb}=67$)
- Hard curation: 100% P1 phenotype transfer, even with only 3 input binders
- Weak-conditioned SA: 0% K/R at P1
- *"Give me your 10 best binders, I'll give you 150 new ones in 30 seconds"*

**[Figure: P1 phenotype inheritance bar chart — strong/weak/full/strong-conditioned/weak-conditioned]**

#### 3b. Binder Scaling: How Many Binders Do You Need?

- Subsample 3, 5, 8, 10, 15, 20, 25, 32 binders from Kunitz strong set
- Phenotype fidelity = 100% at all sample sizes (including $K_b = 3$)
- Diversity increases with $K_b$: 0.36 (3 binders) → 0.50 (32 binders)
- KL divergence decreases with $K_b$: 0.047 → 0.009

**[Figure: Scaling triptych — fidelity / diversity / KL vs. N binders]**

#### 3c. Multiplicity-Weighted Generation

- Sweep $\rho \in [1, 1000]$ on Kunitz full family with multiplicity weighting
- Attention on binders tracks $f_{\text{eff}}$ perfectly at all $\rho$ (confirming Proposition 1)
- $\beta^*(\rho)$ increases from 4.3 to 10.1 as $K_{\text{eff}}$ drops from 99 to 32
- P1 K/R fraction increases monotonically: 0.39 ($\rho=1$) → 0.63 ($\rho=500$)

**[Figure: ρ sweep — 4-panel: fidelity / diversity / KL / loop entropy vs. log₁₀(ρ)]**
**[Figure: Entropy curves at different ρ, showing phase transition shift]**

#### 3d. Calibration Gap Decomposition

- Three layers measured independently: attention / soft decode / hard decode
- Attention gap ≈ 0% — multiplicity works exactly
- PCA gap = dominant — soft P1 score flat at ~0.27 regardless of $\rho$
- Argmax gap = negative — rescues ~0.25 of signal

**[Figure: Gap decomposition — three curves overlaid on calibration plot]**

#### 3e. Cross-Family Validation: Separation Predicts the Gap

Three families, three separation indices, three calibration gaps:

| Family | $K$ | $S$ | $\Delta$ (at $\rho=500$) |
|---|---|---|---|
| WW | 420 | 0.11 | 0.63 |
| Kunitz | 99 | 0.20 | 0.37 |
| SH3 | 55 | 0.34 | 0.01 |

- Linear relationship: $\Delta \approx 0.84 - 2.5 \cdot S$ ($R^2 \approx 1.0$ on 3 points)
- SH3: multiplicity conditioning nearly closes the gap ($f_{\text{obs}} = 0.989$ at $f_{\text{eff}} = 0.999$)
- WW: large gap, hard curation necessary

**[Figure: Separation vs. gap scatter + regression]**
**[Figure: Multi-family calibration curves overlaid]**
**[Figure: Per-family ρ sweep panels showing attention vs. f_obs]**

#### 3f. The β Lever: Operating Above the Phase Transition

- At fixed $\rho$, increasing $\beta/\beta^*$ from 1.0 to 3.0 boosts phenotype transfer
- ρ=200, β=3×β*: f_obs=0.71 (vs 0.61 at β*)
- Cost: diversity drops from 0.53 to 0.46
- Provides a second continuous dial for practitioners

**[Figure: β sweep panels at ρ=10, 50, 200]**

---

### 4. Discussion

#### 4a. The Practitioner's Decision Rule

Given a family alignment and a set of characterized binders:

1. Build the full-family memory matrix (PCA)
2. Compute Fisher separation index $S$ between binder and non-binder subsets
3. Decision:
   - $S > 0.3$: Use multiplicity weighting (preserves fold context, nearly perfect transfer)
   - $0.15 < S < 0.3$: Use multiplicity weighting at high $\rho$ with $\beta > \beta^*$
   - $S < 0.15$: Use hard curation (perfect transfer, but loses non-binder fold context)

#### 4b. Why the Gap Exists and Why It Matters

The calibration gap is not a failure of the theory — it's a well-characterized property of the PCA encoding. The multiplicity-weighted energy is exact; the lossy step is the PCA projection that compresses 20L-dimensional one-hot space to d-dimensional PCA space. When binder-defining variation (e.g., K vs G at P1) is orthogonal to the dominant PCA axes, the tilt in the energy landscape doesn't propagate to the decoded residue identity.

This is *informative*: it tells us exactly when multiplicity conditioning will work (high $S$) and when hard curation is needed (low $S$).

#### 4c. Connection to Importance Weighting in Statistical Mechanics

The multiplicity-weighted Hopfield energy can be viewed as a tilted Boltzmann distribution:

$$p_r(\xi) = \frac{1}{Z_r} \exp(-\beta E_r(\xi))$$

The multiplicity vector $r$ defines a measure over stored patterns, and varying $r$ sweeps out a family of distributions connected by exponential tilting. This connects to importance sampling, annealed importance sampling, and tilted variational inference.

#### 4d. Limitations

- The cross-family validation uses only 3 families; more are needed to confirm the $S$–$\Delta$ relationship
- The functional splits are based on single-position markers; multi-position phenotypes may behave differently
- No experimental validation of generated binder sequences
- The PCA bottleneck may be partially addressable by alternative encodings (e.g., ESM embeddings, learned representations)

#### 4e. Outlook

- Experimental validation: synthesize and test top-ranked candidates from Kunitz strong-conditioned generation
- Multi-position phenotypes: extend to phenotypes defined by multiple correlated positions
- Alternative encodings: replace PCA with learned embeddings that better separate functional subsets
- Combination with post-hoc filtering: multiplicity conditioning as a prior, AF-Multimer scoring as a likelihood

---

### 5. Methods

#### 5a. Data
- Pfam seed alignments: PF00014 (Kunitz), PF00018 (SH3), PF00397 (WW)
- Alignment cleaning: max 50% column gaps, max 30% sequence gaps
- One-hot encoding (20 AA × L positions) → PCA (95% variance retained) → unit-norm

#### 5b. Multiplicity-Weighted Sampling
- ULA with log-multiplicity logit bias (Algorithm 1, extended)
- β* found via entropy inflection on weighted attention
- 20–30 chains × 5000 steps, α=0.01, burn-in=2000, thin=100

#### 5c. Evaluation Metrics
- Phenotype fidelity: fraction of generated sequences displaying marker residue
- Sequence diversity: 1 - mean pairwise sequence identity
- AA composition: KL divergence vs. reference set
- Valid residue fraction
- Attention diagnostics: mean softmax weight on binder patterns during sampling
- Soft decode score: continuous residue probability before argmax

#### 5d. Fisher Separation Index
- Pairwise cosine similarities within and between functional groups in PCA space
- $S = (\bar{c}_{\text{within}} - \bar{c}_{\text{between}}) / ((\sigma_{\text{within}} + \sigma_{\text{between}})/2)$

---

### Figures Summary

1. **Fig 1.** Schematic: multiplicity-weighted Hopfield energy + Langevin dynamics
2. **Fig 2.** Kunitz: P1 phenotype inheritance (strong/weak conditioned)
3. **Fig 3.** Binder scaling: fidelity / diversity / KL vs. N input binders
4. **Fig 4.** ρ sweep: 4-panel generation quality across multiplicity ratio
5. **Fig 5.** Phase transition: entropy curves shifting with ρ; β* vs K_eff
6. **Fig 6.** Calibration gap decomposition: attention / soft / hard decode layers
7. **Fig 7.** Cross-family validation: separation index predicts calibration gap (the money figure)
8. **Fig 8.** Per-family ρ sweep panels: attention vs. f_obs across families
