Dear Editors of the *Journal of Chemical Information and Modeling*,

Please consider our Article, “Conditioning Protein Generation via Hopfield Pattern
Multiplicity,” for publication in *JCIM*.

This work presents a training-free method for conditioning protein sequence generation when
only a small designated subset of a family is available. Pattern multiplicities enter the
modern Hopfield energy as logit biases, giving an exact Gaussian-mixture equilibrium target
without retraining or changing the sampler’s asymptotic cost. The paper also distinguishes
this latent target from the sequence-level distribution obtained after PCA reconstruction and
discrete decoding, which exposes and quantifies a practically important calibration gap.

The computational study includes five Pfam families and a curated omega-conotoxin family,
replicated multiplicity sweeps, exact-equilibrium controls, hard-mask and hard-curation
comparisons, and sequence-, structure-, and language-model assessments. We additionally
introduce a matched HMMER3 benchmark that uses the same cleaned alignments, designation
labels, multiplicity ratios, replicate counts, and output sizes. The comparison gives an
informative tradeoff rather than a blanket superiority claim: the weighted profile HMM more
directly transfers a selected position-specific marginal, whereas stochastic attention gives
lower ESM2 pseudo-perplexity in the matched Kunitz comparison.

The manuscript fits *JCIM* as a new molecular-modeling and chemical-informatics methodology
with explicit theory, reproducible computational benchmarks, and an honest account of its
domain of validity. It is not submitted as a docking or experimental binding study. The
low-confidence Cav2.2 complex predictions are retained in Supporting Information only as a
negative diagnostic and are not used as evidence of binding.

The manuscript and Supporting Information provide a Data and Software Availability statement.
All curated inputs, canonical per-replicate outputs, benchmark emissions, scoring results, and
reproduction scripts are provided in the public study repository. The author declares no
competing financial interest.

A preprint of this manuscript is available on arXiv at
https://arxiv.org/abs/2603.20115 (https://doi.org/10.48550/arXiv.2603.20115). This manuscript
is not under consideration elsewhere.

Thank you for your consideration.

Sincerely,

Jeffrey D. Varner  
Robert Frederick Smith School of Chemical and Biomolecular Engineering  
Cornell University  
jdv27@cornell.edu
