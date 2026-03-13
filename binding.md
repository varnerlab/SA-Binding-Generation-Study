# SA for Binder Design: Discussion Notes (2026-03-13)

## Context
Jeffrey's experimental colleagues want SA to generate sequences that are binders to specific targets, not just plausible family members. This came up as a general "we want designed binders" request (no specific target yet).

## Key Insight
Binding is relational (protein + target). SA encodes fold/family constraints but not target specificity. Two distinct problems:

**Problem A (SA already addresses):** Generate functional family members. For families where all members share a function (defensins, Kunitz inhibitors), SA generation already produces candidate binders. DMS validation (90% tolerated substitutions) is indirect evidence.

**Problem B (requires conditioning):** Generate binders to a specific target. Requires target information to enter the pipeline.

## Proposed Approaches (ranked by feasibility)

1. **Curated memory matrix (training-free, immediate):** Use only known binders as stored patterns. SA was designed for small families. 15 phage display hits is an ideal input. "Give me your 10 best binders, I'll give you 150 new ones in 30 seconds."

2. **Biased energy landscape:** Add binding-proxy term: E_total = E_Hopfield + λ E_bind, where E_bind scores agreement with interface profile. Score function stays analytic. Needs structural knowledge of interface.

3. **Post-hoc filtering:** Generate large pool with SA (cheap), score with AF-Multimer/docking. Generate-then-filter paradigm.

4. **Interface-aware PCA encoding:** Upweight interface positions in one-hot encoding before PCA. Subtle effect, needs binding site definition.

## Cleanest Follow-up Experiment
Take a family with binding data (e.g., Kunitz domains with Ki values against trypsin). Split into strong vs. weak binders. Run SA on each subset. Show generated sequences inherit binding phenotype of input set. No new method development needed.

## Discussion Added to Paper
Added a forward-looking paragraph to the discussion section planting this flag, framing SA as a "small-set amplifier" for functionally characterized subsets. Key points made:
- Binding specificity is relational; family alignment encodes fold but not target
- Memory matrix can be conditioned by curating input to known binders
- For families with shared binding function (Kunitz, defensins), SA already generates candidate binders
- Post-hoc scoring with AF-Multimer or binding energy estimators provides additional filtering
