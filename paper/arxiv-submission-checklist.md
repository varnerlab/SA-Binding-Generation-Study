# Arxiv Submission Checklist — Paper_v1.tex

## RED — Incorrect/Inconsistent Numbers

- [x] **1. β* values conflict between main text and appendix** ✅ FIXED
  - Computed data: β*=4.267 (ρ=1), 10.144 (ρ=1000)
  - Main text 4.3/10.1 was correct (1-decimal rounding)
  - Fixed appendix from 3.85/10.22 → 4.27/10.14 (2-decimal rounding)

- [x] **2. PCA dimension for ω-conotoxin** ✅ FIXED
  - Ran code: d=34 (Table 1 was correct)
  - Fixed appendix from "d ranged from 20" → "d ranged from 34"
  - Also fixed compression ratio from "26x" → "21x"

- [x] **3. "tenfold range of S" in introduction** ✅ FIXED
  - Changed to "threefold" (0.34/0.11=3.1x for 3 Pfam families)

- [x] **4. P1 K/R fraction at ρ=1 inconsistent** ✅ FIXED
  - Data: single-run=0.389, aggregated mean=0.412
  - Changed text from "0.39" → "0.41" to match aggregated/table values

- [x] **5. "39% K/R, matching the natural proportion (32%) within sampling noise"** ✅ FIXED
  - Changed to "41% K/R, close to the natural proportion (32%)"
  - Softened "matching...within sampling noise" → "approximately preserved"

- [x] **6. Methods says "20 to 30 chains" but conotoxin used 50** ✅ FIXED
  - Changed to "Twenty to fifty"

## ORANGE — Reference Problems

- [x] **7. Hallucinated authors in two bib entries** ✅ FIXED
  - satoAlaScanMVIIC2000: corrected to Kim, J. I. and Takahashi, M. and Seagar, M. J.
  - heyneBPTIDMS2021: corrected to Shifman, J. M. (removed 3 fake Bhatt authors)

- [x] **8. Misspelled author name** ✅ FIXED
  - "Candber, Sal" → "Candido, Sal" in esmfold2022 and esm2science2023

- [x] **9. HMMER3 mentioned but never cited** ✅ FIXED
  - Added \cite{hmmer3} in results.tex

- [x] **10. Duplicate bib entries** ✅ FIXED
  - Removed linESMFold2023 (duplicate of esmfold2022)

- [x] **11. Uncited bib entries** ✅ FIXED
  - Removed alamdariProteinGeneration2023, raoMSATransformer2021, linESMFold2023

## YELLOW — Minor Issues (deferred)

- [ ] **12. ~20 unused figure files in sections/figs/ and figs/**
  - Clean out before arxiv upload to keep tarball small

- [ ] **13. varnerSAProtein2026 says "In preparation"**
  - Update when companion paper has a preprint

- [ ] **14. esmfold2022 bib key says 2022 but year=2023**
  - Cosmetic but confusing
