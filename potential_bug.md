# Potential Bug Analysis: Identical 0.29 Scores Issue

## Problem Statement
All SA_strong sequences consistently produce identical pTM scores of 0.29 in AlphaFold2 multimer docking validation. This issue has persisted through multiple fix attempts.

## Primary Suspects

### 1. Score Extraction Logic (`extract_scores.py` and `extract_scores_simple.py`)

**Location:** Lines 34-51 in both files

**Issue:** The JSON file selection always picks the first file alphabetically:
```python
# Line 34: This glob pattern and [0] selection
json_files = list(run_dir.glob("*_scores_rank_001_*.json"))
if not json_files:
    print(f"Warning: No scores JSON found for {run_name}")
    return None

try:
    with open(json_files[0], 'r') as f:  # Always takes first file alphabetically
        scores = json.load(f)
```

**Evidence:**
- Glob finds two files: `Cav2.2_pore_*_scores_*.json` and `{conotoxin}_scores_*.json`
- Always picks Cav2.2 file first (alphabetical order)
- Cav2.2 scores are consistent (0.29) across runs because it's the same target protein

### 2. ColabFold Configuration

**Location:** `config.json` in result directories

**Suspicious settings:**
```json
{
    "num_queries": 2,
    "pair_mode": "unpaired_paired"
}
```

**Issue:** May be treating sequences as separate monomers rather than complex prediction

## Secondary Suspects (Missing Code)

### 3. ColabFold Submission Script
**Status:** Not found in current directory structure
**Potential issues:**
- Wrong ColabFold parameters/flags
- Incorrect sequence formatting
- Wrong submission mode (monomer vs multimer)

### 4. Sequence Generation/Processing Pipeline
**Status:** Not found in current directory structure
**Potential issues:**
- Sequence preprocessing that normalizes scores
- FASTA formatting problems
- HMM tool processing errors (mentioned by user)

### 5. Score Post-processing
**Status:** Unknown if exists
**Potential issues:**
- Code that modifies scores after ColabFold
- Rounding/truncation logic
- Default value assignment

## Observations

1. **Current extraction behavior:**
   - SA_strong: All get Cav2.2 pTM = 0.29 (identical)
   - SA_full: Mix of files selected, showing variation
   - Controls: Slight variation (0.31-0.38) in Cav2.2 scores

2. **ColabFold logs show separate predictions:**
   - Query 1/2: Conotoxin (pTM varies by sequence)
   - Query 2/2: Cav2.2 pore (pTM ~0.29 consistently)

3. **No iptm scores found:** Suggests no interface prediction occurring

## Previous Fix Attempts
- Issue has been "fixed" twice before
- Same 0.29 pattern persists after each fix attempt
- Suggests multiple layers of problems or missed root cause

## Recommended Code Review Focus
1. **extract_scores.py/extract_scores_simple.py** - file selection logic
2. **Missing submission pipeline** - how sequences get to ColabFold
3. **Any hidden score processing** - post-ColabFold modifications
4. **Sequence generation workflow** - upstream preprocessing issues