# Conotoxin-Cav2.2 Docking Validation Pipeline

This pipeline uses AlphaFold2-multimer (via ColabFold) to validate that SA-generated conotoxin sequences are predicted to bind to the Cav2.2 calcium channel.

## Overview

**Goal:** Show that SA strong-seeded conotoxins score higher on binding predictions than SA full-seeded conotoxins, validating that multiplicity weighting produces sequences that are predicted to be functional binders.

**Method:**
- Use PDB 7MIX structure (Cav2.2 + ziconotide/MVIIA) as template
- Predict complexes for 15-20 sequences from each group:
  - SA strong-seeded (generated with strong binder conditioning)
  - SA full-seeded (generated with full family)
  - Known strong binders (positive controls)

**Metrics:**
- **ipTM score**: Interface quality (>0.8 = high confidence binding)
- **pDockQ2**: Overall complex quality
- **Interface pLDDT**: Local confidence at binding interface

## Installation

1. Install ColabFold:
```bash
./install_colabfold.sh
```

2. Test installation:
```bash
python run_docking_test.py
```

## Usage

### Full Pipeline
```bash
cd /path/to/SA-Binding-Generation-Study/code
python experiments/run_conotoxin_docking_validation.py .
```

### Analysis
```julia
cd("experiments")
include("analyze_docking_results.jl")
df, stats, summary = main()
```

## Expected Workflow

1. **Run docking predictions** (~4-6 hours for 50 sequences)
2. **Extract binding scores** from ColabFold outputs
3. **Statistical analysis** comparing groups
4. **Generate figures** for paper

## Expected Results

If the SA method works as expected:
- **SA strong-seeded**: High ipTM scores (>0.7), indicating good predicted binding
- **SA full-seeded**: Lower ipTM scores, more variable
- **Known binders**: Highest scores (validation that method works)

This would provide computational evidence that multiplicity weighting produces sequences predicted to bind Cav2.2.

## Files Generated

- `docking_validation/docking_results.csv` - Raw scores for all sequences
- `docking_validation/summary_statistics.csv` - Group-level statistics
- `docking_validation/statistical_analysis.csv` - P-values and effect sizes
- `figs/docking_validation/binding_scores_comparison.pdf` - Main figure for paper
- Individual ColabFold outputs in subdirectories

## Integration with Paper

The binding scores comparison figure can be added as a new panel or supplementary figure showing computational validation of the generated sequences' predicted binding ability.