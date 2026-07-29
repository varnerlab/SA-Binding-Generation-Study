# Canonical Cav2.2 docking dataset

`docking_results.csv` is the canonical source for the manuscript docking table.
It contains 25 analysis rows:

- 10 SA designated-subset sequences
- 10 SA full-family sequences
- 5 natural controls

The raw directory also contains two pipeline checks, `test_mviia` and
`test_single`. They are valid multimer outputs but are intentionally excluded
from `docking_results.csv`.

The canonical CSV was established in commit `a7defc0` on 2026-03-20. Its
SHA-256 checksum is:

```text
e5968517a0618e190dcf20e7f1b4217cbbb84d8a36a384392c501488336040c5
```

Validation
----------

From the repository root:

```bash
python3 code/experiments/validate_docking_multimer.py \
  --docking-dir code/data/omega_conotoxin/docking_validation
```

The validation performed on 2026-07-28 found 27 of 27 raw runs to be valid
two-chain AlphaFold2-multimer outputs with pTM and iPTM values. The 25 analysis
rows reproduce the manuscript group sizes and summary statistics.

Excluded archives
-----------------

- `docking_validation_full_rerun_20260319_184154` is an incomplete rerun, not
  a competing dataset. Its CSV contains only two rows, most run directories
  lack prediction outputs, and their logs record failures to resolve the
  ColabFold MSA service.
- `docking_validation_stale_20260319_172834` is the explicitly archived stale
  run. Its 34-row CSV has missing iPTM values and is unsuitable for the
  multimer comparison reported in the manuscript.

Do not replace the canonical CSV with either archive. A future replacement
must include the complete 10/10/5 analysis set, retain raw two-chain outputs,
pass `validate_docking_multimer.py`, and record a new checksum here.
