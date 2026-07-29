# Omega-conotoxin data provenance

The canonical family alignment is `omega_conotoxin_full_family_aligned.fasta`
(`K = 74`, aligned length `L = 26`). The conditioning split is defined by the
23 accessions in `strong_cav22_binders_aligned.fasta`; the other 51 sequences
are the background set. “Background” means not included in that designated
list, not experimentally demonstrated nonbinding.

## Accession and marker audit

Run from `code/`:

```bash
julia experiments/generate_conotoxin_label_audit.jl
```

This regenerates:

- `conotoxin_label_audit.csv`, one row per aligned accession.
- `conotoxin_label_agreement_summary.csv`, the contingency table comparing
  accession designation with Tyr at alignment column 13.

The two labels agree for 64 of 74 sequences (86.5%): 19 designated/Tyr-positive,
4 designated/Tyr-negative, 6 background/Tyr-positive, and 45
background/Tyr-negative.

Evidence classes in the row-level audit are repository-local and deterministic:

- `activity_recorded_designated`: exact accession occurs in `binding_data.csv`
  and in the 23-accession designated list.
- `activity_recorded_background`: exact accession occurs in `binding_data.csv`
  but not in the designated list.
- `designation_only`: accession is designated but has no exact-accession record
  in `binding_data.csv`.
- `unannotated_background`: neither condition holds.

These classes record what this repository can support. They do not infer
activity from sequence similarity or treat missing records as evidence of
nonbinding.

## Docking output

The canonical docking table and the status of the two archived reruns are
documented in `docking_validation/README.md`.
