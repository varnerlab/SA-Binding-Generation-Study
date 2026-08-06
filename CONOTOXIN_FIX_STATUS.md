# ω-Conotoxin alignment-frame bug fix — handoff status

Scratch status file, not meant to be committed. Delete once this work is finished and
merged into the paper. If resuming in a fresh Claude Code session, paste this whole file
as context and say "pick up from here."

## The original bug (from section-review.md, Codex review)

`code/experiments/run_omega_conotoxin_experiment.jl` built its "designated" (strong-binder)
memory matrix by independently MAFFT-realigning `strong_cav22_binders_aligned.fasta` on its
own, instead of slicing the designated rows out of the canonical 74-sequence alignment used
everywhere else (SAR table, marker registry, multiplicity sweep). Verified directly: 7 of 23
designated sequences disagreed between the two framings (Cys15/17 and K/R21/22/26 columns
shifted by 1-2 positions), because MAFFT places gaps differently depending on which other
sequences are present during alignment. Column 13 (Tyr13 marker) was never affected.

## What's DONE and verified (do not redo)

1. **Core fix**: `run_omega_conotoxin_experiment.jl` now builds `char_strong` as
   `char_full[group_A, :]` via `canonical_load_alignment`/`canonical_split` from
   `canonical_family_registry.jl`, instead of independently re-aligning. Verified correct by
   Codex (independently reconstructed the generator in-memory, zero mismatches across all
   3,100 generated sequences vs. what's on disk).
2. **Regression test**: `code/test/test_conotoxin_generator_frame.jl` added, passes, registered
   in `runtests.jl`.
3. **Regenerated data** (all verified, already on disk):
   - `code/data/omega_conotoxin/generated_strong_seeded.fasta` and `generated_full_seeded.fasta`
     — regenerated with the fix. (Full-seeded also changed vs. old committed version, but for
     an unrelated reason: the old committed file predated an unrelated earlier code change and
     was already stale; confirmed via in-process determinism check that regeneration with
     current code is 100% reproducible.)
   - `code/data/omega_conotoxin/sar_agreement.csv` — regenerated.
   - `paper-arxiv/sections/generated/tab_sar_agreement.tex` (arXiv AND
     `paper-jcim/sections/generated/tab_sar_agreement.tex`, both regenerated via
     `julia --project=. experiments/generate_paper_tables.jl` — must run from `code/` with
     `--project=.` explicitly, it does not auto-activate). JCIM's `results.tex` is NOT in sync
     with arXiv's (pre-existing gap predating this session, confirmed via git log — out of
     scope per the review's own "arXiv manuscript only" note).
4. **AF2 rerun — COMPLETE.** `code/notebooks/ColabFold_AF2_Conotoxin.ipynb` was hardened
   (SHA-256 upload validation, output-completeness validation, explicit `--random-seed 0
   --num-seeds 1`) after Codex found two real blocking gaps in the first draft. User ran it on
   Colab T4: `sa_strong` and `sa_full` each 50/50 verified, `stored` reused unchanged (confirmed
   byte-identical to the historical run after accounting for ColabFold's `|`/`/` -> `_` header
   sanitization, so it did NOT need rerunning).
   - Raw CSV download from Colab failed (browser blocked the second of 3 sequential downloads);
     worked around by reconstructing the raw data directly from the downloaded PDB zip using
     `code/bin/TMalign` (same Chain_2/reference-length logic as the notebook) — reconstruction
     matched Colab's own reported summary stats to 4 decimal places, so nothing was lost.
   - `code/experiments/merge_conotoxin_af2_rerun.py` (new script, tested against a sandboxed
     copy with synthetic data before real use) merged the new AF2 results with the untouched
     `stored` rows into:
     `code/data/af2_results/omega_conotoxin/conotoxin_af2_validation_raw_corrected.csv` and
     `..._summary_corrected.csv` — **these are now the final, complete, correct AF2 numbers.**
   - Final AF2 summary (n=50 each):
     - stored: pLDDT 67.9±9.8, TM 0.3465±0.1144 (unchanged data; std recomputed as population
       SD (ddof=0) for consistency with the other two rows — was previously published as sample
       SD (ddof=1) giving 9.9/0.1156; trivial difference, disclosed here for the record)
     - sa_strong (designated subset): pLDDT 79.5±4.0, TM 0.4714±0.0203 (barely moved from old
       78.5±4.2 / 0.47±0.02 — expected, matches Codex's earlier same-seed prediction that
       Tyr13/composition barely changed)
     - sa_full: pLDDT 70.0±11.1, TM 0.3492±0.1289 (moved more than sa_strong — from old
       73.9±10.5 / 0.39±0.12 — NOT from the alignment bug, which never affected full-family
       generation; caused by (a) the full-seeded FASTA staleness fix above and (b) chain-
       representative resampling picking a different subset of the 1550 sequences)
   - **IMPORTANT NUANCE for prose (task #10 below)**: the paper claims "both predictors gave
     each generated group a mean TM-score at least as high as that of the stored group." Still
     technically true for sa_full (0.3492 > 0.3465) but the margin shrank from a comfortable
     0.04 to 0.0027 — well within one SD of either group. Word this comparison carefully for
     sa_full specifically rather than stating it as flatly as before; sa_strong's margin is
     still solid and, if anything, slightly wider than before.
5. **AF2 footnote — MUST BE REMOVED.** `paper-arxiv/Paper_v1.tex` currently has a `$^{\dagger}$`
   footnote on the AF2 designated-subset row (Table structure-model-comparison) saying AF2
   predates the fix. This is now FALSE — AF2 was rerun and is current. Remove the `$^{\dagger}$`
   markers (two places: the caption sentence, and the `SA (designated subset)$^{\dagger}$` row
   label) and the corresponding NOTE comment in `code/experiments/prepare_af2_input.py`.
6. **Fixed bugs found along the way** (all committed to working tree, verified):
   - `run_omega_conotoxin_experiment.jl`: Julia 1.12 soft-scope bug in the figure-copy loop
     (`n_copied += 1` -> `global n_copied += 1`), unrelated pre-existing bug, blocked the script
     from completing at all.
   - `render_fold_superposition_figures.py`: TM-score normalization bug (parsed `Chain_1`
     query-length-normalized instead of `Chain_2` reference-length-normalized) — fixed, and the
     hardcoded conotoxin PDB filename replaced with dynamic best-by-TM-score selection reading
     a new `pdb_path` column added to `structure_validation_raw.csv`.
   - `run_conotoxin_structure_validation.jl`: PDB cache was keyed by sequence name only, so
     regenerating sequences under the same names would have silently reused stale pre-fix
     cached structures — fixed with content-hash-based cache keys for `SA_full`/`SA_strong`
     (kept plain-name cache for `stored`, which is provably static). Also fixed non-
     representative "first 50" sampling (was landing entirely in chains 1-2 of 50) with
     `chain_representative_sample` (one state per chain, index `c*31` for `c in 1:50`).
   - `render_conotoxin_sequence_analysis.jl`: removed dead code that loaded the buggy
     independently-realigned matrix but only used it for a log line.
   - `prepare_af2_input.py`: added the same `chain_representative_sample` (Python port,
     verified byte-identical to the Julia version via SHA-256) so AF2 and ESMFold score the
     same sequences.

## What's IN PROGRESS / BLOCKED right now

**ESMFold structure validation — INCOMPLETE, blocked on a degraded/down public API.**
- `stored`: 50/50 done.
- `SA_strong`: 16/50 done (34 missing).
- `SA_full`: 11/50 done (39 missing).
- The free `api.esmatlas.com` REST API has been returning HTTP 504 (Gateway Timeout, ~29s)
  consistently for hours across this session. The most recent full retry got **zero** new
  successes out of 73 attempts — this looks like a sustained outage, not ordinary flakiness.
- Caching is content-hash-keyed and safe to retry indefinitely with zero risk of wasted work
  or reusing stale data — every retry only attempts sequences not already cached.
- **To resume**: `cd code && julia experiments/run_conotoxin_structure_validation.jl`
  (run via the harness's background-tracked Bash so a completion notification fires; do NOT
  manually background with `&` — that was tried once and loses notification tracking).
- **To check exact progress without running the full script** (fast, ~10s), this one-liner
  replicates the script's own cache-lookup logic precisely:
  ```julia
  cd code && julia -e '
  include("Include.jl")
  DATA_DIR = joinpath(pwd(), "data", "omega_conotoxin")
  STRUCT_DIR = joinpath(DATA_DIR, "structures")
  function chain_representative_sample(seqs, n_chains)
      stride, remainder = divrem(length(seqs), n_chains)
      remainder == 0 || error("bad")
      return [seqs[c * stride] for c in 1:n_chains]
  end
  for (label, filename) in [("SA_full", "generated_full_seeded.fasta"), ("SA_strong", "generated_strong_seeded.fasta")]
      raw = parse_fasta(joinpath(DATA_DIR, filename))
      cleaned = [(n, replace(s, r"[.\-~]" => "")) for (n, s) in raw]
      selected = chain_representative_sample(cleaned, 50)
      n_cached = count(((name, seq),) -> begin
          safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
          content_id = string(hash(seq); base=16)
          pdb_path = joinpath(STRUCT_DIR, "$(label)_$(safe_name)_$(content_id).pdb")
          isfile(pdb_path) && filesize(pdb_path) > 100
      end, selected)
      println("$label: $n_cached / 50")
  end'
  ```
- **Decision point deferred**: if the outage persists much longer, the user was about to be
  asked whether to (a) keep waiting/retrying [recommended, free], (b) try a local ESMFold
  install (~15GB download, untested feasibility), or (c) ship with partial/disclosed n=16,n=11
  samples. No decision was made yet — user asked to save state and pause instead.

## What's NOT STARTED (blocked on ESMFold completion above)

- **Task #7**: rerun `code/experiments/render_conotoxin_sequence_analysis.jl` (entropy
  correlation r, top-5-by-pLDDT sequence figure) — needs complete `structure_validation_raw.csv`
  for a representative top-5 selection.
- **Task #8**: rerun `code/experiments/render_fold_superposition_figures.py` (already fixed,
  just needs to be run once ESMFold data is complete) — picks the best-TM-score SA_strong
  structure dynamically now.
- **Task #10**: update hand-typed numbers in `paper-arxiv/sections/results.tex` and
  `paper-arxiv/Paper_v1.tex` (Table conotoxin-pharmacophore, Table structure-model-comparison,
  Table sar-agreement is auto-generated already) with the final regenerated values. Also
  includes: removing the AF2 footnote (see above), the sa_full TM-margin wording nuance (see
  above), and the three original wording fixes from section-review.md:
  1. "five highest-pLDDT" should read "among the 50 modeled sequences" (not implying top-5-of-
     1,550) — results.tex, near "The five highest-pLDDT designated-subset sequences..."
  2. results.tex "The top three sequences" should be singular, matching the fold-superposition
     figure's single selected sequence (Paper_v1.tex already correctly says "single highest
     TM-score" in its own caption — just results.tex's prose is stale).
  3. Paper_v1.tex caption "while full-family seeding reduced both" is backwards — full-family
     Tyr13 33.8%->46.9% and K/R 9.9%->12.0% both INCREASED; caption should say these were lower
     than under designated-subset seeding, not that full-family seeding reduced them.
- **Task #13**: `cd paper-arxiv && make` to rebuild, confirm no undefined refs/citations.

## Known pre-existing, OUT-OF-SCOPE issues found along the way (do not fix unless asked)

- `julia code/test/runtests.jl` has two pre-existing failures unrelated to this work, confirmed
  via `git stash`+rerun on clean HEAD: (1) "JCIM and arXiv section sources stay in sync" (6
  failures — JCIM tree genuinely behind arXiv, predates this session), (2) "Kunitz sequence-
  metric table matches its CSV sources" (ambiguous-match bug in `test_manuscript_consistency.jl`'s
  `table_row` helper once a second "SA (full family)" table row was added elsewhere — confirmed
  via `git diff HEAD` that the affected lines are completely untouched this session).
- `code/experiments/run_conotoxin_multiplicity_sweep.jl` is a legacy/superseded script (the
  canonical `run_canonical_family_sweeps.jl` replaced it) — not touched, not broken, just noted.

## Working tree state

Everything above is uncommitted (`git status` will show all of it as modified/untracked). User
has not asked for a commit yet — do not commit without asking.
