# Code Review Issues (Reproducibility)

Date: 2026-03-20  
Scope: scripts under `code/` in this repository.

## Findings (ordered by severity)

### 1) High — Non-isolated global RNG seeding in shared include
- **File:** [code/Include.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/Include.jl#L40)
- **Issue:** `Random.seed!(1234)` is called at include time, mutating global RNG state.
- **Impact:** Downstream scripts and functions share mutable RNG state implicitly, so experiment ordering or imported modules can change sampling behavior across runs.
- **Recommendation:** Remove global seed initialization from shared include files. Move deterministic seeding into each experiment/repeat with explicit and logged seeds.

### 2) High — Zip extraction can clobber prior outputs
- **File:** [code/experiments/extract_scores.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/extract_scores.py#L46)  
  [code/experiments/validate_docking_multimer.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/validate_docking_multimer.py#L10)  
  [code/experiments/run_conotoxin_docking_validation.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_conotoxin_docking_validation.py#L262-L268)
- **Issue:** Multiple scripts extract ColabFold `*.result.zip` archives into shared directories with overwrite semantics and no run-level guard.
- **Impact:** Re-runs can overwrite previous files and/or reuse stale extraction artifacts, leading to non-idempotent outputs.
- **Recommendation:** Extract into per-run/stamped directories, check for pre-existing expected files, and validate archive contents before/after extraction.

### 3) High — Chain role is assumed instead of validated
- **File:** [code/experiments/run_conotoxin_docking_validation.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_conotoxin_docking_validation.py#L274-L277)
- **Issue:** Interface metric preparation hardcodes chain A as receptor and chain B as conotoxin.
- **Impact:** If chain order differs (different inputs/templates), computed interface features can be assigned incorrectly.
- **Recommendation:** Determine chain roles from metadata/sequence metadata and validate expected mapping before scoring.

### 4) Medium — Input path mismatch / ambiguous workflow contract
- **File:** [code/experiments/prepare_af2_input.py](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/prepare_af2_input.py#L41)
- **Issue:** Script reads `omega_conotoxin_full_family.fasta` while related generated artifacts use an `_aligned` filename in places.
- **Impact:** Different environments/checkouts may produce different input sets due to missing/ambiguous files.
- **Recommendation:** Define a single canonical input filename/config in one place and fail fast with a clear message if absent.

### 5) Medium — Global working directory mutation in orchestrator
- **File:** [code/experiments/run_omega_conotoxin_experiment.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_omega_conotoxin_experiment.jl#L29)
- **Issue:** Script changes cwd globally before running multiple includes/subprocesses.
- **Impact:** Relative-path assumptions become caller-dependent and may diverge across environments.
- **Recommendation:** Use absolute paths or localize directory changes to the smallest deterministic scope and restore cwd after each block.

### 6) Medium — Missing existence checks before artifact copying
- **File:** [code/experiments/run_omega_conotoxin_experiment.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/experiments/run_omega_conotoxin_experiment.jl#L302)
- **Issue:** Figure-copy loop assumes source directories/files exist.
- **Impact:** Silent partial/empty artifacts can be produced when upstream generation is skipped or fails.
- **Recommendation:** Verify expected directories/files and emit explicit failure when missing.

### 7) Medium — Environment bootstrap mutates package state
- **File:** [code/Include.jl](/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code/Include.jl)
- **Issue:** Missing `Manifest.toml` path triggers `Pkg.resolve()/instantiate()/update()`.
- **Impact:** Dependency resolution may differ per machine/run; updates can drift over time.
- **Recommendation:** Prefer explicit dependency snapshots and avoid `Pkg.update()` during experiment execution.

## Non-exhaustive items (not yet line-audited fully)
- `code/src/Binding.jl` was partially reviewed (file is long); full line-level audit is recommended.
- Large Julia experiment scripts with many replicate loops were not all exhaustively validated for RNG scope in this pass.

