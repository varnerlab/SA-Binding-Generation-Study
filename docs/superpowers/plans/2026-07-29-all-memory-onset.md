# All-Memory Entropy-Onset Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the alignment-order-dependent first-20-column entropy-onset rule with an order-independent all-memory rule in every paper-facing calculation, rerun the three affected experiments, and regenerate every downstream artifact.

**Architecture:** One helper function, `all_memory_onset`, becomes the single paper-facing operating-point rule, so the change is made once and tested once rather than repeated across eight scripts. Every producer script is switched to it, including the five whose selected onset does not change, so the committed code matches the submitted Methods. Only three experiments are rerun, because the preflight dependency table (in `remaining-issues-audit.md`) enumerated exactly ten operating points that move out of 94 checked.

**Tech Stack:** Julia 1.12 (`code/Include.jl` bootstrap, `Test`, `CSV`, `DataFrames`), Python 3 with matplotlib for two figure renderers, LaTeX.

## Global Constraints

- Branch is `submission-corrections-2026-07`, currently at `6fef3e2`. Do not branch again. Do not merge to `main`.
- All Julia commands run from `/Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code`. **Use absolute paths in every shell command.** The working directory persists between calls and has silently drifted three times in this project.
- **Positive control before trusting any reproduction.** Before believing a regenerated number, confirm that an unchanged committed value reproduces. If it does not, the setup is wrong; stop and fix the setup.
- **Verify the write, not the writer.** After any file mutation, check the target file's contents. A zero exit code is not evidence the right file changed.
- Full suite: `julia --project=. test/runtests.jl` from `code/`. Currently 263 passing.
- Both manuscript trees must stay in sync. `test_manuscript_consistency.jl` enforces this for the nine shared section files; figure and table files in `Paper_JCIM.tex` and `Paper_v1.tex` are NOT covered and must be edited twice by hand.
- Writing conventions: no em dashes or en dashes; use "designated" and "background", never "binder" and "non-binder", in new prose.
- The exact grids are not uniform and must not be homogenized: **50 points** for canonical and mask calculations, **60** for the exact-GMM benchmark, **80** for the entropy-curve figure.

## The ten changed operating points

Established by the preflight in `remaining-issues-audit.md` and independently reproduced by Codex. Any rerun that moves a value outside this list indicates a bug.

| Experiment | Condition | First-20 | All-memory |
|---|---|---:|---:|
| Canonical sweep | Kunitz rho=2 | 4.5789 | 3.8483 |
| Canonical sweep | SH3 rho=50 | 3.2343 | 6.4825 |
| Canonical sweep | Homeobox rho=100 | 4.5789 | 3.8483 |
| Canonical sweep | Homeobox rho=500 | 4.5789 | 3.8483 |
| Canonical sweep | Forkhead rho=2 | 5.4482 | 4.5789 |
| Canonical sweep | Forkhead rho=10 | 6.4825 | 5.4482 |
| Canonical sweep | Forkhead rho=20 | 6.4825 | 4.5789 |
| Canonical sweep | Conotoxin rho=500 | 7.7131 | 6.4825 |
| Kunitz mask | multiplicity f_target=0.5 | 4.5789 | 3.8483 |
| Exact-GMM benchmark | Kunitz rho=10 | 5.6943 | 4.9289 |

Unchanged and therefore NOT rerun: canonical hard curation (6 of 6), binder scaling (25 of 25), entropy-curve onsets (5 of 5), conotoxin (2 of 2), Kunitz example FASTAs (3 of 3). The entropy-curve **curves** still change even though their onsets do not; see Task 6.

---

### Task 1: Add the all-memory onset helper

**Files:**
- Modify: `code/src/Protein.jl` (add after `find_entropy_transition`, which ends at line 474)
- Modify: `code/test/test_transition_statistic.jl`

**Interfaces:**
- Consumes: `find_entropy_transition(X̂, r; n_betas, β_range, n_probes, seed)` returning a NamedTuple with fields `β_steepest`, `β_onset`, `K_eff`, `βs`, `Hs`.
- Produces: `all_memory_onset(X̂, r=ones(size(X̂,2)); n_betas=50) -> Float64`. Every later task calls this and nothing else.

- [ ] **Step 1: Write the failing tests**

Append to `code/test/test_transition_statistic.jl`:

```julia
@testset "all_memory_onset is the paper-facing operating-point rule" begin
    X = unitcols(10, 24, 91)
    r = abs.(randn(MersenneTwister(3), 24)) .+ 0.2

    # It is exactly find_entropy_transition over every stored memory.
    @test all_memory_onset(X, r; n_betas=40) ==
          find_entropy_transition(X, r; n_betas=40, n_probes=24).β_onset

    # Unweighted default: r defaults to uniform weights.
    @test all_memory_onset(X; n_betas=40) ==
          find_entropy_transition(X; n_betas=40, n_probes=24).β_onset

    # Order independence, which is the entire point. The first-20-column rule
    # fails this; that is the defect being corrected.
    perm = randperm(MersenneTwister(12), 24)
    @test all_memory_onset(X, r; n_betas=40) ==
          all_memory_onset(X[:, perm], r[perm]; n_betas=40)
end
```

Also extend the existing permutation testset at line 21 to assert `β_onset` directly, not only `β_steepest`. Add this line inside that testset after the `β_steepest` assertion:

```julia
    @test isapprox(a.β_onset, b.β_onset; rtol=1e-8)
```

- [ ] **Step 2: Run and verify it FAILS**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code && julia --project=. test/test_transition_statistic.jl
```

Expected: FAIL with `UndefVarError: all_memory_onset not defined`.

- [ ] **Step 3: Implement the helper**

In `code/src/Protein.jl`, immediately after the closing `end` of `find_entropy_transition`:

```julia
"""
    all_memory_onset(X̂, r=ones(size(X̂,2)); n_betas=50) -> Float64

Entropy-crossover onset evaluated over ALL stored memories. This is the paper-facing
operating-point rule. Unlike the earlier `find_entropy_inflection` and
`find_weighted_entropy_inflection`, which probe the first 20 alignment columns, the
result does not depend on alignment row order.

Pass the grid the calling experiment actually uses: 50 points for the canonical and mask
calculations, 60 for the exact-GMM benchmark, 80 for the entropy-curve figure.
"""
all_memory_onset(X̂::Matrix{Float64}, r::Vector{Float64}=ones(size(X̂, 2));
                 n_betas::Int=50) =
    find_entropy_transition(X̂, r; n_betas=n_betas, n_probes=size(X̂, 2)).β_onset
```

Check whether `code/src/Protein.jl` has an explicit export list. If it does, add `all_memory_onset` to it. If symbols are shared by plain `include`, no export is needed.

- [ ] **Step 4: Run and verify it PASSES**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code && julia --project=. test/test_transition_statistic.jl
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/src/Protein.jl code/test/test_transition_statistic.jl
git commit -m "feat: add order-independent all_memory_onset operating-point rule"
```

---

### Task 2: Switch every paper-facing producer script

Codex requires that all paper-facing scripts call the new rule even where the onset is unchanged, so the committed code matches the submitted Methods and a reader can reproduce directly.

**Files:**
- Modify: `code/experiments/run_canonical_family_sweeps.jl:77,104`
- Modify: `code/experiments/run_kunitz_mask_experiment.jl:68,134,144,153`
- Modify: `code/experiments/run_gmm_baseline.jl:31`
- Modify: `code/experiments/dump_entropy_curves.jl:38`
- Modify: `code/experiments/run_kunitz_binding_experiment_with_replicates.jl:112,117,125`
- Modify: `code/experiments/run_kunitz_binding_experiment.jl:125,130,138`
- Modify: `code/experiments/run_omega_conotoxin_experiment.jl:118,119`
- Modify: `code/experiments/run_augmented_memory_deepdive.jl:57,146,225,248,287,332`
- Create: `code/test/test_operating_point_rule.jl`
- Modify: `code/test/runtests.jl`

**Interfaces:**
- Consumes: `all_memory_onset` from Task 1.
- Produces: no new symbols. Later tasks rely on these scripts emitting all-memory operating points.

- [ ] **Step 1: Write the failing driver-selection test**

The real defect was never in the statistic, it was in which statistic the driver selected. This test pins that.

Create `code/test/test_operating_point_rule.jl`:

```julia
using Test

# Paper-facing producer scripts must select the operating point with the
# order-independent all-memory rule. The first-20-column detectors remain in the
# source tree for the legacy scripts, so this is a per-script assertion.
const PAPER_FACING_SCRIPTS = [
    "run_canonical_family_sweeps.jl",
    "run_kunitz_mask_experiment.jl",
    "run_gmm_baseline.jl",
    "dump_entropy_curves.jl",
    "run_kunitz_binding_experiment_with_replicates.jl",
    "run_kunitz_binding_experiment.jl",
    "run_omega_conotoxin_experiment.jl",
    "run_augmented_memory_deepdive.jl",
]

@testset "paper-facing scripts use the all-memory operating point" begin
    exp_dir = normpath(joinpath(@__DIR__, "..", "experiments"))
    for name in PAPER_FACING_SCRIPTS
        path = joinpath(exp_dir, name)
        @test isfile(path)
        source = read(path, String)
        # strip comment lines so a mention in a comment does not fail the test
        code = join(filter(l -> !startswith(strip(l), "#"), split(source, "\n")), "\n")
        @test !occursin("find_entropy_inflection(", code)
        @test !occursin("find_weighted_entropy_inflection(", code)
        @test occursin("all_memory_onset(", code)
    end
end
```

- [ ] **Step 2: Run and verify it FAILS**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code && julia --project=. test/test_operating_point_rule.jl
```

Expected: FAIL for all eight scripts, which still call the old detectors.

- [ ] **Step 3: Rewrite each call site**

The old detectors return a NamedTuple and callers take `.β_star`; the new helper returns a `Float64` directly. Some call sites bind the whole NamedTuple (`pt_full = find_entropy_inflection(X̂_full)`) and later read `pt_full.β_star`. For those, replace the binding with the scalar and update the reader.

Substitutions, preserving each script's own grid:

| Old | New |
|---|---|
| `find_entropy_inflection(X)` then `.β_star` | `all_memory_onset(X)` |
| `find_weighted_entropy_inflection(X, r; n_betas=50).β_star` | `all_memory_onset(X, r; n_betas=50)` |
| `find_weighted_entropy_inflection(X, r; n_betas=60).β_star` | `all_memory_onset(X, r; n_betas=60)` |
| `find_weighted_entropy_inflection(X, r; n_betas=80)` then `.β_star` | `all_memory_onset(X, r; n_betas=80)` |

`dump_entropy_curves.jl:38` is special: it uses `pt.βs` and `pt.Hs` to write the curve, not only `pt.β_star`. Replace the whole call with `find_entropy_transition`, which returns all three:

```julia
        pt = find_entropy_transition(X̂, r; n_betas=n_betas, n_probes=K_total)
```

and change the two readers from `pt.β_star` to `pt.β_onset`. Leave `pt.βs` and `pt.Hs` as they are.

Verify no call site was missed:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -n "find_entropy_inflection(\|find_weighted_entropy_inflection(" \
  code/experiments/run_canonical_family_sweeps.jl \
  code/experiments/run_kunitz_mask_experiment.jl \
  code/experiments/run_gmm_baseline.jl \
  code/experiments/dump_entropy_curves.jl \
  code/experiments/run_kunitz_binding_experiment_with_replicates.jl \
  code/experiments/run_kunitz_binding_experiment.jl \
  code/experiments/run_omega_conotoxin_experiment.jl \
  code/experiments/run_augmented_memory_deepdive.jl
```

Expected: no output.

- [ ] **Step 4: Run and verify it PASSES**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code
julia --project=. test/test_operating_point_rule.jl
```

Expected: PASS.

- [ ] **Step 5: Register the test and run the suite**

Add to `code/test/runtests.jl` after `include("test_transition_statistic.jl")`:

```julia
    include("test_operating_point_rule.jl")
```

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code && julia --project=. test/runtests.jl 2>&1 | tail -4
```

Expected: all pass. Data has not changed yet, so `test_generated_paper_tables.jl` must still pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/experiments code/test/test_operating_point_rule.jl code/test/runtests.jl
git commit -m "refactor: select the operating point with the all-memory rule in paper-facing scripts"
```

---

### Task 3: Rerun the canonical sweep

**Files:**
- Regenerated: `code/data/<slug>/multiplicity_sweep_aggregated.csv`, `multiplicity_sweep_raw_replicates.csv`, `canonical_sweep_execution.csv`, `canonical_family_metadata.csv` for all six slugs
- Regenerated: `code/data/multi_family_comparison_6fam_aggregated.csv`, `code/data/canonical_sweep_provenance.csv`

**Interfaces:**
- Consumes: the switched driver from Task 2.
- Produces: canonical CSVs consumed by `generate_paper_tables.jl` in Task 7.

- [ ] **Step 1: Snapshot the current CSVs for the bit-for-bit check**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
mkdir -p /tmp/canonical_before
for s in kunitz sh3 ww homeobox forkhead omega_conotoxin; do
  cp code/data/$s/multiplicity_sweep_raw_replicates.csv /tmp/canonical_before/$s.csv
done
```

- [ ] **Step 2: Run the sweep**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code
julia --project=. experiments/run_canonical_family_sweeps.jl 2>&1 | tail -20
```

This is the long step. Expect roughly an hour.

- [ ] **Step 3: Verify EXACTLY the eight predicted cells changed**

This is the correctness gate. The 40 unchanged conditions have identical beta and fixed seeds, so they must reproduce bit-for-bit.

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
python3 - <<'PY'
import csv
expected = {("kunitz",2.0),("sh3",50.0),("homeobox",100.0),("homeobox",500.0),
            ("forkhead",2.0),("forkhead",10.0),("forkhead",20.0),
            ("omega_conotoxin",500.0)}
changed=set()
for s in ["kunitz","sh3","ww","homeobox","forkhead","omega_conotoxin"]:
    old={(float(r["rho"]),int(r["replicate"])):r["f_obs"]
         for r in csv.DictReader(open(f"/tmp/canonical_before/{s}.csv"))}
    for r in csv.DictReader(open(f"code/data/{s}/multiplicity_sweep_raw_replicates.csv")):
        k=(float(r["rho"]),int(r["replicate"]))
        if old.get(k) != r["f_obs"]:
            changed.add((s,k[0]))
print("changed:", sorted(changed))
print("unexpected  :", sorted(changed-expected))
print("missing     :", sorted(expected-changed))
assert changed==expected, "rerun moved cells outside the predicted set"
print("OK: exactly the eight predicted conditions changed")
PY
```

Expected: `OK`. If `unexpected` is non-empty, stop and investigate before continuing; something other than the onset rule changed.

- [ ] **Step 4: Record the new cross-family gaps**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
python3 -c "
import csv
for r in csv.DictReader(open('code/data/multi_family_comparison_6fam_aggregated.csv')):
    print(f\"{r['family']:<12}{float(r['separation_index']):>8.4f}{float(r['cal_gap_mean']):>9.4f}\")
"
```

Predicted from the preflight: Homeobox gap moves from 0.0513 toward 0.0689 and conotoxin from 0.1343 toward 0.1640; Kunitz, SH3, WW and Forkhead gaps are unchanged because they are measured at rho=500, which only moved for Homeobox and conotoxin. Treat a departure from that pattern as a signal to investigate.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/data
git commit -m "data: rerun canonical sweep with the all-memory operating point"
```

---

### Task 4: Rerun the Kunitz mask experiment

**Files:**
- Regenerated: `code/data/kunitz/mask_residuals.csv`, `mask_betasweep.csv`, and the results CSV written at `run_kunitz_mask_experiment.jl:160`
- Regenerated: `code/figs/kunitz/fig_mask_betasweep.pdf`

- [ ] **Step 1: Snapshot and rerun**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
cp code/data/kunitz/mask_residuals.csv /tmp/mask_residuals_before.csv
cd code && julia --project=. experiments/run_kunitz_mask_experiment.jl 2>&1 | tail -15
```

- [ ] **Step 2: Verify only the f_target=0.5 row moved**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
diff <(cut -d, -f1,3 /tmp/mask_residuals_before.csv) \
     <(cut -d, -f1,3 code/data/kunitz/mask_residuals.csv)
```

Expected: only the `0.5` row differs, with `beta_w` moving from 4.5789 to 3.8483. Rows 0.7, 0.9, 0.95 and 0.99 must be identical.

- [ ] **Step 3: Record the new residuals for the prose in Task 8**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
column -s, -t code/data/kunitz/mask_residuals.csv
```

Write down the new `residual` and `residual_se` at `f_eff=0.5`. The current Results text quotes 0.111 (SE 0.027) at `f_eff = 0.5`, and that number will change. Task 8 updates it.

- [ ] **Step 4: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/data/kunitz code/figs/kunitz
git commit -m "data: rerun Kunitz mask experiment with the all-memory operating point"
```

---

### Task 5: Rerun the exact-GMM benchmark

**Files:**
- Regenerated: `code/data/kunitz/gmm_baseline_comparison.csv`

- [ ] **Step 1: Snapshot and rerun**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
cp code/data/kunitz/gmm_baseline_comparison.csv /tmp/gmm_before.csv
cd code && julia --project=. experiments/run_gmm_baseline.jl 2>&1 | tail -10
```

- [ ] **Step 2: Verify only rho=10 moved**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
diff /tmp/gmm_before.csv code/data/kunitz/gmm_baseline_comparison.csv
```

Expected: only the `rho=10` row differs, with `beta_star` moving from 5.6943 to 4.9289. The `rho=1` and `rho=500` rows must be identical.

- [ ] **Step 3: Check the assertion in the script still holds**

`run_gmm_baseline.jl:49` asserts `all(rows.aa_kl .< 0.01)`. If the rerun trips it, the script exits non-zero. Confirm it did not:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
python3 -c "
import csv
for r in csv.DictReader(open('code/data/kunitz/gmm_baseline_comparison.csv')):
    print(r['rho'], r['beta_star'], r['p1_exact'], r['p1_ula'], r['aa_kl'])
"
```

Record `p1_exact`, `p1_ula` and `aa_kl` at each rho. Results quotes 0.604 exact, 0.593 ULA, and 1.15e-4 for the KL at rho=500. Since rho=500 is unchanged, those three should be identical. Confirm that, because it is a free positive control on the whole rerun.

- [ ] **Step 4: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/data/kunitz/gmm_baseline_comparison.csv
git commit -m "data: rerun exact-GMM benchmark with the all-memory operating point"
```

---

### Task 6: Regenerate the entropy curves and fix the broken renderer

The five plotted onsets do not change, but the plotted **curves** do, because the curve is a mean over the probe set. Codex measured a maximum absolute difference in H rising from 0.043 at rho=1 to 0.692 at rho=1000; this was independently reproduced.

`code/experiments/render_entropy_curves.py` is also currently broken. Its repo-root loop requires a `paper/sections` directory, which no longer exists, so `REPO_ROOT` is never assigned and the script raises `NameError`.

**Files:**
- Regenerated: `code/data/kunitz/entropy_curves.csv`
- Modify: `code/experiments/render_entropy_curves.py:15-21`
- Regenerated: `paper-jcim/sections/figs/fig5_entropy_curves.pdf` and `paper-arxiv/sections/figs/fig5_entropy_curves.pdf`

- [ ] **Step 1: Regenerate the CSV**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
cp code/data/kunitz/entropy_curves.csv /tmp/entropy_curves_before.csv
cd code && julia --project=. experiments/dump_entropy_curves.jl 2>&1 | tail -8
```

- [ ] **Step 2: Verify onsets held and curves moved**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
python3 - <<'PY'
import csv
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        d.setdefault(r["rho"],{"bs":r["beta_star"],"H":[]})["H"].append(float(r["H"]))
    return d
a,b=load("/tmp/entropy_curves_before.csv"),load("code/data/kunitz/entropy_curves.csv")
for rho in a:
    dH=max(abs(x-y) for x,y in zip(a[rho]["H"],b[rho]["H"]))
    print(f"rho={rho:8} onset {a[rho]['bs'][:6]} -> {b[rho]['bs'][:6]}  max|dH|={dH:.4f}")
PY
```

Expected: every onset identical (4.3530, 4.8485, 6.0152, 7.4627, 9.2585), and max |dH| approximately 0.043, 0.288, 0.458, 0.586, 0.692 in rho order. This is a positive control on both the onset claim and the curve claim at once.

- [ ] **Step 3: Fix the renderer's paths**

In `code/experiments/render_entropy_curves.py`, replace the repo-root discovery block and the `OUT` assignment:

```python
for parent in SCRIPT_PATH.parents:
    if (parent / "code" / "data").is_dir() and (parent / "paper-arxiv" / "sections").is_dir():
        REPO_ROOT = parent
        break

DATA = REPO_ROOT / "code" / "data" / "kunitz" / "entropy_curves.csv"
OUTS = [REPO_ROOT / "paper-jcim" / "sections" / "figs" / "fig5_entropy_curves.pdf",
        REPO_ROOT / "paper-arxiv" / "sections" / "figs" / "fig5_entropy_curves.pdf"]
```

Then replace the single `savefig` call near the end of the file with a loop over `OUTS`, writing both the `.pdf` and the `.png` for each, matching the pattern already used in `render_separation_gap_figure_5fam.py:187-189`.

- [ ] **Step 4: Render and verify both trees got the file**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
python3 code/experiments/render_entropy_curves.py
ls -la paper-jcim/sections/figs/fig5_entropy_curves.pdf paper-arxiv/sections/figs/fig5_entropy_curves.pdf
diff paper-jcim/sections/figs/fig5_entropy_curves.pdf paper-arxiv/sections/figs/fig5_entropy_curves.pdf && echo "both trees identical"
```

Expected: both files exist with a current timestamp and identical contents.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/data/kunitz/entropy_curves.csv code/experiments/render_entropy_curves.py \
        paper-jcim/sections/figs paper-arxiv/sections/figs
git commit -m "fix: regenerate all-memory entropy curves and repair the figure renderer paths"
```

---

### Task 7: Regenerate tables, macros and the separation figure

**Files:**
- Regenerated: `paper-jcim/sections/generated/` and `paper-arxiv/sections/generated/` (all ten files)
- Regenerated: `paper-jcim/sections/figs/fig2_separation_vs_gap.pdf` and the arXiv copy

- [ ] **Step 1: Regenerate the generated tables and macros into both trees**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code
julia --project=. experiments/generate_paper_tables.jl --output=../paper-arxiv/sections/generated
julia --project=. experiments/generate_paper_tables.jl --output=../paper-jcim/sections/generated
```

- [ ] **Step 2: Record the new exploratory-fit macros**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -E "Relation" paper-jcim/sections/generated/numbers.tex
```

Predicted from the preflight: `RelationSlope` moves from -2.0 toward -1.9 and `RelationRsq` from 0.80 toward 0.79. The leave-one-family-out macros will also shift. These are `\input` macros, so the prose picks them up automatically; no prose edit is needed for them.

- [ ] **Step 3: Confirm the SAR table is byte-identical**

The conotoxin operating points did not change, so the SAR table corrected earlier on this branch must be unaffected. This is a positive control.

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git diff --stat paper-jcim/sections/generated/tab_sar_agreement.tex
```

Expected: no output, meaning no change. If it changed, stop; something regenerated conotoxin sampling that should not have.

- [ ] **Step 4: Regenerate the separation-versus-gap figure**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
python3 code/experiments/render_separation_gap_figure_5fam.py
```

That renderer writes only into `paper-arxiv`. Copy to the JCIM tree and confirm:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
cp paper-arxiv/sections/figs/fig2_separation_vs_gap.pdf paper-jcim/sections/figs/
cp paper-arxiv/sections/figs/fig2_separation_vs_gap.png paper-jcim/sections/figs/ 2>/dev/null || true
diff -rq paper-jcim/sections/figs paper-arxiv/sections/figs && echo "figure trees identical"
```

- [ ] **Step 5: Run the anti-drift test and commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code
julia --project=. test/test_generated_paper_tables.jl 2>&1 | tail -3
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv
git commit -m "data: regenerate manuscript tables, macros and figures after the onset correction"
```

---

### Task 8: Update the manuscripts

**Files:**
- Modify: `paper-jcim/sections/methods.tex` and `paper-arxiv/sections/methods.tex`
- Modify: `paper-jcim/sections/results.tex` and `paper-arxiv/sections/results.tex`
- Modify: `paper-jcim/Paper_JCIM.tex` and `paper-arxiv/Paper_v1.tex` if any caption quotes a changed number

- [ ] **Step 1: Replace the Methods operating-point paragraph in both trees**

The current paragraph, added on this branch, describes the first-20 rule and discloses its order dependence. That disclosure is now obsolete. Replace the sentences beginning "The operating inverse temperature $\beta^{*}(\vr)$ was selected as the entropy-crossover onset" through the end of the sensitivity sentence ending "left Kunitz, SH3, WW and Forkhead unchanged." with:

```latex
The operating inverse temperature $\beta^{*}(\vr)$ was selected as the entropy-crossover
onset, the point of maximum downward curvature of the weighted attention entropy with
respect to $\log \beta$ on a fixed logarithmic grid spanning $\beta \in [0.1,\, 500]$,
with the entropy averaged over all stored memories so that the selected value does not
depend on alignment order. The grid used 50 points for the canonical family sweeps and
the hard-mask comparison, 60 points for the exact-equilibrium benchmark, and 80 points
for the entropy curves in Fig.~\ref{fig:phase-transition}. This is a descriptive
operating point rather than an estimate of a thermodynamic phase transition, and entropy
is evaluated at stored memory vectors rather than at samples.
```

- [ ] **Step 2: Update the hard-mask numbers in Results in both trees**

The mask residual at `f_eff = 0.5` changed in Task 4. The current text reads "A difference of 0.5 in intended designated mass produced a difference of 0.111 (SE 0.027) in observed marker fraction" and, earlier in the same paragraph, "by 0.111 (SE 0.027) at $f_{\mathrm{eff}} = 0.5$". Replace both with the regenerated values from `code/data/kunitz/mask_residuals.csv`. Do not retype from memory; read the CSV:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
column -s, -t code/data/kunitz/mask_residuals.csv
```

The 0.075 at `f_eff = 0.7` and 0.002 at `f_eff = 0.99` should be unchanged, since those rows did not move. Confirm before leaving them.

- [ ] **Step 3: Sweep every remaining hand-typed number against its source**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -rn "0\.111\|0\.075\|0\.002\|4\.35\|9\.26\|4\.4 \|9\.3 \|0\.604\|0\.593\|1\.15" \
  paper-jcim/sections/*.tex paper-jcim/Paper_JCIM.tex
```

For each hit, confirm the value against its regenerated CSV. Expected to be unchanged: 4.35 and 9.26 and the rounded 4.4 and 9.3 (entropy-curve onsets held), and 0.604, 0.593 and 1.15e-4 (GMM rho=500 held). Expected to change: the mask residual at `f_eff=0.5`.

Also run the repository's own magic-number audit:

```
/audit-magic-numbers results
```

- [ ] **Step 4: Verify parity and build both**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code
julia --project=. test/test_manuscript_consistency.jl 2>&1 | tail -4
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/paper-jcim && make 2>&1 | tail -2
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/paper-arxiv && make 2>&1 | tail -2
```

Expected: parity passes, both builds exit 0.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv
git commit -m "docs: describe the all-memory operating point and update the affected numbers"
```

---

### Task 9: Final verification

- [ ] **Step 1: Full suite**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/code && julia --project=. test/runtests.jl 2>&1 | tail -6
```

Expected: zero failures, count above 263 by the tests added in Tasks 1 and 2.

- [ ] **Step 2: Clean rebuild of all three documents**

`make` in `paper-jcim` builds the main paper and then the supporting information. The
appendix and the GMM SI table live in `Paper_JCIM_SI.tex`, so the SI log must be checked
too: it can report a clean build from an earlier run while its PDF holds stale numbers.

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/paper-jcim && make 2>&1 | tail -2
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/paper-arxiv && make 2>&1 | tail -2
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
for f in paper-jcim/Paper_JCIM.log paper-jcim/Paper_JCIM_SI.log paper-arxiv/Paper_v1.log; do
  printf "%-30s undefined:%s overfull:%s\n" "$f" "$(grep -ci 'undefined' $f)" "$(grep -c 'Overfull' $f)"
done
```

Expected: all zero.

- [ ] **Step 3: Confirm the data diff matches the approved scope**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git diff --stat main -- code/data
```

Expected changed files: the six families' canonical CSVs, `multi_family_comparison_6fam_aggregated.csv`, `canonical_sweep_provenance.csv`, `kunitz/mask_residuals.csv`, `kunitz/mask_betasweep.csv`, `kunitz/gmm_baseline_comparison.csv`, `kunitz/entropy_curves.csv`, and `omega_conotoxin/sar_agreement.csv` from the earlier work. Anything else, especially structure, docking or ESM2 outputs, means something ran that should not have.

- [ ] **Step 4: Visually inspect the three regenerated figures**

Open `fig5_entropy_curves.pdf`, `fig2_separation_vs_gap.pdf` and `fig_mask_betasweep.pdf` in both trees and confirm no clipping, no missing series, and readable axes.

- [ ] **Step 5: Record the outcome in the audit document**

Append a section to `remaining-issues-audit.md` recording, for each of the ten predicted operating points, the observed before and after values, and confirming that no cell outside the predicted set moved.

Use an absolute path for the append and verify the write landed in the repository root, not in `code/`:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
wc -l /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/remaining-issues-audit.md
find /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study -name "remaining-issues-audit.md" -not -path "*/.git/*"
```

Expected: exactly one path.

- [ ] **Step 6: Report, do not merge**

Summarize what changed, present the diff, and stop. The author's prose and tone pass is still outstanding and merging is the author's decision.

---

## Deliberately out of scope

- AlphaFold, ESMFold, ESM2, docking and HMM baseline reruns. The preflight showed the three Kunitz example FASTA bases and both conotoxin bases are unchanged, so their inputs are identical.
- Binder-scaling sampling. All 25 bases are unchanged; the script is switched to the new rule in Task 2 but not rerun.
- Conotoxin sampling. Both operating points are unchanged, so the SAR table, Table 5, the loop heatmap and the sequence-analysis figures stand.
- The prose and tone pass, which is the author's.
- Refactoring legacy scripts that produce no reported artifact.
