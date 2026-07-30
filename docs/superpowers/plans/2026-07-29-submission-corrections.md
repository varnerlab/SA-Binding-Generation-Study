# Pre-Submission Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the seven agreed pre-submission corrections from `submission-audit-results.md` to both the JCIM and arXiv manuscripts, fixing one real data error and one inference error, and locking each correction behind a regression test so it cannot silently return.

**Architecture:** Two halves. The first half is code and data: the conotoxin SAR script reads its designated input from the wrong coordinate frame, so it is fixed, its output CSV is regenerated, and the resulting table is moved into the existing anti-drift generator so the manuscript number can never drift from the CSV again. The second half is manuscript text: each prose correction is applied to both `paper-jcim/sections/` and `paper-arxiv/sections/`, which are duplicated real files rather than symlinks, and a parity guard test added in Task 1 makes a one-sided edit fail the suite.

**Tech Stack:** Julia 1.12 (`code/Include.jl` bootstrap, `Test` stdlib, `CSV`, `DataFrames`), LaTeX (`paper-jcim/Makefile`, `paper-arxiv/Makefile`), git.

## Global Constraints

- All Julia commands run from `code/`. Every script starts with `include("Include.jl")`.
- Full suite command: `cd code && julia --project=. test/runtests.jl`. It currently passes at 149 tests.
- Every manuscript prose edit MUST be applied to BOTH `paper-jcim/sections/<file>.tex` and `paper-arxiv/sections/<file>.tex`. The two trees are independent copies. As of 2026-07-29 all nine shared section files are byte-identical except `theory.tex`, which differs only by an arXiv-only `\begin{samepage}` at line 42 and `\end{samepage}` at line 60.
- Writing conventions (from `CLAUDE.md`, non-negotiable): no em dashes or en dashes anywhere in prose, use periods or commas instead. No subsection headings inside Introduction, Results, or Discussion. Flowing prose only.
- Terminology: the method is target-agnostic. Use "designated" and "background", never "binder" and "non-binder", in all new or edited prose. Legacy function names in code keep "binder" and are not renamed by this plan.
- Do NOT regenerate the canonical family sweeps. The evidence in `submission-audit-results.md` does not support it. The only data regeneration in this plan is `code/data/omega_conotoxin/sar_agreement.csv`.
- Numbers appearing in prose must be traceable to a table, a figure, or `sections/generated/numbers.tex`. Do not hand-type a value that a generator can emit.
- Work on a branch off `main`, not on `main` directly. Commit after every task.

---

### Task 0: Create the working branch

**Files:**
- No file changes.

- [ ] **Step 1: Confirm a clean tree and branch**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git status --short
git checkout -b submission-corrections-2026-07
```

Expected: `git status --short` shows at most `submission-audit-results.md` and the new plan file as untracked or modified. If anything else is dirty, stop and ask.

- [ ] **Step 2: Record the baseline test count**

```bash
cd code && julia --project=. test/runtests.jl 2>&1 | tail -5
```

Expected: a summary line reporting 149 passing tests and 0 failures. Write the exact number down; later tasks compare against it.

---

### Task 1: Guard JCIM and arXiv section parity

Every later task edits prose twice. This guard makes a one-sided edit a test failure instead of a silent divergence found at proof stage.

**Files:**
- Create: `code/test/test_manuscript_consistency.jl`
- Modify: `code/test/runtests.jl`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a testset named `"JCIM and arXiv section sources stay in sync"`. Tasks 6 and 7 append banned-claim assertions to this same file.

- [ ] **Step 1: Write the test**

Create `code/test/test_manuscript_consistency.jl`:

```julia
include(joinpath(@__DIR__, "..", "Include.jl"))
using Test

const SHARED_SECTIONS = [
    "abstract.tex", "appendix.tex", "discussion.tex", "introduction.tex",
    "methods.tex", "results.tex", "si_gmm_derivation.tex",
    "significance_statement.tex", "theory.tex",
]

# The arXiv theory.tex wraps the exact-mixture proposition in a samepage box for
# layout only. That wrapper is the single sanctioned difference between the trees.
const LAYOUT_ONLY_LINES = Set(["\\begin{samepage}", "\\end{samepage}"])

manuscript_body(path) =
    filter(line -> !(strip(line) in LAYOUT_ONLY_LINES), readlines(path))

@testset "JCIM and arXiv section sources stay in sync" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    jcim = joinpath(repo, "paper-jcim", "sections")
    arxiv = joinpath(repo, "paper-arxiv", "sections")
    for name in SHARED_SECTIONS
        jcim_path = joinpath(jcim, name)
        arxiv_path = joinpath(arxiv, name)
        @test isfile(jcim_path)
        @test isfile(arxiv_path)
        @test manuscript_body(jcim_path) == manuscript_body(arxiv_path)
    end
end
```

- [ ] **Step 2: Register it in the suite**

In `code/test/runtests.jl`, add the include immediately after `include("test_generated_paper_tables.jl")`:

```julia
    include("test_generated_paper_tables.jl")
    include("test_manuscript_consistency.jl")
    include("test_rng_reproducibility.jl")
```

- [ ] **Step 3: Run the new test on its own and verify it PASSES**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl
```

Expected: PASS. This guard is written against the current, already-consistent state, so it passes immediately. If it fails, the trees have drifted since 2026-07-29. Stop and report the diff rather than editing the test to accommodate it.

- [ ] **Step 4: Prove the guard actually bites**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
printf '\n%% temporary guard check\n' >> paper-jcim/sections/methods.tex
cd code && julia --project=. test/test_manuscript_consistency.jl
```

Expected: FAIL on `methods.tex`. Then revert:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git checkout paper-jcim/sections/methods.tex
cd code && julia --project=. test/test_manuscript_consistency.jl
```

Expected: PASS again.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/test/test_manuscript_consistency.jl code/test/runtests.jl
git commit -m "test: guard JCIM and arXiv section parity"
```

---

### Task 2: Fix the conotoxin SAR coordinate frame

`code/experiments/compute_conotoxin_sar_agreement.jl:79` loads `strong_cav22_binders.fasta`, whose 23 sequences are raw and range in length from 24 to 31, then indexes raw character positions as if they were MVIIA-numbered alignment columns. The generated sequences it is compared against live in the canonical cleaned 26-column frame. This is the only genuine data error in the audit.

**Files:**
- Create: `code/test/test_conotoxin_sar_frame.jl`
- Modify: `code/experiments/compute_conotoxin_sar_agreement.jl:78-80`
- Modify (regenerated output): `code/data/omega_conotoxin/sar_agreement.csv`
- Modify: `code/test/runtests.jl`

**Interfaces:**
- Consumes: `canonical_load_alignment(spec, data_dir)` and `canonical_split(spec, char_mat, names, auxiliary)` from `code/experiments/canonical_family_registry.jl`. `canonical_split` returns `(designated, background, marker_col, marker_residues)` where `designated` is a vector of row indices into `char_mat`. For the Conotoxin spec, `char_mat` is 74 by 26 and `length(designated) == 23`. This is the same API already used by `code/test/test_conotoxin_labels.jl`.
- Produces: `code/data/omega_conotoxin/sar_agreement.csv` with column `Input_strong` in the canonical frame. Task 3 reads this CSV.

- [ ] **Step 1: Write the failing test**

Create `code/test/test_conotoxin_sar_frame.jl`:

```julia
include(joinpath(@__DIR__, "..", "Include.jl"))
include(joinpath(@__DIR__, "..", "experiments", "canonical_family_registry.jl"))
using Test

# (MVIIA position, accepted residues). Basic positions accept the K/R class,
# matching the permissive counting rule used in the SAR table.
const SAR_POSITIONS = [
    (13, ('Y',)), (2, ('K', 'R')), (10, ('K', 'R')), (11, ('L',)),
    (1, ('C',)), (8, ('C',)), (15, ('C',)), (16, ('C',)),
    (20, ('C',)), (25, ('C',)), (21, ('K', 'R')), (4, ('K', 'R')),
]

@testset "conotoxin SAR input is read in the canonical alignment frame" begin
    data_dir = joinpath(@__DIR__, "..", "data")
    spec = only(filter(s -> s.family == "Conotoxin", CANONICAL_FAMILIES))
    char_mat, names, auxiliary = canonical_load_alignment(spec, data_dir)
    designated, _, _, _ = canonical_split(spec, char_mat, names, auxiliary)

    @test size(char_mat) == (74, 26)
    @test length(designated) == 23

    expected = Dict(pos => count(i -> char_mat[i, pos] in residues, designated) /
                           length(designated)
                    for (pos, residues) in SAR_POSITIONS)

    # All six framework cysteines are invariant across the designated accessions
    # once they are read in the aligned frame. The pre-fix script reported 0.87,
    # 0.70 and 0.52 at positions 16, 20 and 25 because it indexed raw sequences.
    for pos in (1, 8, 15, 16, 20, 25)
        @test expected[pos] == 1.0
    end

    table = CSV.read(joinpath(data_dir, "omega_conotoxin", "sar_agreement.csv"),
                     DataFrame)
    @test nrow(table) == 12
    for row in eachrow(table)
        @test isapprox(row.Input_strong, expected[row.Position]; atol = 5e-4)
    end
end
```

- [ ] **Step 2: Run it and verify it FAILS**

```bash
cd code && julia --project=. test/test_conotoxin_sar_frame.jl
```

Expected: FAIL. The committed `sar_agreement.csv` holds `Input_strong` of 0.87, 0.696 and 0.522 at positions 16, 20 and 25, against an expected 1.0 at each.

- [ ] **Step 3: Fix the script**

In `code/experiments/compute_conotoxin_sar_agreement.jl`, add the registry include next to the existing `Include.jl` line near the top of the file:

```julia
include(joinpath(_CODE_DIR, "Include.jl"))
include(joinpath(_CODE_DIR, "experiments", "canonical_family_registry.jl"))
```

Then replace the three lines at `:78-80`:

```julia
# Also load strong binder input
raw_strong = parse_fasta(joinpath(DATA_DIR, "strong_cav22_binders.fasta"))
@info "Strong binder input: $(length(raw_strong)) sequences"
```

with:

```julia
# Designated input, read in the canonical cleaned alignment frame. Reading the raw
# FASTA here would index variable-length sequences (24 to 31 residues) as if their
# character positions were MVIIA-numbered alignment columns.
_spec = only(filter(s -> s.family == "Conotoxin", CANONICAL_FAMILIES))
_char_mat, _names, _auxiliary = canonical_load_alignment(_spec, joinpath(_CODE_DIR, "data"))
_designated, _, _, _ = canonical_split(_spec, _char_mat, _names, _auxiliary)
raw_strong = [(_names[i], String(_char_mat[i, :])) for i in _designated]
@info "Designated input (canonical frame): $(length(raw_strong)) sequences"
```

`compute_sar_agreement` consumes `x[2]` as the sequence string and is unchanged.

- [ ] **Step 4: Regenerate the CSV**

```bash
cd code && julia --project=. experiments/compute_conotoxin_sar_agreement.jl
```

Expected log line: `Designated input (canonical frame): 23 sequences`. Then confirm the corrected column:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
column -s, -t code/data/omega_conotoxin/sar_agreement.csv
```

Expected `Input_strong`: 0.826 at position 13, 0.957 at 2, 0.696 at 10, 0.174 at 11, 1.0 at positions 1, 8, 15, 16, 20 and 25, 0.478 at 21, and 0.522 at 4. The `SA_strong` and `SA_full` columns must be unchanged, because only the input branch moved frames.

- [ ] **Step 5: Run the test and verify it PASSES**

```bash
cd code && julia --project=. test/test_conotoxin_sar_frame.jl
```

Expected: PASS.

- [ ] **Step 6: Register the test and run the full suite**

In `code/test/runtests.jl`, add after `include("test_conotoxin_labels.jl")`:

```julia
    include("test_conotoxin_labels.jl")
    include("test_conotoxin_sar_frame.jl")
```

```bash
cd code && julia --project=. test/runtests.jl 2>&1 | tail -5
```

Expected: all pass, count above the Task 0 baseline.

- [ ] **Step 7: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/experiments/compute_conotoxin_sar_agreement.jl \
        code/data/omega_conotoxin/sar_agreement.csv \
        code/test/test_conotoxin_sar_frame.jl code/test/runtests.jl
git commit -m "fix: read conotoxin SAR designated input in the canonical alignment frame"
```

---

### Task 3: Move the SAR table into the anti-drift generator

The SAR table body is currently hand-typed in two places, `paper-jcim/Paper_JCIM.tex:339-350` and `paper-arxiv/Paper_v1.tex:392-403`. That is how the wrong numbers survived. Emit it from the CSV instead, and close the existing coverage gap where `test_generated_paper_tables.jl` checks only the arXiv copy.

**Files:**
- Modify: `code/experiments/generate_paper_tables.jl` (add a `tab_sar_agreement.tex` emitter inside `generate`, and a `sar_agreement.csv` check inside `validate_inputs`)
- Modify: `code/test/test_generated_paper_tables.jl`
- Create (generated): `paper-arxiv/sections/generated/tab_sar_agreement.tex` and `paper-jcim/sections/generated/tab_sar_agreement.tex`
- Modify: `paper-jcim/Paper_JCIM.tex:339-350`
- Modify: `paper-arxiv/Paper_v1.tex:392-403`

**Interfaces:**
- Consumes: `code/data/omega_conotoxin/sar_agreement.csv` from Task 2, columns `Position`, `WT_Residue`, `Role`, `Effect_of_mutation`, `Input_strong`, `SA_strong`, `SA_full`.
- Produces: `tab_sar_agreement.tex`, containing only body rows plus a trailing `\bottomrule`, matching the established convention of `tab_cross_family.tex`.

- [ ] **Step 1: Extend the generated-file test first**

In `code/test/test_generated_paper_tables.jl`, add `"tab_sar_agreement.tex"` to `expected_files`, and change the single-tree check into a both-trees check. Replace the `committed` binding and the two `readdir` assertions with:

```julia
    committed_trees = [
        joinpath(repo_dir, "paper-arxiv", "sections", "generated"),
        joinpath(repo_dir, "paper-jcim", "sections", "generated"),
    ]
```

and replace the comparison block with:

```julia
        @test sort(readdir(generated)) == expected_files
        for committed in committed_trees
            @test sort(readdir(committed)) == expected_files
            for filename in expected_files
                @test read(joinpath(generated, filename)) == read(joinpath(committed, filename))
            end
        end
```

- [ ] **Step 2: Run it and verify it FAILS**

```bash
cd code && julia --project=. test/test_generated_paper_tables.jl
```

Expected: FAIL. `tab_sar_agreement.tex` is in `expected_files` but the generator does not emit it and neither tree contains it.

- [ ] **Step 3: Emit the table from the generator**

In `code/experiments/generate_paper_tables.jl`, inside `validate_inputs()`, add before the return:

```julia
    sar_path = joinpath(DATA_DIR, "omega_conotoxin", "sar_agreement.csv")
    isfile(sar_path) || error("Missing conotoxin SAR table: $sar_path")
    sar = CSV.read(sar_path, DataFrame)
    nrow(sar) == 12 || error("Conotoxin SAR table must have twelve rows")
```

and return `sar` alongside the existing values, updating the destructuring in `generate` accordingly.

In `generate(output_dir)`, add:

```julia
    sar_rows = String[]
    for row in eachrow(sar)
        bold = @sprintf("\\textbf{%.2f}", row.SA_strong)
        push!(sar_rows, @sprintf(
            "%d & %s & %s & %s & %.2f & %s & %.2f \\\\",
            Int(row.Position), row.WT_Residue, row.Role, row.Effect_of_mutation,
            row.Input_strong, bold, row.SA_full))
    end
    write_text(joinpath(output_dir, "tab_sar_agreement.tex"),
               join(sar_rows, "\n") * "\n" * raw"\bottomrule")
```

- [ ] **Step 4: Generate into both trees**

```bash
cd code
julia --project=. experiments/generate_paper_tables.jl --output=../paper-arxiv/sections/generated
julia --project=. experiments/generate_paper_tables.jl --output=../paper-jcim/sections/generated
```

Then inspect the result:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
cat paper-arxiv/sections/generated/tab_sar_agreement.tex
```

Expected: twelve rows, positions in the CSV order 13, 2, 10, 11, 1, 8, 15, 16, 20, 25, 21, 4, with 1.00 in the Input column for every cysteine row, followed by `\bottomrule`.

- [ ] **Step 5: Point both manuscripts at the generated file**

In `paper-jcim/Paper_JCIM.tex`, delete lines 339 through 351 (the twelve hand-typed rows and the `\bottomrule`) and put in their place:

```latex
    \input{sections/generated/tab_sar_agreement}
```

Do the same in `paper-arxiv/Paper_v1.tex` for lines 392 through 404. Leave the surrounding `\begin{tabular}{clllccc}`, `\toprule`, header row and `\midrule` untouched in both files.

- [ ] **Step 6: Run the test and verify it PASSES**

```bash
cd code && julia --project=. test/test_generated_paper_tables.jl
```

Expected: PASS.

- [ ] **Step 7: Build both PDFs**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/paper-jcim && make
cd ../paper-arxiv && make
```

Expected: both build with no unresolved references. Open the SAR table page in each PDF and confirm twelve rows render with the corrected Input column.

- [ ] **Step 8: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add code/experiments/generate_paper_tables.jl code/test/test_generated_paper_tables.jl \
        paper-arxiv/sections/generated paper-jcim/sections/generated \
        paper-jcim/Paper_JCIM.tex paper-arxiv/Paper_v1.tex
git commit -m "feat: generate the conotoxin SAR table from its canonical CSV"
```

---

### Task 4: Correct the SAR caption and surrounding prose

With the corrected input, the framework cysteines read 1.00 in the designated input, so the apparent enrichment from 0.52 or 0.70 up to 1.00 at positions 20 and 25 does not exist. Enrichment does still hold at Tyr13, Arg10, position 21, Lys4 and Leu11, so this is a precision fix, not a retraction.

**Files:**
- Modify: `paper-jcim/Paper_JCIM.tex:325-332` and `paper-arxiv/Paper_v1.tex:378-384` (caption)
- Modify: `paper-jcim/sections/results.tex:220-228` and `paper-arxiv/sections/results.tex:220-228`

- [ ] **Step 1: Rewrite the caption in both manuscript files**

Replace the caption sentence that currently reads:

```latex
  Designated-subset generation retains the Tyr13 signal and cysteine scaffold and
  enriches several reported positions.
```

with:

```latex
  The six framework cysteines are invariant across the designated input, so the
  table reports their retention rather than their enrichment. Designated-subset
  generation retains the Tyr13 signal and the cysteine scaffold, and raises the
  frequency at Tyr13, Arg10, position~21 and Lys4 relative to the designated input.
```

Keep the surrounding caption sentences, including the closing sentence stating that these frequencies do not test mutation effects in the generated backgrounds.

- [ ] **Step 2: Correct the results prose in both copies**

In `results.tex`, the sentence beginning "designated-subset-seeded sequences retained the Tyr13 marker (98.3\%)" claims enrichment "relative to both the 23-sequence input and full-family generation". Enrichment relative to the input holds for the basic-residue positions but not for the cysteines. Replace the clause:

```latex
($\geq 97.4\%$), and enriched basic residues at loop~2 positions critical for channel
interaction (Lys2: 97.5\% K/R; Arg10: 82.3\% K/R) relative to both the 23-sequence input
and full-family generation.
```

with:

```latex
($\geq 97.4\%$), which are already invariant across the designated input. It raised the
frequency of basic residues at loop~2 positions reported as important for channel
interaction (Arg10: 82.3\% K/R against 69.6\% in the designated input) and at
position~21 (66.2\% K/R against 47.8\%), and exceeded full-family generation at every
reported position.
```

- [ ] **Step 3: Verify the numbers against the regenerated table**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
column -s, -t code/data/omega_conotoxin/sar_agreement.csv
```

Every percentage newly written into the prose must appear in this CSV. Confirm 0.696 and 0.478 in `Input_strong`, and 0.823 and 0.662 in `SA_strong`.

- [ ] **Step 4: Run the parity guard and build**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl
cd ../paper-jcim && make
```

Expected: parity PASSES, meaning the edit landed in both trees. Build is clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv
git commit -m "docs: correct SAR caption and prose for the aligned designated input"
```

---

### Task 5: Rewrite the hard-mask paragraph

`results.tex:153-164` argues that because attention tracks `f_eff` closely, the hard-mask advantage "is not an attention effect but a decode effect". `code/data/kunitz/mask_residuals.csv` shows `mask_p1kr` is 0.5387 on both the `f_eff=0.5` and `f_eff=0.7` rows, because the mask condition is one condition with `f_eff = 1` by construction. The comparison therefore spans conditions with different intended designated mass, and the inference does not follow.

**Files:**
- Modify: `paper-jcim/sections/results.tex:153-164` and `paper-arxiv/sections/results.tex:153-164`

- [ ] **Step 1: Re-read the evidence before writing**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
column -s, -t code/data/kunitz/mask_residuals.csv
```

Confirm that `mask_p1kr` is identical on the first two rows and that `residual` contracts from -0.111 to -0.002 as `f_eff` runs from 0.5 to 0.99.

- [ ] **Step 2: Replace the inference sentence in both copies**

Replace:

```latex
Because the attention weights track $f_{\mathrm{eff}}$ to within a fraction of a percent
($\Delta_{\mathrm{attn}} \approx 0$, Eq.~\ref{eq:gap-decomp}), this masking advantage is not an
attention effect but a decode effect: eliminating background attention concentrates the weighted
superposition on designated patterns alone, placing the state where the leading PCA components
resolve the designated residue most reliably.
```

with:

```latex
Hard masking removes the background components outright, so it raises the intended designated
mass from $f_{\mathrm{eff}}$ to one, and the comparison at matched $\beta$ therefore spans
conditions with different intended mass. Attention realized the intended mass in each condition
to within a fraction of a percent ($\Delta_{\mathrm{attn}} \approx 0$,
Eq.~\ref{eq:gap-decomp}), and the advantage contracted as $f_{\mathrm{eff}}$ approached one.
The informative quantity is the size of the response. A difference of 0.5 in intended designated
mass produced a difference of only 0.111 in observed marker fraction, so the latent-to-sequence
map compressed the conditioning signal by roughly a factor of four. That compression is imposed
by the full map, comprising finite-temperature Gaussian noise in the latent space, the
normalized-coordinate PCA reconstruction, and argmax decoding.
```

- [ ] **Step 3: Correct the following sentence**

The sentence beginning "Recovery on this intermediate-separation family is thus governed by two factors" attributes both factors to "the decode gap rather than through the attention term". Replace "acting through the decode gap rather than through the attention term" with "acting through the latent-to-sequence map rather than through a failure of attention to realize its target".

- [ ] **Step 4: Check that no other passage repeats the retracted inference**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -rn "decode effect\|not an attention effect" paper-jcim/sections paper-arxiv/sections
```

Expected: no matches. If `discussion.tex` or `appendix.tex` repeats the claim, apply the same correction there in both trees.

- [ ] **Step 5: Run the parity guard and build**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl
cd ../paper-jcim && make
```

Expected: PASS, clean build.

- [ ] **Step 6: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv
git commit -m "docs: state the hard-mask comparison as a change in intended designated mass"
```

---

### Task 6: Fix the entropy identity and delete the false proportionality

Two related errors. The identity `H_r(0) = log K_eff` is false for unequal weights: at `beta = 0` the attention weights are `w_k = r_k / sum_j r_j`, so the limit is the Shannon entropy `-sum_k w_k log w_k`, while `log K_eff = -log sum_k w_k^2` is the Renyi-2 entropy. Separately, `appendix.tex:190` asserts `$\beta^{*} \propto \log K_{\mathrm{eff}}$`, which is directionally wrong, because the reported `beta*` rises from 4.35 to 9.26 while `K_eff` falls from 99 to 32.1. That line also contradicts `appendix.tex:156-160` in the same file, which argues that a lower `log K_eff` forces a higher `beta*`.

`code/src/Binding.jl:848-873` already documents the correct distinction, so the manuscript currently contradicts its own source code.

**Files:**
- Modify: `paper-jcim/sections/theory.tex:118-125` and `paper-arxiv/sections/theory.tex` (same passage, two lines lower)
- Modify: `paper-jcim/sections/appendix.tex:154-160` and `paper-arxiv/sections/appendix.tex:154-160`
- Modify: `paper-jcim/sections/appendix.tex:189-192` and `paper-arxiv/sections/appendix.tex:189-192`
- Modify: `code/test/test_manuscript_consistency.jl`

- [ ] **Step 1: Write the failing guard**

Append to `code/test/test_manuscript_consistency.jl`:

The two claims straddle line breaks in the source, so match on whitespace-normalized text
rather than with line-oriented regexes.

```julia
# Both claims are false. The beta -> 0 attention entropy is the Shannon entropy of the
# weights, not the Renyi-2 entropy log K_eff; see the shannon_entropy docstring in
# code/src/Binding.jl. And beta* rises while log K_eff falls, so the proportionality in
# the appendix has the wrong direction and contradicts the mechanism argued earlier in
# the same file.
const RETRACTED_CLAIMS = [
    "H_{\\vr}(0) = \\log K_{\\mathrm{eff}}",
    "H_{\\mathbf{r}}(0) = \\log K_{\\mathrm{eff}}",
    "\\beta^{*} \\propto \\log K_{\\mathrm{eff}}",
]

squeeze_whitespace(text) = replace(text, r"\s+" => " ")

@testset "retracted entropy claims stay out of the manuscript" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    for tree in ("paper-jcim", "paper-arxiv"), name in SHARED_SECTIONS
        text = squeeze_whitespace(read(joinpath(repo, tree, "sections", name), String))
        for claim in RETRACTED_CLAIMS
            @test !occursin(claim, text)
        end
    end
end
```

- [ ] **Step 2: Run it and verify it FAILS**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl
```

Expected: FAIL in both trees. `theory.tex` trips the `\vr` form, `appendix.tex` trips both the `\mathbf{r}` form at line 154 and the `\propto` form at line 190. That is six failing assertions across the two trees. Confirm the exact source strings first if any assertion does not fire:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -n "K_{\\\\mathrm{eff}}" paper-jcim/sections/theory.tex paper-jcim/sections/appendix.tex
```

Adjust the literal to match the real source. Never adjust the source to match the test.

- [ ] **Step 3: Correct the theory passage in both copies**

In `theory.tex`, replace:

```latex
attention distribution at $\beta=0$: the weighted entropy at zero temperature is $H_{\vr}(0) = \log
K_{\mathrm{eff}}(\vr) < \log K$, a lower starting entropy than the unweighted case.
Reaching the retrieval-phase inflection from this pre-concentrated starting point requires
a higher inverse temperature, so $\beta^{*}(\rho) \geq \beta^{*}(1)$.
```

with:

```latex
attention distribution at $\beta = 0$, where the attention weights reduce to
$w_k = r_k / \sum_j r_j$ and the limiting entropy is the Shannon entropy
$H_{\vr}(0) = -\sum_k w_k \log w_k < \log K$. The effective pattern count
$K_{\mathrm{eff}}(\vr) = (\sum_k r_k)^2 / \sum_k r_k^2$ is a separate inverse-Simpson
descriptor of the same weights, and $\log K_{\mathrm{eff}}$ is their Renyi-2 entropy, which
coincides with $H_{\vr}(0)$ only when the weights are equal. We observed that the entropy
curve shifts to higher inverse temperature as $\rho$ increases. We report that shift as an
observation and do not derive it from $K_{\mathrm{eff}}$.
```

Also replace the phrase "the retrieval-phase inflection" wherever it survives in this remark, per Task 7.

- [ ] **Step 4: Correct the appendix passage in both copies**

In `appendix.tex:154-160`, replace:

```latex
phase transition: the logit biases pre-concentrate the attention distribution
at $\beta = 0$, reducing the weighted entropy to
$H_{\mathbf{r}}(0) = \log K_{\mathrm{eff}}(\mathbf{r}) < \log K$, so
reaching the retrieval-phase inflection requires a higher inverse temperature
$\beta^{*}(\rho) \geq \beta^{*}(1)$. We observed this empirically:
```

with:

```latex
entropy crossover: the logit biases pre-concentrate the attention distribution
at $\beta = 0$, reducing the weighted entropy to the Shannon entropy
$H_{\mathbf{r}}(0) = -\sum_k w_k \log w_k < \log K$ with
$w_k = r_k / \sum_j r_j$. We observed that the crossover moved to higher inverse
temperature as $\rho$ increased:
```

- [ ] **Step 5: Delete the false proportionality in both copies**

In `appendix.tex:189-192`, replace:

```latex
with $\beta^{*}(\rho)$ increasing from 4.35 ($\rho=1$) to 9.26
($\rho=1000$), consistent with $\beta^{*} \propto \log K_{\mathrm{eff}}$
predicted by the mean-field argument in the main text
(Fig.~\MainRef{fig:phase-transition}).
```

with:

```latex
with the operating point increasing from 4.35 ($\rho=1$) to 9.26 ($\rho=1000$)
(Fig.~\MainRef{fig:phase-transition}). We report this displacement without asserting a
functional relationship to $K_{\mathrm{eff}}$.
```

- [ ] **Step 6: Run the guard and verify it PASSES**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl
```

Expected: PASS, including the parity testset.

- [ ] **Step 7: Confirm no numeric result moved**

The identity appears only in prose. The operating-point code computes attention entropy directly, so nothing regenerates. Verify:

```bash
cd code && julia --project=. test/test_entropy_identities.jl
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study && git status --short code/data
```

Expected: entropy tests pass, and `code/data` shows no changes.

- [ ] **Step 8: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv code/test/test_manuscript_consistency.jl
git commit -m "docs: use Shannon entropy for the beta to zero limit and drop the false proportionality"
```

---

### Task 7: Rename the operating point and disclose its sensitivity

`code/src/Binding.jl:881-913` selects the most negative second derivative of entropy with respect to `log beta`. That is the onset of the entropy drop, not an inflection, which is a zero crossing of the second derivative. `code/src/Protein.jl:423-474` already documents this correctly and already names the two statistics `beta_onset` and `beta_steepest`. Only the manuscript language is stale.

The order dependence is real but small. The detector probes the first 20 memory columns, and switching to an all-memory calculation moved the selected value by exactly one step on the fixed 50-point logarithmic grid `10 .^ range(log10(0.1), log10(500), length=50)` for Homeobox and conotoxin, with four of six families unchanged. Disclose it; do not regenerate the sweeps.

**Files:**
- Modify: `paper-jcim/sections/theory.tex` and `paper-arxiv/sections/theory.tex`
- Modify: `paper-jcim/sections/methods.tex` and `paper-arxiv/sections/methods.tex`
- Modify: `paper-jcim/sections/appendix.tex` and `paper-arxiv/sections/appendix.tex`

- [ ] **Step 1: Inventory every occurrence**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -rn "phase transition\|phase-transition\|inflection" paper-jcim/sections paper-arxiv/sections | grep -v "fig:phase-transition"
```

Work through the list. Leave the LaTeX label `fig:phase-transition` alone, since renaming a label touches every cross-reference for no scientific gain. The figure caption text, however, is prose and should be updated.

- [ ] **Step 2: Apply the substitutions in both trees**

- "phase transition" becomes "entropy crossover".
- "inflection" becomes "onset".
- `$\beta^{*}$` keeps its symbol, but its first definition gains the words "the entropy-crossover onset".

- [ ] **Step 3: Add the operating-point disclosure to Methods in both copies**

Insert into `methods.tex`, in the paragraph that describes how the operating inverse temperature is chosen:

```latex
The operating inverse temperature was selected as the entropy-crossover onset, the point of
maximum downward curvature of the weighted attention entropy with respect to $\log \beta$ on a
fixed 50-point logarithmic grid spanning $\beta \in [0.1, 500]$, with entropy evaluated at the
first 20 stored memory columns. This is a descriptive operating point, not an estimate of a
thermodynamic phase transition, and entropy is evaluated at stored memories rather than at
samples. Recomputing the onset from all stored memories moved the selected value by one grid
step, a factor of 1.19, for Homeobox and $\omega$-conotoxin and left Kunitz, SH3, WW and
Forkhead unchanged.
```

- [ ] **Step 4: Verify the disclosure matches the code**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
sed -n '878,900p' code/src/Binding.jl
grep -n "n_betas\|find_weighted_entropy_inflection" code/experiments/run_canonical_family_sweeps.jl
```

Confirm the grid endpoints, the 50-point length, and `n_probes = min(K, 20)` on the first columns. If the canonical driver passes different arguments, the Methods text must match the driver, not the default.

- [ ] **Step 5: Run the guard and build**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl && julia --project=. test/test_transition_statistic.jl
cd ../paper-jcim && make
```

Expected: PASS, clean build with no unresolved references.

- [ ] **Step 6: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv
git commit -m "docs: rename the operating point to the entropy-crossover onset and disclose its sensitivity"
```

---

### Task 8: Explain why ULA is retained, and correct the replication claim

For unit-norm memories the Gibbs density is exactly a Gaussian mixture, and `build_memory_matrix` does project to unit norm, so this applies exactly to every run in the paper. A referee will ask why an approximate correlated sampler is used at all. Answer it in the text. Separately, `theory.tex:38-40` claims a real-valued `r_k` is equivalent to storing `r_k` copies, which holds only for integer multiplicities.

**Files:**
- Modify: `paper-jcim/sections/theory.tex:36-60` and `paper-arxiv/sections/theory.tex:36-62`
- Modify: `paper-jcim/sections/discussion.tex:6-10` and `paper-arxiv/sections/discussion.tex:6-10`

- [ ] **Step 1: Add the sampler-choice statement after Proposition 1 in both copies**

Immediately after the sentence "The derivation is given in the Supporting Information.", insert:

```latex
Direct Gaussian-mixture sampling is therefore the preferred equilibrium implementation for the
unit-norm model, and we use it as the exact reference throughout. We retain ULA for the reported
generation runs to maintain comparability with the original stochastic-attention workflow and to
characterize its finite-step behavior. ULA is not required to sample this target.
```

- [ ] **Step 2: Correct the replication claim in both copies**

Replace:

```latex
Patterns with higher multiplicity deepen the energy wells in their vicinity; this is equivalent
to storing $r_k$ copies of pattern $\vm_k$ without the $\mathcal{O}(d \cdot \sum_k r_k)$
memory cost.
```

with:

```latex
Patterns with higher multiplicity deepen the energy wells in their vicinity. For integer $r_k$
this is exactly equivalent to storing $r_k$ copies of pattern $\vm_k$ without the
$\mathcal{O}(d \cdot \sum_k r_k)$ memory cost. For positive real $r_k$ it is the continuous
generalization of that replication.
```

- [ ] **Step 3: Narrow the ULA claim in Discussion in both copies**

Replace "ruling out ULA convergence error as the source of that gap" with "indicating that finite-step ULA error was not the main source of that gap".

- [ ] **Step 4: State the KL direction**

`code/src/Protein.jl:316-321` defines `aa_composition_kl(gen_seqs, stored_seqs)` as
$D_{\mathrm{KL}}(\text{stored} \parallel \text{generated})$, that is, reference to generated.
The manuscript reports the value without stating a direction. Find every reporting site:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -rn "KL" paper-jcim/sections paper-arxiv/sections
```

At each site, and in the Methods definition at `methods.tex:47`, write the direction explicitly
as `$D_{\mathrm{KL}}(\text{reference} \parallel \text{generated})$`. Do not describe it as a
distance or as symmetric.

- [ ] **Step 5: Run the guard and build**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl
cd ../paper-jcim && make
```

- [ ] **Step 6: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv
git commit -m "docs: justify retaining ULA and qualify the replication and convergence claims"
```

---

### Task 9: Fix the two citations and the two grammar breaks

**Files:**
- Modify: `paper-jcim/References_v1.bib` and `paper-arxiv/References_v1.bib`
- Modify: `paper-jcim/sections/introduction.tex:4` and `paper-arxiv/sections/introduction.tex:4`
- Modify: `paper-jcim/sections/methods.tex:22-23` and `:55-57`, plus the arXiv copies
- Modify: `paper-jcim/sections/discussion.tex:37-40` and `paper-arxiv/sections/discussion.tex:37-40`

- [ ] **Step 1: Add the two replacement bib entries to both bib files**

```bibtex
@article{hawkinsHookerVAE2021,
  author={Hawkins-Hooker, Alex and Depardieu, Florence and Baur, Sebastien and Couairon, Guillaume and Chen, Arthur and Bikard, David},
  title={Generating functional protein variants with variational autoencoders},
  journal={PLOS Computational Biology},
  volume={17},
  number={2},
  pages={e1008736},
  year={2021},
  doi={10.1371/journal.pcbi.1008736}
}

@inproceedings{meierZeroShot2021,
  author={Meier, Joshua and Rao, Roshan and Verkuil, Robert and Liu, Jason and Sercu, Tom and Rives, Alexander},
  title={Language models enable zero-shot prediction of the effects of mutations on protein function},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

Verify both references before committing. The audit supplied them and they have not been independently checked.

- [ ] **Step 2: Repoint the VAE citation**

In `introduction.tex:4`, change `\cite{sinaiAdaptiveMLDE2020}` to `\cite{hawkinsHookerVAE2021}`. The existing entry keyed `sinaiAdaptiveMLDE2020` is in fact Yang, Wu and Arnold, *Nature Methods* 16, 687 to 694, 2019, so its key name, its year and its use as a variational-autoencoder example are all wrong. Keep the entry in the bib only if it is cited elsewhere:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -rn "sinaiAdaptiveMLDE2020" paper-jcim paper-arxiv --include=*.tex
```

If the only use was the VAE citation, delete the entry from both bib files. If it is cited elsewhere as a directed-evolution review, rename the key to `yangMLDE2019` and fix every use.

- [ ] **Step 3: Repoint the ESM2 citation**

In `methods.tex:55-57`, the masked-marginal pseudo-perplexity sentence cites the ESMFold paper. Add `\cite{meierZeroShot2021}` as the method reference, keeping the ESMFold citation only where ESMFold itself is used for structure prediction.

- [ ] **Step 4: Fix the SH3 marker sentence in both copies**

Replace:

```latex
for SH3, the selected Trp marker
binding groove was the marker ($K_{\mathrm{des}} = 33$);
```

with a sentence that states the actual selection rule. `canonical_split` at
`code/experiments/canonical_family_registry.jl:58-66` does not use a fixed column. It computes
the Trp frequency of every column over its non-gap rows, restricts to columns where that
frequency lies strictly between 0.15 and 0.85, and takes the highest-frequency column among
those. The other families in this same sentence are described by their selection rule, so match
that pattern:

```latex
for SH3, the partially conserved Trp column of the peptide-binding groove, selected as the
column with the highest Trp frequency among those where Trp occurs in 15 to 85\% of
sequences, was the marker ($K_{\mathrm{des}} = 33$);
```

- [ ] **Step 5: Fix the broken sentence in Discussion in both copies**

Replace:

```latex
Hard curation achieves complete marker retention with as few as three designated
input sequences. If the designation comes from an independent experimental campaign, such as phage
display or yeast surface display, functional selections, or even literature curation) produce
exactly this kind of small, functionally characterized set.
```

with:

```latex
Hard curation achieves complete marker retention with as few as three designated
input sequences. Independent experimental campaigns produce exactly this kind of small,
characterized set. Phage display, yeast surface display, functional selections and literature
curation all yield a handful of sequences that the practitioner already trusts.
```

- [ ] **Step 6: Build and confirm no unresolved citations**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/paper-jcim && make
grep -i "undefined\|unresolved" Paper_JCIM.log Paper_JCIM_SI.log | head
cd ../paper-arxiv && make
grep -i "undefined\|unresolved" Paper_v1.log | head
```

Expected: no undefined citation warnings in either log.

- [ ] **Step 7: Run the guard and commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
cd code && julia --project=. test/test_manuscript_consistency.jl
cd .. && git add paper-jcim paper-arxiv
git commit -m "docs: correct the VAE and ESM2 citations and two broken sentences"
```

---

### Task 10: Remove the remaining overstatements

**Files:**
- Modify: `paper-jcim/sections/methods.tex:47` and the arXiv copy
- Modify: `paper-jcim/sections/introduction.tex` and the arXiv copy
- Modify: `paper-jcim/sections/results.tex` and `discussion.tex`, both trees

- [ ] **Step 1: Rename the separation index**

At `methods.tex:47` the measure is defined from pairwise cosine similarities within and between groups, which is not the classical Fisher discriminant ratio. The canonical CSV column is already named `separation_index`, so only the prose needs changing. Replace "The Fisher separation index $S$" with "The separation index $S$ used in this study", and drop "Fisher" from every other occurrence:

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -rn "Fisher" paper-jcim/sections paper-arxiv/sections
```

- [ ] **Step 2: Qualify the two broad introductory claims**

The statement that profile HMMs generally produce low structural confidence and poor compositional fidelity is supported by the reported Kunitz comparison only. Add "in the Kunitz comparison reported here" to that sentence. The statement that most learned models lack sufficient training signal in the small-family regime overlooks pretrained models, which do not rely only on family-specific data. Narrow it to models trained on the target family alone.

- [ ] **Step 3: Correct the remaining precision claims**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -rn "preserves.*exactly\|near zero\|near-zero" paper-jcim/sections paper-arxiv/sections
```

Replace "preserves exactly" with "matched within" plus the reported tolerance wherever the values differ. Where residuals approach 0.15, state the largest deviation rather than calling them near zero.

- [ ] **Step 4: Keep the five-family regression explicitly exploratory**

Confirm the regression is already labelled exploratory and includes the leave-one-family-out sensitivity. If the label is missing at any point of use, add it.

- [ ] **Step 5: Apply the conotoxin functional-language rules in both copies**

Only five designated and two background accessions have exact activity metadata matches, as `methods.tex:29-34` already states, and no generated sequence was tested for Cav2.2 binding. Apply:

- "Cav2.2-specific pharmacophore" becomes "Tyr13-associated marker".
- "binders" becomes "designated accessions" wherever activity is unverified.
- Claims of preserved binding, specificity or pharmacological activity become claims about preserved or enriched sequence markers.

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
grep -rn "pharmacophore\|binder" paper-jcim/sections paper-arxiv/sections
```

- [ ] **Step 6: Run the guard, build, and commit**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl
cd ../paper-jcim && make
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv
git commit -m "docs: qualify overstated claims and tighten conotoxin functional language"
```

---

### Task 11: Tone and sentence-length pass

**Files:**
- Modify: `paper-jcim/sections/results.tex` and `methods.tex`, plus the arXiv copies

- [ ] **Step 1: Remove the promotional phrases in both trees**

Verified counts across `sections/*.tex` and `Paper_JCIM.tex` as of 2026-07-29: "small-set amplifier" 4, "principled" 2, "central finding" 1, "negligible computation" 1, "natural axis" 1, "near-perfect" 1, "faithful recapitulation" 1. The phrase "confirming high fidelity" does not appear.

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
for w in "small-set amplifier" "central finding" "principled" "negligible computation" \
         "natural axis" "near-perfect" "faithful recapitulation"; do
  printf "%-26s " "$w"; grep -rin "$w" paper-jcim/sections paper-arxiv/sections | wc -l
done
```

Prefer "increased", "was associated with", "matched within", "suggested", "we observed", "provides a way to". Rerun the loop after editing and expect zero for every phrase.

- [ ] **Step 2: Shorten the long sentences**

Concentrations are at `results.tex:89-100`, `153-168`, `212-228`, `271-279`, and `methods.tex:2-17`, `18-34`. Target most sentences below 30 to 35 words. Put one result or one interpretation in each sentence. Separate methods, results and interpretation rather than joining them with commas and semicolons.

- [ ] **Step 3: Rewrite captions to describe rather than conclude**

Figure and table captions should say what the panel shows, not what the reader should conclude from it.

- [ ] **Step 4: Run the style linter and the section reviewer**

Use the repository's own tooling rather than eyeballing:

```
/style-check
/review-section results
/audit-magic-numbers results
```

`style-check` enforces the no-dash and no-subsection-heading rules. `audit-magic-numbers` catches any number newly written into prose without a supporting table or figure reference, which matters because Task 4 and Task 5 both added figures to prose.

- [ ] **Step 5: Run the guard, build both, and commit**

```bash
cd code && julia --project=. test/test_manuscript_consistency.jl
cd ../paper-jcim && make
cd ../paper-arxiv && make
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add paper-jcim paper-arxiv
git commit -m "docs: shorten sentences and neutralize promotional phrasing"
```

---

### Task 12: Final verification and audit closeout

**Files:**
- Modify: `submission-audit-results.md`

- [ ] **Step 1: Run the full suite**

```bash
cd code && julia --project=. test/runtests.jl 2>&1 | tail -8
```

Expected: zero failures, zero errors, a count above the Task 0 baseline of 149 by the tests added in Tasks 1, 2 and 6.

- [ ] **Step 2: Rebuild both PDFs from clean**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study/paper-jcim && make
cd ../paper-arxiv && make
grep -ci "overfull\|underfull" paper-jcim/Paper_JCIM.log \
  paper-jcim/Paper_JCIM_SI.log paper-arxiv/Paper_v1.log
```

Expected: builds succeed, no undefined references, overfull box count no worse than the pre-change baseline.

- [ ] **Step 3: Read the SAR table and the hard-mask paragraph in the rendered PDFs**

These are the two substantive corrections. Confirm visually that the SAR table shows 1.00 for every cysteine in the Input column, and that the hard-mask paragraph no longer claims a pure decode effect.

- [ ] **Step 4: Confirm the canonical sweeps were not disturbed**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git diff --stat main -- code/data
```

Expected: `code/data/omega_conotoxin/sar_agreement.csv` is the only changed data file. Any other change means a sweep was regenerated by accident. Investigate before merging.

- [ ] **Step 5: Update the audit document**

In `submission-audit-results.md`, add a closing section recording, for each of the seven agreed items, the commit that resolved it and the test that now guards it. Record explicitly that the `\propto` claim at `appendix.tex:190` was real, that the second verification pass missed it by searching for the word "proportional" rather than the LaTeX symbol, and that the appendix had contradicted itself between line 156 and line 190.

- [ ] **Step 6: Final commit**

```bash
cd /Users/jdv27/Desktop/julia_work/SA-Binding-Generation-Study
git add submission-audit-results.md
git commit -m "docs: record resolution of the pre-submission audit"
```

- [ ] **Step 7: Report before merging**

Summarize what changed, what was deliberately not done (no canonical sweep regeneration, no move of the hard-mask subsection to the SI), and any item where the replacement text needs the author's judgment rather than a mechanical fix. Do not merge to `main` without the author's approval.

---

## Deliberately out of scope

- Regenerating the canonical family sweeps. The one-grid-step sensitivity does not justify it, and Task 7 discloses the sensitivity instead.
- Switching the canonical driver from `find_weighted_entropy_inflection` to the order-independent `find_entropy_transition`. That changes generated results and belongs in a separate plan with its own replication.
- Moving the hard-mask subsection to the SI, and shortening the Discussion. Both are structural editorial calls for the author.
- Reformatting the SAR table's Role and Mutation-effect columns. Task 3 emits them verbatim from the CSV to keep the change reviewable.
