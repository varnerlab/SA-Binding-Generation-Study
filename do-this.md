# Documentation cleanup to do next

Current state:

- Repository: `SA-Binding-Generation-Study`
- Branch: `main`
- Verified repair baseline: commit `29c3f78`
- Before this handoff was recorded, local `main` was three commits ahead of
  `origin/main`.
- The implementation and manuscript are correct.
- The full test suite passes 126/126.
- Two tracked files currently appear as uncommitted deletions:
  - `no-test-decode-issues.md`
  - `paper-submission-audit.md`

## Objective

Clean up the audit documentation without changing code, data, figures, manuscript
results, or scientific conclusions.

## Required work

### 1. Restore the two deleted audit files

Restore:

```bash
git restore -- no-test-decode-issues.md paper-submission-audit.md
```

These files are useful historical records. They should not disappear as unexplained,
unstaged deletions.

Add a short status banner to each if necessary:

- `no-test-decode-issues.md` was resolved by commits `094a66e` and `29c3f78`.
- `paper-submission-audit.md` is a historical July pre-submission audit. Preserve it
  as an audit record even where later commits resolved or corrected individual items.

Do not silently rewrite the historical body of either file.

### 2. Rewrite `remaining-issues-list.md` as the final corrected resolution

The current file puts Codex's correction above Claude's superseded claims. Replace
that duplicated structure with one concise, authoritative account.

The rewritten file should contain:

1. Current verified state: `main` at `29c3f78`, 126/126 tests passing, and no
   manuscript number affected.
2. Issue 1's corrected conclusion:
   - exactly 23 historical CSVs have gap-sensitive fields;
   - the original column attribution was partly wrong;
   - binding and HMM `novelty` are PCA cosine metrics, while their `seqid` fields
     are gap-sensitive;
   - `approach_comparison.csv` is unaffected;
   - there is no tracked bare `binding_experiment.csv`;
   - the definitive per-file inventory and regeneration policy are in
     `code/data/kunitz/README.md`.
3. Issue 2's corrected conclusion:
   - one-hot encoding, sequence identity, and validity have different deliberate
     contracts;
   - ambiguity symbols cannot be represented by the 20-channel encoder;
   - `sequence_identity` compares non-gap ambiguity symbols literally, so `B`
     can match `B`;
   - `valid_residue_fraction` must penalize non-standard symbols;
   - no numerical behavior or manuscript result changed.
4. Issue 3's corrected conclusion:
   - the claim that `~` was untested was false;
   - it was already covered by synthetic decoder, identity, and validity tests;
   - commit `29c3f78` added a direct predicate assertion as an extra guard.
5. A short review-outcome section preserving Claude's useful acknowledgment:
   the recurring error was generalizing from a partial check—fixture inventory to
   branch coverage, grep counts to column semantics, and stored-memory measurements
   to general decoder recommendations.

Remove the long superseded issue descriptions, incorrect file tables, and obsolete
recommendations from the final version. Git history already preserves them.

### 3. Validate the documentation cleanup

Run:

```bash
git diff --check
cd code
julia --project=. test/runtests.jl
cd ..
git status --short
git diff --name-status
```

Expected:

- 126/126 tests pass.
- No tracked files remain deleted.
- Only the three audit Markdown files should be modified:
  - `no-test-decode-issues.md`
  - `paper-submission-audit.md`
  - `remaining-issues-list.md`
- No code, CSV, figure, LaTeX, or PDF changes.

### 4. Commit directly on `main`

Make one documentation-only commit:

```bash
git add no-test-decode-issues.md paper-submission-audit.md remaining-issues-list.md
git commit -m "docs: consolidate decoder audit resolution"
```

Do not create another branch, PR, worktree, stash, or decoder implementation plan.

### 5. Publish the accumulated repair

After confirming the final diff and clean working tree, push `main` to
`origin/main` so the decoder repair and its audit record are no longer local-only:

```bash
git push origin main
```

Report the final commit hash, test result, push result, and clean working-tree status.
