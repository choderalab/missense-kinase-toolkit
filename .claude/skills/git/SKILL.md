---
name: git
description: >-
  missense-kinase-toolkit-specific git conventions; extends my central `git`
  skill. Use when creating commits, branches, or pull requests.
---

# missense-kinase-toolkit git & pull-request conventions

## Baseline — fetch first

Apply my canonical `git` conventions (branch first, scoped imperative commits,
Claude attribution, PR-template handling) before the repo-specific notes below.
WebFetch and follow:

https://raw.githubusercontent.com/jessicaw9910/skills/main/.claude/skills/git/SKILL.md

If the fetch fails (no network / non-200), **tell me the central `git` skill
could not be retrieved** and confirm how to proceed — do not silently skip the
baseline.

## Repo-specific additions

- This is a mono repo: scope each commit to one sub-package (`schema/`,
  `databases/`, `ml/`, `app/`) where practical — CI workflows are path-filtered
  per sub-package (see the `ci` skill).

## Refreshing a stale topic branch from `main`

I keep long-lived topic branches (e.g. `databases`, `ml`, `app`, `schema`,
`docs`, `cbioportal`, `oncokb`, `pkis`, `notebooks`) that can sit untouched for
months. When I pick one up to develop a feature, I want it synced with `main`
**accepting all incoming changes from `main` on conflict** (I stash unsaved work
first).

Use:

```bash
git switch <topic-branch>
git stash                 # if there are unsaved changes
git fetch origin
git merge -X theirs origin/main
git stash pop             # if you stashed
```

- `-X theirs` favors the **incoming** branch (`main`). During a merge while on
  the topic branch, `ours` = the topic branch and `theirs` = `main` — so
  `theirs` is correct for "take main's version." This only forces `main`'s
  version on **conflicting** hunks; non-conflicting topic-branch work is kept.
  It resolves binary-file conflicts (e.g. `KinaseInfo.tar.gz`) cleanly by taking
  `main`'s blob.
- This **updates the topic branch** so I can keep developing on a current base;
  the actual merge into `main` happens later via a PR.
- For a small, isolated fix, branch off `main` instead — these topic branches
  can be ~70 commits ahead, so a PR from one drags in unrelated commits.

## Splitting intermingled changes across topic branches

When one working tree has uncommitted changes spanning several sub-packages and
each belongs on a different stale topic branch, don't try to carry them across
`git switch` (stale branches diverge and the switch will conflict or refuse).
Instead, patch each set onto its freshly-synced branch:

1. Capture per-sub-package patches: `git diff -- <subpkg> > <subpkg>.patch`.
2. Reset the working tree: `git checkout -- <paths>` (untracked WIP files stay
   put and travel harmlessly across switches — confirm they aren't tracked on
   the target branch first).
3. For each branch: `git switch <branch>`, refresh from `main` (see above), then
   `git apply --3way <subpkg>.patch`, and commit.

Stage with `git add -u <subpkg>` (not `git add <subpkg>`, which also stages
untracked WIP files) so pre-commit only lints what you're committing.
