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
