---
name: git
description: >-
  Git and pull-request conventions for the missense-kinase-toolkit repo (PR
  template, branching, commit attribution). Use when creating commits,
  branches, or pull requests.
---

# missense-kinase-toolkit git & pull-request conventions

## Branching

- Don't commit directly to the default branch (`main`). Branch first.
- Only commit or push when explicitly asked.

## Pull requests

Use `.github/PULL_REQUEST_TEMPLATE.md`, which has these sections:
- **Description** — the purpose of the PR.
- **Todos** — checklist of what was accomplished.
- **Questions** — open items / things to revisit.
- **Status** — readiness checkbox.

Fill in each section; leave the Status box reflecting the true state. Use the
`gh` CLI for GitHub operations (PRs, issues, API).

## Commits

- Keep commits scoped and message subjects in the imperative.
- Because this is a mono repo, scope each commit to one sub-package
  (`schema/`, `databases/`, `ml/`, `app/`) where practical — CI workflows are
  path-filtered per sub-package (see the `ci` skill).
- End commit messages with:

  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- End PR bodies with:

  ```
  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  ```
