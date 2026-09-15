---
name: plotting
description: >-
  missense-kinase-toolkit-specific plotting conventions; extends my central
  `plotting` skill. Use whenever writing or editing figure code (e.g.
  `mkt.databases.plot`) or generating project figures.
---

# missense-kinase-toolkit plotting conventions

## Baseline — fetch first

Apply my canonical `plotting` conventions (Arial, NPG palette, fully-opaque
colors for PowerPoint/PDF export, spine removal, high-DPI saving) before the
repo-specific notes below. WebFetch and follow:

https://raw.githubusercontent.com/jessicaw9910/skills/main/.claude/skills/plotting/SKILL.md

If the fetch fails (no network / non-200), **tell me the central `plotting`
skill could not be retrieved** and confirm how to proceed — do not silently skip
the baseline.

## Repo-specific additions

- Plotting code lives mainly in `mkt.databases.plot` / `mkt.databases.plot_config`.
- Palette / color-map sources:
  - `LIST_NPG_COLORS` — `mkt.databases.colors` (the NPG categorical palette).
  - `DICT_POCKET_KLIFS_REGIONS` — `mkt.databases.klifs` (KLIFS region colors).
  - `DICT_KINASE_GROUP_COLORS` — `mkt.schema.constants` (kinase-group colors).
- See CLAUDE.md → "PowerPoint/PDF Transparency Best Practices" for additional
  repo context.
