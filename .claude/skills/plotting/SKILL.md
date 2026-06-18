---
name: plotting
description: >-
  Conventions for creating or editing matplotlib/seaborn figures in
  missense-kinase-toolkit (Arial font, Nature/NPG palette LIST_NPG_COLORS,
  PowerPoint/PDF-safe transparency, spine removal). Use whenever writing or
  modifying plotting/figure code (e.g. mkt.databases.plot) or generating
  project figures.
---

# missense-kinase-toolkit plotting conventions

Apply these whenever you create or modify a figure in this repo. Plotting code
lives mainly in `mkt.databases.plot` / `mkt.databases.plot_config`; the color
palette lives in `mkt.databases.colors` (`LIST_NPG_COLORS`).

## Font + palette

```python
import seaborn as sns
import matplotlib.pyplot as plt
from mkt.databases.colors import LIST_NPG_COLORS

plt.rcParams["font.family"] = "Arial"
sns.set_palette(sns.color_palette(LIST_NPG_COLORS))
sns.set_style("white")
```

For KLIFS region colors use `DICT_POCKET_KLIFS_REGIONS` (from
`mkt.databases.klifs`); for kinase-group colors use `DICT_KINASE_GROUP_COLORS`
(from `mkt.schema.constants`).

## Transparency — PowerPoint/PDF safe (critical)

These figures get dropped into PowerPoint and exported to PDF, where SVG
transparency (`opacity` / `fill-opacity` < 1) renders as a **grey hue**. Prefer
one of these (see CLAUDE.md "PowerPoint/PDF Transparency Best Practices"):

- **Recommended:** bake alpha into RGBA tuples rather than a global `alpha=`:

  ```python
  import matplotlib.colors as mcolors

  facecolor_rgba = mcolors.to_rgba("#FFD700", alpha=0.5)
  ellipse = Ellipse((x, y), w, h, facecolor=facecolor_rgba, edgecolor="#FFD700")
  ax.legend(..., framealpha=1.0, facecolor="white", edgecolor="black")
  ```

- Or save a high-DPI PNG for slides:
  `fig.savefig("plot.png", dpi=300, bbox_inches="tight", facecolor="white")`.
- Or, if SVG is required, set `mpl.rcParams["svg.fonttype"] = "none"` and
  `mpl.rcParams["svg.hashsalt"] = "42"`, then in PowerPoint drag the SVG →
  right-click → Convert to Shape.

## Spines

Remove spines while keeping tick marks for a cleaner look:

```python
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
```
