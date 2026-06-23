---
name: pymol_viz
description: >-
  How kinase structure visualization works in this repo: the
  StructureConfig/StructureVisualizer/PyMOLGenerator pipeline, the
  generate_pymol_files CLI, and when to use generated scripts (licensed PyMOL)
  vs in-process pymol2/SASA. Use when generating or editing PyMOL files,
  structure highlighting configs, or SASA computation.
---

# PyMOL / structure visualization

## Two distinct PyMOL paths — do not conflate

- **Rendering / images** → generate a standalone PyMOL **script** (+ PDB +
  instructions) and run it in the user's **licensed PyMOL GUI** for image
  quality. This is what `mkt.databases.pymol.PyMOLGenerator` and the
  `generate_pymol_files` CLI produce.
- **Numeric/compute work** (e.g. SASA in `mkt.databases.sasa`) → use in-process
  open-source `pymol2` only. Open-source PyMOL is for computation, not
  rendering.

## Pipeline

```
SequenceAlignment  →  StructureConfig (mkt.databases.app.schema)
   (kinase info,        ├─ generate_list_idx() / generate_style_color_lists()
    dict_align)         └─ produces list_idx (1-indexed), list_color, list_style, list_label
        ↓
StructureVisualizer (mkt.databases.app.structures)
   loads KinCore CIF → Bio.PDB Structure → PDB text; get_highlight_data()
        ↓
PyMOLGenerator (mkt.databases.pymol)
   writes <gene>_<attr>_{structure.pdb, pymol_script.py, instructions.txt}
```

Config types are the `StandardConfig` / `StandardConfigChoice` enums
(KLIFS_IMPORTANT, PHOSPHOSITES, MUTATIONS_KLIFS, MUTATIONS_GROUP, KLIFS_CUSTOM,
…). To add a highlight mode, subclass `StructureConfig` and implement
`generate_list_idx()` + `generate_style_color_lists()`.

## CLI

```bash
generate_pymol_files --gene ABL1 --config KLIFS_IMPORTANT
generate_pymol_files --gene EGFR --config PHOSPHOSITES
generate_pymol_files --gene ABL1 --config MUTATIONS_KLIFS --json-mutations muts.json
generate_pymol_files --gene ABL1 --config KLIFS_CUSTOM --indices 315,317 --colors red,blue
```

`MUTATIONS_*` configs require `--json-mutations`; `KLIFS_CUSTOM` requires
matched `--indices` (1-indexed full-length UniProt) and `--colors`. Output lands
in `<repo_root>/images/pymol_output/<gene>/<subdir>/` by default.

## Label tuning — do NOT add CLI flags for it

Fine label placement (`label_offset`, `label_min_dist`, `label_spring_strength`,
connector lines) is tuned by the user in the PyMOL GUI, not exposed as new CLI
flags. Leave those as `StructureConfig` defaults.
