---
name: app_integration
description: >-
  How the Streamlit app and the reusable mkt.databases.app package fit together
  (SequenceAlignment, StructureConfig/Visualizer, PropertyTables) and the
  schema → databases → app dependency chain. Use when editing the Streamlit app
  or the mkt.databases.app modules, or wiring visualization logic between them.
---

# App integration (mkt.databases.app ↔ Streamlit app)

## Two pieces, clear split

- **`mkt.databases.app`** (importable, shipped with `mkt-databases`) holds the
  reusable, framework-agnostic visualization logic:
  - `sequences.py` — `SequenceAlignment` (kinase info + aligned sequences,
    `dict_align`).
  - `schema.py` — `StructureConfig` (abstract) + the `StandardConfig` /
    `StandardConfigChoice` config enums; uses dependency injection (takes a
    `SequenceAlignment`).
  - `structures.py` — `StructureVisualizer` (CIF → Bio.PDB → PDB text,
    `get_highlight_data()`).
  - `properties.py` — `PropertyTables`.
  - `utils.py` — `create_structure_visualizer`, `validate_uniprot_indices`.
- **`missense_kinase_toolkit/app/`** is the Streamlit front-end (not
  pip-installable, deployed at https://mkt-app.streamlit.app). `app.py` /
  `visualizers.py` **import from `mkt.databases.app`** and render with Streamlit.

## Rule: logic in the package, UI in Streamlit

Put reusable data/structure/sequence logic in `mkt.databases.app` so both the
Streamlit app and the `generate_pymol_files` CLI can share it (the CLI imports
the same `SequenceAlignment` / `StructureConfig` / `create_structure_visualizer`
— see the `pymol_viz` skill). Keep Streamlit-specific widgets/layout in
`missense_kinase_toolkit/app/`. When adding a feature used by both, add it to
the package and call it from the front-end, not the reverse.

## Dependency chain

```
mkt-schema  →  mkt-databases (incl. mkt.databases.app)  →  Streamlit app
```

The Streamlit app depends on both `mkt-schema` and `mkt-databases`; never make
`mkt.databases.app` import from the Streamlit `app/` directory.
