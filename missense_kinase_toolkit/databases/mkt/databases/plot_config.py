"""Configuration dataclasses for plot_dataset_data.py.

Loads plot aesthetics and data source paths from YAML config files
via OmegaConf, following the pattern used in mkt_impact.
"""

from dataclasses import dataclass, field
from pathlib import Path

import seaborn as sns
from mkt.schema.constants import DICT_KINASE_GROUP_COLORS
from omegaconf import OmegaConf

# --- matplotlib rcParams ---


@dataclass
class MatplotlibRCConfig:
    """Global matplotlib rcParams applied before any plot."""

    svg_fonttype: str = "path"
    pdf_fonttype: int = 42
    text_usetex: bool = False


# --- family color palette ---


@dataclass
class FamilyColorConfig:
    """Color palette for kinase families.

    Two modes:
      1. ``use_kinase_group_colors=True`` (default): uses ``DICT_KINASE_GROUP_COLORS``
         from ``mkt.schema.constants`` — a curated, colorblind-friendly mapping.
      2. ``use_kinase_group_colors=False``: builds colors from a seaborn palette
         (``palette_name`` / ``palette_n_colors``) with ``other_color`` for "Other".

    In both modes, ``families`` controls which families appear and their order.
    When ``families`` is None, the keys of ``DICT_KINASE_GROUP_COLORS`` are used.
    """

    use_kinase_group_colors: bool = True
    palette_name: str = "tab10"
    palette_n_colors: int = 10
    other_color: str = "#808080"
    families: list[str] | None = None

    def get_colors(self) -> dict:
        """Return a dict mapping family name to color.

        Returns:
        --------
        dict
            Mapping of family names to color values.
        """
        if self.use_kinase_group_colors:
            # start from the curated constant dict
            base = dict(DICT_KINASE_GROUP_COLORS)
            if self.families is not None:
                # filter + reorder to only the requested families
                return {f: base.get(f, self.other_color) for f in self.families}
            return base

        # seaborn palette mode
        families = self.families
        if families is None:
            # fall back to curated dict when no explicit families given
            return dict(DICT_KINASE_GROUP_COLORS)

        colors = sns.color_palette(self.palette_name, n_colors=self.palette_n_colors)

        family_colors = {}
        non_other = [f for f in families if f != "Other"]
        for family in families:
            if family == "Other":
                family_colors[family] = self.other_color
            else:
                idx = non_other.index(family)
                family_colors[family] = colors[idx]

        return family_colors


# --- col_kinase / sequence-type colors (shared by venn + boxplot) ---


@dataclass
class ColKinaseColorConfig:
    """RGB colors for sequence-type categories."""

    construct_unaligned: list[float] = field(default_factory=lambda: [242, 101, 41])
    klifs_region_aligned: list[float] = field(default_factory=lambda: [0, 51, 113])
    klifs_residues_only: list[float] = field(default_factory=lambda: [88, 152, 255])

    def as_rgb_dict(self) -> dict[str, tuple[float, float, float]]:
        """Return colors as 0-1 scaled RGB tuples keyed by category name."""
        return {
            "construct_unaligned": tuple(v / 255 for v in self.construct_unaligned),
            "klifs_region_aligned": tuple(v / 255 for v in self.klifs_region_aligned),
            "klifs_residues_only": tuple(v / 255 for v in self.klifs_residues_only),
        }


# --- per-plot configs ---


@dataclass
class DynamicRangePlotConfig:
    """Aesthetics for the dynamic-range histogram."""

    figsize: list[float] = field(default_factory=lambda: [11, 6])
    font_size: int = 14
    axes_titlesize: int = 16
    axes_labelsize: int = 14
    figure_titlesize: int = 20
    alpha: float = 0.25
    bins: int = 100
    color_pkis2: str = "blue"
    color_davis: str = "green"
    axvline_x: float = 99
    axvline_color: str = "red"
    title_fontsize: int = 20
    title_fontweight: str = "bold"
    title_y: float = 1.25
    subtitle_fontsize: int = 16
    subtitle_alpha: float = 0.75
    subtitle_y: float = 1.16
    axis_label_fontsize: int = 16
    tick_labelsize: int = 14
    filename: str = "dynamic_range_histogram"


@dataclass
class RidgelinePlotConfig:
    """Aesthetics for the ridgeline plot."""

    figsize: list[float] = field(default_factory=lambda: [10.5, 7.5])
    overlap: float = 0.1
    scale: float = 1.5
    fill_alpha: float = 0.5
    edgecolor: str = "black"
    edge_linewidth: float = 1.5
    baseline_linewidth: float = 0.5
    baseline_alpha: float = 0.3
    ytick_fontsize: int = 20
    title_fontsize: int = 22
    title_fontweight: str = "bold"
    title_color_davis: str = "black"
    title_color_pkis2: str = "black"
    xtick_fontsize: int = 18
    xlabel_fontsize: int = 20
    xlabel_text: str = "% of RefSeq sequence contained in construct"
    grid_alpha: float = 0.3
    filename: str = "ridgeline_plot"


@dataclass
class StackedBarchartConfig:
    """Aesthetics for the stacked bar chart."""

    figsize_width_per_source: float = 12
    figsize_height: float = 7
    layout_nrows: int = 1
    stack_color_true: str = "#d3d3d3"
    stack_color_false: str = "#505050"
    bar_edgecolor: str = "black"
    bar_linewidth: float = 0.5
    bar_alpha: float = 1.0
    pct_label_fontsize: int = 20
    pct_label_fontweight: str = "bold"
    pct_label_min_threshold: float = 5
    xtick_fontsize: int = 16
    xlabel_fontsize: int = 24
    ylabel_fontsize: int = 24
    title_fontsize: int = 26
    title_fontweight: str = "bold"
    ytick_fontsize: int = 18
    ylim_max: float = 105
    legend_fontsize: int = 20
    legend_title_fontsize: int = 20
    legend_bbox_y: float = -0.1
    bottom_adjust: float = 0.2
    title_color_davis: str = "black"
    title_color_pkis2: str = "black"
    filename: str = "stacked_barchart"


@dataclass
class VennDiagramConfig:
    """Aesthetics for the Venn diagram."""

    figsize: list[float] = field(default_factory=lambda: [8, 8])
    circle_alpha: float = 0.6
    intersection_color: str = "lightgray"
    intersection_alpha: float = 0.4
    set_label_fontsize: int = 16
    set_label_fontweight: str = "bold"
    subset_label_fontsize: int = 14
    title_fontsize: int = 22
    title_fontweight: str = "bold"
    filename: str = "venn_diagram"


@dataclass
class MetricsBoxplotConfig:
    """Aesthetics for the metrics boxplot."""

    figsize: list[float] = field(default_factory=lambda: [12, 4.5])
    box_widths: float = 0.6
    box_alpha: float = 0.7
    median_color: str = "black"
    median_linewidth: int = 2
    whisker_color: str = "black"
    whisker_linewidth: float = 1.5
    cap_color: str = "black"
    cap_linewidth: float = 1.5
    jitter_std: float = 0.04
    jitter_alpha: float = 0.6
    jitter_size: int = 50
    jitter_color: str = "black"
    xtick_fontsize: int = 14
    ylabel_fontsize: int = 20
    ylabel_text: str = "MSE (Z-Score)"
    title_fontsize: int = 22
    title_fontweight: str = "bold"
    ytick_fontsize: int = 18
    grid_alpha: float = 0.3
    bracket_start_pct: float = 0.08
    bracket_spacing_pct: float = 0.15
    bracket_height_pct: float = 0.02
    bracket_linewidth: float = 1.5
    pvalue_fontsize: int = 14
    pvalue_fontweight: str = "bold"
    title_color_davis: str = "black"
    title_color_pkis2: str = "black"
    filename: str = "metrics_boxplot"


@dataclass
class SequenceSchematicConfig:
    """Aesthetics for the sequence input schematic."""

    figsize: list[float] = field(default_factory=lambda: [14, 7])
    rect_height: float = 0.6
    gap_color: str = "white"
    ellipsis_color: str = "#888888"
    ellipsis_fontsize: int = 10
    label_fontsize: int = 10
    title_fontsize: int = 12
    title_fontweight: str = "bold"
    panel_title_fontsize: int = 11
    panel_title_fontweight: str = "bold"
    n_show_start: int = 40
    n_show_end: int = 20
    n_ellipsis: int = 5
    filename: str = "sequence_input_schematic"


# --- data sources ---


@dataclass
class DataSourceConfig:
    """Paths to input data files (relative to repo root)."""

    davis_csv: str = "data/davis_data_processed.csv"
    pkis2_csv: str = "data/pkis2_data_processed.csv"
    metrics_csv: str = "data/2025_val_stable_metrics.csv"


# --- output ---


@dataclass
class OutputConfig:
    """Output directory settings."""

    subdir: str = "images"
    bool_svg: bool = True
    bool_png: bool = True


# --- top-level config ---


@dataclass
class PlotDatasetConfig:
    """Top-level config aggregating all sub-configs."""

    matplotlib_rc: MatplotlibRCConfig = field(default_factory=MatplotlibRCConfig)
    family_colors: FamilyColorConfig = field(default_factory=FamilyColorConfig)
    col_kinase_colors: ColKinaseColorConfig = field(
        default_factory=ColKinaseColorConfig
    )
    dynamic_range: DynamicRangePlotConfig = field(
        default_factory=DynamicRangePlotConfig
    )
    ridgeline: RidgelinePlotConfig = field(default_factory=RidgelinePlotConfig)
    stacked_barchart: StackedBarchartConfig = field(
        default_factory=StackedBarchartConfig
    )
    venn_diagram: VennDiagramConfig = field(default_factory=VennDiagramConfig)
    metrics_boxplot: MetricsBoxplotConfig = field(default_factory=MetricsBoxplotConfig)
    sequence_schematic: SequenceSchematicConfig = field(
        default_factory=SequenceSchematicConfig
    )
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PlotDatasetConfig":
        """Load a PlotDatasetConfig from a YAML file.

        Parameters:
        -----------
        config_path : str | Path
            Path to the YAML configuration file.

        Returns:
        --------
        PlotDatasetConfig
            Fully populated config instance.
        """
        omega = OmegaConf.load(config_path)
        raw = OmegaConf.to_container(omega, resolve=True)

        cfg = cls()
        if "matplotlib_rc" in raw:
            cfg.matplotlib_rc = MatplotlibRCConfig(**raw["matplotlib_rc"])
        if "family_colors" in raw:
            cfg.family_colors = FamilyColorConfig(**raw["family_colors"])
        if "col_kinase_colors" in raw:
            cfg.col_kinase_colors = ColKinaseColorConfig(**raw["col_kinase_colors"])
        if "dynamic_range" in raw:
            cfg.dynamic_range = DynamicRangePlotConfig(**raw["dynamic_range"])
        if "ridgeline" in raw:
            cfg.ridgeline = RidgelinePlotConfig(**raw["ridgeline"])
        if "stacked_barchart" in raw:
            cfg.stacked_barchart = StackedBarchartConfig(**raw["stacked_barchart"])
        if "venn_diagram" in raw:
            cfg.venn_diagram = VennDiagramConfig(**raw["venn_diagram"])
        if "metrics_boxplot" in raw:
            cfg.metrics_boxplot = MetricsBoxplotConfig(**raw["metrics_boxplot"])
        if "sequence_schematic" in raw:
            cfg.sequence_schematic = SequenceSchematicConfig(
                **raw["sequence_schematic"]
            )
        if "data_sources" in raw:
            cfg.data_sources = DataSourceConfig(**raw["data_sources"])
        if "output" in raw:
            cfg.output = OutputConfig(**raw["output"])

        return cfg
