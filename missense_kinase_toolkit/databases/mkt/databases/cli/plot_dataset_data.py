#!/usr/bin/env python3

import logging
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mkt.databases import config
from mkt.databases.datasets.process import (
    generate_ridgeline_df,
    generate_stacked_barchart_df,
)
from mkt.databases.log_config import configure_logging
from mkt.schema.io_utils import get_repo_root

logger = logging.getLogger(__name__)


def get_family_colors():
    """Define a consistent color palette for kinase families.

    Returns a dictionary mapping family names to colors.
    Uses a vibrant, colorblind-friendly palette with 'Other' in grey.
    """
    families = [
        "AGC",
        "Atypical",
        "CAMK",
        "CK1",
        "CMGC",
        "Lipid",
        "NEK",
        "STE",
        "TK",
        "TKL",
        "Other",
    ]

    # more vibrant color palette (tab10 is more saturated than colorblind)
    # combination of tab10 and bright colors for better visibility
    colors = sns.color_palette("tab10", n_colors=10)

    family_colors = {}
    for i, family in enumerate(families):
        if family == "Other":
            family_colors[family] = "#808080"  # grey
        else:
            # map non-Other families to vibrant palette
            idx = [f for f in families if f != "Other"].index(family)
            family_colors[family] = colors[idx]

    return family_colors


def generate_venn_diagram_dict(df_in: pd.DataFrame) -> dict:
    """Generate a dictionary for Venn diagram plotting.

    Parameters:
    -----------
    df_in : pd.DataFrame
        DataFrame with columns: kinase_name, seq_construct_unaligned, seq_klifs_region_aligned, seq_klifs_residues_only

    Returns:
    --------
    dict_out : dict
        Dictionary with keys as sequence types and values as lists of kinase names.
    """

    dict_out = {
        "Construct Unaligned": df_in.loc[
            ~df_in["seq_construct_unaligned"].isna(), "kinase_name"
        ]
        .unique()
        .tolist(),
        "KLIFS Region Aligned": df_in.loc[
            ~df_in["seq_klifs_region_aligned"].isna(), "kinase_name"
        ]
        .unique()
        .tolist(),
        "Klifs Residues Only": df_in.loc[
            ~df_in["seq_klifs_residues_only"].isna(), "kinase_name"
        ]
        .unique()
        .tolist(),
    }

    return dict_out


def convert_to_percentile(input, orig_max=10) -> float:
    """Convert Kd values to percentile scale."""
    return (input / orig_max) * 100


def convert_from_percentile(input, orig_max=10, precision=3) -> float:
    """Convert percentile values back to Kd scale."""
    if precision is not None:
        try:
            return round((input / 100) * orig_max, precision)
        except TypeError:
            return [np.round(i, precision) for i in (input / 100) * orig_max]
    else:
        return (input / 100) * max(orig_max)


def plot_dynamic_range(df_davis, df_pkis2, output_path) -> None:
    """Create a histogram comparing dynamic assay ranges between Davis and PKIS2.

    Parameters:
    -----------
    df_davis : pd.DataFrame
        DataFrame with 'y' column containing Kd values
    df_pkis2 : pd.DataFrame
        DataFrame with 'y' column containing percent inhibition values
    output_path : str
        Path to save the plot files (will save both SVG and PNG)
    """
    import matplotlib
    import matplotlib.ticker

    # ensure vector output - disable rasterization
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["text.usetex"] = False  # Don't use LaTeX rendering

    # process Davis data
    col_davis_y = "y"
    col_davis_y_transformed = "y_trans"
    df_davis = df_davis.copy()
    df_davis[col_davis_y] = 10 ** (-df_davis[col_davis_y])
    df_davis[col_davis_y_transformed] = convert_to_percentile(df_davis[col_davis_y])

    # process PKIS2 data
    df_pkis2 = df_pkis2.copy()
    df_pkis2["1-Percent Inhibition"] = 100 - df_pkis2["y"]

    # calculate no binding percentages
    na_davis = (
        sum(df_davis[col_davis_y] == df_davis[col_davis_y].max()) / df_davis.shape[0]
    )
    na_pkis2 = sum(df_pkis2["y"] == 0) / df_pkis2.shape[0]

    _, ax1 = plt.subplots()
    plt.gcf().set_size_inches(11, 6)
    matplotlib.rcParams.update({"font.size": 14})
    matplotlib.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14})
    matplotlib.rcParams.update({"figure.titlesize": 20})

    alpha = 0.25

    sns.histplot(
        data=df_pkis2,
        x="1-Percent Inhibition",
        ax=ax1,
        bins=100,
        log=True,
        color="blue",
        alpha=alpha,
        label=f"PKIS2 (n={df_pkis2.shape[0]:,})",
    )

    sns.histplot(
        data=df_davis,
        x=col_davis_y_transformed,
        ax=ax1,
        bins=100,
        log=True,
        color="green",
        alpha=alpha,
        label=f"Davis (n={df_davis.shape[0]:,})",
    )

    ax1.xaxis.label.set_size(16)
    ax1.yaxis.label.set_size(16)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.set_xlabel("1-% inhibition (PKIS2)", color="blue", fontsize=16)
    ax1.axvline(x=99, color="red", linestyle="--")

    ax1.text(
        x=0.5,
        y=1.25,
        s="Comparing dynamic assay ranges",
        fontsize=20,
        weight="bold",
        ha="center",
        va="bottom",
        transform=ax1.transAxes,
    )

    ax1.text(
        x=0.5,
        y=1.16,
        s=f"No binding detected: {na_davis:.1%} Davis, {na_pkis2:.1%} PKIS2",
        fontsize=16,
        alpha=0.75,
        ha="center",
        va="bottom",
        transform=ax1.transAxes,
    )

    ax1.get_xaxis().set_major_formatter(lambda x, p: format(x / 100, ".0%"))
    ax2 = ax1.secondary_xaxis(
        "top", functions=(convert_from_percentile, convert_to_percentile)
    )
    ax2.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax2.xaxis.label.set_size(16)
    # Use mathtext for subscript and micro symbol
    ax2.set_xlabel(r"$\mathregular{K_d}$ ($\mu$M) (Davis)", color="green", fontsize=16)
    ax1.set_ylabel(r"$\mathregular{log_{10}}$(count)", fontsize=16)
    plt.legend(loc="upper left")
    plt.xlim(0, 100)
    plt.tight_layout()

    # save both SVG and PNG formats
    svg_path = (
        output_path.replace(".png", ".svg")
        if output_path.endswith(".png")
        else output_path
    )
    png_path = svg_path.replace(".svg", ".png")

    plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Dynamic range plot saved to {svg_path} and {png_path}")


def plot_ridgeline(df, output_path) -> None:
    """Create a ridgeline plot showing distribution of fraction_construct by family.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: kinase_name, family, fraction_construct, source
    output_path : str
        Path to save the SVG file
    """
    import matplotlib
    from scipy import stats

    # ensure vector output - disable rasterization
    matplotlib.rcParams["svg.fonttype"] = "none"  # Keep fonts as text, not paths
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts

    # remove rows with None values
    df_clean = df.dropna(subset=["family", "fraction_construct"])

    # get color palette
    family_colors = get_family_colors()

    # get unique sources
    sources = sorted(df_clean["source"].unique())

    # sort families by Davis median
    df_davis = df_clean[df_clean["source"] == "Davis"]
    family_medians = (
        df_davis.groupby("family")["fraction_construct"]
        .median()
        .sort_values(ascending=False)
    )
    families = family_medians.index.tolist()

    # create figure with subplots for each source (horizontal faceting)
    fig, axes = plt.subplots(1, len(sources), figsize=(10.5, 7.5), sharey=True)
    if len(sources) == 1:
        axes = [axes]

    for idx, source in enumerate(sources):
        ax = axes[idx]
        df_source = df_clean[df_clean["source"] == source]

        # count number of unique constructs
        n_constructs = df_source["kinase_name"].nunique()

        # create ridgeline plot for each family with reduced overlap
        overlap = 0.1
        scale = 1.5

        for i, family in enumerate(families):
            df_family = df_source[df_source["family"] == family]
            if len(df_family) > 0:
                data = df_family["fraction_construct"].values

                # calculate kernel density estimate
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(0, 1, 200)  # Use full 0-1 range
                density = kde(x_range)

                # normalize and scale density
                density = density / density.max() * scale

                # plot the density as a filled curve with overlap
                # convert to percent for plotting
                y_base = i * (1 - overlap)
                x_range_pct = x_range * 100
                ax.fill_between(
                    x_range_pct,
                    y_base,
                    y_base + density,
                    color=family_colors.get(family, "grey"),
                    alpha=0.5,
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=len(families) - i,
                )

                # add a baseline for reference
                ax.plot(
                    x_range_pct,
                    [y_base] * len(x_range_pct),
                    color="black",
                    linewidth=0.5,
                    alpha=0.3,
                )

        # set labels and title with construct count
        y_positions = [i * (1 - overlap) for i in range(len(families))]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(families, fontsize=20)  # 10 * 1.3 ≈ 16
        ax.set_title(
            f"{source} (n={n_constructs})", fontsize=22, fontweight="bold"
        )  # 14 * 1.3 ≈ 18
        ax.tick_params(axis="x", labelsize=18)  # x-axis tick labels

        # set y limits with padding
        ax.set_ylim(-0.5, max(y_positions) + scale + 0.5)
        ax.set_xlim(0, 100)  # 0-100 percent range
        ax.grid(axis="x", alpha=0.3)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # only show y-axis labels on the leftmost plot
    for idx in range(1, len(axes)):
        axes[idx].set_ylabel("")
        axes[idx].tick_params(left=False)

    # add shared x-axis label
    fig.text(
        0.5,
        0.02,
        "% of RefSeq sequence contained in construct",
        ha="center",
        fontsize=20,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for shared x-axis label

    # Save both SVG and PNG formats
    svg_path = (
        output_path.replace(".png", ".svg")
        if output_path.endswith(".png")
        else output_path
    )
    png_path = svg_path.replace(".svg", ".png")

    plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Ridgeline plot saved to {svg_path} and {png_path}")


def plot_stacked_barchart(df, output_path) -> None:
    """Create a stacked bar chart showing counts by family.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: family, bool_uniprot2refseq, count, source
    output_path : str
        Path to save the SVG file
    """
    import matplotlib

    # ensure vector output - disable rasterization
    matplotlib.rcParams["svg.fonttype"] = "none"  # Keep fonts as text, not paths
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts

    # remove rows with None values
    df_clean = df.dropna(subset=["family"])

    # get color palette
    family_colors = get_family_colors()

    # pivot data for stacking
    sources = sorted(df_clean["source"].unique())

    # create figure with subplots for each source - match ridgeline aspect ratio
    fig, axes = plt.subplots(1, len(sources), figsize=(12 * len(sources), 7))
    if len(sources) == 1:
        axes = [axes]

    for idx, source in enumerate(sources):
        ax = axes[idx]
        df_source = df_clean[df_clean["source"] == source]

        # pivot to get families vs bool_uniprot2refseq
        df_pivot = df_source.pivot_table(
            index="family", columns="bool_uniprot2refseq", values="count", fill_value=0
        )

        # calculate percentages (sum across bool_uniprot2refseq for each family = 100%)
        df_pivot_pct = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

        # calculate total counts for each family
        df_counts = df_pivot.sum(axis=1)

        # total number of constructs is sum of all counts
        n_constructs = int(df_counts.sum())

        # sort families by highest % False to lowest % False
        if False in df_pivot_pct.columns:
            family_order = (
                df_pivot_pct[False].sort_values(ascending=False).index.tolist()
            )
        else:
            family_order = sorted(df_pivot_pct.index.tolist())

        # prepare data for stacked bar chart
        x_pos = np.arange(len(family_order))

        # colors for True/False stacks (greyscale: lighter for True, darker for False)
        stack_colors = {"True": "#d3d3d3", "False": "#505050"}

        # plot False on top, True on bottom (reverse order)
        bool_values = [True, False]

        bottom = np.zeros(len(family_order))
        for bool_val in bool_values:
            if bool_val in df_pivot_pct.columns:
                values = [
                    df_pivot_pct.loc[fam, bool_val] if fam in df_pivot_pct.index else 0
                    for fam in family_order
                ]

                bars = ax.bar(
                    x_pos,
                    values,
                    bottom=bottom,
                    color=stack_colors[str(bool_val)],
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=1.0,
                )

                # add percentage labels - white for False, black for True
                text_color = "white" if bool_val is False else "black"
                for i, (bar, val) in enumerate(zip(bars, values)):
                    if val > 5:  # Only show label if segment is large enough
                        height = bar.get_height()
                        # round to nearest percent
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bottom[i] + height / 2.0,
                            f"{val:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=20,
                            fontweight="bold",  # 9 * 1.3 ≈ 12
                            color=text_color,
                        )

                bottom += values

        # create x-axis labels with family colors and counts
        family_labels = []
        for fam in family_order:
            count = int(df_counts[fam]) if fam in df_counts else 0
            family_labels.append(f"{fam}\n(n={count})")

        # set labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels(family_labels, ha="center", fontsize=16)  # 10 * 1.3 ≈ 13

        # color the x-axis labels
        for i, (tick_label, fam) in enumerate(zip(ax.get_xticklabels(), family_order)):
            tick_label.set_color(family_colors.get(fam, "grey"))
            tick_label.set_fontweight("bold")

        ax.set_xlabel("Kinase Family", fontsize=24)
        ax.set_ylabel("Percentage (%)", fontsize=24)
        ax.set_title(f"{source} (n={n_constructs})", fontsize=26, fontweight="bold")
        # y-axis tick labels
        ax.tick_params(axis="y", labelsize=18)
        # extra space at top for legend
        ax.set_ylim(0, 105)

    # create shared legend patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=stack_colors["True"], edgecolor="black", label="True"),
        Patch(facecolor=stack_colors["False"], edgecolor="black", label="False"),
    ]

    # add legend outside plot area at the bottom center
    fig.legend(
        handles=legend_elements,
        title="RefSeq sequence identical to UniProt sequence",
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=20,
        title_fontsize=20,
        bbox_to_anchor=(0.5, -0.1),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # save both SVG and PNG formats
    svg_path = (
        output_path.replace(".png", ".svg")
        if output_path.endswith(".png")
        else output_path
    )
    png_path = svg_path.replace(".svg", ".png")

    plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Stacked bar chart saved to {svg_path} and {png_path}")


def plot_venn_diagram(df, output_path, source_name) -> None:
    """Create a Venn diagram showing overlap of kinases across different sequence types.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: kinase_name, seq_construct_unaligned, seq_klifs_region_aligned, seq_klifs_residues_only
    output_path : str
        Path to save the SVG file
    source_name : str
        Name of the source (e.g., 'Davis', 'PKIS2')
    """
    import matplotlib
    from matplotlib_venn import venn3

    # ensure vector output - disable rasterization
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42

    # generate the Venn diagram dictionary
    venn_dict = generate_venn_diagram_dict(df)

    # convert lists to sets for Venn diagram
    set_construct_unaligned = set(venn_dict["Construct Unaligned"])
    set_klifs_region_aligned = set(venn_dict["KLIFS Region Aligned"])
    set_klifs_residues_only = set(venn_dict["Klifs Residues Only"])

    # define colors (RGB converted to 0-1 scale) - matching boxplot colors
    colors = {
        "construct_unaligned": (242 / 255, 101 / 255, 41 / 255),
        "klifs_region_aligned": (0 / 255, 51 / 255, 113 / 255),
        "klifs_residues_only": (88 / 255, 152 / 255, 255 / 255),
    }

    # create figure
    _, ax = plt.subplots(figsize=(8, 8))

    # create Venn diagram
    venn = venn3(
        [set_construct_unaligned, set_klifs_region_aligned, set_klifs_residues_only],
        set_labels=(
            "Construct Unaligned",
            "KLIFS Region Aligned",
            "KLIFS Residues Only",
        ),
        ax=ax,
    )

    # color the circles
    if venn.get_patch_by_id("100"):
        venn.get_patch_by_id("100").set_color(colors["construct_unaligned"])
        venn.get_patch_by_id("100").set_alpha(0.6)
    if venn.get_patch_by_id("010"):
        venn.get_patch_by_id("010").set_color(colors["klifs_region_aligned"])
        venn.get_patch_by_id("010").set_alpha(0.6)
    if venn.get_patch_by_id("001"):
        venn.get_patch_by_id("001").set_color(colors["klifs_residues_only"])
        venn.get_patch_by_id("001").set_alpha(0.6)

    # color intersections (lighter versions)
    for patch_id in ["110", "101", "011", "111"]:
        patch = venn.get_patch_by_id(patch_id)
        if patch:
            patch.set_color("lightgray")
            patch.set_alpha(0.4)

    # increase label font sizes
    for label in venn.set_labels:
        if label:
            label.set_fontsize(16)
            label.set_fontweight("bold")

    # increase count font sizes
    for label in venn.subset_labels:
        if label:
            label.set_fontsize(14)

    ax.set_title(f"{source_name} Kinase Coverage", fontsize=22, fontweight="bold")

    plt.tight_layout()

    # save both SVG and PNG formats
    svg_path = (
        output_path.replace(".png", ".svg")
        if output_path.endswith(".png")
        else output_path
    )
    png_path = svg_path.replace(".svg", ".png")

    plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Venn diagram saved to {svg_path} and {png_path}")


def plot_metrics_boxplot(df, output_path) -> None:
    """Create a boxplot with jitter showing MSE values by col_kinase and source.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: col_kinase, source, fold, avg_stable_epoch, mse
    output_path : str
        Path to save the SVG file
    """
    import matplotlib

    # ensure vector output - disable rasterization
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42

    # define colors for each col_kinase (RGB converted to 0-1 scale)
    col_kinase_colors = {
        "construct_unaligned": (242 / 255, 101 / 255, 41 / 255),
        "klifs_region_aligned": (0 / 255, 51 / 255, 113 / 255),
        "klifs_residues_only": (88 / 255, 152 / 255, 255 / 255),
    }

    # format col_kinase labels: remove underscores, title case, Klifs > KLIFS
    def format_col_kinase(col_name, avg_epoch):
        formatted = col_name.replace("_", " ").title()
        formatted = formatted.replace("Klifs", "KLIFS")
        return f"{formatted}\n(Epoch={int(avg_epoch)})"

    # get unique col_kinase values and their average stable epochs per source
    col_kinase_source_info = (
        df.groupby(["col_kinase", "source"])["avg_stable_epoch"].first().to_dict()
    )

    # get unique sources and sort
    sources = sorted(df["source"].unique())

    # calculate width ratios based on number of unique col_kinase values per source
    width_ratios = []
    for source in sources:
        df_source = df[df["source"] == source]
        n_col_kinases = df_source["col_kinase"].nunique()
        width_ratios.append(n_col_kinases)

    # create figure with subplots for each source - proportional widths, shared y-axis
    fig, axes = plt.subplots(
        1,
        len(sources),
        figsize=(12, 4.5),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    if len(sources) == 1:
        axes = [axes]

    for idx, source in enumerate(sources):
        ax = axes[idx]
        df_source = df[df["source"] == source]

        # get unique col_kinase values for this source
        col_kinases = sorted(df_source["col_kinase"].unique())

        # prepare data for boxplot
        data_for_boxplot = []
        labels = []
        colors = []
        for col_kinase in col_kinases:
            df_col = df_source[df_source["col_kinase"] == col_kinase]
            data_for_boxplot.append(df_col["mse"].values)
            # Get the correct average epoch for this col_kinase and source combination
            avg_epoch = col_kinase_source_info[(col_kinase, source)]
            labels.append(format_col_kinase(col_kinase, avg_epoch))
            colors.append(col_kinase_colors.get(col_kinase, (0.5, 0.5, 0.5)))

        # create boxplot
        bp = ax.boxplot(
            data_for_boxplot,
            positions=range(len(col_kinases)),
            widths=0.6,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black", linewidth=1.5),
            capprops=dict(color="black", linewidth=1.5),
        )

        # color each box
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # addd jittered points
        for i, col_kinase in enumerate(col_kinases):
            df_col = df_source[df_source["col_kinase"] == col_kinase]
            y = df_col["mse"].values
            # add jitter to x positions
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=50, color="black", zorder=3)

        # set labels and title
        ax.set_xticks(range(len(col_kinases)))
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylabel("MSE (Z-Score)", fontsize=20)
        ax.set_title(
            source.title() if source == "davis" else source.upper(),
            fontsize=22,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(axis="y", alpha=0.3)

    # store comparison info for later (will draw after all subplots are created)
    comparison_info = []
    for idx, source in enumerate(sources):
        ax = axes[idx]
        df_source = df[df["source"] == source]
        col_kinases = sorted(df_source["col_kinase"].unique())

        # add p-value comparisons for Davis dataset only
        if source == "davis" and "construct_unaligned" in col_kinases:
            from scipy import stats as scipy_stats

            # get data for construct_unaligned
            df_construct_unaligned = df_source[
                df_source["col_kinase"] == "construct_unaligned"
            ]
            data_construct_unaligned = df_construct_unaligned["mse"].values

            # position of construct_unaligned
            pos_construct_unaligned = col_kinases.index("construct_unaligned")

            # comparisons to make
            comparisons = []
            if "klifs_region_aligned" in col_kinases:
                df_klifs_region = df_source[
                    df_source["col_kinase"] == "klifs_region_aligned"
                ]
                data_klifs_region = df_klifs_region["mse"].values
                pos_klifs_region = col_kinases.index("klifs_region_aligned")

                # Perform t-test
                t_stat, p_val = scipy_stats.ttest_ind(
                    data_construct_unaligned, data_klifs_region
                )
                comparisons.append((pos_construct_unaligned, pos_klifs_region, p_val))

            if "klifs_residues_only" in col_kinases:
                df_klifs_residues = df_source[
                    df_source["col_kinase"] == "klifs_residues_only"
                ]
                data_klifs_residues = df_klifs_residues["mse"].values
                pos_klifs_residues = col_kinases.index("klifs_residues_only")

                # perform t-test
                t_stat, p_val = scipy_stats.ttest_ind(
                    data_construct_unaligned, data_klifs_residues
                )
                comparisons.append((pos_construct_unaligned, pos_klifs_residues, p_val))

            comparison_info.append((idx, comparisons))

    # now draw comparison brackets after determining the shared y-axis limits
    if comparison_info:
        # get the maximum y value across all axes
        y_max = max(ax.get_ylim()[1] for ax in axes)
        y_min = min(ax.get_ylim()[0] for ax in axes)
        y_range = y_max - y_min

        # calculate bracket positions with increased spacing
        bracket_start = y_max + 0.08 * y_range  # Start 8% above the top
        bracket_spacing = 0.15 * y_range  # Increased spacing between brackets
        bracket_height = 0.02 * y_range  # Height of bracket

        for ax_idx, comparisons in comparison_info:
            ax = axes[ax_idx]

            for i, (pos1, pos2, p_val) in enumerate(comparisons):
                # calculate height for this comparison line
                y = bracket_start + i * bracket_spacing
                h = bracket_height

                # draw bracket
                ax.plot(
                    [pos1, pos1, pos2, pos2], [y, y + h, y + h, y], lw=1.5, c="black"
                )

                # format p-value
                if p_val < 0.001:
                    p_text = "p < 0.001"
                elif p_val < 0.01:
                    p_text = f"p = {p_val:.3f}**"
                elif p_val < 0.05:
                    p_text = f"p = {p_val:.2f}*"
                else:
                    p_text = f"p = {p_val:.2f}"

                # add p-value text with increased font size
                ax.text(
                    (pos1 + pos2) * 0.5,
                    y + h,
                    p_text,
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )

        # adjust y-axis limits for all axes to accommodate brackets
        new_y_max = (
            bracket_start
            + max(len(comps) for _, comps in comparison_info) * bracket_spacing
            + 0.1 * y_range
        )
        for ax in axes:
            ax.set_ylim(y_min, new_y_max)

    # only show y-axis label on the leftmost plot
    for idx in range(1, len(axes)):
        axes[idx].set_ylabel("")

    plt.tight_layout()

    # save both SVG and PNG formats
    svg_path = (
        output_path.replace(".png", ".svg")
        if output_path.endswith(".png")
        else output_path
    )
    png_path = svg_path.replace(".svg", ".png")

    plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Metrics boxplot saved to {svg_path} and {png_path}")


def main():
    """Main function to generate plots for dataset data."""

    configure_logging()

    try:
        config.set_request_cache(path.join(get_repo_root(), "requests_cache.sqlite"))
    except Exception as e:
        logger.warning(f"Failed to set request cache, using current directory: {e}")
        config.set_request_cache(path.join(".", "requests_cache.sqlite"))

    # load processed data
    df_davis = pd.read_csv(path.join(get_repo_root(), "data/davis_data_processed.csv"))
    df_pkis2 = pd.read_csv(path.join(get_repo_root(), "data/pkis2_data_processed.csv"))

    # generate ridgeline data
    df_davis_ridgeline = generate_ridgeline_df(df_davis, source="Davis")
    df_pkis2_ridgeline = generate_ridgeline_df(df_pkis2, source="PKIS2")
    df_ridgeline = pd.concat([df_davis_ridgeline, df_pkis2_ridgeline], axis=0)

    # generate stacked barchart data
    df_davis_stack = generate_stacked_barchart_df(df_davis, source="Davis")
    df_pkis2_stack = generate_stacked_barchart_df(df_pkis2, source="PKIS2")
    df_stack = pd.concat([df_davis_stack, df_pkis2_stack], axis=0)

    # create output directory if it doesn't exist
    output_dir = path.join(get_repo_root(), "images")
    if not path.exists(output_dir):
        import os

        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # generate and save plots
    ridgeline_path = path.join(output_dir, "ridgeline_plot.svg")
    plot_ridgeline(df_ridgeline, ridgeline_path)

    stacked_path = path.join(output_dir, "stacked_barchart.svg")
    plot_stacked_barchart(df_stack, stacked_path)

    # generate dynamic range comparison plot
    dynamic_range_path = path.join(output_dir, "dynamic_range_histogram.svg")
    plot_dynamic_range(df_davis, df_pkis2, dynamic_range_path)

    # load metrics data and generate boxplot
    metrics_path = path.join(get_repo_root(), "data/2025_val_stable_metrics.csv")
    if path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        boxplot_path = path.join(output_dir, "metrics_boxplot.svg")
        plot_metrics_boxplot(df_metrics, boxplot_path)
    else:
        logger.warning(f"Metrics file not found: {metrics_path}")

    # generate Venn diagrams for Davis and PKIS2
    venn_davis_path = path.join(output_dir, "venn_diagram_davis.svg")
    plot_venn_diagram(df_davis, venn_davis_path, "Davis")

    venn_pkis2_path = path.join(output_dir, "venn_diagram_pkis2.svg")
    plot_venn_diagram(df_pkis2, venn_pkis2_path, "PKIS2")

    logger.info("All plots generated successfully!")


if __name__ == "__main__":
    main()
