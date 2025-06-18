import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Patch
from mkt.databases.colors import DICT_BIOCHEM_PROP_COLORS, DICT_KINASE_GROUP_COLORS
from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS
from mkt.databases.utils import add_one_hot_encoding_to_dataframe
from mkt.schema import io_utils

logger = logging.getLogger(__name__)


@dataclass
class PoissonRegressionNoInteraction:
    """Class for Poisson regression without interaction terms."""

    df: pd.DataFrame
    """DataFrame containing the filtered missense kinase mutations and all columns to be one-hot encoded."""
    list_oh_cols: list[str] = field(
        default_factory=lambda: ["gene_hugoGeneSymbol", "klifs_region"]
    )
    """List of columns to be one-hot encoded."""
    list_drop_cols: list[str, str] = field(
        default_factory=lambda: ["gene_PLK2", "klifs_I:1"]
    )
    """List of columns to be dropped from the DataFrame."""
    df_onehot: pd.DataFrame = field(init=False)
    """DataFrame containing the one-hot encoded columns."""
    col_endog: str = "counts"
    """Column name for the endog/dependent variable."""
    df_count: pd.DataFrame = field(init=False)
    """DataFrame containing the counts of each kinase group."""
    model: Any | None = field(init=False, default=None)
    """Results of the Poisson regression model."""
    results: Any | None = field(init=False, default=None)
    """Fitted results of the Poisson regression model."""

    def __post_init__(self):
        if self.df.empty:
            logger.error("Input DataFrame is empty. Please provide a valid DataFrame.")

        self.df_onehot = self.return_onehot_dataframe()
        self.df_count = self.return_count_dataframe()

    def return_onehot_dataframe(self) -> pd.DataFrame:
        """Return the one-hot encoded DataFrame."""
        list_prefix = [col.split("_")[0] for col in self.list_oh_cols]

        df_onehot = add_one_hot_encoding_to_dataframe(
            df=self.df,
            col_name=self.list_oh_cols,
            prefix=list_prefix,
            col_drop=self.list_drop_cols,
        )
        return df_onehot

    def return_count_dataframe(self) -> pd.DataFrame:
        """Return the count DataFrame."""
        df_count = self.df_onehot.value_counts().reset_index(name=self.col_endog)

        return df_count

    def fit(self) -> tuple[Any, Any]:
        """Run Poisson regression on the one-hot encoded DataFrame.

        Returns
        -------
        Any
            The fitted Poisson regression model.
        """
        if self.df_count.empty:
            logger.error(
                "One-hot encoded DataFrame is empty. Please check the input DataFrame."
            )

        X = self.df_count.drop(columns=[self.col_endog]).astype(float)
        X = sm.add_constant(X)
        y = self.df_count[self.col_endog].astype(float)

        self.model = sm.GLM(y, X, family=sm.families.Poisson())
        self.results = self.model.fit()

        return self.model, self.results

    def summary(self) -> Any:
        """Return the summary of the Poisson regression model."""
        if self.model is None:
            logger.error("Model has not been fitted yet. Please run `fit()` first.")

        return self.results.summary()

    def convert_coef2df(self) -> pd.DataFrame:
        """Convert the results of the Poisson regression model to a DataFrame."""

        df = pd.read_csv(io.StringIO(self.summary().tables[1].as_csv()))
        df = df.set_index(df.columns[0])
        df.columns = [i.strip() for i in df.columns]
        df.index = [i.strip() for i in df.index]

        return df

    def return_results_summary(self, filename: str | None = None) -> None:
        """Return the results summary of the Poisson regression and optionally save it to a file.

        Parameters
        ----------
        filename: str | None
            If provided, the summary will be saved to this file. If None, the summary will not be saved.

        Returns
        -------
        None
        """
        if filename is not None:
            plt.ioff()

        summary_text = str(self.summary())

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()

        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            plt.ion()
        else:
            plt.show()
            plt.close()

    def create_coefficient_barplot(
        self,
        col_plot: str,
        dict_color: dict[str, str],
        fn_color: Callable,
        col_color: str,
        str_xaxis: str,
        str_legend: str,
        bool_keep_sig: bool,
        p_value: float = 0.05,
        figsize: tuple[int, int] = (14, 8),
        xpos_legend: float = 0.9,
        use_symlog: bool = False,
    ) -> tuple:
        """Create a seaborn-styled barplot with error bars and significance asterisks

        Parameters:
        -----------
        col_plot: str
            Column prefix to plot (e.g., "gene_", "klifs_", "_")
        dict_color: dict[str, str]
            Color to shade barplots
        fn_color: Callable
            Function to map colors to the DataFrame; should take a DataFrame and return a dict
        col_color: str
            Column to use for shading bars (should match keys of dict_color)
        str_xaxis: str
            Label for x-axis
        str_legend: str
            Legend title
        bool_keep_sig: bool
            Whether to keep only significant results based on p_value
        p_value: float
            Cut-off significance value for astrices; default: 0.05
        figsize: tuple[int, int]
            Figure size; default: (14, 8)
        xpos_legend: float
            Position of legend in x direction; default: 0.9
        use_symlog: bool
            Whether to use symmetric log scale for better visualization of values near zero; default: False

        Returns
        -------
        tuple
            Containing fig and ax
        """
        df = self.convert_coef2df()
        if df.empty:
            logger.error("DataFrame is empty. Please run `fit()` first.")
        if not all(col in df.columns for col in ["coef", "[0.025", "0.975]", "P>|z|"]):
            logger.error(
                "DataFrame must contain 'coef', '[0.025', '0.975]', and 'P>|z|' columns for plotting."
            )
        if not any(df.index.map(lambda x: x.startswith(col_plot))):
            logger.error(
                f"Columns starting with '{col_plot}' not found in DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )

        # filter and sort DataFrame based on col_plot
        df = df.loc[df.index.map(lambda x: str(x).startswith(col_plot)), :]
        if bool_keep_sig:
            df = df[df["P>|z|"] < p_value].sort_values(by="coef", ascending=False)
        df = df.sort_values(by="coef", ascending=False)
        # add any incremental data needed for plotting
        if col_plot == "_":
            df.index = df.index.map(
                lambda x: x[1:].replace("_", " ").replace("-", ", ").title()
            )
        if col_color == "group":
            dict_kinase = io_utils.deserialize_kinase_dict()
            df["group"] = df.index.map(
                lambda x: dict_kinase[x.split("_")[1]].adjudicate_group()
            )

        if all(df.index.map(lambda x: "_" in x)):
            df["x_labels"] = df.index.map(lambda x: x.split("_")[1])
        else:
            df["x_labels"] = df.index.map(lambda x: x.split("_")[0])

        if col_color not in df.columns:
            logger.error(
                f"Column '{col_color}' not found in DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )

        dict_color = fn_color(df, dict_color)

        sns.set_style("white")
        sns.set_palette("husl")

        fig, ax = plt.subplots(figsize=figsize)

        # calculate error bars (distance from coefficient to CI bounds)
        lower_error = df["coef"] - df["[0.025"]
        upper_error = df["0.975]"] - df["coef"]

        # map colors based on group
        bar_colors = [dict_color.get(group, "#000000") for group in df[col_color]]

        # create bar positions
        x_pos = range(len(df))

        # create the barplot with seaborn styling
        bars = ax.bar(
            x_pos,
            df["coef"],
            color=bar_colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

        # add subtle gradient effect to bars
        for bar, color in zip(bars, bar_colors):
            # Add a subtle shadow effect
            bar.set_edgecolor("darkgrey")
            bar.set_linewidth(0.8)

        # add error bars with lighter, thinner styling
        ax.errorbar(
            x_pos,
            df["coef"],
            yerr=[lower_error, upper_error],
            fmt="none",
            color="#666666",
            capsize=3,
            capthick=1,
            elinewidth=1,
            alpha=0.7,
            zorder=4,
        )

        # add significance asterisks
        dist_star = 0.05
        if p_value and not bool_keep_sig:
            for i, (idx, row) in enumerate(df.iterrows()):
                stars = "*" if row["P>|z|"] < p_value else ""
                if stars:
                    coef_val = row["coef"]
                    if coef_val >= 0:
                        # for positive values, place above the upper whisker
                        y_pos = row["0.975]"] + dist_star
                    else:
                        # for negative values, place below the lower whisker
                        y_pos = row["[0.025"] - dist_star

                    ax.text(
                        i,
                        y_pos,
                        stars,
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                        color="black",
                    )

        ax.set_xlabel(f"{str_xaxis}", fontsize=14, fontweight="bold", color="#2E2E2E")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            df["x_labels"].apply(
                lambda x: x.replace(", ", ",\n") if len(x) > 15 else x
            ),
            rotation=90,
            ha="center",
            fontsize=10,
            color="#2E2E2E",
        )
        ax.tick_params(axis="y", labelsize=11, colors="#2E2E2E")

        if use_symlog:
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylabel(
                "Coefficient (symlog scale)",
                fontsize=14,
                fontweight="bold",
                color="#2E2E2E",
            )
        else:
            ax.set_ylabel(
                "Coefficient", fontsize=14, fontweight="bold", color="#2E2E2E"
            )

        ax.set_xlim(-0.5, len(df) - 0.5)
        ax.axhline(
            y=0, color="black", linestyle="-", alpha=0.6, linewidth=1.2, zorder=1
        )
        ax.grid(False)

        # set up legend
        dict_color_rev = {
            k: v for k, v in dict_color.items() if k in df[col_color].tolist()
        }
        dict_color_rev = self.revise_dict_col(dict_color_rev)
        legend_elements = [
            Patch(
                facecolor=v,
                edgecolor="white",
                linewidth=1.2,
                label=k,
            )
            for k, v in sorted(dict_color_rev.items())
        ]
        legend = ax.legend(
            handles=legend_elements,
            title=f"{str_legend}",
            title_fontsize=12,
            fontsize=11,
            loc="upper left",
            bbox_to_anchor=(xpos_legend, 1),
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        legend.get_title().set_fontweight("bold")

        plt.tight_layout()

        if p_value:
            set_sig_levels = set(
                df["P>|z|"].apply(lambda x: "*" if x < p_value else "").tolist()
            )
            set_sig_levels.discard("")
            if set_sig_levels:
                sig_text = "Significance: * p<0.05"
                plt.figtext(
                    0.02, 0.02, sig_text, fontsize=10, style="italic", color="#2E2E2E"
                )

        # remove top and right spines for cleaner look
        sns.despine()

        return fig, ax

    @staticmethod
    def revise_dict_col(dict_in):
        """Revise the input dictionary to ensure keys are formatted correctly for clean legend.

        Parameters
        ----------
        dict_in : dict
            Input dictionary with keys that may contain multiple values separated by commas.

        Returns
        -------
        dict
            Revised dictionary with keys formatted as single values or grouped by color.
        """
        if all([":" in k for k in dict_in.keys()]):
            # KLIFS keys
            dict_temp = {k.split(":")[0]: v for k, v in dict_in.items()}
            set_colors = set(dict_temp.values())
            list_keys = []
            for color in set_colors:
                list_temp = [k for k, v in dict_temp.items() if v == color]
                list_temp.sort()
                list_keys.append(", ".join(list_temp))
            return dict(zip(list_keys, set_colors))
        else:
            # kinase and biochem properties keys
            return {k.split(" ")[0]: v for k, v in dict_in.items()}

    @staticmethod
    def return_klifs_dict_colors(df_in, dict_in) -> dict[str, str]:
        """Return a dictionary of KLIFS region colors.

        Parameters
        ----------
        df_in : pd.DataFrame
            DataFrame containing the KLIFS regions.
        dict_in : dict[str, str]
            Dictionary mapping KLIFS regions to colors.

        Returns
        -------
        dict[str, str]
            Dictionary of KLIFS region colors.
        """
        index_temp = df_in.index.map(lambda x: x.split("_")[1])

        dict_out = dict(
            zip(index_temp, [dict_in[i.split(":")[0]]["color"] for i in index_temp])
        )

        return dict_out

    @staticmethod
    def return_biochem_dict_colors(df_in, dict_in) -> dict[str, str]:
        """Return a dictionary of biochemical property colors.

        Parameters
        ----------
        df_in : pd.DataFrame
            DataFrame containing the KLIFS regions.
        dict_in : dict[str, str]
            Dictionary mapping KLIFS regions to colors.

        Returns
        -------
        dict[str, str]
            Dictionary of biochemical property region colors.
        """
        dict_out = dict(
            zip(df_in.index, df_in.index.map(lambda x: dict_in[x.split(" ")[0]]))
        )
        return dict_out


DICT_BARPLOT_OPTIONS = {
    "kinase": {
        "col_plot": "gene_",
        "dict_color": DICT_KINASE_GROUP_COLORS,
        "fn_color": lambda x, y: y,
        "col_color": "group",
        "str_xaxis": "Kinases",
        "str_legend": "Kinase Groups",
        "bool_keep_sig": True,
    },
    "klifs": {
        "col_plot": "klifs_",
        "dict_color": DICT_POCKET_KLIFS_REGIONS,
        "fn_color": PoissonRegressionNoInteraction.return_klifs_dict_colors,
        "col_color": "x_labels",
        "str_xaxis": "KLIFS Residue",
        "xpos_legend": 0.85,
        "str_legend": "KLIFS Regions",
        "bool_keep_sig": True,
    },
    "biochem": {
        "col_plot": "_",
        "dict_color": DICT_BIOCHEM_PROP_COLORS,
        "fn_color": PoissonRegressionNoInteraction.return_biochem_dict_colors,
        "col_color": "x_labels",
        "str_xaxis": "Biochemical Properties",
        "str_legend": "KLIFS Regions",
        "bool_keep_sig": False,
    },
}
