import logging
import re
from dataclasses import dataclass

import streamlit as st
from constants import DICT_RESOURCE_URLS, LIST_CAPTIONS, LIST_OPTIONS
from mkt.databases.alphafold import adjudicate_structure
from mkt.databases.app.properties import PropertyTables
from mkt.databases.app.schema import (
    DefaultConfig,
    KLIFSImportantConfig,
    PhosphositesConfig,
)
from mkt.databases.app.structures import StructureVisualizer
from mkt.databases.colors import DICT_COLORS
from mkt.databases.log_config import configure_logging
from mkt.schema.io_utils import (
    DICT_FUNCS,
    deserialize_kinase_dict,
    return_str_path_from_pkg_data,
    untar_files_in_memory,
)
from mkt.schema.kinase_schema import KinaseInfo
from mkt.schema.utils import rgetattr
from streamlit_bokeh import streamlit_bokeh
from visualizers import SequenceAlignmentGenerator, StructureVisualizerGenerator

logger = logging.getLogger(__name__)


@dataclass
class DashboardState:
    """Class to hold the state of the dashboard."""

    kinase: str
    """Selected kinase."""
    palette: str
    """Selected color palette."""


# adapted from InterPLM (https://github.com/ElanaPearl/InterPLM/blob/main/interplm)
class Dashboard:
    """Class to visualize the kinase dashboard."""

    def __init__(self):
        """Initialize the Dashboard class."""
        self.list_kinases = self._load_data()

    @staticmethod
    @st.cache_resource
    def _load_data():
        """Load and cache the data - only load filenames and unload KinaseInfo objects separately."""
        str_path = return_str_path_from_pkg_data()

        list_kinases, _ = untar_files_in_memory(str_path, bool_extract=False)
        list_kinases.sort()

        return list_kinases

    @staticmethod
    def generate_json_file(obj_kinase: KinaseInfo) -> str:
        """Generate a JSON file with the kinase data."""
        str_json = DICT_FUNCS["json"]["serialize"](
            obj_kinase.model_dump(),
            **DICT_FUNCS["json"]["kwargs_serialize"],
        )

        return str_json

    def setup_sidebar(self) -> DashboardState:
        """Set up the inputs for the dashboard.

        Returns
        -------
        DashboardState
            The state of the dashboard.

        """
        st.sidebar.title("KinaseInfo options")
        st.sidebar.markdown(
            "This tool allows you to visualize the aligned, harmonized sequence, structure, and property information of human kinases derived from the resources linked below."
        )

        # select kinase to visualize
        st.sidebar.markdown(
            "## Kinase selection\n"
            "Select a kinase from the dropdown menu to visualize its data."
        )
        kinase_selection = st.sidebar.selectbox(
            "Kinase by HGNC name",
            options=self.list_kinases,
            index=0,
            label_visibility="collapsed",
            help="Select a kinase to visualize its data.",
        )

        # select color palette
        st.sidebar.markdown(
            "## Sequence color palette\n"
            "Select a color palette for the visualization."
        )
        palette_selection = st.sidebar.selectbox(
            "Select palette",
            options=DICT_COLORS.keys(),
            index=0,
            label_visibility="collapsed",
            help="Select a color palette for the visualization.",
        )

        state_dashboard = DashboardState(
            kinase=kinase_selection,
            palette=palette_selection,
        )

        st.sidebar.markdown(
            "## About\n"
            "This tool is developed by Jess White in the labs of John Chodera and Wesley Tansey at Memorial Sloan Kettering Cancer Center. "
            "For more information, visit the corresponding [Github repo](https://github.com/choderalab/missense-kinase-toolkit)."
        )

        st.sidebar.markdown("## Database resource")
        st.sidebar.markdown("This tool uses data from the following resources:\n")
        for link_text, link_url in DICT_RESOURCE_URLS.items():
            st.sidebar.link_button(link_text, link_url)

        return state_dashboard

    def display_dashboard(self, dashboard_state: DashboardState) -> None:
        """Run the dashboard.

        Parameters
        ----------
        dashboard_state : DashboardState
            The state of the dashboard containing the selected kinase and color palette.

        """
        # load KinaseInfo model
        obj_temp = deserialize_kinase_dict(list_ids=[dashboard_state.kinase])[
            dashboard_state.kinase
        ]
        str_json = self.generate_json_file(obj_temp)
        st.download_button(
            label="Download JSON file",
            data=str_json,
            file_name=f"{dashboard_state.kinase}.json",
            mime="application/json",
            help="Download the KinaseInfo object as a JSON file.",
            icon=":material/download:",
        )

        with st.expander("Sequences", expanded=True):
            st.markdown("### Sequence alignment\n")
            st.markdown(
                f"All non-KLIFS sequence residues shaded using {dashboard_state.palette} palette. "
                "KLIFS sequence residues shaded using KLIFS pocket color scheme. "
                "Residues that mismatch with the canonical UniProt sequence are shown in crimson. "
                "Crimson y-axis labels indicate the absense of a sequence for the chosen kinase in the database queried.\n"
            )

            obj_alignment = SequenceAlignmentGenerator(
                str_kinase=dashboard_state.kinase,
                dict_color=DICT_COLORS[dashboard_state.palette]["DICT_COLORS"],
                obj_kinase=obj_temp,
            )

            streamlit_bokeh(
                obj_alignment.plot,
                use_container_width=True,
                key="plot_alignment",
            )

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Structure", expanded=True):
                # adjudicate the structure source (KinCoRe CIF preferred, AF fallback)
                _, structure_source = adjudicate_structure(obj_temp)

                if structure_source is None:
                    st.error("No structure available for this kinase.", icon="⚠️")
                else:
                    st.markdown("### Kinase Domain\n" f"#### {structure_source}\n")
                    try:
                        plot_spot = st.empty()

                        # allow for annotations if present in the KinaseInfo object
                        list_idx = [0] + [
                            idx + 1
                            for idx, i in enumerate(
                                [
                                    "uniprot.phospho_sites",
                                    "KLIFS2UniProtIdx",
                                ]
                            )
                            if rgetattr(obj_temp, i) is not None
                        ]

                        annotation = st.radio(  # noqa: F841
                            "Select an annotation to render (select one):",
                            options=[LIST_OPTIONS[i] for i in list_idx],
                            captions=[LIST_CAPTIONS[i] for i in list_idx],
                            index=0,
                        )

                        with plot_spot:
                            # map annotation choice to config class
                            dict_annotation_config = {
                                "None": DefaultConfig,
                                "Phosphosites": PhosphositesConfig,
                                "KLIFS": KLIFSImportantConfig,
                            }

                            config = dict_annotation_config[annotation](
                                seq_align=obj_alignment,
                            )
                            struct_viz = StructureVisualizer(config)
                            viz = StructureVisualizerGenerator(struct_viz)
                            st.components.v1.html(
                                viz.html, height=600, width=None, scrolling=False
                            )
                    except Exception as e:
                        logger.exception(
                            f"Error generating structure for {dashboard_state.kinase}: {e}",
                        )
                        st.error("No structure available for this kinase.", icon="⚠️")

        with col2:
            with st.expander("Properties", expanded=True):
                st.markdown("### Kinase properties\n")

                table = PropertyTables(obj_temp)

                # share one column geometry across all four tables: each column is sized to the
                # widest content across every table (labels in col 1, values in col 2), measuring
                # values by their visible text so HTML links don't inflate the width
                _tables = [
                    table.df_kinhub,
                    table.df_klifs,
                    table.df_kincore,
                    table.df_computed,
                ]

                def _visible_len(cell) -> int:
                    return len(re.sub(r"<[^>]+>", "", str(cell)))

                label_ch = max(
                    (len(str(i)) for df in _tables if df is not None for i in df.index),
                    default=10,
                )
                # cap the value column so a very long value (e.g. SRMS's ~92-char KLIFS name)
                # wraps instead of widening the table past its half-page column -- otherwise the
                # browser scales the whole fixed-layout table (label column included) down to fit
                VALUE_MAX_CH = 36
                value_ch = min(
                    VALUE_MAX_CH,
                    max(
                        (
                            _visible_len(v)
                            for df in _tables
                            if df is not None
                            for v in df["Property"]
                        ),
                        default=10,
                    ),
                )

                # column geometry shared across all four tables. only the label column is a fixed
                # width; the table fills its container up to a content-fit max-width, so on a wide
                # monitor it stays content-sized while on a laptop the value column (not the label)
                # absorbs the shortfall -- avoiding the browser scaling the whole fixed table down
                label_w = label_ch + 8
                table_max_w = label_w + value_ch + 2

                def render_property_table(df, str_source):
                    # render the Styler HTML directly: st.table/st.dataframe cannot hide the
                    # column header, so drop the redundant "Property" header (key-value tables)
                    # via Styler.hide + st.markdown; row labels stay, saving a header row.
                    if df is not None:
                        styler = df.style.hide(axis="columns").set_table_styles(
                            [
                                {
                                    "selector": "td, th",
                                    "props": [
                                        ("text-align", "left"),
                                        ("padding", "2px 10px"),
                                        ("font-weight", "normal"),
                                        ("overflow-wrap", "anywhere"),
                                    ],
                                },
                                {
                                    "selector": "table",
                                    "props": [
                                        ("table-layout", "fixed"),
                                        ("width", "100%"),
                                        ("max-width", f"{table_max_w}ch"),
                                    ],
                                },
                                {
                                    "selector": "td:first-child, th:first-child",
                                    # labels are uppercase (wider than the `ch` glyph), so pad
                                    # generously to keep the widest label on one line
                                    "props": [("width", f"{label_w}ch")],
                                },
                            ]
                        )
                        st.markdown(styler.to_html(), unsafe_allow_html=True)
                    else:
                        st.error(
                            f"No {str_source} objects available for this kinase.",
                            icon="⚠️",
                        )

                st.markdown("#### KinHub\n")
                render_property_table(table.df_kinhub, "KinHub")

                st.markdown("#### KLIFS\n")
                render_property_table(table.df_klifs, "KLIFS")

                st.markdown("#### KinCoRe\n")
                render_property_table(table.df_kincore, "KinCoRe")

                st.markdown("#### Computed\n")
                render_property_table(table.df_computed, "computed")


def main():
    configure_logging()

    st.set_page_config(
        layout="wide",
        page_title="mkt",
        page_icon="🛍️",
    )

    st.title("KinaseInfo Dashboard")

    visualizer = Dashboard()
    state = visualizer.setup_sidebar()
    st.subheader(f"Selected Kinase: {state.kinase}")
    visualizer.display_dashboard(state)


if __name__ == "__main__":
    main()
