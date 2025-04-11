import logging
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from generate_alignments import SequenceAlignment
from generate_structures import StructureVisualizer
from mkt.databases.colors import DICT_COLORS

# from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS
from mkt.schema import io_utils
from streamlit_bokeh import streamlit_bokeh

logger = logging.getLogger(__name__)


@dataclass
class DashboardState:
    """Class to hold the state of the dashboard."""

    kinase: str
    """Selected kinase."""
    palette: str
    """Selected color palette."""
    # check_phospho: bool
    # """Check to highlight phosphorylation sites."""
    # list_seq: list[list[str]] | None = None
    # """List of sequences to show in aligner."""
    # list_colors: list[list[str]] | None = None
    # """List of colors to show in aligner."""


# adapted from InterPLM (https://github.com/ElanaPearl/InterPLM/blob/main/interplm)
class Dashboard(StructureVisualizer):
    def __init__(self):
        """Initialize the Dashboard class."""
        super().__init__()
        self.list_kinases = self._load_data()

    @staticmethod
    @st.cache_resource
    def _load_data():
        """Load and cache the data - only load filenames and unload KinaseInfo objects separately."""
        str_path = io_utils.return_str_path_from_pkg_data()

        list_kinases, _ = io_utils.untar_files_in_memory(str_path, bool_extract=False)
        list_kinases.sort()

        return list_kinases

    def setup_sidebar(self) -> DashboardState:
        """Set up the inputs for the dashboard.

        Returns
        -------
        DashboardState
            The state of the dashboard.

        """
        st.sidebar.title("KinaseInfo options")
        st.sidebar.markdown(
            "This tool allows you to visualize the aligned sequences and structures of kinases and their phosphorylation sites. "
            "Select a kinase from the dropdown menu to get started."
        )

        # select kinase to visualize
        st.sidebar.markdown(
            "## Kinase selection\n"
            "Select a kinase from the dropdown menu to visualize its structure."
        )
        kinase_selection = st.sidebar.selectbox(
            "Kinase by HGNC name",
            options=self.list_kinases,
            index=0,
            label_visibility="collapsed",
            help="Select a kinase to visualize its structure.",
        )

        # select color palette
        st.sidebar.markdown(
            "## Sequence color palette\n"
            "Select a color palette for the visualization. "
            "The default palette is 'default'."
        )
        palette_selection = st.sidebar.selectbox(
            "Select palette",
            options=DICT_COLORS.keys(),
            index=0,
            label_visibility="collapsed",
            help="Select a color palette for the visualization.",
        )

        # # select highlight phosphorylation sites
        # st.sidebar.markdown(
        #     "## Phosphositess\n"
        #     "Check the box to highlight phosphorylation sites adjudicated by UniProt."
        # )
        # st.sidebar.markdown(
        #     "## About\n"
        #     "This tool is developed by the Kinase Research Group at the University of XYZ. "
        #     "For more information, visit our [website](https://example.com)."
        # )
        # add_highlight = st.sidebar.checkbox(
        #     "Highlight phosphosites?",
        #     value=False,
        #     help="Check to highlight sites adjudicated as phosphorylation sites by UniProt.",
        # )

        state_dashboard = DashboardState(
            kinase=kinase_selection,
            palette=palette_selection,
            # check_phospho=add_highlight,
        )

        return state_dashboard

    def display_dashboard(self, dashboard_state: DashboardState) -> None:
        """Run the dashboard.

        Parameters
        ----------
        dashboard_state : DashboardState
            The state of the dashboard containing the selected kinase and color palette.

        """
        # load KinaseInfo model
        obj_temp = io_utils.deserialize_kinase_dict(list_ids=[dashboard_state.kinase])[
            dashboard_state.kinase
        ]

        # plot_width = st.sidebar.slider("Plot Width", min_value=400, max_value=1200, value=800)

        with st.expander("Sequences", expanded=True):
            st.markdown(
                "### Sequence alignment\n"
                "This section shows the aligned sequences of the selected kinase. "
                "The colors represent different regions of the kinase."
            )

            obj_alignment = SequenceAlignment(
                list_sequences=[
                    obj_temp.uniprot.canonical_seq,
                    obj_temp.klifs.pocket_seq,
                ],
                list_ids=["UniProt", "KLIFS"],
                dict_colors=DICT_COLORS[dashboard_state.palette]["DICT_COLORS"],
                plot_width=1200,
            )

            # st.bokeh_chart(obj_alignment.plot, use_container_width=True)
            streamlit_bokeh(obj_alignment.plot, use_container_width=True, key="plot1")

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Structure", expanded=True):
                st.markdown("### KinCore active structure\n")
                try:
                    structure_html = self.visualize_structure(
                        mmcif_dict=obj_temp.kincore.cif.cif,
                        str_id=dashboard_state.kinase,
                    )
                    st.components.v1.html(structure_html, height=600)
                    st.checkbox("Show phosphosites", value=False)
                    st.checkbox("Show KLIFS pocket", value=False)
                    st.checkbox("Show mutational density", value=False)
                except Exception as e:
                    logger.exception(
                        f"Error generating structure for {dashboard_state.kinase}: {e}",
                    )
                    if obj_temp.kincore is None:
                        st.error(
                            "No KinCore objects available for this kinase.", icon="⚠️"
                        )
                    else:
                        st.error("No structure available for this kinase.", icon="⚠️")

        with col2:
            with st.expander("Properties", expanded=True):
                st.markdown("### Kinase properties\n")

                st.markdown("#### KinHub\n")
                try:
                    df_kinhub = pd.DataFrame(
                        {
                            k.replace("_", " ").upper(): v
                            for k, v in dict(obj_temp.kinhub).items()
                        },
                        index=[0],
                    ).T
                    df_kinhub.columns = ["Property"]
                    st.table(df_kinhub)
                except Exception as e:
                    logger.exception(
                        f"Error generating KinHub properties for {dashboard_state.kinase}: {e}",
                    )
                    st.error("No KinHub objects available for this kinase.", icon="⚠️")


def main():
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
