import logging
from dataclasses import dataclass

import streamlit as st

# from mkt.schema.kinase_schema import KinaseInfo
from generate_structures import StructureVisualizer

# from mkt.databases.colors import DICT_COLORS
# from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS
from mkt.schema import io_utils

logger = logging.getLogger(__name__)


@dataclass
class DashboardState:
    """Class to hold the state of the dashboard."""

    kinase: str
    """Selected kinase."""
    # palette: str
    # """Selected color palette."""
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
        self.dict_kinase = self._load_data()

    @staticmethod
    @st.cache_resource
    def _load_data():
        """Load and cache the data."""
        dict_kinase = io_utils.deserialize_kinase_dict()

        dict_reverse = {
            (
                v.hgnc_name + "_" + v.uniprot_id.split("_")[1]
                if "_" in v.uniprot_id
                else v.hgnc_name
            ): v
            for v in dict_kinase.values()
        }

        dict_reverse = dict(sorted(dict_reverse.items()))

        return dict_reverse

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
            options=self.dict_kinase.keys(),
            index=0,
            label_visibility="collapsed",
            help="Select a kinase to visualize its structure.",
        )

        # select color palette
        # st.sidebar.markdown(
        #     "## Color palette\n"
        #     "Select a color palette for the visualization. "
        #     "The default palette is 'default'."
        # )
        # palette_selection = st.sidebar.selectbox(
        #     "Select palette",
        #     options=DICT_COLORS.keys(),
        #     index=0,
        #     label_visibility="collapsed",
        #     help="Select a color palette for the visualization.",
        # )

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
            # palette=palette_selection,
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
        # load Pydantic model
        obj_temp = self.dict_kinase[dashboard_state.kinase]

        try:
            structure_html = self.visualize_structure(
                mmcif_dict=obj_temp.kincore.cif.cif,
                str_id=dashboard_state.kinase,
            )
            st.components.v1.html(structure_html, height=500)
        except Exception as e:
            print(
                # logger.exception(
                "Error generating structure for %s: %s",
                dashboard_state.kinase,
                e,
            )
            if obj_temp.kincore is None:
                st.error("No KinCore objects available for this kinase.", icon="⚠️")
            else:
                st.error("No structure available for this kinase.", icon="⚠️")


def main():
    st.set_page_config(
        layout="wide",
        page_title="mkt",
        page_icon="🛍️",
    )

    st.title("KinaseInfo Dashboard")
    st.markdown(
        "This tool allows you to visualize the aligned functional sequences and structures of kinases and their phosphorylation sites. "
        "Select a kinase from the dropdown menu to get started."
    )

    visualizer = Dashboard()
    state = visualizer.setup_sidebar()
    st.subheader(f"Selected Kinase: {state.kinase}")

    visualizer.display_dashboard(state)
    

if __name__ == "__main__":
    main()
