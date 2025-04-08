import logging
from dataclasses import dataclass
from io import StringIO

import py3Dmol
import streamlit as st
from Bio.PDB.mmcifio import MMCIFIO
from mkt.databases.colors import DICT_COLORS

# from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS
from mkt.schema import io_utils

# from mkt.schema.kinase_schema import KinaseInfo

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
class SequenceStructureVisualizer:
    def __init__(self):
        """Initialize the SequenceStructureVisualizer class."""
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
        st.sidebar.markdown(
            "## Color palette\n"
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

    @staticmethod
    def dict_to_mmcif_text(mmcif_dict: dict[str, str]) -> str:
        """Convert MMCIF2Dict object back to mmCIF text format

        Parameters
        ----------
        mmcif_dict : dict[str, str]
            Dictionary containing mmCIF data.

        Returns
        -------
        str
            mmCIF text format as a string.

        """
        mmcif_io = MMCIFIO()
        mmcif_io.set_dict(mmcif_dict)

        # Write to a StringIO object instead of a file
        mmcif_text = StringIO()
        mmcif_io.save(mmcif_text)

        return mmcif_text.getvalue()

    def _generate_structure(
        self,
        hgnc_name: str,
        dict_col: dict[str, str],
        dict_params: dict[str, str] = {"width": 300, "height": 300},
    ) -> str:
        """Display the structure of the selected kinase.

        Parameters
        ----------
        hgnc_name : str
            HGNC gene name for which to generate structure.

        """
        # TODO: Use colors
        obj_temp = self.dict_kinase[hgnc_name]

        # get the selected kinase CIF, if exists
        mmcif_text = self.dict_to_mmcif_text(obj_temp.kincore.cif.cif)

        view = py3Dmol.view(width=500, height=500)
        view.addModel(mmcif_text, "cif")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()

        return view._make_html()

    def display_structure(self, dashboard_state: DashboardState) -> None:
        """Run the dashboard.

        Parameters
        ----------
        dashboard_state : DashboardState
            The state of the dashboard containing the selected kinase and color palette.

        """
        try:
            structure_html = self._generate_structure(dashboard_state.kinase)
            st.components.v1.html(structure_html, height=500)
        except Exception as e:
            logger.exception(
                "Error generating structure for %s: %s", dashboard_state.kinase, e
            )
            if self.dict_kinase[dashboard_state.kinase].kincore is None:
                st.error("No KinCore objects available for this kinase.", icon="⚠️")
            else:
                st.error("No structure available for this kinase.", icon="⚠️")


def main():
    st.set_page_config(
        layout="wide",
        page_title="mkt",
        page_icon="🛍️",
    )

    st.title("Kinase Structure/Sequence Visualizer")
    st.markdown(
        "This tool allows you to visualize the aligned functional sequences and structures of kinases and their phosphorylation sites. "
        "Select a kinase from the dropdown menu to get started."
    )

    visualizer = SequenceStructureVisualizer()
    state = visualizer.setup_sidebar()
    st.subheader(f"Selected Kinase: {state.kinase}")

    visualizer.display_structure(state)


if __name__ == "__main__":
    main()
