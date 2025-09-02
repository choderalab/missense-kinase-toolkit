import logging
from dataclasses import dataclass

import streamlit as st
from constants import DICT_RESOURCE_URLS, LIST_CAPTIONS, LIST_OPTIONS
from generate_alignments import SequenceAlignment
from generate_properties import PropertyTables
from generate_structures import StructureVisualizer
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

            obj_alignment = SequenceAlignment(
                obj_temp,
                DICT_COLORS[dashboard_state.palette]["DICT_COLORS"],
            )

            streamlit_bokeh(
                obj_alignment.plot,
                use_container_width=True,
                key="plot_alignment",
            )

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Structure", expanded=True):
                if obj_temp.kincore is None:
                    st.error("No KinCore objects available for this kinase.", icon="⚠️")
                else:
                    st.markdown("### KinCore active structure\n")
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
                            viz = StructureVisualizer(
                                obj_kinase=obj_temp,
                                dict_align=obj_alignment.dict_align,
                                str_attr=annotation,
                            )
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

                st.markdown("#### KinHub\n")
                if table.df_kinhub is not None:
                    st.table(table.df_kinhub)
                else:
                    st.error("No KinHub objects available for this kinase.", icon="⚠️")

                st.markdown("#### KLIFS\n")
                if table.df_klifs is not None:
                    st.table(table.df_klifs)
                else:
                    st.error("No KLIFS objects available for this kinase.", icon="⚠️")

                st.markdown("#### KinCore\n")
                if table.df_kincore is not None:
                    st.table(table.df_kincore)
                else:
                    st.error("No KinCore objects available for this kinase.", icon="⚠️")


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
