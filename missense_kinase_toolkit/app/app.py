import logging
from dataclasses import dataclass

import streamlit as st
from generate_alignments import SequenceAlignment
from generate_properties import PropertyTables
from generate_structures import StructureVisualizer
from mkt.databases.colors import DICT_COLORS

# from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS
from mkt.schema import io_utils
from mkt.schema.io_utils import DICT_FUNCS
from mkt.schema.kinase_schema import KinaseInfo
from streamlit_bokeh import streamlit_bokeh

logger = logging.getLogger(__name__)

DICT_RESOURCE_URLS = {
    "KinHub": "http://www.kinhub.org/",
    "KLIFS": "https://klifs.net/",
    "KinCore": "http://dunbrack.fccc.edu/kincore/home",
    "UniProt": "https://www.uniprot.org/",
    "Pfam": "https://www.ebi.ac.uk/interpro/entry/pfam",
}


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
class Dashboard:
    """Class to visualize the kinase dashboard."""

    def __init__(self):
        """Initialize the Dashboard class."""
        self.list_kinases = self._load_data()

    @staticmethod
    @st.cache_resource
    def _load_data():
        """Load and cache the data - only load filenames and unload KinaseInfo objects separately."""
        str_path = io_utils.return_str_path_from_pkg_data()

        list_kinases, _ = io_utils.untar_files_in_memory(str_path, bool_extract=False)
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
        obj_temp = io_utils.deserialize_kinase_dict(list_ids=[dashboard_state.kinase])[
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

            obj_alignment = SequenceAlignment(
                list_sequences=[
                    obj_temp.uniprot.canonical_seq,
                    obj_temp.klifs.pocket_seq,
                ],
                list_ids=["UniProt", "KLIFS"],
                dict_colors=DICT_COLORS[dashboard_state.palette]["DICT_COLORS"],
                plot_width=1200,
                plot_height=50,
                bool_top=False,
                bool_reverse=False,
            )

            streamlit_bokeh(
                # obj_alignment.plot, use_container_width=True, key="plot1"
                obj_alignment.plot_bottom,
                use_container_width=True,
                key="plot1",
            )

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Structure", expanded=True):
                st.markdown("### KinCore active structure\n")
                # st.markdown(f"### [KinCore]({DICT_RESOURCE_URLS['KinCore']}) active structure\n")
                viz = StructureVisualizer()
                try:
                    structure_html = viz.visualize_structure(
                        mmcif_dict=obj_temp.kincore.cif.cif,
                        str_id=dashboard_state.kinase,
                    )
                    st.components.v1.html(structure_html, height=600)
                    annotation = st.radio(  # noqa: F841
                        "Select an annotation to render (select one):",
                        options=[
                            "None",
                            "Phosphosites",
                            "KLIFS pocket",
                            "Mutational density",
                        ],
                        captions=[
                            "No additional annotation",
                            "Phosphorylation sites as adjudicated by UniProt",
                            "Residues that belong to the KLIFS binding pocket",
                            "Missense mutational density within cBioPortal MSK-IMPACT cohort ([Zehir et al, 2017.](https://www.nature.com/articles/nm.4333))",
                        ],
                        index=0,
                    )
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

                table = PropertyTables()
                table.extract_properties(obj_temp)

                st.markdown("#### KinHub\n")
                # st.markdown(f"#### [KinHub]({DICT_RESOURCE_URLS['KinHub']})\n")
                if table.df_kinhub is not None:
                    st.table(table.df_kinhub)
                else:
                    st.error("No KinHub objects available for this kinase.", icon="⚠️")

                st.markdown("#### KLIFS\n")
                # st.markdown(f"#### [KLIFS]({DICT_RESOURCE_URLS['KLIFS']})\n")
                if table.df_klifs is not None:
                    st.table(table.df_klifs)
                else:
                    st.error("No KLIFS objects available for this kinase.", icon="⚠️")

                st.markdown("#### KinCore\n")
                # st.markdown(f"#### [KinCore]({DICT_RESOURCE_URLS['KinCore']})\n")
                if table.df_kincore is not None:
                    st.table(table.df_kincore)
                else:
                    st.error("No KinCore objects available for this kinase.", icon="⚠️")


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
