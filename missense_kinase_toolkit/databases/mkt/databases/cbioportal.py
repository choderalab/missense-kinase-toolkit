import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from bravado.client import SwaggerClient
from mkt.databases import klifs
from mkt.databases.api_schema import APIKeySwaggerClient
from mkt.databases.config import get_cbioportal_instance, maybe_get_cbioportal_token
from mkt.databases.io_utils import (
    parse_iterabc2dataframe,
    return_kinase_dict,
    save_dataframe_to_csv,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class cBioPortal(APIKeySwaggerClient):
    """Class to interact with the cBioPortal API."""

    def __init__(self):
        """Initialize cBioPortal Class object.

        Upon initialization, cBioPortal API is queried.

        Attributes
        ----------
        instance : str
            cBioPortal instance
        url : str
            cBioPortal API URL
        _cbioportal : bravado.client.SwaggerClient | None
            cBioPortal API object (post-init)

        """
        self.instance = get_cbioportal_instance()
        self.url: str = f"https://{self.instance}/api/v2/api-docs"
        self._cbioportal = self.query_api()

    def maybe_get_token(self):
        return maybe_get_cbioportal_token()

    def query_api(self):
        http_client = self.set_api_key()

        cbioportal_api = SwaggerClient.from_url(
            self.url,
            http_client=http_client,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
        )

        return cbioportal_api

    def get_instance(self):
        """Get cBioPortal instance."""
        return self.instance

    def get_url(self):
        """Get cBioPortal API URL."""
        return self.url

    def get_cbioportal(self):
        """Get cBioPortal API object."""
        return self._cbioportal


@dataclass
class Mutations(cBioPortal):
    """Class to get mutations from a cBioPortal study."""

    study_id: str
    """cBioPortal study ID."""
    bool_prefix: bool = True
    """Add prefix to ABC column names if True; default is True."""
    _mutations: list | None = field(init=False, default=None)
    """List of cBioPortal mutations; None if study not found (post-init)."""

    def __post_init__(self):
        """Post-initialization to get mutations from cBioPortal."""
        super().__init__()
        self._mutations = self.get_all_mutations_by_study()
        if self._mutations is None:
            logger.error(f"Mutations for study {self.study_id} not found.")

    def get_all_mutations_by_study(
        self,
    ) -> list | None:
        """Get mutations  cBioPortal data

        Returns
        -------
        list | None
            cBioPortal data of Abstract Base Classes objects if successful, otherwise None

        """
        studies = self._cbioportal.Studies.getAllStudiesUsingGET().result()
        study_ids = [study.studyId for study in studies]

        if self.study_id in study_ids:
            # TODO: add incremental error handling beyond missing study
            muts = self._cbioportal.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
                molecularProfileId=f"{self.study_id}_mutations",
                sampleListId=f"{self.study_id}_all",
                projection="DETAILED",
            ).result()
        else:
            logging.error(
                f"Study {self.study_id} not found in cBioPortal instance {self.instance}"
            )

        return muts

    def get_cbioportal_cohort_mutations(
        self,
        bool_save: bool = False,
    ) -> None | pd.DataFrame:
        """Get cBioPortal cohort mutations and optionally save as a CSV file.

        Parameters
        ----------
        bool_save : bool
            Save cBioPortal cohort mutations as a CSV file if True

        Returns
        -------
        pd.DataFrame | None
            DataFrame of cBioPortal cohort mutations if successful, otherwise None

        Notes
        -----
            The CSV file will be saved in the output directory specified in the configuration file.
            As the "gene" ABC object is nested within the "mutation" ABC object,
                the two dataframes are parsed and concatenated.

        """
        df_muts = parse_iterabc2dataframe(self._mutations)

        # extract columns that are ABC objects
        list_abc_cols = df_muts.columns[
            df_muts.map(lambda x: type(x).__module__ == "abc").sum() == df_muts.shape[0]
        ].tolist()

        # parse the ABC object cols and concatenate with main dataframe
        df_combo = df_muts.copy()
        if len(list_abc_cols) > 0:
            for col in list_abc_cols:
                if self.bool_prefix:
                    df_abc = parse_iterabc2dataframe(df_muts[col], str_prefix=col)
                else:
                    df_abc = parse_iterabc2dataframe(df_muts[col])
                df_combo = pd.concat([df_combo, df_abc], axis=1).drop([col], axis=1)

        if bool_save:
            filename = f"{self.study_id}_mutations.csv"
            save_dataframe_to_csv(df_combo, filename)
        else:
            return df_combo

    def filter_single_aa_missense_mutations(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter DataFrame for single amino acid missense mutations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of mutations

        Returns
        -------
        pd.DataFrame
            DataFrame of single amino acid missense mutations

        """
        # filter for missense mutations
        df_missense = df.loc[df["mutationType"] == "Missense_Mutation", :].reset_index(
            drop=True
        )

        # filter for single amino acid changes
        df_missense = df_missense.loc[
            df_missense["proteinChange"].apply(
                lambda x: type(self.try_except_middle_int(x)) is int
            ),
            :,
        ].reset_index(drop=True)

        return df_missense

    def return_adjusted_colname(
        self,
        colname: str,
        prefix: str = "gene",
    ) -> str:
        """Return adjusted column name based on bool_prefix.

        Parameters
        ----------
        colname : str
            Column name to adjust

        Returns
        -------
        str
            Adjusted column name

        """
        if self.bool_prefix:
            return f"{prefix}_{colname}"
        else:
            return colname

    @staticmethod
    def try_except_middle_int(str_in):
        """Try to convert string [1:-1] characters to integer.

        Parameters
        ----------
        str_in : str
            String to convert to integer

        Returns
        -------
        int | None
            Integer if successful, otherwise None

        """

        try:
            return int(str_in[1:-1])
        except ValueError:
            return None

    def get_study_id(self):
        """Get cBioPortal study ID."""
        return self.study_id

    def get_mutations(self):
        """Get cBioPortal mutations."""
        return self._mutations


@dataclass
class KinaseMissenseMutations(Mutations):
    """Class to get kinase mutations from a cBioPortal study."""

    dict_replace: dict[str, str] = field(default_factory=lambda: {"STK19": "WHR1"})
    """Dictionary mapping cBioPortal to MKT HGNC gene names for mismatches; default is {"STK19": "WHR1"}."""

    def __post_init__(self):
        super().__post_init__()

    def query_hgnc_gene_names(
        self,
        list_hgnc: list[str],
        dict_kinase: dict[str, object],
    ) -> dict[str, str | None]:
        """Query HGNC gene names from a DataFrame of mutations.

        Parameters
        ----------
        list_hgnc : list[str]
            List of HGNC gene names to query
        dict_kinase : dict[str, object]
            Dictionary mapping kinase names to KinaseInfo objects

        Returns
        -------
        dict
            Dictionary mapping HGNC gene names from cBioPortal to Uniprot IDs

        """
        from mkt.databases import hgnc

        dict_hgnc2uniprot = dict.fromkeys(set(list_hgnc))

        list_err = []
        for hgnc_name in tqdm(dict_hgnc2uniprot.keys(), desc="Querying HGNC..."):
            temp = hgnc.HGNC(input_symbol_or_id=hgnc_name)
            try:
                uniprot_id = temp.maybe_get_info_from_hgnc_fetch(
                    list_to_extract=["uniprot_ids"]
                )["uniprot_ids"][0][0]
                dict_hgnc2uniprot[hgnc_name] = uniprot_id
            except Exception as e:
                logger.error(f"Error retrieving Uniprot ID for {hgnc_name}: {e}")
                list_err.append(hgnc_name)
        logger.error(f"List errors:\n{list_err}")

        # replace any HGNC gene names in the dictionary
        for cbio_name, mkt_name in self.dict_replace.items():
            if cbio_name in dict_hgnc2uniprot:
                dict_hgnc2uniprot[cbio_name] = dict_kinase[mkt_name].uniprot_id

        return dict_hgnc2uniprot

    def get_kinase_missense_mutations(
        self,
        bool_save=False,
    ) -> None | pd.DataFrame:
        """Get cBioPortal kinase mutations and optionally save as a CSV file.

        Parameters
        ----------
        bool_save : bool
            Save cBioPortal kinase mutations as a CSV file if True

        Returns
        -------
        pd.DataFrame | None
            DataFrame of kinase mutations if successful, otherwise None

        """
        dict_in = return_kinase_dict()

        df_muts = self.get_cbioportal_cohort_mutations(bool_save=False)

        logger.info(
            f"\nPatients in {self.study_id} with mutations: {df_muts['patientId'].nunique():,}\n"
            f"Samples in {self.study_id} with mutations: {df_muts['sampleId'].nunique():,}"
        )

        col_hgnc = self.return_adjusted_colname("hugoGeneSymbol")

        # filter for single amino acid missense mutations
        df_muts_missense = self.filter_single_aa_missense_mutations(df_muts)

        # extract the HGNC gene names from the mutations
        dict_hgnc2uniprot = self.query_hgnc_gene_names(
            list_hgnc=df_muts_missense[col_hgnc].tolist(),
            dict_kinase=dict_in,
        )

        # hgnc2uniprot filtered to mutations in the kinase dictionary
        dict_hgnc2uniprot_kin = {
            k: v
            for k, v in dict_hgnc2uniprot.items()
            if v in [v.uniprot_id for v in dict_in.values()]
        }

        # dict_kinase is filtered to only include kinases with mutations
        dict_kinase_cbio = {
            k: v
            for k, v in dict_in.items()
            if v.uniprot_id.split("_")[0] in dict_hgnc2uniprot_kin.values()
        }
        # replace any mismatched gene names in the dictionary
        for cbio_name, mkt_name in self.dict_replace.items():
            dict_kinase_cbio[cbio_name] = dict_kinase_cbio.pop(mkt_name)
        # BRD4 and STK19 don't have KLIFS - if want to remove them, uncomment below
        # dict_kinase_cbio = {
        #     k: v for k, v in dict_kinase_cbio.items()
        #     if v.KLIFS2UniProtIdx is not None
        # }

        # filter mutations for kinase genes
        df_muts_missense_kin = df_muts_missense.loc[
            df_muts_missense[col_hgnc].isin(dict_kinase_cbio.keys()), :
        ].reset_index(drop=True)

        # remove mutations with mismatches to canonical Uniprot sequence
        df_muts_missense_kin_filtered = self.remove_mismatched_uniprot_mutations(
            df_muts_missense_kin,
            dict_in=dict_kinase_cbio,
        )

        df_muts_missense_kin_filtered_annotated = self.annotate_kinase_regions(
            df=df_muts_missense_kin_filtered,
            dict_in=dict_kinase_cbio,
        )

        if df_muts_missense_kin_filtered_annotated is not None:
            if bool_save:
                filename = f"{self.study_id}_kinase_missense_mutations.csv"
                save_dataframe_to_csv(df_muts_missense_kin_filtered_annotated, filename)
            else:
                return df_muts_missense_kin_filtered_annotated

    def remove_mismatched_uniprot_mutations(
        self,
        df: pd.DataFrame,
        dict_in: dict,
    ) -> pd.DataFrame:
        """Remove mutations with mismatches to canonical Uniprot sequence.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of mutations
        dict_in : dict
            Dictionary of HGNC gene names to Uniprot IDs

        Returns
        -------
        pd.DataFrame
            DataFrame of mutations with mismatched Uniprot IDs removed

        """
        list_mismatch, list_err = [], []

        for _, row in df.iterrows():

            hgnc_name = row[self.return_adjusted_colname("hugoGeneSymbol")]
            codon = row["proteinChange"]
            sample_id = row["sampleId"]
            idx = self.try_except_middle_int(codon)

            try:
                if dict_in[hgnc_name].uniprot.canonical_seq[idx - 1] != codon[0]:
                    list_mismatch.append(f"{sample_id}_{hgnc_name}_{codon}")
            except Exception as e:
                list_err.append(f"{sample_id}_{hgnc_name}_{codon}: {e}")

        # TODO: check non-mismatches for list_set_kinase_mismatch gene_hugoGeneSymbol
        set_kinase_mismatch = {i.split("_")[1] for i in list_mismatch + list_err}
        logger.error(f"HGNC gene names with mismatches: {set_kinase_mismatch}")
        df_filtered = df.loc[
            ~df["gene_hugoGeneSymbol"].isin(set_kinase_mismatch), :
        ].reset_index(drop=True)

        return df_filtered

    def annotate_kinase_regions(
        self,
        df: pd.DataFrame,
        dict_in: dict,
    ) -> pd.DataFrame:
        """Annotate kinase regions in a DataFrame of mutations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of mutations
        dict_in : dict
            Dictionary of HGNC gene names to KinaseInfo objects

        Returns
        -------
        pd.DataFrame
            DataFrame of mutations with kinase regions annotated

        """
        list_klifs, list_kincore = [], []
        for _, row in df.iterrows():

            hgnc_name = row[self.return_adjusted_colname("hugoGeneSymbol")]
            codon = row["proteinChange"]
            idx = self.try_except_middle_int(codon)

            # KLIFS
            if dict_in[hgnc_name].KLIFS2UniProtIdx is None:
                list_klifs.append(None)
            elif idx in dict_in[hgnc_name].KLIFS2UniProtIdx.values():
                klifs_region = [
                    k
                    for k, v in dict_in[hgnc_name].KLIFS2UniProtIdx.items()
                    if v == idx
                ][0]
                list_klifs.append(klifs_region)
            else:
                list_klifs.append(None)

            # KinCore
            if dict_in[hgnc_name].kincore is None:
                list_kincore.append(None)
            elif (
                dict_in[hgnc_name].kincore.fasta.start
                <= idx
                <= dict_in[hgnc_name].kincore.fasta.end
            ):
                list_kincore.append(True)
            else:
                list_kincore.append(False)

        df["klifs_region"] = list_klifs
        df["kincore_region"] = list_kincore

        return df

    def generate_heatmap_fig(
        self,
        df_in,
        filename: str | None = None,
    ) -> None:
        """Generate a heatmap figure of missense mutation counts by KLIFS region.

        Parameters
        ----------
        df_in : pd.DataFrame
            DataFrame of missense mutations with 'gene_hugoGeneSymbol' and 'klifs_region' columns
        filename : str | None
            Path and filename (incl format) to save the heatmap figure;
                if None, the figure will not be saved

        Returns
        -------
        None
            Displays the heatmap figure; saves it if bool_save is True

        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        pivot_table = (
            df_in.groupby(self.return_adjusted_colname("hugoGeneSymbol"))[
                "klifs_region"
            ]
            .value_counts(dropna=True)
            .unstack(fill_value=0)
        )

        logger.info(
            "\nPercent of KLIFS residues + kinase with no documented missense mutation: "
            f"{pivot_table.apply(lambda x: x == 0).sum().sum() / pivot_table.size:.1%}"
        )

        sorted_columns = pivot_table.columns[
            pivot_table.columns.map(lambda x: int(x.split(":")[1])).argsort()
        ]
        pivot_table = pivot_table[sorted_columns]
        pivot_table = pivot_table.apply(
            lambda x: np.log10(x + 1) if pd.api.types.is_numeric_dtype(x) else x
        )

        custom_palette = dict(
            zip(
                pivot_table.columns,
                pivot_table.columns.map(
                    lambda x: klifs.DICT_POCKET_KLIFS_REGIONS[x.split(":")[0]]["color"]
                ),
            )
        )

        vmax_value = int(np.ceil(pivot_table.values.max()))

        g = sns.clustermap(
            pivot_table,
            fmt="d",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax_value,
            cbar_kws={
                "label": "$log_{10}$(count + 1)",
                "shrink": 0.5,
                "orientation": "horizontal",
            },
            cbar_pos=(0.85, 1.02, 0.1, 0.01),
            figsize=(20, 20),
            dendrogram_ratio=(0.05, 0.05),
            row_cluster=True,
            col_cluster=True,
            method="average",
            metric="euclidean",
        )

        g.fig.suptitle("Missense Mutation Counts by KLIFS Region", y=1.02, fontsize=20)
        g.ax_heatmap.set_xlabel("KLIFS Region", fontsize=16)
        g.ax_heatmap.set_ylabel("Gene Symbol", fontsize=16)
        g.ax_heatmap.tick_params(axis="x", which="major", labelsize=12)
        g.ax_heatmap.tick_params(axis="y", which="major", labelsize=12)

        x_labels = g.ax_heatmap.get_xticklabels()
        for label in x_labels:
            label_text = label.get_text()
            if label_text in custom_palette:
                label.set_color(custom_palette[label_text])

        # plt.figure(figsize=(20, 20))
        # sns.heatmap(
        #     pivot_table,
        #     annot=False,
        #     cmap="Blues",
        #     linewidths=0.5,
        #     cbar_kws={"label": "log10(count + 1)", "shrink": 0.5},
        #     square=True,
        # )
        # plt.title("Missense Mutation Counts by KLIFS Region", fontsize=20, pad=20)
        # plt.xlabel("KLIFS Region", fontsize=12)
        # plt.ylabel("Gene Symbol", fontsize=12)

        plt.xticks(rotation=90, ha="center")
        plt.yticks(rotation=0)
        # plt.tight_layout()

        if filename:
            plt.savefig(
                filename, format=filename.split(".")[-1], bbox_inches="tight", dpi=300
            )

        plt.show()


# TODO: implement clinical annotations class
