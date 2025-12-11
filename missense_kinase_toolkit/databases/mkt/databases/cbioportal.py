import logging
import os
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from Bio import Align
from bravado.client import SwaggerClient
from mkt.databases import klifs, properties
from mkt.databases.api_schema import APIKeySwaggerClient
from mkt.databases.colors import DICT_KINASE_GROUP_COLORS
from mkt.databases.config import get_cbioportal_instance, maybe_get_cbioportal_token
from mkt.databases.io_utils import (
    parse_iterabc2dataframe,
    return_kinase_dict,
    save_dataframe_to_csv,
)
from mkt.databases.utils import add_one_hot_encoding_to_dataframe
from tqdm import tqdm

logger = logging.getLogger(__name__)


DICT_KINASE = return_kinase_dict()


@dataclass
class cBioPortal(APIKeySwaggerClient):
    """Class to interact with the cBioPortal API."""

    instance: str = field(init=False, default="")
    """cBioPortal instance."""
    url: str = field(init=False, default="")
    """cBioPortal API URL."""
    _cbioportal: SwaggerClient | None = field(init=False, default=None)
    """cBioPortal API object (post-init)."""

    def __post_init__(self):
        """Post-initialization to set up cBioPortal API client."""
        self.instance = get_cbioportal_instance()
        self.url = f"https://{self.instance}/api/v2/api-docs"
        try:
            self._cbioportal = self.query_api()
        except Exception as e:
            logger.warning(
                f"Error initializing cBioPortal API client: {e}\n"
                "Can still load data from CSV files if pathfile(s) provided."
            )

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
class cBioPortalQuery(cBioPortal):
    """Class to get data from a cBioPortal instance."""

    bool_prefix: bool = True
    """Add prefix to ABC column names if True; default is True."""
    list_col_explode: list[str] | None = field(default=None)
    """List of columns to explode in convert_api_query_to_dataframe;
        None if no columns to explode (post-init)."""
    pathfile: str | None = None
    """Path to load dataframe from CSV file; if None, regenerate (post-init)."""
    _data: list | None = field(init=False, default=None)
    """List of cBioPortal sub-API queries; None if ID not found (post-init)."""
    _df: pd.DataFrame | None = field(init=False, default=None)

    def __post_init__(self):
        """Post-initialization to check study ID in instance and query API data."""
        super().__post_init__()
        if not self.check_entity_id():
            logger.warning(
                f"Study {self.get_entity_id()} not found "
                f"in cBioPortal instance {self.instance}"
            )
        if self.pathfile is not None:
            try:
                self._df = self.load_from_csv()
            except Exception as e:
                logger.error(
                    f"Error loading DataFrame from {self.pathfile}: {e}\n"
                    "Regenerating DataFrame from API query..."
                )
                self.regenerate_dataframe()
        else:
            self.regenerate_dataframe()

    @abstractmethod
    def get_entity_id(self):
        """Get the entity ID (study_id or panel_id).

        Returns
        -------
        str
            Entity ID
        """
        ...

    @abstractmethod
    def check_entity_id(self) -> bool:
        """Check if the entity ID is valid.

        Returns
        -------
        bool
            True if the entity ID is valid, False otherwise
        """
        ...

    @abstractmethod
    def query_sub_api(self):
        """Query a sub-API of cBioPortal and return result.

        Returns
        -------
        SwaggerClient
            API response
        """
        ...

    def load_from_csv(
        self,
        str_path: str | None = None,
    ) -> pd.DataFrame | None:
        """Load DataFrame from CSV file.

        Parameters
        ----------
        str_path : str | None
            Path to CSV file; if None, use self.pathfile

        Returns
        -------
        pd.DataFrame | None
            DataFrame loaded from CSV file if successful, otherwise None
        """
        if str_path is not None:
            path_to_use = str_path
        else:
            path_to_use = self.pathfile
            logger.info(f"Loading DataFrame from CSV file: {path_to_use}.")

        if path_to_use is not None and os.path.exists(path_to_use):
            try:
                df = pd.read_csv(path_to_use)
                return df
            except Exception as e:
                logger.error(f"Error loading DataFrame from {path_to_use}: {e}")
                return None
        else:
            logger.error(f"Path {path_to_use} does not exist or is not specified.")
            return None

    def regenerate_dataframe(self) -> pd.DataFrame | None:
        """Regenerate DataFrame from API query.

        Returns
        -------
        pd.DataFrame | None
            DataFrame of API query if successful, otherwise None
        """
        self._data = self.query_sub_api()
        if self._data is None:
            logger.error(
                f"Data for {self.get_entity_id()} not found "
                f"in cBioPortal instance {self.instance}"
            )
        else:
            self._df = self.convert_api_query_to_dataframe()
            if self._df is None:
                logger.error(
                    f"DataFrame for {self.get_entity_id()} could not be created."
                )

    def convert_api_query_to_dataframe(self) -> pd.DataFrame | None:
        """Convert API to query to a dataframe.

        Returns
        -------
        pd.DataFrame | None
            DataFrame of API query if successful, otherwise None
        """
        try:
            df = parse_iterabc2dataframe(self._data)

            # explode columns, if specified
            if self.list_col_explode is not None:
                for col in self.list_col_explode:
                    if col in df.columns:
                        df = df.explode(col).reset_index(drop=True)

            # extract columns that are ABC objects
            list_abc_cols = df.columns[
                df.map(lambda x: type(x).__module__ == "abc").sum() == df.shape[0]
            ].tolist()

            # parse the ABC object cols and concatenate with main dataframe
            df_combo = df.copy()
            if len(list_abc_cols) > 0:
                for col in list_abc_cols:
                    if self.bool_prefix:
                        df_abc = parse_iterabc2dataframe(df[col], str_prefix=col)
                    else:
                        df_abc = parse_iterabc2dataframe(df[col])
                    df_combo = pd.concat([df_combo, df_abc], axis=1).drop([col], axis=1)

            return df_combo

        except Exception as e:
            logger.error(f"Error converting API query to DataFrame: {e}")
            return None

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
        prefix : str
            Prefix to add to the column name if bool_prefix is True; default is "gene"

        Returns
        -------
        str
            Adjusted column name
        """
        if self.bool_prefix:
            return f"{prefix}_{colname}"
        else:
            return colname

    def get_data(self):
        """Get cBioPortal data."""
        if self._data is not None:
            return self._data
        else:
            logger.error(f"Data for {self.get_entity_id()} not found.")
            return None

    def get_df(self):
        """Get DataFrame of cBioPortal data in dataframe."""
        if self._df is not None:
            # defensive copy to avoid modifying original DataFrame
            return self._df.copy()
        else:
            logger.error(f"DataFrame for {self.get_entity_id()} not found.")
            return None


@dataclass
class StudyData(cBioPortalQuery):
    """Class to get mutations from a cBioPortal study."""

    study_id: str = field(kw_only=True)
    """cBioPortal study ID."""

    def __post_init__(self):
        super().__post_init__()

    def get_entity_id(self):
        """Get cBioPortal study ID."""
        return self.study_id

    def check_entity_id(self) -> bool:
        """Check if the study ID is valid.

        Returns
        -------
        bool
            True if the study ID is valid, False otherwise
        """
        try:
            studies = self._cbioportal.Studies.getAllStudiesUsingGET().result()
            study_ids = [study.studyId for study in studies]
            return self.study_id in study_ids
        except Exception as e:
            logger.warning(f"Error checking study ID {self.study_id}: {e}")
            return False


@dataclass
class Mutations(StudyData):
    """Class to get mutations from a cBioPortal study."""

    def __post_init__(self):
        """Post-initialization to get mutations from cBioPortal."""
        super().__post_init__()

    def query_sub_api(self) -> list | None:
        """Get mutations cBioPortal data.

        Returns
        -------
        list | None
            cBioPortal data as list of Abstract Base Classes
                objects if successful, otherwise None.

        """
        try:
            # TODO: add incremental error handling beyond missing study
            muts = self._cbioportal.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
                molecularProfileId=f"{self.study_id}_mutations",
                sampleListId=f"{self.study_id}_all",
                projection="DETAILED",
            ).result()
        except Exception as e:
            logger.error(f"Error retrieving mutations for study {self.study_id}: {e}")
            muts = None
        return muts


@dataclass
class KinaseMissenseMutations(Mutations):
    """Class to get kinase mutations from a cBioPortal study."""

    dict_replace: dict[str, str] = field(default_factory=lambda: {"STK19": "WHR1"})
    """Dictionary mapping cBioPortal to mkt gene names for mismatches; default is {"STK19": "WHR1"}."""
    str_blosom: str = "BLOSUM80"
    """BLOSUM matrix to use for mutation analysis; default is "BLOSUM80"."""
    pathfile_filter: str | None = None
    """Path to CSV file for filtered kinase missense mutations; default is None."""
    _df_filter: pd.DataFrame | None = field(init=False, default=None)
    """DataFrame of kinase missense mutations; None if DataFrame could not be created (post-init)."""

    def __post_init__(self):
        super().__post_init__()
        if self.pathfile_filter is not None:
            str_temp = "loaded"
            logger.info(
                f"Loading filtered DataFrame from CSV file: {self.pathfile_filter}."
            )
            self._df_filter = self.load_from_csv(str_path=self.pathfile_filter)
        else:
            str_temp = "generated"
            self._df_filter = self.get_kinase_missense_mutations()

        if self._df_filter is None:
            logger.error(
                "DataFrame for kinase missense mutations in study "
                f"{self.study_id} could not be {str_temp}."
            )

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
                list_err.append(f"{hgnc_name}: {e}")
        if len(list_err) > 0:
            str_errors = "\n".join(list_err)
            logger.error(f"Errors retrieving HGNC gene names:\n{str_errors}")

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

        col_hgnc = self.return_adjusted_colname("hugoGeneSymbol")

        # filter for single amino acid missense mutations
        df = self._df.copy()  # defensive copy to avoid modifying original DataFrame
        df_muts_missense = self.filter_single_aa_missense_mutations(df)

        # extract the HGNC gene names from the mutations
        dict_hgnc2uniprot = self.query_hgnc_gene_names(
            list_hgnc=df_muts_missense[col_hgnc].tolist(),
            dict_kinase=DICT_KINASE,
        )

        # hgnc2uniprot filtered to mutations in the kinase dictionary
        dict_hgnc2uniprot_kin = {
            k: v
            for k, v in dict_hgnc2uniprot.items()
            if v in [v.uniprot_id for v in DICT_KINASE.values()]
        }

        # dict_kinase is filtered to only include kinases with mutations
        dict_kinase_cbio = {
            k: v
            for k, v in DICT_KINASE.items()
            if v.uniprot_id.split("_")[0] in dict_hgnc2uniprot_kin.values()
        }

        # replace any mismatched gene names in the dictionary
        for cbio_name, mkt_name in self.dict_replace.items():
            if mkt_name in dict_kinase_cbio:
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
        if len(set_kinase_mismatch) > 0:
            str_errors = "\n".join(set_kinase_mismatch)
            logger.error(
                "HGNC gene names of kinases with mismatches between "
                f"cBioPortal and canonical Uniprot sequences:\n{str_errors}"
            )
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
        mx_blosum = Align.substitution_matrices.load(self.str_blosom)

        dict_out = {
            "klifs_region": [],
            "kincore_kd": [],
            "blosum_penalty": [],
            "charge": [],
            "polarity": [],
            "volume": [],
        }
        for _, row in df.iterrows():

            hgnc_name = row[self.return_adjusted_colname("hugoGeneSymbol")]
            codon = row["proteinChange"]
            idx = self.try_except_middle_int(codon)
            aa_from = codon[0].upper()
            aa_to = codon[-1].upper()

            # KLIFS
            if dict_in[hgnc_name].KLIFS2UniProtIdx is None:
                dict_out["klifs_region"].append(None)
            elif idx in dict_in[hgnc_name].KLIFS2UniProtIdx.values():
                klifs_region = [
                    k
                    for k, v in dict_in[hgnc_name].KLIFS2UniProtIdx.items()
                    if v == idx
                ][0]
                dict_out["klifs_region"].append(klifs_region)
            else:
                dict_out["klifs_region"].append(None)

            # KinCore
            if dict_in[hgnc_name].kincore is None:
                dict_out["kincore_kd"].append(None)
            elif (
                dict_in[hgnc_name].kincore.fasta.start
                <= idx
                <= dict_in[hgnc_name].kincore.fasta.end
            ):
                dict_out["kincore_kd"].append(True)
            else:
                dict_out["kincore_kd"].append(False)

            # BLOSUM penalty
            dict_out["blosum_penalty"].append(mx_blosum[aa_from, aa_to])

            # property changes
            dict_temp = properties.classify_aa_change(aa_from=aa_from, aa_to=aa_to)
            for k, v in dict_temp.items():
                if type(v) is str:
                    v = v.replace(", ", "-").replace(" ", "_")
                dict_out[k].append(v)

        for key, value in dict_out.items():
            df[key] = value

        df = add_one_hot_encoding_to_dataframe(
            df, col_name=["charge", "polarity", "volume"]
        )

        return df

    def generate_pivot_table(
        self,
        colname: str,
        bool_onehot: bool,
        bool_log10: bool,
        max_value: int | None,
    ) -> pd.DataFrame:
        """Generate a pivot table of missense mutation counts by KLIFS region.

        Parameters
        ----------
        colname : str
            Column name to pivot on; default is "klifs_region"
        bool_onehot : bool
            Column name to pivot on; default is "klifs_region" (just counts);
                if "blosum_penalty", the mean BLOSUM penalty is used instead;
                    if starts with "_", it is treated as a one-hot encoded column
        bool_log10 : bool
            Convert counts to log10 if True; default is True
        max_value : int | None
            Maximum value to truncate the log10 counts to if bool_log10 is True;
                if None, no truncation is applied; default is None

        Returns
        -------
        pd.DataFrame
            Pivot table of missense mutation counts by KLIFS region

        """
        df = self._df_filter.copy()
        if colname not in df.columns:
            colname = self.return_adjusted_colname(colname)
            if colname not in df.columns:
                logger.warning(
                    f"Column {colname} not found in DataFrame. "
                    f"Available columns: {df.columns.tolist()}"
                )
                return None

        dict_out = dict.fromkeys(["dataframe", "title"])
        dict_out["title"] = "Missense mutation counts by KLIFS region"
        col_hgnc = self.return_adjusted_colname("hugoGeneSymbol")
        col_klifs = "klifs_region"
        # BLOSUM take mean, others take value counts
        if colname == "blosum_penalty":
            pivot_table = (
                df.groupby([col_hgnc, col_klifs])[colname]
                .agg("mean")
                .unstack(fill_value=0)
            )
            title = ", BLOSUM Penalty (Mean)"
        # one-hot encoding columns
        elif colname.startswith("_"):
            # keep only values that correpond to the one-hot encoding
            df_temp = df.loc[df[colname] == bool_onehot, :].reset_index(drop=True)
            pivot_table = (
                df_temp.groupby([col_hgnc, col_klifs])[colname]
                .value_counts(dropna=True)
                .unstack(fill_value=0)
                .unstack(fill_value=0)
            )
            # drop True/False index level
            pivot_table.columns = [col[1] for col in pivot_table.columns]
            title = (
                f"{' (' if bool_onehot else ' (Not '}"
                f"{colname[1:].replace('-', ', ').replace('_', ' ').title()})"
            )
        # value count of KLIFS regions by gene only
        else:
            pivot_table = (
                df.groupby(col_hgnc)[colname]
                .value_counts(dropna=True)
                .unstack(fill_value=0)
            )
            title = ""

        sorted_columns = pivot_table.columns[
            pivot_table.columns.map(lambda x: int(x.split(":")[1])).argsort()
        ]
        pivot_table = pivot_table[sorted_columns]

        if colname != "blosum_penalty":
            logger.info(
                "\nPercent of KLIFS residues + kinase with no documented missense mutation: "
                f"{pivot_table.apply(lambda x: x == 0).sum().sum() / pivot_table.size:.1%}"
            )
            pivot_table = pivot_table.map(
                lambda x: self.convert_log_and_truncate(x, bool_log10, max_value),
            )

        dict_out["dataframe"] = pivot_table
        dict_out["title"] = dict_out["title"] + title

        return dict_out

    def generate_heatmap_fig(
        self,
        filename: str | None = None,
        colname: str = "klifs_region",
        bool_onehot: bool = True,
        bool_log10: bool = True,
        max_value: int | None = None,
        dict_clustermap_args: dict | None = None,
    ) -> None:
        """Generate a heatmap figure of missense mutation counts by KLIFS region.

        Parameters
        ----------
        df_in : pd.DataFrame
            DataFrame of missense mutations with 'gene_hugoGeneSymbol' and 'klifs_region' columns
        filename : str | None
            Path and filename (incl format) to save the heatmap figure;
                if None, the figure will not be saved
        colname : str
            Column name to pivot on; default is "klifs_region" (just counts);
                if "blosum_penalty", the mean BLOSUM penalty is used instead;
                    if starts with "_", it is treated as a one-hot encoded column
        bool_onehot : bool
            If colname corresponds to one-hot encoded columns, which values to keep;
                default is True
        bool_log10 : bool
            Convert counts to log10 if True; default is True
        max_value : int | None
            Maximum value to truncate the log10 counts to if bool_log10 is True;
                if None, no truncation is applied; default is None
        dict_clustermap_args : dict | None
            Additional arguments for the seaborn clustermap function;
            if None, default arguments are used; default is None

        Returns
        -------
        None
            Displays the heatmap figure; saves it if bool_save is True

        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import ListedColormap

        if filename is not None:
            plt.ioff()

        dict_data = self.generate_pivot_table(
            colname=colname,
            bool_onehot=bool_onehot,
            bool_log10=bool_log10,
            max_value=max_value,
        )
        if dict_data is None:
            logger.error(
                f"Could not generate pivot table for column {colname} in DataFrame."
            )
            return

        pivot_table = dict_data["dataframe"]
        title = dict_data["title"]
        if pivot_table.empty:
            logger.error(
                f"Pivot table for column {colname} is empty. "
                "No heatmap will be generated."
            )
            return

        custom_palette = dict(
            zip(
                pivot_table.columns,
                pivot_table.columns.map(
                    lambda x: klifs.DICT_POCKET_KLIFS_REGIONS[x.split(":")[0]]["color"]
                ),
            )
        )

        kinfam_palette = dict(
            zip(
                pivot_table.index,
                pivot_table.index.map(
                    lambda x: DICT_KINASE_GROUP_COLORS[
                        DICT_KINASE[x].adjudicate_group()
                    ]
                ),
            )
        )

        vmax_value = int(np.ceil(pivot_table.values.max()))

        # create custom colormap where grey is for 0 values and YlOrRd for >0 to max
        ylord_cmap = plt.cm.get_cmap("YlOrRd")
        n_colors = 100
        # create colors: 1 for grey (0) + n_colors for gradient (>0 to max)
        colors = ["lightgrey"]  # Grey for exactly 0
        colors.extend([ylord_cmap(i) for i in np.linspace(0.1, 1, n_colors)])
        custom_cmap = ListedColormap(colors)
        # create boundaries: n_colors + 2 boundaries for n_colors + 1 colors
        bounds = [0]  # Start at 0
        bounds.extend(
            np.linspace(0.001, vmax_value, n_colors + 1)
        )  # n_colors + 1 boundaries from >0 to max
        norm = mcolors.BoundaryNorm(
            bounds, len(colors)
        )  # use len(colors) instead of custom_cmap.N

        dict_kwargs = {
            "fmt": "d",
            "cmap": custom_cmap,
            "norm": norm,
            "vmin": 0,
            "vmax": vmax_value,
            "linewidths": 0.25,
            "linecolor": "white",
            "cbar_kws": {
                "label": "$log_{10}$(count)",
                "shrink": 0.5,
                "orientation": "horizontal",
                "ticks": np.arange(0, vmax_value + 0.5, 0.5),
            },
            "cbar_pos": (0.85, 0.98, 0.1, 0.01),
            "figsize": (20, 20),
            "dendrogram_ratio": (0.05, 0.05),
            "row_cluster": True,
            "col_cluster": False,
            "method": "average",
            "metric": "correlation",
            "row_colors": pivot_table.index.map(kinfam_palette),
        }

        if dict_clustermap_args is not None:
            dict_kwargs.update(dict_clustermap_args)

        try:
            g = sns.clustermap(pivot_table, **dict_kwargs)
        except Exception as e:
            plt.close()
            logger.error(
                f"Error generating clustermap: {e}\n"
                f"Inputs: method={dict_kwargs['method'].title()}, "
                f"metric={dict_kwargs['metric'].title()}\n"
                f"Adding small, random noise to pivot table to avoid error."
            )
            np.random.seed(42)
            g = sns.clustermap(
                pivot_table + np.random.normal(0, 1e-10, pivot_table.shape),
                **dict_kwargs,
            )

        g.fig.suptitle(
            f"{title}\n"
            f"{dict_kwargs['method'].title()} Linkage, "
            f"{dict_kwargs['metric'].title()} Metric",
            y=0.98,
            fontsize=20,
        )
        g.ax_heatmap.set_xlabel("KLIFS Region", fontsize=16)
        g.ax_heatmap.set_ylabel("Gene Symbol", fontsize=16)
        g.ax_heatmap.tick_params(axis="x", which="major", labelsize=12)
        g.ax_heatmap.tick_params(axis="y", which="major", labelsize=12)

        # kinase group legend
        custom_handles = [
            plt.Line2D([], [], color=color, marker="s", linestyle="None", markersize=8)
            for color in DICT_KINASE_GROUP_COLORS.values()
        ]
        custom_labels = list(DICT_KINASE_GROUP_COLORS.keys())
        g.fig.legend(
            handles=custom_handles,
            labels=custom_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=len(custom_labels),
            title="Kinase Groups",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        x_labels = g.ax_heatmap.get_xticklabels()
        for label in x_labels:
            label_text = label.get_text()
            if label_text in custom_palette:
                label.set_color(custom_palette[label_text])

        plt.xticks(rotation=90, ha="center")
        plt.yticks(rotation=0)

        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(
                filename,
                format=filename.split(".")[-1],
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
            plt.ion()
        else:
            plt.show()
            plt.close()

    @staticmethod
    def convert_log_and_truncate(
        x: int | float | str,
        bool_log10: bool,
        max_value: int | None,
    ) -> int | float | str:
        """Convert a value to log10 and truncate if necessary.

        Parameters
        ----------
        x : int | float | str
            Value to convert to log10
        bool_truncate : bool
            Truncate the value to max_value if True; default is True
        max_value : int
            Maximum value to truncate to if bool_truncate is True; default is 1.5

        Returns
        -------
        int | float | str
            Log10 converted value if numeric, otherwise original value;
            truncated to max_value if bool_truncate is True

        """
        # if x is not numeric, try to convert to float or return as is
        if not isinstance(x, (int, float)):
            try:
                x = float(x)
            except ValueError:
                logger.error(f"Value {x} cannot be converted to float.")
            return x

        # nan handling
        if pd.isna(x):
            return np.nan

        # if zezro or negative, return 0
        if x <= 0:
            return 0

        # log conversion
        if bool_log10:
            x = np.log10(x)
        else:
            x = np.log2(x)

        # truncate to max_value if provided
        if max_value is not None:
            if x > max_value:
                return max_value
            else:
                return x
        else:
            return x


@dataclass
class Treatment(StudyData):
    """Class to get treatment information from a cBioPortal study."""

    list_col_explode: list[str] | None = field(default_factory=lambda: ["samples"])
    """List of columns to explode in convert_api_query_to_dataframe;
        ["samples"] if no columns to explode (post-init)."""

    def __post_init__(self):
        """Post-initialization to get mutations from cBioPortal."""
        super().__post_init__()

    def query_sub_api(self) -> list | None:
        """Get mutations cBioPortal data.

        Returns
        -------
        list | None
            cBioPortal data as list of Abstract Base Classes
                objects if successful, otherwise None.

        """
        try:
            # TODO: add incremental error handling beyond missing study
            treatment = self._cbioportal.Treatments.getAllSampleTreatmentsUsingPOST(
                studyViewFilter={"studyIds": [self.study_id], "tiersBooleanMap": {}}
            ).result()
        except Exception as e:
            logger.error(f"Error retrieving treatments for study {self.study_id}: {e}")
            treatment = None
        return treatment


@dataclass
class Clinical(StudyData):
    """Class to get clinical information from a cBioPortal study."""

    def __post_init__(self):
        """Post-initialization to get clinical info from cBioPortal."""
        super().__post_init__()

    def query_sub_api(self) -> list | None:
        """Get clinical info cBioPortal data.

        Returns
        -------
        list | None
            cBioPortal data as list of Abstract Base Classes
                objects if successful, otherwise None.

        """
        try:
            clinical = self._cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(
                studyId=self.study_id
            ).result()
        except Exception as e:
            logger.error(
                f"Error retrieving clinical data for study {self.study_id}: {e}"
            )
            clinical = None
        return clinical


@dataclass
class PanelData(cBioPortalQuery):
    """Class to get gene panel information from a cBioPortal instance."""

    panel_id: str = field(kw_only=True)
    """cBioPortal panel ID."""

    def __post_init__(self):
        super().__post_init__()

    def get_entity_id(self):
        """Get cBioPortal panel ID."""
        return self.panel_id

    def check_entity_id(self) -> bool:
        """Check if the panel ID is valid.

        Returns
        -------
        bool
            True if the panel ID is valid, False otherwise
        """
        panels = self._cbioportal.Gene_Panels.getAllGenePanelsUsingGET().result()
        panel_ids = [panel.genePanelId for panel in panels]
        return self.panel_id in panel_ids


@dataclass
class GenePanel(PanelData):
    """Class to get gene panel information from a cBioPortal study."""

    def __post_init__(self):
        """Post-initialization to get clinical info from cBioPortal."""
        super().__post_init__()

    def query_sub_api(self) -> list | None:
        """Get gene panel genes cBioPortal data.

        Returns
        -------
        list | None
            cBioPortal data as list of Abstract Base Classes
                objects if successful, otherwise None.

        """
        try:
            gene_panels = (
                self._cbioportal.Gene_Panels.getGenePanelUsingGET(
                    genePanelId=self.panel_id
                )
                .result()
                .genes
            )
        except Exception as e:
            logger.error(
                f"Error retrieving gene panel data for panel {self.panel_id}: {e}"
            )
            gene_panels = None
        return gene_panels
