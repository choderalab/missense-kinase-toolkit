"""cBioPortal API client and extraction of missense kinase mutations, treatments, and panels.

Builds on :class:`cBioPortal`/:class:`cBioPortalQuery` to pull study, mutation,
treatment, and gene-panel data; :class:`KinaseMissenseMutations` extracts missense
mutations restricted to kinase genes, reconciled onto canonical UniProt coordinates and
named by the kinase domain that contains them.
"""

import logging
import os
import time
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
from Bio import Align
from bravado.client import SwaggerClient
from mkt.databases import properties
from mkt.databases.api_schema import APIKeySwaggerClient
from mkt.databases.config import get_cbioportal_instance, maybe_get_cbioportal_token
from mkt.databases.constants import normalize_build
from mkt.databases.io_utils import (
    parse_iterabc2dataframe,
    return_kinase_dict,
    save_dataframe_to_csv,
)
from mkt.databases.isoform import (
    TUPLE_DEFAULT_TIERS,
    CanonicalReconciler,
    SourceTier,
    select_domain_name,
)
from mkt.databases.utils import add_one_hot_encoding_to_dataframe
from mkt.schema.utils import TQDM_BAR_FORMAT, split_domain_suffix
from tqdm import tqdm

logger = logging.getLogger(__name__)


DICT_KINASE = return_kinase_dict()

INT_CLIENT_RETRIES = 2
"""int: attempts made to construct the cBioPortal Swagger client before giving up;
the session already retries at the HTTP level, so this only covers a failure that
survives the response (e.g. an unparseable Swagger spec)."""

FLOAT_CLIENT_BACKOFF = 2.0
"""float: seconds to wait before the second client-construction attempt, doubling
thereafter."""


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
        """Post-initialization to set up cBioPortal API client.

        Retries client construction so a transient failure on first contact -- one
        the session-level retries cannot cover, such as a truncated or unparseable
        Swagger spec -- does not leave the client permanently unusable.
        """
        self.instance = get_cbioportal_instance()
        self.url = f"https://{self.instance}/api/v2/api-docs"
        for int_attempt in range(1, INT_CLIENT_RETRIES + 1):
            try:
                self._cbioportal = self.query_api()
                break
            except Exception as e:
                logger.warning(
                    f"Error initializing cBioPortal API client "
                    f"(attempt {int_attempt} of {INT_CLIENT_RETRIES}): {e}\n"
                    "Can still load data from CSV files if pathfile(s) provided.",
                    exc_info=True,
                )
                if int_attempt < INT_CLIENT_RETRIES:
                    time.sleep(FLOAT_CLIENT_BACKOFF * 2 ** (int_attempt - 1))

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
        self._stamp_now()

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
    """DataFrame of cBioPortal data; None if DataFrame could not be created (post-init)."""

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
            self._stamp_now()
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
        if self._cbioportal is None:
            logger.warning(
                f"No cBioPortal client available to check study ID {self.study_id}."
            )
            return False
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
            # use POST endpoint since GET now requires entrezGeneId
            muts = self._cbioportal.Mutations.fetchMutationsInMolecularProfileUsingPOST(
                molecularProfileId=f"{self.study_id}_mutations",
                mutationFilter={"sampleListId": f"{self.study_id}_all"},
                projection="DETAILED",
            ).result()
        except Exception as e:
            logger.error(f"Error retrieving mutations for study {self.study_id}: {e}")
            muts = None
        return muts


@dataclass
class StructuralVariant(StudyData):
    """Class to get structural variants (gene fusions) from a cBioPortal study.

    Fetches from the ``{study_id}_structural_variants`` molecular profile via the
    cBioPortal ``StructuralVariants`` POST endpoint. Pass ``list_entrez`` to restrict
    to specific genes (e.g. FGFR2 / FGFR3 for fusion candidacy) — cBioPortal's
    ``StructuralVariantFilter`` requires either ``entrezGeneIds`` or
    ``sampleMolecularIdentifiers``, so a gene filter is the efficient path; leave it
    ``None`` only if the study is small.
    """

    list_entrez: list[int] | None = None
    """Entrez gene IDs to restrict the fetch to (e.g. FGFR2=2263, FGFR3=2261). None
    fetches across all genes (may require the study to expose a default sample list)."""

    def __post_init__(self):
        super().__post_init__()

    def query_sub_api(self) -> list | None:
        """Get structural-variant cBioPortal data.

        The ``/api/v2/api-docs`` swagger spec used by the base client predates
        structural-variant support (its resource list has no ``StructuralVariants``),
        so this queries the REST endpoint ``POST /api/structural-variant/fetch``
        directly. Response rows are already flat (``site1*`` / ``site2*`` scalar
        fields), so no ABC flattening is required downstream.

        Returns
        -------
        list | None
            cBioPortal structural variants as a list of dicts if successful,
            otherwise None.
        """
        sv_filter: dict = {
            "molecularProfileIds": [f"{self.study_id}_structural_variants"],
        }
        if self.list_entrez is not None:
            sv_filter["entrezGeneIds"] = self.list_entrez
        token = maybe_get_cbioportal_token()
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = requests.post(
                f"https://{self.instance}/api/structural-variant/fetch",
                json=sv_filter,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            svs = resp.json()
        except Exception as e:
            logger.error(
                f"Error retrieving structural variants for study {self.study_id}: {e}"
            )
            svs = None
        return svs

    def convert_api_query_to_dataframe(self) -> pd.DataFrame | None:
        """Build a flat DataFrame from the REST response (a list of dicts).

        Overrides the ABC-flattening base implementation: the structural-variant
        REST payload is already flat, so a direct ``pd.DataFrame`` is sufficient.

        Returns
        -------
        pd.DataFrame | None
            DataFrame of structural variants if successful, otherwise None.
        """
        try:
            return pd.DataFrame(self._data)
        except Exception as e:
            logger.error(f"Error converting structural variants to DataFrame: {e}")
            return None


def apply_gene_replacements(
    dict_hgnc2uniprot: dict[str, str | None],
    dict_kinase: dict[str, object],
    dict_replace: dict[str, str],
) -> dict[str, str | None]:
    """Point renamed cBioPortal genes at their mkt kinase's UniProt accession.

    A replacement (e.g. ``STK19 -> WHR1``) applies only when the target kinase exists.

    Parameters
    ----------
    dict_hgnc2uniprot : dict[str, str | None]
        cBioPortal gene symbol -> UniProt accession from HGNC.
    dict_kinase : dict[str, KinaseInfo]
        Kinase name -> kinase object.
    dict_replace : dict[str, str]
        cBioPortal gene symbol -> mkt kinase name.

    Returns
    -------
    dict[str, str | None]
        A copy of ``dict_hgnc2uniprot`` with the applicable replacements.
    """
    dict_out = dict(dict_hgnc2uniprot)
    for cbio_name, mkt_name in dict_replace.items():
        if cbio_name not in dict_out:
            continue
        if mkt_name not in dict_kinase:
            logger.info(
                f"Skipping {cbio_name} -> {mkt_name} replacement; "
                f"{mkt_name} is not in DICT_KINASE."
            )
            continue
        dict_out[cbio_name] = split_domain_suffix(dict_kinase[mkt_name].uniprot_id)[0]
    return dict_out


def return_gene2names(
    dict_hgnc2uniprot: dict[str, str | None],
    dict_kinase: dict[str, object],
) -> dict[str, list[str]]:
    """Map cBioPortal gene symbols to their ordered mkt kinase names.

    A symbol matches every kinase whose base UniProt accession equals the symbol's, so a
    multi-domain gene maps to all its domains (e.g. ``JAK1 -> ["JAK1_1", "JAK1_2"]``).

    Parameters
    ----------
    dict_hgnc2uniprot : dict[str, str | None]
        cBioPortal gene symbol -> UniProt accession.
    dict_kinase : dict[str, KinaseInfo]
        Kinase name -> kinase object.

    Returns
    -------
    dict[str, list[str]]
        Gene symbol -> sorted kinase names; symbols matching no kinase are omitted.
    """
    dict_accession2names: dict[str, list[str]] = {}
    for name, obj in dict_kinase.items():
        accession = split_domain_suffix(obj.uniprot_id)[0]
        dict_accession2names.setdefault(accession, []).append(name)
    return {
        symbol: sorted(dict_accession2names[accession])
        for symbol, accession in dict_hgnc2uniprot.items()
        if accession in dict_accession2names
    }


def return_gene2seq(
    dict_gene2names: dict[str, list[str]],
    dict_kinase: dict[str, object],
) -> dict[str, str]:
    """Return each gene's UniProt canonical sequence, shared by all its domains.

    Parameters
    ----------
    dict_gene2names : dict[str, list[str]]
        Gene symbol -> kinase names (see :func:`return_gene2names`).
    dict_kinase : dict[str, KinaseInfo]
        Kinase name -> kinase object.

    Returns
    -------
    dict[str, str]
        Gene symbol -> canonical sequence.

    Raises
    ------
    ValueError
        If a gene's domains carry different canonical sequences, which would make the
        residue check in reconciliation unreliable.
    """
    dict_out = {}
    for symbol, list_names in dict_gene2names.items():
        set_seq = {dict_kinase[name].uniprot.canonical_seq for name in list_names}
        if len(set_seq) != 1:
            raise ValueError(
                f"{symbol} domains {list_names} have different canonical sequences."
            )
        dict_out[symbol] = set_seq.pop()
    return dict_out


def return_hgvsg_list(df: pd.DataFrame, str_build: str) -> list[str | None]:
    """Return a Genome Nexus genomic HGVS string per single-base substitution row.

    Parameters
    ----------
    df : pd.DataFrame
        cBioPortal mutations with ``chr``, ``startPosition``, ``referenceAllele``,
        ``variantAllele`` and (optionally) ``ncbiBuild``.
    str_build : str
        Genome build the strings are valid for; rows on another build get None. Builds
        are compared after alias normalization (``"37"``/``"hg19"`` mean GRCh37).

    Returns
    -------
    list[str | None]
        ``"7:g.140453136A>T"``-style strings, or None for non-SNV rows, rows on another
        build, or when the coordinate columns are missing.
    """
    list_cols = ["chr", "startPosition", "referenceAllele", "variantAllele"]
    if not set(list_cols) <= set(df.columns):
        return [None] * len(df)
    str_canonical = normalize_build(str_build)
    list_build = (
        df["ncbiBuild"].tolist() if "ncbiBuild" in df.columns else [str_build] * len(df)
    )
    list_hgvsg = []
    for chrom, start, ref, alt, build in zip(
        df["chr"],
        df["startPosition"],
        df["referenceAllele"],
        df["variantAllele"],
        list_build,
    ):
        bool_snv = (
            str_canonical is not None
            and normalize_build(build) == str_canonical
            and isinstance(ref, str)
            and isinstance(alt, str)
            and len(ref) == len(alt) == 1
            and ref in "ACGT"
            and alt in "ACGT"
            and pd.notna(start)
        )
        list_hgvsg.append(f"{chrom}:g.{int(start)}{ref}>{alt}" if bool_snv else None)
    return list_hgvsg


def return_genomic_location_list(df: pd.DataFrame, str_build: str) -> list[str | None]:
    """Return an OncoKB ``genomicLocation`` string per mutation row.

    OncoKB's ``byGenomicChange`` endpoint takes ``chromosome,start,end,ref,alt`` and
    resolves the alteration on its own transcript, which is the only frame-independent
    way to annotate a cohort: OncoKB annotates FGFR1 on the MSKCC-override isoform but
    TGFBR2 on the UniProt canonical, so a ``proteinChange`` from one study transcript
    asks about the wrong residue for some genes.

    Unlike :func:`return_hgvsg_list`, which is limited to single-base substitutions,
    this covers indels too, since the endpoint takes an explicit end coordinate.

    Parameters
    ----------
    df : pd.DataFrame
        cBioPortal mutations with ``chr``, ``startPosition``, ``endPosition``,
        ``referenceAllele``, ``variantAllele`` and (optionally) ``ncbiBuild``.
    str_build : str
        Genome build the coordinates are valid for; rows on another build get None.
        Builds are compared after alias normalization (``"37"``/``"hg19"`` mean GRCh37).

    Returns
    -------
    list[str | None]
        ``"7,140453136,140453136,A,T"``-style strings, or None for rows on another
        build or when the coordinate columns are missing.
    """
    list_cols = [
        "chr",
        "startPosition",
        "endPosition",
        "referenceAllele",
        "variantAllele",
    ]
    if not set(list_cols) <= set(df.columns):
        return [None] * len(df)
    str_canonical = normalize_build(str_build)
    list_build = (
        df["ncbiBuild"].tolist() if "ncbiBuild" in df.columns else [str_build] * len(df)
    )
    list_location = []
    for chrom, start, end, ref, alt, build in zip(
        df["chr"],
        df["startPosition"],
        df["endPosition"],
        df["referenceAllele"],
        df["variantAllele"],
        list_build,
    ):
        bool_usable = (
            str_canonical is not None
            and normalize_build(build) == str_canonical
            and pd.notna(chrom)
            and pd.notna(start)
            and pd.notna(end)
            and isinstance(ref, str)
            and isinstance(alt, str)
        )
        list_location.append(
            f"{chrom},{int(start)},{int(end)},{ref},{alt}" if bool_usable else None
        )
    return list_location


def assign_mkt_name(
    df: pd.DataFrame,
    dict_gene2names: dict[str, list[str]],
    dict_kinase: dict[str, object],
    col_gene: str = "gene_hugoGeneSymbol",
    col_idx: str = "uniprot_idx",
) -> pd.DataFrame:
    """Add ``mkt_name`` and ``in_kinase_domain`` per mutation.

    A multi-domain gene resolves to the domain whose adjudicated kinase-domain span holds
    the canonical position; a position outside every domain keeps the mkt base name.

    Parameters
    ----------
    df : pd.DataFrame
        Mutations with a gene symbol and canonical position column.
    dict_gene2names : dict[str, list[str]]
        Gene symbol -> kinase names (see :func:`return_gene2names`).
    dict_kinase : dict[str, KinaseInfo]
        Kinase name -> kinase object.
    col_gene : str, optional
        Gene symbol column, by default "gene_hugoGeneSymbol".
    col_idx : str, optional
        Canonical position column, by default "uniprot_idx".

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with ``mkt_name`` and ``in_kinase_domain`` (None where the gene
        is unknown or the position is missing).
    """
    dict_name2span = {
        name: (
            dict_kinase[name].adjudicate_kd_start(),
            dict_kinase[name].adjudicate_kd_end(),
        )
        for list_names in dict_gene2names.values()
        for name in list_names
    }
    list_mkt_name, list_in_kd = [], []
    for symbol, idx in zip(df[col_gene], df[col_idx]):
        list_names = dict_gene2names.get(symbol)
        if not list_names:
            list_mkt_name.append(None)
            list_in_kd.append(None)
            continue
        str_base = split_domain_suffix(list_names[0])[0]
        name, bool_in_kd = select_domain_name(
            str_base, list_names, dict_name2span, None if pd.isna(idx) else int(idx)
        )
        list_mkt_name.append(name)
        list_in_kd.append(bool_in_kd)
    df = df.copy()
    df["mkt_name"] = list_mkt_name
    df["in_kinase_domain"] = list_in_kd
    return df


@dataclass
class KinaseMissenseMutations(Mutations):
    """Class to get kinase mutations from a cBioPortal study."""

    dict_replace: dict[str, str] = field(default_factory=lambda: {"STK19": "WHR1"})
    """Dictionary mapping cBioPortal to mkt gene names for mismatches; default is {"STK19": "WHR1"}."""
    str_blosom: str = "BLOSUM80"
    """BLOSUM matrix to use for mutation analysis; default is "BLOSUM80"."""
    pathfile_filter: str | None = None
    """Path to CSV file for filtered kinase missense mutations; default is None."""
    bool_drop_mismatch: bool = False
    """Use the legacy filter that drops every mutation of a gene with any canonical-residue mismatch instead of per-row reconciliation, by default False."""
    tuple_sources: tuple[SourceTier, ...] = TUPLE_DEFAULT_TIERS
    """Reconciliation tiers tried in order, by default direct, mskcc, refseq, genomenexus."""
    str_isoform_override: str = "mskcc"
    """Isoform-override source for the transcript tier, by default "mskcc"."""
    str_build: str = "GRCh37"
    """Genome build for transcript and variant lookups; only rows on this build get a Genome Nexus variant, by default "GRCh37"."""
    bool_drop_unreconciled: bool = True
    """Drop rows whose position could not be reconciled onto the canonical sequence, by default True."""
    str_col_gene: str = "mkt_name"
    """Column :meth:`generate_pivot_table` groups mutations by, by default "mkt_name"."""
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
        for hgnc_name in tqdm(
            dict_hgnc2uniprot.keys(),
            desc="Querying HGNC...",
            bar_format=TQDM_BAR_FORMAT,
        ):
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

        return apply_gene_replacements(
            dict_hgnc2uniprot, dict_kinase, self.dict_replace
        )

    def get_kinase_missense_mutations(
        self,
        bool_save: bool = False,
    ) -> pd.DataFrame | None:
        """Get kinase missense mutations on canonical UniProt coordinates.

        Parameters
        ----------
        bool_save : bool, optional
            Also save the mutations as a CSV file, by default False.

        Returns
        -------
        pd.DataFrame | None
            Mutations with ``uniprot_idx``, ``reconcile_source``, ``mkt_name``,
            ``in_kinase_domain`` and region annotations, or None if the study has no
            mutation data.
        """
        if self._df is None:
            logger.error(f"No mutation data for study {self.study_id}.")
            return None

        col_gene = self.return_adjusted_colname("hugoGeneSymbol")
        df_missense = self.filter_single_aa_missense_mutations(self._df.copy())

        dict_hgnc2uniprot = self.query_hgnc_gene_names(
            list_hgnc=df_missense[col_gene].tolist(),
            dict_kinase=DICT_KINASE,
        )
        dict_gene2names = return_gene2names(dict_hgnc2uniprot, DICT_KINASE)
        dict_gene2seq = return_gene2seq(dict_gene2names, DICT_KINASE)

        df_kinase = df_missense.loc[
            df_missense[col_gene].isin(dict_gene2names.keys()), :
        ].reset_index(drop=True)

        if self.bool_drop_mismatch:
            df_kinase = self.remove_mismatched_uniprot_mutations(
                df_kinase, dict_gene2seq
            )
        else:
            df_kinase = self.reconcile_uniprot_positions(df_kinase, dict_gene2seq)

        df_kinase = assign_mkt_name(
            df_kinase, dict_gene2names, DICT_KINASE, col_gene=col_gene
        )
        df_kinase = self.annotate_kinase_regions(df=df_kinase, dict_kinase=DICT_KINASE)

        if bool_save:
            filename = f"{self.study_id}_kinase_missense_mutations.csv"
            save_dataframe_to_csv(df_kinase, filename)
        return df_kinase

    def remove_mismatched_uniprot_mutations(
        self,
        df: pd.DataFrame,
        dict_gene2seq: dict[str, str],
    ) -> pd.DataFrame:
        """Drop every mutation of any gene with a mismatch to its canonical sequence.

        Legacy alternative to :meth:`reconcile_uniprot_positions`; emits the same
        ``uniprot_idx`` and ``reconcile_source`` columns, with every kept row ``"direct"``.

        Parameters
        ----------
        df : pd.DataFrame
            Kinase missense mutations.
        dict_gene2seq : dict[str, str]
            Gene symbol -> UniProt canonical sequence.

        Returns
        -------
        pd.DataFrame
            Mutations of genes whose every reported residue matches the canonical.
        """
        col_gene = self.return_adjusted_colname("hugoGeneSymbol")
        set_mismatch = set()
        for symbol, codon in zip(df[col_gene], df["proteinChange"]):
            idx = self.try_except_middle_int(codon)
            seq = dict_gene2seq.get(symbol)
            if seq is None or idx is None or not 1 <= idx <= len(seq):
                set_mismatch.add(symbol)
            elif seq[idx - 1] != codon[0]:
                set_mismatch.add(symbol)

        if set_mismatch:
            logger.error(
                "Dropping every mutation of kinases with a mismatch between cBioPortal "
                f"and canonical UniProt sequences: {sorted(set_mismatch)}"
            )
        df = df.loc[~df[col_gene].isin(set_mismatch), :].reset_index(drop=True)
        df["uniprot_idx"] = pd.array(
            [self.try_except_middle_int(codon) for codon in df["proteinChange"]],
            dtype="Int64",
        )
        df["reconcile_source"] = str(SourceTier.direct)
        return df

    def reconcile_uniprot_positions(
        self,
        df: pd.DataFrame,
        dict_gene2seq: dict[str, str],
    ) -> pd.DataFrame:
        """Reconcile each mutation's position onto the canonical UniProt sequence.

        Runs :class:`~mkt.databases.isoform.CanonicalReconciler` over the rows; rows that
        cannot be reconciled are logged per gene and, with :attr:`bool_drop_unreconciled`,
        dropped individually.

        Parameters
        ----------
        df : pd.DataFrame
            Kinase missense mutations.
        dict_gene2seq : dict[str, str]
            Gene symbol -> UniProt canonical sequence.

        Returns
        -------
        pd.DataFrame
            Mutations with ``uniprot_idx`` and ``reconcile_source`` added.
        """
        col_gene = self.return_adjusted_colname("hugoGeneSymbol")
        list_codon = df["proteinChange"].tolist()
        list_refseq = (
            df["refseqMrnaId"].tolist() if "refseqMrnaId" in df.columns else None
        )

        reconciler = CanonicalReconciler(
            dict_canonical=dict_gene2seq,
            tuple_sources=self.tuple_sources,
            str_override=self.str_isoform_override,
            str_build=self.str_build,
        )
        list_idx, list_source = reconciler.reconcile_many(
            df[col_gene].tolist(),
            [self.try_except_middle_int(codon) for codon in list_codon],
            [
                codon[0].upper() if isinstance(codon, str) and codon else None
                for codon in list_codon
            ],
            list_refseq=list_refseq,
            list_hgvsg=return_hgvsg_list(df, self.str_build),
        )

        df = df.copy()
        df["uniprot_idx"] = pd.array(list_idx, dtype="Int64")
        df["reconcile_source"] = [None if s is None else str(s) for s in list_source]

        mask_drop = df["uniprot_idx"].isna()
        if mask_drop.any():
            df_drop = df.loc[mask_drop]
            list_summary = [
                f"{symbol} (n={len(group)}, residues {group['proteinChange'].map(self.try_except_middle_int).min()}-"
                f"{group['proteinChange'].map(self.try_except_middle_int).max()})"
                for symbol, group in df_drop.groupby(col_gene)
            ]
            logger.warning(
                f"{int(mask_drop.sum())} of {len(df)} mutations could not be reconciled "
                f"onto canonical UniProt sequences: {', '.join(list_summary)}"
            )
            if self.bool_drop_unreconciled:
                df = df.loc[~mask_drop, :].reset_index(drop=True)
        return df

    def annotate_kinase_regions(
        self,
        df: pd.DataFrame,
        dict_kinase: dict[str, object],
    ) -> pd.DataFrame:
        """Annotate KLIFS region, KinCoRe domain membership and residue-change properties.

        Keyed on ``mkt_name`` and the canonical ``uniprot_idx``; a row whose ``mkt_name``
        is not a kinase entry (a bare multi-domain gene symbol) gets no region annotation.

        Parameters
        ----------
        df : pd.DataFrame
            Mutations with ``mkt_name``, ``uniprot_idx`` and ``proteinChange``.
        dict_kinase : dict[str, KinaseInfo]
            Kinase name -> kinase object.

        Returns
        -------
        pd.DataFrame
            Mutations with ``klifs_region``, ``kincore_kd``, ``blosum_penalty`` and
            one-hot charge/polarity/volume columns.
        """
        mx_blosum = Align.substitution_matrices.load(self.str_blosom)

        # reverse KLIFS maps built once per kinase rather than scanned per row
        dict_idx2klifs: dict[str, dict[int, str]] = {}
        for name in df["mkt_name"].dropna().unique():
            obj = dict_kinase.get(name)
            if obj is None or obj.KLIFS2UniProtIdx is None:
                continue
            dict_rev: dict[int, str] = {}
            for region, idx_region in obj.KLIFS2UniProtIdx.items():
                if idx_region is not None:
                    dict_rev.setdefault(idx_region, region)
            dict_idx2klifs[name] = dict_rev

        dict_out = {
            "klifs_region": [],
            "kincore_kd": [],
            "blosum_penalty": [],
            "charge": [],
            "polarity": [],
            "volume": [],
        }
        for name, idx, codon in zip(
            df["mkt_name"], df["uniprot_idx"], df["proteinChange"]
        ):
            aa_from = codon[0].upper()
            aa_to = codon[-1].upper()
            obj = dict_kinase.get(name)
            idx = None if pd.isna(idx) else int(idx)

            # KLIFS
            dict_rev = dict_idx2klifs.get(name)
            dict_out["klifs_region"].append(
                dict_rev.get(idx) if dict_rev is not None and idx is not None else None
            )

            # KinCoRe (an MSA-only shell has no FASTA -> treat as no KinCoRe KD info)
            fasta = (
                obj.kincore.fasta
                if obj is not None and obj.kincore is not None
                else None
            )
            if fasta is None or idx is None:
                dict_out["kincore_kd"].append(None)
            else:
                dict_out["kincore_kd"].append(fasta.start <= idx <= fasta.end)

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
        col_gene = self.str_col_gene
        if col_gene not in df.columns:
            col_fallback = self.return_adjusted_colname("hugoGeneSymbol")
            logger.warning(
                f"Column {col_gene} not in DataFrame (e.g. loaded from an older CSV); "
                f"grouping by {col_fallback}."
            )
            col_gene = col_fallback
        col_klifs = "klifs_region"
        # BLOSUM take mean, others take value counts
        if colname == "blosum_penalty":
            pivot_table = (
                df.groupby([col_gene, col_klifs])[colname]
                .agg("mean")
                .unstack(fill_value=0)
            )
            title = ", BLOSUM Penalty (Mean)"
        # one-hot encoding columns
        elif colname.startswith("_"):
            # keep only values that correpond to the one-hot encoding
            df_temp = df.loc[df[colname] == bool_onehot, :].reset_index(drop=True)
            pivot_table = (
                df_temp.groupby([col_gene, col_klifs])[colname]
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
                df.groupby(col_gene)[colname]
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


@dataclass(kw_only=True)
class Clinical(StudyData):
    """Class to get clinical information from a cBioPortal study."""

    bool_sample: bool
    """If True, return sample-level clinical data; if False, return patient-level clinical data."""

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
        bool_sample : bool
            If True, return sample-level clinical data; if False, return patient-level clinical data

        """
        try:
            if self.bool_sample:
                clinical = (
                    self._cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(
                        studyId=self.study_id,
                        clinicalDataType="SAMPLE",
                    ).result()
                )
            else:
                clinical = (
                    self._cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(
                        studyId=self.study_id,
                        clinicalDataType="PATIENT",
                    ).result()
                )
        except Exception as e:
            logger.error(
                f"Error retrieving clinical data for study {self.study_id}: {e}"
            )
            clinical = None
        return clinical


@dataclass
class ClinicalSample(Clinical):
    """Class to get sample-level clinical information from a cBioPortal study."""

    bool_sample: bool = field(init=False, default=True)
    """If True, return sample-level clinical data; if False, return patient-level clinical data"""

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ClinicalPatient(Clinical):
    """Class to get patient-level clinical information from a cBioPortal study."""

    bool_sample: bool = field(init=False, default=False)
    """If True, return sample-level clinical data; if False, return patient-level clinical data"""

    def __post_init__(self):
        super().__post_init__()


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
        if self._cbioportal is None:
            logger.warning(
                f"No cBioPortal client available to check panel ID {self.panel_id}."
            )
            return False
        try:
            panels = self._cbioportal.Gene_Panels.getAllGenePanelsUsingGET().result()
            panel_ids = [panel.genePanelId for panel in panels]
            return self.panel_id in panel_ids
        except Exception as e:
            logger.warning(f"Error checking panel ID {self.panel_id}: {e}")
            return False


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
