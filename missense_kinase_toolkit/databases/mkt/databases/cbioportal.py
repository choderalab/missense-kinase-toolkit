"""cBioPortal API client and extraction of missense kinase mutations, treatments, and panels.

Builds on :class:`cBioPortal`/:class:`cBioPortalQuery` to pull study, mutation,
treatment, and gene-panel data; :class:`KinaseMissenseMutations` extracts missense
mutations restricted to kinase genes, reconciled onto canonical UniProt coordinates and
named by the kinase domain that contains them.
"""

import logging
import os
import re
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
from mkt.databases.constants import (
    EntrezGeneIDPattern,
    MissenseProteinChangePattern,
    normalize_build,
)
from mkt.databases.genomenexus import annotate_genomic_locations
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
        """Post-initialization to set up the cBioPortal API client."""
        self.set_instance()
        self.init_client()

    def set_instance(self) -> None:
        """Set the cBioPortal instance and its Swagger spec URL (no network)."""
        self.instance = get_cbioportal_instance()
        self.url = f"https://{self.instance}/api/v2/api-docs"

    def init_client(self) -> None:
        """Build the cBioPortal API client.

        Retries client construction so a transient failure on first contact -- one
        the session-level retries cannot cover, such as a truncated or unparseable
        Swagger spec -- does not leave the client permanently unusable.
        """
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
        """Load from ``pathfile`` if given, else query the API.

        The API client is only built (and the entity ID checked) when a query is
        needed, so loading cached CSVs makes no cBioPortal requests.
        """
        self.set_instance()
        if self.pathfile is not None:
            try:
                self._df = self.load_from_csv()
                return
            except Exception as e:
                logger.error(
                    f"Error loading DataFrame from {self.pathfile}: {e}\n"
                    "Regenerating DataFrame from API query..."
                )
        self.init_client()
        # None means the lookup itself failed (already logged); only warn on a miss
        if self.check_entity_id() is False:
            logger.warning(
                f"Study {self.get_entity_id()} not found "
                f"in cBioPortal instance {self.instance}"
            )
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
    def check_entity_id(self) -> bool | None:
        """Check if the entity ID is valid.

        Returns
        -------
        bool | None
            True if the entity ID is valid, False if not; None if the lookup
            could not be made (no client or a failed request)
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

    def check_entity_id(self) -> bool | None:
        """Check if the study ID is valid.

        Returns
        -------
        bool | None
            True if the study ID is valid, False if not; None if the lookup
            could not be made (no client or a failed request)
        """
        if self._cbioportal is None:
            logger.warning(
                f"No cBioPortal client available to check study ID {self.study_id}."
            )
            return None
        try:
            studies = self._cbioportal.Studies.getAllStudiesUsingGET().result()
            study_ids = [study.studyId for study in studies]
            return self.study_id in study_ids
        except Exception as e:
            logger.warning(f"Error checking study ID {self.study_id}: {e}")
            return None


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


def clean_entrez_id(value: object) -> str | None:
    """Return an Entrez Gene ID as a string if it is a valid ID.

    Parameters
    ----------
    value : object
        Entrez Gene ID as found in cBioPortal's ``entrezGeneId`` (an int or string);
        missing values are allowed. A float such as ``8358.0`` is rejected, so read
        the column as a nullable integer (``Int64``) first.

    Returns
    -------
    str | None
        The ID (e.g. ``"8358"``) if it matches
        :data:`~mkt.databases.constants.EntrezGeneIDPattern`, else None.
    """
    str_entrez = str(value).strip()
    if re.fullmatch(EntrezGeneIDPattern, str_entrez):
        return str_entrez
    return None


def is_missense_protein_change(value: object) -> bool:
    """Return whether a ``proteinChange`` is a single-residue missense substitution.

    Parameters
    ----------
    value : object
        cBioPortal ``proteinChange`` (e.g. ``"V600E"``); missing values are allowed.

    Returns
    -------
    bool
        True if it matches
        :data:`~mkt.databases.constants.MissenseProteinChangePattern` with different
        reference and alternate residues (so not a stop-retained ``*28*``).
    """
    match = re.fullmatch(MissenseProteinChangePattern, str(value).strip())
    return match is not None and match.group(1) != match.group(3)


def log_hgnc_query_summary(
    list_resolved: list[str],
    list_no_uniprot: list[str],
    list_not_found: list[str],
    list_err: list[str],
) -> None:
    """Log the outcome of an HGNC gene-name query, one gene per line.

    Parameters
    ----------
    list_resolved : list[str]
        ``"old -> new (via field)"`` for symbols found by a fallback (INFO).
    list_no_uniprot : list[str]
        Genes HGNC knows but that have no UniProt ID, e.g. readthroughs (WARNING).
    list_not_found : list[str]
        Symbols no lookup found (WARNING).
    list_err : list[str]
        ``"symbol: error"`` for unexpected failures (ERROR).
    """
    for str_msg, list_items, int_level in (
        (
            "Resolved by fallback (HGNC no longer uses the symbol)",
            list_resolved,
            logging.INFO,
        ),
        ("Found in HGNC but without a UniProt ID", list_no_uniprot, logging.WARNING),
        (
            "Not found in HGNC by symbol, Entrez ID or previous symbol",
            list_not_found,
            logging.WARNING,
        ),
        ("Errors retrieving HGNC gene names", list_err, logging.ERROR),
    ):
        if list_items:
            str_items = "\n".join(f"  {item}" for item in sorted(list_items))
            logger.log(int_level, f"{str_msg} ({len(list_items)}):\n{str_items}")


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


def return_hgvsg_list(
    df: pd.DataFrame, str_build: str, list_build: list | None = None
) -> list[str | None]:
    """Return a Genome Nexus genomic HGVS string per single-base substitution row.

    Parameters
    ----------
    df : pd.DataFrame
        cBioPortal mutations with ``chr``, ``startPosition``, ``referenceAllele``,
        ``variantAllele`` and (optionally) ``ncbiBuild``.
    str_build : str
        Genome build the strings are valid for; rows on another build get None. Builds
        are compared after alias normalization (``"37"``/``"hg19"`` mean GRCh37).
    list_build : list | None
        Per-row builds used instead of ``ncbiBuild`` (e.g. from
        :func:`return_verified_build_list`); None reads ``ncbiBuild``.

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
    if list_build is None:
        list_build = (
            df["ncbiBuild"].tolist()
            if "ncbiBuild" in df.columns
            else [str_build] * len(df)
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


def return_genomic_location_list(
    df: pd.DataFrame, str_build: str, list_build: list | None = None
) -> list[str | None]:
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
    list_build : list | None
        Per-row builds used instead of ``ncbiBuild`` (e.g. from
        :func:`return_verified_build_list`); None reads ``ncbiBuild``.

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
    if list_build is None:
        list_build = (
            df["ncbiBuild"].tolist()
            if "ncbiBuild" in df.columns
            else [str_build] * len(df)
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


def return_verified_build_list(
    df: pd.DataFrame, str_build: str, str_col_gene: str | None = None
) -> list[str | None]:
    """Return each row's genome build, confirming untagged/mismatched rows via Genome Nexus.

    A row whose ``ncbiBuild`` is missing or differs from ``str_build`` is annotated on
    ``str_build`` (MSKCC isoform override, cBioPortal's frame) and taken as ``str_build``
    when Genome Nexus returns the row's own gene and protein change; otherwise it keeps
    its normalized tag (None when missing), so the locus builders still skip it.

    Parameters
    ----------
    df : pd.DataFrame
        cBioPortal mutations with the coordinate columns, ``proteinChange``, a gene
        column and (optionally) ``ncbiBuild``.
    str_build : str
        Cohort genome build.
    str_col_gene : str | None
        Gene-symbol column; None picks ``gene_hugoGeneSymbol`` or ``hugoGeneSymbol``.

    Returns
    -------
    list[str | None]
        Normalized build per row, for the ``list_build`` of :func:`return_hgvsg_list`
        and :func:`return_genomic_location_list`.
    """
    str_canonical = normalize_build(str_build)
    if "ncbiBuild" not in df.columns:
        return [str_canonical] * len(df)
    list_tag = [normalize_build(build) for build in df["ncbiBuild"]]
    if str_canonical is None:
        return list_tag

    if str_col_gene is None:
        str_col_gene = next(
            (c for c in ("gene_hugoGeneSymbol", "hugoGeneSymbol") if c in df.columns),
            None,
        )
    if str_col_gene is None or "proteinChange" not in df.columns:
        return list_tag

    # locations as if every row were on the cohort build; only disputed rows are checked
    list_location = return_genomic_location_list(
        df, str_build, list_build=[str_canonical] * len(df)
    )
    list_idx = [
        i
        for i, (tag, loc) in enumerate(zip(list_tag, list_location))
        if tag != str_canonical and loc is not None
    ]
    if not list_idx:
        return list_tag

    dict_annotation = annotate_genomic_locations(
        sorted({list_location[i] for i in list_idx}),
        build=str_canonical,
        isoform_override="mskcc",
    )
    list_gene = df[str_col_gene].tolist()
    list_change = df["proteinChange"].tolist()
    n_verified = 0
    for i in list_idx:
        summary = dict_annotation.get(list_location[i]) or {}
        str_change = (summary.get("hgvspShort") or "").removeprefix("p.")
        if (
            summary.get("hugoGeneSymbol") == list_gene[i]
            and str_change == list_change[i]
        ):
            list_tag[i] = str_canonical
            n_verified += 1
    logger.info(
        f"{len(list_idx)} row(s) had a missing or non-{str_canonical} ncbiBuild; "
        f"{n_verified} verified on {str_canonical} via Genome Nexus, "
        f"{len(list_idx) - n_verified} left unresolved."
    )
    return list_tag


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
    bool_drop_unreconciled: bool = False
    """Drop rows whose position could not be reconciled onto the canonical sequence, by default False (kept with unreconciled_reason so callers can report them)."""
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
            df_missense["proteinChange"].apply(is_missense_protein_change), :
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

    def query_hgnc_uniprot_ids(
        self,
        list_hgnc: list[str],
        dict_kinase: dict[str, object],
        dict_entrez: dict[str, object] | None = None,
    ) -> dict[str, str | None]:
        """Query HGNC gene names from a DataFrame of mutations.

        A symbol HGNC no longer recognizes (e.g. the renamed histones, ``HIST1H3B``
        -> ``H3C2``) is retried by its Entrez Gene ID, then as a previous symbol; a
        fallback is used only when it matches exactly one current gene.

        Parameters
        ----------
        list_hgnc : list[str]
            List of HGNC gene names to query
        dict_kinase : dict[str, object]
            Dictionary mapping kinase names to KinaseInfo objects
        dict_entrez : dict[str, object] | None
            Optional mapping of gene name to Entrez Gene ID (cBioPortal's
            ``entrezGeneId``), used for the Entrez fallback

        Returns
        -------
        dict
            Dictionary mapping HGNC gene names from cBioPortal to Uniprot IDs

        """
        from mkt.databases import hgnc

        dict_entrez = dict_entrez or {}
        dict_hgnc2uniprot = dict.fromkeys(set(list_hgnc))

        list_resolved, list_no_uniprot, list_not_found, list_err = [], [], [], []
        for hgnc_name in tqdm(
            dict_hgnc2uniprot.keys(),
            desc="Querying HGNC...",
            bar_format=TQDM_BAR_FORMAT,
        ):
            obj_hgnc = hgnc.HGNC(input_symbol_or_id=hgnc_name)
            try:
                dict_info, str_via = self._query_hgnc_with_fallback(
                    obj_hgnc, clean_entrez_id(dict_entrez.get(hgnc_name))
                )
            except Exception as e:
                list_err.append(f"{hgnc_name}: {e}")
                continue
            if dict_info is None:
                list_not_found.append(hgnc_name)
                continue
            if str_via is not None:
                list_resolved.append(f"{hgnc_name} -> {obj_hgnc.hgnc} (via {str_via})")
            list_uniprot = (dict_info["uniprot_ids"] or [[]])[0]
            if len(list_uniprot) == 0:
                list_no_uniprot.append(obj_hgnc.hgnc)
                continue
            dict_hgnc2uniprot[hgnc_name] = list_uniprot[0]

        log_hgnc_query_summary(list_resolved, list_no_uniprot, list_not_found, list_err)

        return apply_gene_replacements(
            dict_hgnc2uniprot, dict_kinase, self.dict_replace
        )

    @staticmethod
    def _query_hgnc_with_fallback(
        obj_hgnc: object, str_entrez: str | None
    ) -> tuple[dict | None, str | None]:
        """Fetch a gene's HGNC symbol and UniProt IDs, falling back on a miss.

        Tries the symbol, then the Entrez Gene ID, then the symbol as a previous
        symbol. A fallback search matching exactly one gene sets ``obj_hgnc.hgnc``
        to the current symbol, which is then fetched.

        Parameters
        ----------
        obj_hgnc : mkt.databases.hgnc.HGNC
            HGNC client for the cohort's gene symbol
        str_entrez : str | None
            The gene's Entrez Gene ID, if known

        Returns
        -------
        tuple[dict | None, str | None]
            The ``symbol``/``uniprot_ids`` fetch (None if no lookup found the gene)
            and the fallback field that found it (None for a direct symbol hit)
        """
        list_fields = ["symbol", "uniprot_ids"]
        dict_info = obj_hgnc.maybe_get_info_from_hgnc_fetch(list_to_extract=list_fields)
        if dict_info is not None and dict_info["symbol"] is not None:
            return dict_info, None

        for str_field, str_term in (
            ("entrez_id", str_entrez),
            ("prev_symbol", obj_hgnc.input_symbol_or_id),
        ):
            if str_term is None:
                continue
            list_match = obj_hgnc.maybe_get_symbol_from_hgnc_search(
                custom_field=str_field, custom_term=str_term
            )
            # only an unambiguous match; the search then updated obj_hgnc.hgnc
            if list_match is None or len(list_match) != 1:
                continue
            dict_info = obj_hgnc.maybe_get_info_from_hgnc_fetch(
                list_to_extract=list_fields
            )
            if dict_info is not None and dict_info["symbol"] is not None:
                return dict_info, str_field

        return None, None

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
        col_entrez = self.return_adjusted_colname("entrezGeneId")
        df_missense = self.filter_single_aa_missense_mutations(self._df.copy())

        # nullable integers: a column with missing IDs would otherwise read as floats
        dict_entrez = (
            df_missense.drop_duplicates(subset=col_gene)
            .set_index(col_gene)[col_entrez]
            .pipe(pd.to_numeric, errors="coerce")
            .astype("Int64")
            .to_dict()
            if col_entrez in df_missense.columns
            else None
        )
        dict_hgnc2uniprot = self.query_hgnc_uniprot_ids(
            list_hgnc=df_missense[col_gene].tolist(),
            dict_kinase=DICT_KINASE,
            dict_entrez=dict_entrez,
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
        ``uniprot_idx``, ``reconcile_source`` and ``unreconciled_reason`` columns, with
        every kept row ``"direct"``.

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
        df["unreconciled_reason"] = None
        return df

    def reconcile_uniprot_positions(
        self,
        df: pd.DataFrame,
        dict_gene2seq: dict[str, str],
    ) -> pd.DataFrame:
        """Reconcile each mutation's position onto the canonical UniProt sequence.

        Runs :class:`~mkt.databases.isoform.CanonicalReconciler` over the rows; rows that
        cannot be reconciled get an ``unreconciled_reason``
        (:class:`~mkt.databases.isoform.UnreconciledReason`), are logged per gene and,
        with :attr:`bool_drop_unreconciled`, dropped individually.

        Parameters
        ----------
        df : pd.DataFrame
            Kinase missense mutations.
        dict_gene2seq : dict[str, str]
            Gene symbol -> UniProt canonical sequence.

        Returns
        -------
        pd.DataFrame
            Mutations with ``uniprot_idx``, ``reconcile_source`` and
            ``unreconciled_reason`` (None for reconciled rows) added.
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
        list_symbol = df[col_gene].tolist()
        list_position = [self.try_except_middle_int(codon) for codon in list_codon]
        list_aa_ref = [
            codon[0].upper() if isinstance(codon, str) and codon else None
            for codon in list_codon
        ]
        list_idx, list_source = reconciler.reconcile_many(
            list_symbol,
            list_position,
            list_aa_ref,
            list_refseq=list_refseq,
            list_hgvsg=return_hgvsg_list(
                df,
                self.str_build,
                list_build=return_verified_build_list(
                    df, self.str_build, str_col_gene=col_gene
                ),
            ),
        )

        df = df.copy()
        df["uniprot_idx"] = pd.array(list_idx, dtype="Int64")
        df["reconcile_source"] = [None if s is None else str(s) for s in list_source]
        list_refseq = list_refseq or [None] * len(df)
        df["unreconciled_reason"] = [
            (
                None
                if idx is not None
                else str(
                    reconciler.return_unreconciled_reason(
                        list_symbol[i], list_position[i], list_aa_ref[i], list_refseq[i]
                    )
                )
            )
            for i, idx in enumerate(list_idx)
        ]

        mask_drop = df["uniprot_idx"].isna()
        if mask_drop.any():
            df_drop = df.loc[mask_drop]
            list_summary = [
                f"{symbol} (n={len(group)}, residues {group['proteinChange'].map(self.try_except_middle_int).min()}-"
                f"{group['proteinChange'].map(self.try_except_middle_int).max()}, "
                f"{', '.join(f'{k}={v}' for k, v in group['unreconciled_reason'].value_counts().items())})"
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
            "variant_canonical": [],
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

            # canonical-frame variant label: the reported proteinChange is in the
            # study's transcript frame (MSK-IMPACT annotates FGFR1 on the MSKCC
            # override), so a canonical label is the only key that joins against
            # UniProt-numbered resources such as ProtVar or the KLIFS maps
            dict_out["variant_canonical"].append(
                None
                if name is None or pd.isna(name) or idx is None
                else f"{split_domain_suffix(str(name))[0]}_{aa_from}{idx}{aa_to}"
            )

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
            Transform counts to log10(count + 1) if True, else log2(count + 1)
        max_value : int | None
            Cap on the transformed counts; None applies no cap

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
        """Log-transform a count with a pseudocount of 1, capped at ``max_value``.

        The pseudocount keeps a count of 1 distinct from 0: log10(1 + 1) ~ 0.30,
        while 0 (and any negative value) maps to 0.

        Parameters
        ----------
        x : int | float | str
            Count to transform; a numeric string is converted to float first
        bool_log10 : bool
            Use log10(x + 1) if True, else log2(x + 1)
        max_value : int | None
            Cap on the transformed value; None applies no cap

        Returns
        -------
        int | float | str
            Transformed value (NaN stays NaN); a non-numeric string is returned as is

        """
        # numeric strings become floats; anything else non-numeric is returned as is
        if not isinstance(x, (int, float, np.number)):
            try:
                x = float(x)
            except ValueError:
                logger.error(f"Value {x} cannot be converted to float.")
                return x

        # nan handling
        if pd.isna(x):
            return np.nan

        # zero or negative counts map to 0 (= log of the pseudocount alone)
        if x <= 0:
            return 0

        # log conversion with a pseudocount of 1
        if bool_log10:
            x = np.log10(x + 1)
        else:
            x = np.log2(x + 1)

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

    def check_entity_id(self) -> bool | None:
        """Check if the panel ID is valid.

        Returns
        -------
        bool | None
            True if the panel ID is valid, False if not; None if the lookup
            could not be made (no client or a failed request)
        """
        if self._cbioportal is None:
            logger.warning(
                f"No cBioPortal client available to check panel ID {self.panel_id}."
            )
            return None
        try:
            panels = self._cbioportal.Gene_Panels.getAllGenePanelsUsingGET().result()
            panel_ids = [panel.genePanelId for panel in panels]
            return self.panel_id in panel_ids
        except Exception as e:
            logger.warning(f"Error checking panel ID {self.panel_id}: {e}")
            return None


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
