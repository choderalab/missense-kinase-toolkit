import ast
import logging
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from mkt.databases import requests_wrapper, utils_requests
from mkt.databases.api_schema import RESTAPIClient
from mkt.schema.kinase_schema import SwissProtPattern, TrEMBLPattern
from nf_rnaseq import variables  # noqa
from nf_rnaseq.uniprot import UniProtGET  # noqa

logger = logging.getLogger(__name__)


@dataclass
class UniProt:
    """Class to interact with the UniProt API."""

    uniprot_id: str
    """UniProt ID."""
    url: str = "https://rest.uniprot.org/uniprotkb"
    """URL for the UniProt API."""


@dataclass
class UniProtFASTA(UniProt, RESTAPIClient):
    """Class to interact UniProt API for FASTA download."""

    url_fasta: str | None = None
    """URL for the UniProt FASTA API including UniProt ID."""
    _header: str | None = None
    """FASTA header."""
    _sequence: str | None = None
    """FASTA sequence."""

    def __post_init__(self):
        self.url = self.set_url()
        self.url_fasta = self.url.replace("<ID>", self.uniprot_id)
        self._header, self._sequence = self.query_api()

    def set_url(self):
        """Set URL for UniProt API; UniProtKB if SwissProt or Unisave if TrEMBL."""
        import re

        if re.match(SwissProtPattern, self.uniprot_id):
            return "https://rest.uniprot.org/uniprotkb/<ID>.fasta"
        elif re.match(TrEMBLPattern, self.uniprot_id):
            return "https://rest.uniprot.org/unisave/<ID>?format=fasta&versions=67"
        else:
            logging.error(f"Invalid UniProt ID: {self.uniprot_id}")
            raise ValueError(f"Invalid UniProt ID: {self.uniprot_id}")

    def query_api(
        self,
        bool_seq: bool = True,
    ) -> str | None:
        """Get FASTA sequence for UniProt ID.

        Parameters
        ----------
        bool_seq : bool
            If True, return sequence string only (i.e., no header or line breaks); otherwise return full FASTA string

        Returns
        -------
        str | None
            FASTA sequences for UniProt ID; None if request fails

        """
        res = requests_wrapper.get_cached_session().get(self.url_fasta)
        if res.ok:
            str_fasta = res.text
            if bool_seq:
                header, seq = self._convert_fasta2seq(str_fasta)
        else:
            utils_requests.print_status_code_if_res_not_ok(res)
            print(f"UniProt ID: {self.uniprot_id}\n")
            header, seq = None, None
        return header, seq

    @staticmethod
    def _convert_fasta2seq(str_fasta) -> tuple[str, str]:
        """Convert FASTA sequence to string sequence (i.e., remove header line breaks).

        Parameters
        ----------
        str_fasta : str
            FASTA string (including header and line breaks)

        Returns
        -------
        header, seq : tuple[str, str]
            FASTA header and sequence strings (excluding line breaks)

        """
        header = str_fasta.split("\n")[0]
        seq = [i.split("\n", 1)[1].replace("\n", "") for i in [str_fasta]][0]
        return header, seq


DICT_UNIPROT_EVIDENCE = {
    "ECO:0000250": "Computational, ISS",
    "ECO:0000255": "Computational, ISM",
    "ECO:0000269": "Manual, experimental",
    "ECO:0000305": "Inferred by curator",
    "ECO:0007744": "Manual, combo",
}
"""dict[str, str]: UniProt evidence codes mapped to descriptions."""


DICT_PHOSPHO_EMPTY = {
    "phospho_sites": None,
    "phospho_evidence": None,
    "phospho_description": None,
}
"""dict[str, None]: Dictionary for empty phospho sites, evidence, and description."""


@dataclass
class UniProtJSON(UniProt, RESTAPIClient):
    """Class to interact UniProt API for JSON download."""

    headers: str = "{'Accept': 'application/json'}"
    """Header for the API request."""
    _json: dict | None = None
    """JSON response from the API."""
    dict_mod_res: dict | None = None
    """Dictionary of modified residues."""

    def __post_init__(self):
        self._json = self.query_api()
        self.dict_mod_res = self.extract_modified_residues()

    def query_api(self) -> dict | None:
        """Get JSON for UniProt ID.

        Returns
        -------
        dict | None
            JSON for UniProt ID; None if request fails

        """
        self.url_json = f"{self.url}/{self.uniprot_id}"

        res = requests_wrapper.get_cached_session().get(
            self.url_json,
            headers=ast.literal_eval(self.headers),
        )
        if res.ok:
            json = res.json()
        else:
            json = None
            utils_requests.print_status_code_if_res_not_ok(res)
        return json

    def extract_modified_residues(self):
        """Extract modified residues from JSON.

        Returns
        -------
        list[str]
            List of modified residues

        """
        if self._json is not None:
            if "features" not in self._json:
                logger.warning(f"No features found for {self.uniprot_id}")
                return DICT_PHOSPHO_EMPTY
            # filter for modified residues and convert to DataFrame
            df_mod_res = pd.DataFrame(
                [
                    entry
                    for entry in self._json["features"]
                    if entry["type"] == "Modified residue"
                ]
            )
            if df_mod_res.shape[0] == 0:
                logger.warning(f"No modified residues found for {self.uniprot_id}...")
                return DICT_PHOSPHO_EMPTY
            df_temp = pd.json_normalize(df_mod_res["location"]).reset_index(drop=True)
            df_mod_res = pd.concat([df_mod_res, df_temp], axis=1)
            # return human-readable evidence codes as a set
            df_mod_res["evidenceCode"] = df_mod_res["evidences"].apply(
                lambda x: {
                    (
                        DICT_UNIPROT_EVIDENCE[entry["evidenceCode"]]
                        if entry["evidenceCode"] in DICT_UNIPROT_EVIDENCE
                        else entry["evidenceCode"]
                    )
                    for entry in x
                }
            )
            # canonical sequence is NaN all others provide isoform mapping
            if "sequence" in df_mod_res.columns:
                df_mod_res = df_mod_res.loc[
                    df_mod_res["sequence"].isna(), :
                ].reset_index(drop=True)
                if df_mod_res.shape[0] == 0:
                    logger.warning(
                        f"No modified residues found in canonical sequence for {self.uniprot_id}..."
                    )
                    return DICT_PHOSPHO_EMPTY
            # filter for phospho sites
            df_mod_res = df_mod_res.loc[
                df_mod_res["description"].apply(lambda x: x.startswith("Phospho")), :
            ].reset_index(drop=True)
            if df_mod_res.shape[0] == 0:
                logger.warning(
                    f"No phospho sites found in cannonical sequence for {self.uniprot_id}"
                )
                return DICT_PHOSPHO_EMPTY
            list_site, list_desc, list_evidence = [], [], []
            for _, row in df_mod_res.iterrows():
                if (
                    row["start.value"] == row["end.value"]
                    and row["start.modifier"] == row["end.modifier"] == "EXACT"
                ):
                    list_site.append(row["start.value"])
                    list_desc.append(row["description"])
                    list_evidence.append(row["evidenceCode"])
            if len(list_site) == 0:
                logger.warning(f"No phospho sites found for {self.uniprot_id}...")
                dict_out = DICT_PHOSPHO_EMPTY
            else:
                dict_out = dict(
                    zip(
                        DICT_PHOSPHO_EMPTY.keys(),
                        [list_site, list_evidence, list_desc],
                    )
                )
            return dict_out
        else:
            DICT_PHOSPHO_EMPTY


@dataclass
class UniProtRefSeqProteinGET(UniProtGET):
    """Class to interact with UniProt API bulk download for list of RefSeqProtein identifiers via GET."""

    def check_if_job_ready(self):
        """Check if the job is ready and add json if so."""
        i = 0
        while True:
            response = requests.get(self.url_query)
            self.check_response(response)
            j = response.json()
            if "results" in j or "failedIds" in j:
                self.json = self.concatenate_json_batches()
                # Log summary instead of full JSON
                num_results = len(self.json.get("results", []))
                num_failed = len(self.json.get("failedIds", []))
                num_suggested = len(self.json.get("suggestedIds", []))
                logger.info(
                    f"\nJob {self.jobId} complete:\n"
                    f"{self.term_in} to {self.term_out} mapping returned: "
                    f"{num_results} results, "
                    f"{num_failed} failed, "
                    f"{num_suggested} suggested"
                )
                return True
            else:
                i += 1
                if i >= 10:
                    raise Exception(f"{self.jobId}: {j['jobStatus']}")
                else:
                    time.sleep(self.polling_interval)

    def concatenate_json_batches(self):
        """Concatenate json from batches and return as dictionary."""
        list_results, list_suggestedIds = [], []
        # failedIds occur on each page so need to use a set instead of list
        set_failedIds = set()
        for batch in self.get_batch():
            batch_json = batch.json()
            if "results" in batch_json:
                list_results.extend(batch_json["results"])
            if "failedIds" in batch_json:
                set_failedIds.update(batch_json["failedIds"])
            # this is new
            if "suggestedIds" in batch_json:
                list_suggestedIds.extend(batch_json["suggestedIds"])

        # deduplicate suggestedIds
        set_suggestedIds = {f"{i['from']};{i['to']}" for i in list_suggestedIds}
        list_suggestedIds = [
            {"from": i.split(";")[0], "to": i.split(";")[1]} for i in set_suggestedIds
        ]

        dict_temp = {
            "results": list_results,
            "failedIds": list(set_failedIds),
            "suggestedIds": list_suggestedIds,
        }
        return dict_temp

    def maybe_get_gene_names(self):
        """Get list of gene names from UniProt ID and add as list_gene_names attr."""
        list_identifier = []
        list_gene_names = []

        str_results = "results"
        if str_results in self.json:
            list_results = self.json[str_results]
            list_identifier.extend([i["from"] for i in list_results])
            list_gene_names.extend([i["to"]["primaryAccession"] for i in list_results])

        str_failedIds = "failedIds"
        if str_failedIds in self.json:
            list_failed = self.json[str_failedIds]
            list_identifier.extend(list_failed)
            list_gene_names.extend(list(np.repeat(np.nan, len(list_failed))))

        str_suggestedIds = "suggestedIds"
        if str_suggestedIds in self.json:
            list_suggested = [i["from"] for i in self.json[str_suggestedIds]]
            list_identifier.extend(list_suggested)
            list_gene_names_temp = [i["to"] for i in self.json[str_suggestedIds]]
            list_gene_names.extend(list_gene_names_temp)

        df = pd.DataFrame({"in": list_identifier, "out": list_gene_names})
        df_agg = df.groupby("in", sort=False).agg(set).reset_index()
        df_agg["out"] = df_agg["out"].apply(lambda x: list(x))

        list_check = [i for i in self.list_identifier if i in df_agg["in"].tolist()]
        assert len(list_check) == len(self.list_identifier)

        self.list_identifier = df["in"].tolist()
        self.list_gene_names = df["out"].tolist()
        self.df = df_agg


def query_uniprotbulk_api(
    input_ids: str,
    term_in: str = "RefSeq_Protein",
    term_out: str = "UniProtKB",
    database: str = "UniProtBULK",
) -> pd.DataFrame:
    """Query UniProt bulk download API for list of RefSeq Protein identifiers and return DataFrame.

    Parameters:
    -----------
    inputs_ids : str
        Comma-separated string of RefSeq Protein identifiers to query UniProt API for.
    term_in : str
        Term to query UniProt API for. Default is "RefSeq_Protein".
    term_out : str
        Term to retrieve from UniProt API. Default is "UniProtKB".
    database : str
        Database to query UniProt API for. Default is "UniProtBULK".

    Returns:
    --------
    pd.DataFrame
        DataFrame with UniProt mapping results.
    """
    DICT_DATABASES = deepcopy(variables.DICT_DATABASES)
    for k1, v1 in DICT_DATABASES.items():
        if k1 == database:
            for v2 in v1.values():
                v2["term_in"] = term_in
                v2["term_out"] = term_out

    dict_post = DICT_DATABASES[database]["POST"]
    post_obj = dict_post["api_object"](
        identifier=input_ids,
        term_in=dict_post["term_in"],
        term_out=dict_post["term_out"],
        url_base=dict_post["url_base"],
    )

    dict_get = DICT_DATABASES[database]["GET"]
    api_obj = UniProtRefSeqProteinGET(
        identifier=input_ids,
        term_in=dict_get["term_in"],
        term_out=dict_get["term_out"],
        url_base=dict_get["url_base"],
        headers=dict_get["headers"],
        jobId=post_obj.jobId,
    )

    return api_obj.df.copy()
