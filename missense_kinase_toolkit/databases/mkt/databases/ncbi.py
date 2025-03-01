import ast
import logging
import os.path
from dataclasses import field
from io import BytesIO, StringIO
from zipfile import ZipFile

from Bio import SeqIO
from pydantic.dataclasses import dataclass

from mkt.databases import requests_wrapper
from mkt.databases.api_schema import RESTAPIClient

logger = logging.getLogger(__name__)


@dataclass
class ProteinNCBI(RESTAPIClient):
    """Class to interact with query NCBI Protein API; only FASTA download supported."""

    accession: str
    """Accession ID for the protein."""
    url: str = (
        "https://api.ncbi.nlm.nih.gov/datasets/v2/protein/accession/<ACC>/download?"
    )
    """URL for the NCBI Protein API."""
    annotation: str | None = "FASTA_PROTEIN"
    """Annotation type to include in the download: FASTA_UNSPECIFIED ┃ FASTA_GENE ┃ FASTA_RNA ┃
        FASTA_PROTEIN ┃ FASTA_GENE_FLANK ┃ FASTA_CDS ┃ FASTA_5P_UTR ┃ FASTA_3P_UTR."""
    headers: str = "{'Accept': 'application/zip'}"
    """Header for the API request."""
    list_headers: list[str | None] = field(default_factory=list)
    """List of FASTA headers."""
    list_seq: list[str | None] = field(default_factory=list)
    """List of FASTA sequences."""

    def __post_init__(self):
        self.create_query_url()
        self.query_api()

    def create_query_url(self):
        """Create URL for NCBI Protein API query."""

        self.url_query = self.url.replace("<ACC>", str(self.accession))

        if self.annotation is not None:
            if self.annotation not in [
                "FASTA_UNSPECIFIED",
                "FASTA_GENE",
                "FASTA_RNA",
                "FASTA_PROTEIN",
                "FASTA_GENE_FLANK",
                "FASTA_CDS",
                "FASTA_5P_UTR",
                "FASTA_3P_UTR",
            ]:
                logger.error("Annotation type not valid.")
            else:
                self.url_query += f"include_annotation_type={self.annotation}"

    def query_api(self) -> dict:
        res = requests_wrapper.get_cached_session().get(
            self.url_query,
            headers=ast.literal_eval(self.headers),
        )

        if res.ok:
            zip_ref = ZipFile(BytesIO(res.content))
            info_list = zip_ref.infolist()
            list_ext = [
                os.path.splitext(file_info.filename)[1] for file_info in info_list
            ]
            list_idx = [idx for idx, i in enumerate(list_ext) if i == ".faa"]

            if len(list_idx) == 0:
                logging.error(
                    f"Failed to download any FASTA files using following query: {self.url_query}."
                )

            for idx in list_idx:
                str_fasta = zip_ref.open(info_list[idx]).read().decode()
                fastas = SeqIO.parse(StringIO(str_fasta), "fasta")
                list_fasta = [(fasta.description, str(fasta.seq)) for fasta in fastas]
                self.list_headers.extend([fasta[0] for fasta in list_fasta])
                self.list_seq.extend([fasta[1] for fasta in list_fasta])
        else:
            logging.error(
                f"Status code {res.status_code} using following query: {self.url_query}."
            )
