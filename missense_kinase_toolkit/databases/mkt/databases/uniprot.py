from mkt.databases import requests_wrapper, utils_requests


class UniProt:
    """Class to interact with the UniProt API."""

    def __init__(
        self,
        uniprot_id: str,
    ) -> None:
        """Initialize UniProt Class object.

        Parameters
        ----------
        uniprot_id : str
            UniProt ID

        Attributes
        ----------
        url : str
            UniProt API URL
        uniprot_id : str
            UniProt ID

        """
        self.url = "https://rest.uniprot.org/uniprotkb"
        self.uniprot_id = uniprot_id
        self._sequence = self.get_uniprot_fasta()

    def get_uniprot_fasta(
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
        url_fasta = f"{self.url}/{self.uniprot_id}.fasta"

        res = requests_wrapper.get_cached_session().get(url_fasta)
        if res.ok:
            str_fasta = res.text
            if bool_seq:
                str_fasta = self._convert_fasta2seq(str_fasta)
        else:
            str_fasta = None
            utils_requests.print_status_code_if_res_not_ok(res)
        return str_fasta

    @staticmethod
    def _convert_fasta2seq(str_fasta):
        """Convert FASTA sequence to string sequence (i.e., remove header line breaks).

        Parameters
        ----------
        str_fasta : str
            FASTA string (including header and line breaks)

        Returns
        -------
        str_seq : str
            Sequence string (excluding header and line breaks)

        """
        str_seq = [i.split("\n", 1)[1].replace("\n", "") for i in [str_fasta]][0]
        return str_seq
