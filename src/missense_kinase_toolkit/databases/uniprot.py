import requests

from missense_kinase_toolkit.databases import requests_wrapper, utils_requests


class UniProt():
    """Class to interact with the UniProt API."""
    def __init__(
            self,
    ) -> None:
        """Initialize UniProt Class object.

        Attributes
        ----------
        url : str
            UniProt API URL
        """
        self.url = "https://www.uniprot.org"


    def get_uniprot_fasta(list_input, url_1, url_2):
        list_fasta = []
        for keyword in list_input:
            url = url_1 + keyword + url_2
            fasta = requests.get(url).text
            if fasta == "":
                print(f"{keyword} not found in UniProt...")
            else:
                list_fasta.append(fasta)
        return list_fasta

    def convert_fasta2seq(list_fasta):
        list_seq = [i.split("\n", 1)[1].replace("\n", "") for i in list_fasta]
        return list_seq
