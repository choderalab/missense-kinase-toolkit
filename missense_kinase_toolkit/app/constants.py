DICT_RESOURCE_URLS = {
    "KinHub": "http://www.kinhub.org/",
    "KLIFS": "https://klifs.net/",
    "KinCore": "http://dunbrack.fccc.edu/kincore/home",
    "UniProt": "https://www.uniprot.org/",
    "Pfam": "https://www.ebi.ac.uk/interpro/entry/pfam",
}
"""dict[str, str]: Dictionary containing the resource URLs for the dashboard."""

LIST_OPTIONS = [
    "None",
    "Phosphosites",
    "KLIFS",
    "Mutational density",
]
"""list[str]: List of structure options for the dashboard."""

LIST_CAPTIONS = [
    "No additional annotation",
    "Phosphorylation sites as adjudicated by UniProt",
    "Residues that belong to the KLIFS binding pocket (hinge, HRD, xDFG regions represented as sticks)",
    "Missense mutational density within cBioPortal MSK-IMPACT cohort ([Zehir et al, 2017.](https://www.nature.com/articles/nm.4333))",
]
"""list[str]: List of captions for the structure options in the dashboard."""
