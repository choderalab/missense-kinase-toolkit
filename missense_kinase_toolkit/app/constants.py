DICT_RESOURCE_URLS = {
    "KinHub": "http://www.kinhub.org/",
    "KLIFS": "https://klifs.net/",
    "KinCoRe": "http://dunbrack.fccc.edu/kincore/home",
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
    (
        "Residues that belong to the KLIFS binding pocket  \n"
        "*(hinge, gatekeeper, HRD, xDFG regions represented as sticks)*"
    ),
    "Missense mutational density within cBioPortal MSK-IMPACT cohort ([Zehir et al, 2017.](https://www.nature.com/articles/nm.4333))",
]
"""list[str]: List of captions for the structure options in the dashboard."""

DICT_COMPUTED_HELP = {
    "is_pseudokinase": (
        "Predicted catalytically inactive kinases are missing any of the "
        "following residues, sourced from the KLIFS or else the Dunbrack MSA:\n"
        "\t1. Catalytic Lys (KLIFS III:17 or II:13 in WNKs)\n"
        "\t2. HRD Asp (c.l:70)\n"
        "\t3. DFG Asp (xDFG:81)\n"
        "The presence of a KinCoRe active-state structure means False.\n"
        "Curated overrides:\n"
        "\t• True: BUB1B, ROR1/2, RYK.\n"
        "\t• False: CAMKK1, STYK1.\n"
        "\t• None: neither alignment is available."
    ),
    "is_pseudogene": (
        "Flagged as a pseudogene if either:\n"
        '\t• UniProt header contains "putative"\n'
        '\t• KLIFS name contains "pseudogene"'
    ),
    "is_lipid_kinase": (
        'HGNC symbol starts with "PI" (PI3K, PI4K, PIP4K/PIP5K).\n'
        "Excluding:\n"
        "\t• PIM\n"
        "\t• PIN\n"
        "\t• PIK3R4 (actually a protein kinase)"
    ),
    "catalytic Lys": (
        "The β3 (VAIK) lysine at KLIFS III:17 positions the ATP α/β-phosphates.\n"
        "The residue is sourced from KLIFS pocket or else Dunbrack MSA."
    ),
    "HRD motif": (
        "Catalytic-loop His-Arg-Asp; the Asp is the catalytic base.\n"
        "KLIFS positions:\n"
        "\t• Most kinases: c.l:68-69-70\n"
        "\t• PIK, PIKK: c.l:72-71-70 (loop reads D-R-H)"
    ),
    "DFG motif": (
        "Asp-Phe-Gly starting the activation segment (KLIFS xDFG:81-83).\n"
        "The Asp chelates the catalytic Mg²⁺ ions."
    ),
    "APE motif": (
        "Ala-Pro-Glu ending the activation segment (Dunbrack MSA ALC:152-154).\n"
        "None if:\n"
        "\t• No MSA row is available\n"
        "\t• The motif is gapped"
    ),
    "molecular brake": (
        "Autoinhibitory Asn-Glu-Lys triad characterized in FGFR "
        "([Chen et al., Mol Cell 2007.](https://doi.org/10.1016/j.molcel.2007.06.028)).\n"
        "KLIFS positions:\n"
        "\t• Asn: b.l:37\n"
        "\t• Glu: hinge:46\n"
        "\t• Lys: one residue before VIII:79"
    ),
}
"""dict[str, str]: Hover description of each computed property, keyed by its base label."""
