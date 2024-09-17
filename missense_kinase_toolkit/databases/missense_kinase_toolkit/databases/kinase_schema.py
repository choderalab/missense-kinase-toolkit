from dataclasses import dataclass

#TODO create a Pydantic model to incorporate and UniProt, Pfam, and KLIFS data for all 
# eventually populate with AF2 active BLAminus+ structures from Dunbrack lab

#TODO: Make Pydantic model instead of dataclass
@dataclass
class KLIFSPocket:
    """Dataclass to hold KLIFS pocket alignment information per kinase.

    Attributes
    ----------
    uniprotID : str
        UniProt ID
    hgncName : str
        HGNC name
    uniprotSeq : str
        UniProt canonical sequence
    klifsSeq : str
        KLIFS pocket sequence
    list_klifs_region : list[str]
        List of start and end regions of KLIFS pocket separated by ":"; end region will be the
            same as start region if no concatenation necessary to find a single exact match
    list_klifs_substr_actual : list[str]
        List of substring of KLIFS pocket that maps to the *start region* of the KLIFS pocket
    list_klifs_substr_match : list[str]
        List of the actual substring used to match to the KLIFS pocket for the region(s) provided;
            will be the same as list_klifs_substr_actual if no concatenation necessary to find a single exact match
    list_substring_idxs : list[list[int] | None]
        List of indices in UniProt sequence where KLIFS substring match starts;
            offset by length of preceding KLIFS region with gaps removed
        
    """
    uniprotID                   :   str
    hgncName                    :   str
    uniprotSeq                  :   str
    klifsSeq                    :   str
    list_klifs_region           :   list[str]
    list_klifs_substr_actual    :   list[str]
    list_klifs_substr_match     :   list[str]
    list_substring_idxs         :   list[list[int] | None]

    def remove_klifs_list_gaps(self):
        """Remove gaps from KLIFS pocket substring list.
        
        Returns
        -------
        list_substring_klifs_narm = list[str]
            List of KLIFS pocket substrings with gaps removed
        
        """
        from missense_kinase_toolkit.databases.klifs import remove_gaps_from_klifs

        list_substring_klifs_narm = [remove_gaps_from_klifs(substring_klifs) \
                                     for substring_klifs in self.list_klifs_substring]
        return list_substring_klifs_narm