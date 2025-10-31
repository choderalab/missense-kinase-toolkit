import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mkt.databases import hgnc
from mkt.databases.aligners import BL2UniProtAligner
from mkt.databases.ncbi import ProteinNCBI
from mkt.databases.uniprot import UniProtFASTA, query_uniprotbulk_api
from mkt.schema.io_utils import deserialize_kinase_dict
from mkt.schema.kinase_schema import SwissProtPattern
from pydantic import BaseModel, Field, constr, model_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)

tqdm.pandas()


DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")
DICT_KINASE_REV = {v.uniprot_id: v for v in DICT_KINASE.values()}


DICT_DAVIS_DROP = {
    "DiscoverX Gene Symbol": [
        "-phosphorylated",  # no way to handle phosphorylated residues yet
        "-cyclin",  # no way to handle cyclin proteins yet
        "ALK(1151Tins)",  # insertion at III:18 - not sure how to handle KLIFS
    ],
    "Species": [
        "P.falciparum",  # not human kinase
        "M.tuberculosis",  # not human kinase
    ],
    "Construct Description": [
        "(L747-T751del,Sins)",  # not sure what Sins mutation is in EGFR
        "(ITD",  # internal tandem duplication - complex mutation, need more details
    ],
}
"""Dict[str, list[str]]: terms to drop where key is colname and value is list of terms."""

DICT_DAVIS_MERGE_FIX = {
    "GCN2(Kin.Dom.2,S808G)": {
        "Construct Description": "Mutation (S808G)",  # given name, this seems to be a mutant
    },
    # "RIPK5" : {
    #     "Entrez Gene Symbol": "DSTYK", # updated HGNC gene symbol - just a typo
    # },
    "EGFR(E746-A750del)": {
        "AA Start/Stop": "R669/V1011",  # added construct boundaries (missing from all dels) - used EGFR for all but T790M
    },
    "EGFR(L747-E749del, A750P)": {
        "AA Start/Stop": "R669/V1011",
    },
    "EGFR(L747-S752del, P753S)": {
        "AA Start/Stop": "R669/V1011",
    },
    "EGFR(S752-I759del)": {
        "AA Start/Stop": "R669/V1011",
    },
    "RPS6KA4(Kin.Dom.1-N-terminal)": {
        "AA Start/Stop": "M1/V374",  # compared RPS6KA constructs - made N-terminal construct full length up to start of Kin.Dom.2
    },
    "RPS6KA5(Kin.Dom.1-N-terminal)": {
        "AA Start/Stop": "M1/A389",
    },
}
"""Dictionary of {`DiscoverX Gene Symbol` : {column name : replacement string}} to fix entries in the Davis DiscoverX dataset."""

SeqStartStop = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWY]{1}$")
"""Pydantic model for start/stop sequence constraints."""
SequenceDEL = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWXY\-]+$")
"""Pydantic model for UniProt/RefSeq sequence constraints, allowing for deletions (-)."""


class DiscoverXInfo(BaseModel):
    """DiscoverX dataset processing class."""

    discoverx_gene_symbol: str
    """DiscoverX gene symbol."""
    key: str | None
    """Dictionary key for the kinase; HGNC name + '_1/2' for multi-mapping."""
    accession: str
    """str: DiscoverX accession of the kinase construct."""
    uniprot_id: str
    """str | None: UniProt accession of the kinase."""
    str_start: SeqStartStop | None  # AKT1, AKT2, and AKT3 have None start/stop
    """SeqStartStop: Amino acid at the start of the construct."""
    int_start: int | None  # AKT1, AKT2, and AKT3 have None start/stop
    """int: Position of the start of the construct in the RefSeq sequence."""
    str_stop: SeqStartStop | None  # AKT1, AKT2, and AKT3 have None start/stop
    """SeqStartStop: Amino acid at the stop of the construct."""
    int_stop: int | None  # AKT1, AKT2, and AKT3 have None start/stop
    """int: Position of the stop of the construct in the RefSeq sequence."""
    seq_refseq: SequenceDEL | None
    """SequenceDEL: RefSeq sequence with deletions allowed."""
    seq_uniprot: SequenceDEL | None = None
    """SequenceDEL: UniProt canonical sequence with deletions allowed."""
    bool_wt: bool
    """bool: Whether the construct is wild-type (True) or mutant (False)."""
    list_missense_mutations: list[tuple[str, int, str]] | None = None
    """list[tuple[str, int, str] | None]: List of missense mutations as tuples of \
        (wild-type amino acid, position, mutant amino acid)."""
    list_deletion_mutations: list[tuple[str, list[int], str]] | None = None
    """list[tuple[str, list[int], str] | None]: List of deletion mutations as tuples of \
        (start WT amino acid, list of positions deleted, end WT amino acid)."""
    # these are populated after initialization
    list_uniprot2refseq: list[int | None] | None = None
    """list[int | None]: List mapping UniProt sequence indices to RefSeq sequence indices;
        length must match the length of the UniProt sequence and entries correspond to RefSeq."""
    list_refseq2uniprot: list[int | None] | None = None
    """list[int | None]: List mapping RefSeq sequence indices to UniProt sequence indices
        length must match the length of the RefSeq sequence and entries correspond to UniProt."""
    bool_has_kd: bool | None = None
    """bool | None: Whether the kinase has a kinase domain defined in DICT_KINASE (True), \
        not defined (False), or key not in DICT_KINASE (None)."""
    bool_has_klifs: bool | None = None
    """bool | None: Whether the kinase has KLIFS residues defined in DICT_KINASE (True), \
        not defined (False), or key not in DICT_KINASE (None)."""
    bool_mutations_in_kd_region: bool | None = None
    """bool | None: Whether all mutations fall within the kinase domain (True), \
        outside (False), or no mutations/kinase domain/DICT_KINASE (None)."""
    bool_mutations_in_klifs_region: bool | None = None
    """bool | None: Whether all mutations fall within the span of the KLIFS region (True), \
        outside (False), or no mutations/KLIFS region/DICT_KINASE (None)."""
    bool_mutations_in_klifs_residues: bool | None = None
    """bool | None: Whether all mutations fall within a KLIFS residue (True), \
        outside (False), or no mutations/KLIFS residue/DICT_KINASE (None)."""
    # dict_refseq_indices: dict[int, str | None] = Field(default_factory=dict, initialize=False)
    # """dict[int, str | None]: Dictionary with 'start' and 'stop' indices for RefSeq sequence."""
    bool_offset: bool = True
    """bool: Whether to use 1-based indexing (True) or 0-based indexing (False). Default is True."""

    @staticmethod
    def check_construct_boundaries(
        str_seq: str,
        int_idx: int,
        int_offset: int,
        str_aa: str,
        str_id: str,
        bool_start: bool = True,
    ) -> None:
        """Raise error if construct boundaries do not match sequence.

        Parameters
        ----------
        str_seq : str
            Amino acid sequence.
        int_idx : int
            Index of the amino acid in the sequence.
        str_aa : str
            Amino acid expected at the index.

        Raises
        ------
        ValueError
            If the amino acid at the index does not match the expected amino acid.
        """
        if str_seq[int_idx - int_offset] != str_aa:
            raise ValueError(
                f"{'Start' if bool_start else 'Stop'} amino acid {str_aa} in "
                f"{str_id} does not match RefSeq sequence at position "
                f"{int_idx} ({str_seq[int_idx - int_offset]})"
            )

    # validate construct boundaries against RefSeq sequence
    @model_validator(mode="before")
    @classmethod
    def validate_construct_boundaries(cls, data: dict) -> dict:
        int_start, int_stop = data.get("int_start"), data.get("int_stop")
        if int_start is not None and int_stop is not None:
            offset = 1 if data.get("bool_offset", True) else 0

            seq_refseq = data["seq_refseq"]
            gene_symbol = data["discoverx_gene_symbol"]
            key = data.get("key")

            # construct start codon
            str_start = data["str_start"]
            cls.check_construct_boundaries(
                str_seq=seq_refseq,
                int_idx=int_start,
                int_offset=offset,
                str_aa=str_start,
                str_id=f"{gene_symbol}/{key}",
                bool_start=True,
            )
            # construct stop codon
            str_stop = data["str_stop"]
            cls.check_construct_boundaries(
                str_seq=seq_refseq,
                int_idx=int_stop,
                int_offset=offset,
                str_aa=str_stop,
                str_id=f"{gene_symbol}/{key}",
                bool_start=False,
            )

        return data

    # validate missense mutations against RefSeq sequence
    @model_validator(mode="before")
    @classmethod
    def validate_missense_mutations(cls, data: dict) -> dict:

        list_missense_mutations = data.get("list_missense_mutations")
        gene_symbol = data["discoverx_gene_symbol"]
        key = data.get("key")

        if list_missense_mutations is not None:
            offset = 1 if data.get("bool_offset", True) else 0

            for tuple_mut in list_missense_mutations:
                str_wt, idx_codon, _ = tuple_mut
                aa_refseq = data["seq_refseq"][idx_codon - offset]
                if aa_refseq != str_wt:
                    raise ValueError(
                        f"Missense mutation {tuple_mut} does not match RefSeq sequence "
                        f"at position {idx_codon} ({aa_refseq}) for {gene_symbol}/{key}"
                    )

        return data

    @staticmethod
    def check_deletion_boundaries(
        str_seq: str,
        int_idx: int,
        int_offset: int,
        str_check: str,
        str_id: str,
        bool_start: bool = True,
    ) -> None:
        """Raise error if deletion boundaries do not match sequence.

        Parameters
        ----------
        str_seq : str
            Amino acid sequence.
        int_idx : int
            Index of the amino acid deleted in the sequence.
        int_offset : int
            Offset to apply to the index.
        str_check : str
            Amino acid expected at the index.
        str_id : str
            Identifier for logging.
        bool_start : bool, optional
            Whether checking start (True) or stop (False) of deletion; defaults to True.

        Raises
        ------
        ValueError
            If the amino acid at the index does not match the expected amino acid.
        """
        if str_seq[int_idx - int_offset] != str_check:
            raise ValueError(
                f"{'Deletion start' if bool_start else 'Deletion end'} amino acid "
                f"{str_check} in {str_id} does not match sequence at position "
                f"{int_idx} ({str_seq[int_idx - int_offset]})"
            )

    # validate start/stop deletion mutations against RefSeq sequence
    @model_validator(mode="before")
    @classmethod
    def validate_deletions(cls, data: dict) -> dict:
        list_deletion_mutations = data.get("list_deletion_mutations")
        gene_symbol = data["discoverx_gene_symbol"]
        key = data.get("key")
        seq_refseq = data["seq_refseq"]

        if list_deletion_mutations is not None:
            offset = 1 if data.get("bool_offset", True) else 0

            for tuple_del in list_deletion_mutations:
                str_start, list_idx_del, str_end = tuple_del

                # RefSeq start codon
                cls.check_deletion_boundaries(
                    str_seq=seq_refseq,
                    int_idx=list_idx_del[0],
                    int_offset=offset,
                    str_check=str_start,
                    str_id=f"{gene_symbol}/{key}",
                    bool_start=True,
                )

                # RefSeq end codon
                cls.check_deletion_boundaries(
                    str_seq=seq_refseq,
                    int_idx=list_idx_del[-1],
                    int_offset=offset,
                    str_check=str_end,
                    str_id=f"{gene_symbol}/{key}",
                    bool_start=False,
                )

        return data

    def model_post_init(self, __context: any) -> None:
        """Post-initialization processing to populate additional fields."""

        # map between RefSeq and UniProt sequences
        self.list_refseq2uniprot, self.list_uniprot2refseq = (
            self.return_refseq2uniprot_mapping()
        )

        # check if kinase has defined kinase domain or KLIFS residues
        if self.key in DICT_KINASE:
            self.bool_has_kd = (
                DICT_KINASE[self.key].adjudicate_kd_sequence() is not None
            )
            self.bool_has_klifs = DICT_KINASE[self.key].KLIFS2UniProtIdx is not None

        # process mutations if not wild-type
        if not self.bool_wt:
            (
                str_refseq_mut,
                str_uniprot_mut,
                bool_kd_region,
                bool_klifs_region,
                bool_klifs_residues,
            ) = self.replace_mutations_in_sequences()
            self.seq_refseq = str_refseq_mut
            self.seq_uniprot = str_uniprot_mut
            self.bool_mutations_in_kd_region = bool_kd_region
            self.bool_mutations_in_klifs_region = bool_klifs_region
            self.bool_mutations_in_klifs_residues = bool_klifs_residues

    # TODO: mutation validation @model_validator(mode="after") for str_mut instead of str_wt

    def return_index(self, idx_in: int) -> int:
        """Return adjusted index based on offset setting.

        Parameters:
        -----------
        idx_in : int
            Input index.

        Returns:
        --------
        int
            Adjusted index.
        """
        return idx_in - (1 if self.bool_offset else 0)

    def return_refseq2uniprot_mapping(
        self,
    ) -> tuple[list[int | None] | None, list[int | None] | None]:
        """Return mapping of RefSeq to UniProt indices (and visa versa) using global alignment.

        Parameters:
        -----------
        str_refseq : str
            RefSeq sequence string.
        str_uniprot : str
            UniProt sequence string.

        Returns:
        --------
        list[int], list[int]
            Two lists mapping RefSeq to UniProt indices and vice versa;
                index in initial string given by position in list.
        """
        if self.seq_refseq is None or self.seq_uniprot is None:
            return None, None

        # if sequences are identical, no need to align just add offset
        if self.seq_refseq == self.seq_uniprot:
            list_idx_uniprot2refseq = [
                i + (1 if self.bool_offset else 0) for i in range(len(self.seq_uniprot))
            ]
            list_idx_refseq2uniprot = [
                i + (1 if self.bool_offset else 0) for i in range(len(self.seq_refseq))
            ]
        # if sequences differ, align and create mapping using global alignment
        else:
            aligner = BL2UniProtAligner()
            alignments = aligner.align(self.seq_refseq, self.seq_uniprot)

            # if True, use 1-based indexing
            if self.bool_offset:
                idx_refseq, idx_uniprot = 1, 1
            else:
                idx_refseq, idx_uniprot = 0, 0

            align_refseq, align_uniprot = alignments[0][0], alignments[0][1]
            list_idx_uniprot2refseq, list_idx_refseq2uniprot = [], []
            for char_refseq, char_uniprot in zip(align_refseq, align_uniprot):
                # no match in refseq (but match in uniprot)
                if char_refseq == "-":
                    list_idx_uniprot2refseq.append(None)
                    idx_uniprot += 1
                # no match in uniprot (but match in refseq)
                elif char_uniprot == "-":
                    list_idx_refseq2uniprot.append(None)
                    idx_refseq += 1
                # allows for mismatch characters (".") - do not record them
                else:
                    list_idx_refseq2uniprot.append(idx_uniprot)
                    idx_uniprot += 1

                    list_idx_uniprot2refseq.append(idx_refseq)
                    idx_refseq += 1

        assert len(list_idx_uniprot2refseq) == len(self.seq_uniprot)
        assert len(list_idx_refseq2uniprot) == len(self.seq_refseq)

        return list_idx_refseq2uniprot, list_idx_uniprot2refseq

    def extract_region_indices(self, bool_kd: bool = True) -> tuple[int, int]:
        """Extract kinase domain or KLIFS region indices from DICT_KINASE.

        Parameters:
        -----------
        bool_kd : bool
            Whether to check kinase domain (True) or KLIFS region (False); default is True.

        Returns:
        --------
        tuple[int, int]
            Tuple of (start index, end index).
        """
        if bool_kd:
            idx_start = DICT_KINASE[self.key].adjudicate_kd_start()
            idx_end = DICT_KINASE[self.key].adjudicate_kd_end()
            return idx_start, idx_end
        else:
            list_klifs_refseq = self.convert_uniprot2refseq_klifs_residues()
            idx_start = min([v for v in list_klifs_refseq if v is not None])
            idx_end = max([v for v in list_klifs_refseq if v is not None])
            return idx_start, idx_end

    def check_region_of_mutation(
        self,
        idx_start: int | None,
        idx_end: int | None,
        list_codons: int | list[int],
        bool_kd: bool = True,
    ) -> bool:
        """Check if mutation falls within kinase domain or KLIFS region.

        Parameters:
        -----------
        idx_start : int | None
            Start index of the region.
        idx_end : int | None
            End index of the region.
        idx_codon : int
            Index of the mutation codon.
        bool_kd : bool
            Whether to check kinase domain (True) or KLIFS region (False); default is True.

        Returns:
        --------
        bool
            Updated bool indicating if mutation falls within region.
        """
        list_bool = [idx_start <= i <= idx_end for i in list_codons]
        if all(list_bool):
            return True
        else:
            region = "adjudicated kinase domain" if bool_kd else "KLIFS region"
            for idx, bool_val in zip(list_codons, list_bool):
                if not bool_val:
                    logger.info(
                        f"Mutation at {idx} in {self.discoverx_gene_symbol} "
                        f"falls outside {region} range {idx_start}-{idx_end}."
                    )
            return False

    def convert_uniprot2refseq_klifs_residues(self) -> list[int] | None:
        """Convert KLIFS residues from UniProt to RefSeq indices.

        Parameters:
        -----------
        None

        Returns:
        --------
        list[int | None]
            List of RefSeq indices corresponding to KLIFS residues or None if not available.
        """
        KLIFS2UniProtIdx = DICT_KINASE[self.key].KLIFS2UniProtIdx
        list_klifs_uniprot = list(KLIFS2UniProtIdx.values())
        list_klifs_refseq = [
            (
                self.list_uniprot2refseq.index(v)
                if (v in self.list_uniprot2refseq and v is not None)
                else None
            )
            for v in list_klifs_uniprot
        ]
        return list_klifs_refseq

    def convert_refseq2uniprot_mutations(
        self,
        idx_refseq: int | list[int],
    ) -> int | list[int]:
        """Convert RefSeq mutation index to UniProt mutation index.

        Parameters:
        -----------
        idx_refseq : int | list[int]
            Index or list of indices of the mutation in the RefSeq sequence.

        Returns:
        --------
        str
            UniProt sequence with the mutation replaced - need to run return_index() before using.
        """
        # index is zero-indexed in list, so no need to adjust for offset in opposite direction
        if isinstance(idx_refseq, list):
            idx_uniprot_start = self.list_uniprot2refseq.index(idx_refseq[0]) + (
                1 if self.bool_offset else 0
            )
            idx_uniprot_end = self.list_uniprot2refseq.index(idx_refseq[-1]) + (
                1 if self.bool_offset else 0
            )
            idx_uniprot = list(range(idx_uniprot_start, idx_uniprot_end + 1))
        else:
            idx_uniprot = self.list_uniprot2refseq.index(idx_refseq) + (
                1 if self.bool_offset else 0
            )

        return idx_uniprot

    def replace_mutations_in_sequences(self) -> tuple[
        str | None,  # RefSeq sequence
        str | None,  # UniProt sequence
        bool | None,  # all mutations in KD
        bool | None,  # all mutations in KLIFS region
        bool | None,  # all mutations in KLIFS residues
    ]:
        """Replace mutations in RefSeq and UniProt full sequences in the Davis dataset.

        Parameters:
        -----------
        df_in : pd.DataFrame
            Input DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame with mutations replaced in RefSeq and UniProt full sequences.
        list[tuple]
            Tuple of (RefSeq sequence, UniProt sequence, bool mutation in KD, \
                bool mutation in KLIFS region, bool mutation in KLIFS residues).
        """
        dict_seq = {
            "refseq_full": self.seq_refseq,
            "uniprot_full": self.seq_uniprot,
        }

        list_idx = []
        # missense mutations
        if self.list_missense_mutations is not None:
            for tuple_mut in self.list_missense_mutations:
                str_wt, idx_codon, str_mut = tuple_mut
                list_idx.append(idx_codon)
                # check that codon from UniProt is in RefSeq sequence
                assert idx_codon in self.list_refseq2uniprot
                # convert to uniprot index
                idx_codon_uniprot = self.convert_refseq2uniprot_mutations(idx_codon)
                # check here since can't do in validator before mapping
                assert self.seq_uniprot[self.return_index(idx_codon_uniprot)] == str_wt
                # replace in sequences
                for key_seq, str_seq in dict_seq.items():
                    idx_temp = self.return_index(
                        idx_codon_uniprot if key_seq == "uniprot_full" else idx_codon
                    )
                    str_replace = "".join(
                        [
                            i if idx != idx_temp else str_mut
                            for idx, i in enumerate(str_seq)
                        ]
                    )
                    dict_seq[key_seq] = str_replace

        # deletion mutations
        if self.list_deletion_mutations is not None:
            for tuple_mut in self.list_deletion_mutations:
                str_start, list_idx_del, str_end = tuple_mut
                list_idx.extend(list_idx_del)
                # check that codon from UniProt is in RefSeq sequence
                assert all([i in self.list_refseq2uniprot for i in list_idx_del])
                # convert to uniprot index
                list_idx_del_uniprot = self.convert_refseq2uniprot_mutations(
                    list_idx_del
                )
                # check here since can't do in validator before mapping
                assert (
                    self.seq_uniprot[self.return_index(list_idx_del_uniprot[0])]
                    == str_start
                )
                assert (
                    self.seq_uniprot[self.return_index(list_idx_del_uniprot[-1])]
                    == str_end
                )
                # replace in sequences
                list_idx_del_rev = [self.return_index(i) for i in list_idx_del]
                list_idx_del_uniprot_rev = [
                    self.return_index(i) for i in list_idx_del_uniprot
                ]
                for key_seq, str_seq in dict_seq.items():
                    idx_temp = (
                        list_idx_del_uniprot_rev
                        if key_seq == "uniprot_full"
                        else list_idx_del_rev
                    )
                    str_replace = "".join(
                        [
                            i if idx not in idx_temp else "-"
                            for idx, i in enumerate(str_seq)
                        ]
                    )
                    dict_seq[key_seq] = str_replace

        # check if all mutations are within KD/KLIFS region or KLIFS residues
        bool_kd_region, bool_klifs_region, bool_klifs_residue = None, None, None
        if len(list_idx) > 0:
            # check kinase domain
            # TODO: technically these should be converted from uniprot2refseq indices,
            # waiting because currently need to handle mapping out of construct
            if self.bool_has_kd:
                idx_kd_start, idx_kd_end = self.extract_region_indices(bool_kd=True)
                bool_kd_region = self.check_region_of_mutation(
                    idx_kd_start, idx_kd_end, list_idx, bool_kd=True
                )

            # check KLIFS region/residues
            if self.bool_has_klifs:
                # check KLIFS region
                idx_klifs_start, idx_klifs_end = self.extract_region_indices(
                    bool_kd=False
                )
                bool_klifs_region = self.check_region_of_mutation(
                    idx_klifs_start, idx_klifs_end, list_idx, bool_kd=False
                )
                # check KLIFS residues
                list_klifs_refseq = self.convert_uniprot2refseq_klifs_residues()
                bool_klifs_residue = all([i in list_klifs_refseq for i in list_idx])

        return (
            dict_seq["refseq_full"],
            dict_seq["uniprot_full"],
            bool_kd_region,
            bool_klifs_region,
            bool_klifs_residue,
        )

    def generate_construct_dictionary(self) -> dict:
        """Generate dictionary representation of the DiscoverXInfo object.

        Returns:
        --------
        dict
            Dictionary of length == construct where keys are RefSeq indices and values are properties
                (e.g., KD start/end, KLIFS residues).
        """
        # dict_temp = DICT_KINASE

        if self.idx_start is None or self.idx_end is None:
            logger.info(
                f"Cannot generate construct dictionary for {self.discoverx_gene_symbol} "
                f"as construct boundaries are not defined."
            )
        # list_construct_idx = list(range(self.idx_start, self.idx_end + 1))


@dataclass
class DiscoverXInfoGenerator:
    """Class to generate DiscoverXInfo objects from DiscoverX kinase construct information CSV."""

    str_url: str = (
        "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/DiscoverX_489_Kinase_Assay_Construct_Information.csv"
    )
    """str: URL to the DiscoverX kinase construct information CSV."""
    bool_offset: bool = True
    """bool: Whether to use 1-based indexing (True) or 0-based indexing (False). Default is True."""
    df: pd.DataFrame | None = None
    """pd.DataFrame: DataFrame of the DiscoverX kinase construct information CSV."""
    df_id: pd.DataFrame | None = None
    """pd.DataFrame: DataFrame of the UniProt ID mapping results."""
    dict_discoverx_info: dict[str, DiscoverXInfo] = Field(
        default_factory=dict, initialize=False
    )
    """dict[str, DiscoverXInfo]: Dictionary of DiscoverXInfo objects keyed by DiscoverX gene symbol."""

    def __post_init__(self):
        """Post-initialization to generate DataFrame and ID mapping."""
        self.generate_dataframes()
        self.dict_discoverx_info = self.generate_discoverx_info_dict()

    def generate_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate DataFrame of DiscoverX kinase construct information with UniProt mapping.

        Returns:
        --------
        df : pd.DataFrame
            DataFrame of the DiscoverX kinase construct information.
        df_id : pd.DataFrame
            DataFrame of the UniProt ID mapping results.
        """
        df = pd.read_csv(self.str_url)

        # drop things we cannot currently handle
        for k, v in DICT_DAVIS_DROP.items():
            df = df.loc[
                ~df[k].apply(lambda x: any([i in str(x) for i in v])), :
            ].reset_index(drop=True)

        # apply manual fixes for issues found
        for k1, v1 in DICT_DAVIS_MERGE_FIX.items():
            for k2, v2 in v1.items():
                df.loc[df["DiscoverX Gene Symbol"] == k1, k2] = v2

        # 4 entries are UniProt IDs already
        bool_uniprot = df["Accession Number"].apply(
            lambda x: True if re.match(SwissProtPattern, x.split(".")[0]) else False
        )
        df_uniprot = pd.DataFrame(
            {
                "in": df.loc[bool_uniprot, "Accession Number"],
                "out": df.loc[bool_uniprot, "Accession Number"].apply(
                    lambda x: [x.split(".")[0]]
                ),
            }
        ).reset_index(drop=True)
        df_uniprot = df_uniprot.drop_duplicates(
            subset=["in"],
        ).reset_index(drop=True)

        # MAPPING FROM OTHER IDS TO UNIPROT USING BULK API
        database = "UniProtBULK"
        bool_np = df["Accession Number"].apply(lambda x: x.startswith("NP_"))
        # mapping from RefSeq Protein to UniProtKB
        input_ids = ",".join(
            df.loc[~bool_uniprot & bool_np, "Accession Number"].unique().tolist()
        )
        df_refseq = query_uniprotbulk_api(
            input_ids=input_ids,
            term_in="RefSeq_Protein",
            term_out="UniProtKB",
            database=database,
        )
        # mapping from EMBL-GenBank-DDBJ_CDS to UniProtKB
        input_ids = ",".join(
            df.loc[~bool_uniprot & ~bool_np, "Accession Number"].unique().tolist()
        )
        df_cds = query_uniprotbulk_api(
            input_ids=input_ids,
            term_in="EMBL-GenBank-DDBJ_CDS",
            term_out="UniProtKB",
            database=database,
        )
        # concatenate all previous queries
        df_id = pd.concat([df_uniprot, df_refseq, df_cds], axis=0, ignore_index=True)
        # suggestedIds are UPIDs - mapping from UniParc to UniProtKB
        set_upid = set(
            df_id.loc[
                df_id["out"].apply(
                    lambda x: True if str(x[0]).startswith("UPI") else False
                ),
                "out",
            ]
            .apply(lambda x: x[0])
            .tolist()
        )
        input_ids = ",".join(set_upid)
        df_uniparc = query_uniprotbulk_api(
            input_ids=input_ids,
            term_in="UniParc",
            term_out="UniProtKB",
            database=database,
        )
        mapping = dict(zip(df_uniparc["in"], df_uniparc["out"]))
        df_id["out"] = df_id["out"].apply(
            lambda out_list: next(
                (mapping[item] for item in out_list if item in mapping), out_list
            )
        )

        # add UniProt IDs from DICT_KINASE where possible
        set_uniprot = {v.uniprot_id.split("_")[0] for v in DICT_KINASE.values()}
        df_id["key_uniprot"] = df_id["out"].apply(
            lambda x: (
                np.nan
                if pd.isna(x[0])
                else (lambda f: f[0] if f else None)([i for i in x if i in set_uniprot])
            )
        )
        self.df_id = df_id.copy()

        # merge ID mapping to original dataframe
        df_merge = df.merge(
            df_id[["in", "key_uniprot"]],
            how="left",
            left_on="Accession Number",
            right_on="in",
        )
        df_merge.drop(columns="in", inplace=True)

        # mapping from HGNC to UniProt if "DiscoverX Gene Symbol" matches
        dict_hgnc2uniprot = {
            k.split("_")[0]: v.uniprot_id.split("_")[0] for k, v in DICT_KINASE.items()
        }
        df_merge.loc[df_merge["key_uniprot"].isna(), "key_uniprot"] = list(
            map(
                lambda x, y: dict_hgnc2uniprot.get(x.split("(")[0], y),
                df_merge.loc[df_merge["key_uniprot"].isna(), "DiscoverX Gene Symbol"],
                df_merge.loc[df_merge["key_uniprot"].isna(), "key_uniprot"],
            )
        )

        # use HGNC to check remaining unmapped entries - only PIKFYVE is absent from DICT_KINASE
        dict_hgnc_check = dict.fromkeys(["prev_symbol", "alias_symbol"])
        for k in dict_hgnc_check.keys():
            dict_hgnc_check[k] = (
                df_merge.loc[df_merge["key_uniprot"].isna(), "DiscoverX Gene Symbol"]
                .apply(
                    lambda x: hgnc.HGNC(
                        input_symbol_or_id=x.split("(")[0]
                    ).maybe_get_symbol_from_hgnc_search(
                        custom_field=k, custom_term=x.split("(")[0]
                    )
                )
                .tolist()
            )
        list_missing_combo = [
            i[0] if i != [] else (j[0] if j != [] else None)
            for i, j in zip(*dict_hgnc_check.values())
        ]
        list_missing_final = [
            dict_hgnc2uniprot[i] if i is not None else None for i in list_missing_combo
        ]
        df_merge.loc[df_merge["key_uniprot"].isna(), "key_uniprot"] = list_missing_final

        self.df = df_merge.copy()
        self.df = self.replace_multi_mapping_keys()
        self.df["key_hgnc"] = self.df["key_uniprot"].apply(
            lambda x: None if pd.isna(x) else DICT_KINASE_REV.get(x).hgnc_name
        )

        # add full RefSeq sequences
        self.df["refseq_full"] = self.df["Accession Number"].progress_apply(
            lambda x: ProteinNCBI(accession=x.strip()).list_seq[0]
        )

        # add canonical uniprot sequences
        self.df["uniprot_full"] = self.df["key_hgnc"].apply(
            lambda x: None if pd.isna(x) else DICT_KINASE.get(x).uniprot.canonical_seq
        )

        # PIKFYVE does have a UniProt sequence, just not in our DICT_KINASE
        str_pikfyve_hgnc_name = "PIKFYVE"
        idx_pikfyve_bool = self.df["DiscoverX Gene Symbol"] == str_pikfyve_hgnc_name
        str_pikfyve_uniprot_id = hgnc.HGNC(
            input_symbol_or_id=str_pikfyve_hgnc_name
        ).maybe_get_info_from_hgnc_fetch(list_to_extract=["uniprot_ids"])[
            "uniprot_ids"
        ][
            0
        ][
            0
        ]
        self.df.loc[idx_pikfyve_bool, "key_uniprot"] = str_pikfyve_uniprot_id
        str_pikfyve_uniprot_seq = UniProtFASTA(
            uniprot_id=str_pikfyve_uniprot_id
        )._sequence
        self.df.loc[idx_pikfyve_bool, "uniprot_full"] = str_pikfyve_uniprot_seq

    def generate_discoverx_info_dict(self) -> dict[str, DiscoverXInfo]:
        """Generate dictionary of DiscoverXInfo objects from the DataFrame.

        Returns:
        --------
        dict[str, DiscoverXInfo]
            Dictionary of DiscoverXInfo objects keyed by DiscoverX gene symbol.
        """
        dict_out = {}
        for _, row in self.df.iterrows():

            # extract relevant info
            str_symbol = row["DiscoverX Gene Symbol"]
            str_key = row["key_hgnc"]
            str_accession = row["Accession Number"]
            str_uniprot_id = row["key_uniprot"]
            str_mut = row["Construct Description"]
            str_aa_start_stop = row["AA Start/Stop"]
            str_refseq = row["refseq_full"]
            str_uniprot = row["uniprot_full"]

            # parse construct boundaries
            str_start, int_start, str_stop, int_stop = self.return_construct_boundaries(
                str_in=str_aa_start_stop
            )

            # parse mutation info
            bool_wt, dict_mut = self.parse_mutation_info(str_mut=str_mut)

            obj_info = DiscoverXInfo(
                discoverx_gene_symbol=str_symbol,
                key=str_key,
                accession=str_accession,
                uniprot_id=str_uniprot_id,
                bool_wt=bool_wt,
                list_missense_mutations=dict_mut.get("missense", None),
                list_deletion_mutations=dict_mut.get("deletion", None),
                str_start=str_start,
                int_start=int_start,
                str_stop=str_stop,
                int_stop=int_stop,
                seq_refseq=str_refseq,
                seq_uniprot=str_uniprot,
                bool_offset=self.bool_offset,
            )

            dict_out[str_symbol] = obj_info

        return dict_out

    def check_multimatch_str(
        self,
        list_in: list[tuple[int, str]],
    ) -> list[str]:
        """Check which multi-mapping kinase IDs correspond to a given region.

        Parameters
        ----------
        list_in : list[tuple[int, str]]
            List of tuples of (row index, kinase ID) for multi-mapping kinase IDs.

        Returns
        -------
        list[str]
            List of strings containing the resolved kinase IDs.
        """
        # check which multi-mapping IDs correspond to a given region
        list_multi_match = []
        for _, key, aa in list_in:
            _, idx_start, _, idx_end = self.return_construct_boundaries(aa)
            if idx_start is None or idx_end is None:
                list_multi_match.append(key + "_?")
            else:
                for suffix in ["_1", "_2"]:
                    str_match = key + suffix
                    # TODO: should really use list_refseq2uniprot mapping
                    min_klifs_idx = min(
                        DICT_KINASE_REV[str_match].KLIFS2UniProtIdx.values()
                    )
                    if idx_start <= min_klifs_idx <= idx_end:
                        list_multi_match.append(str_match)

        # 2 are missing start/stop for one region but other KD is in data
        list_out = []
        for str_in in list_multi_match:
            if "?" not in str_in:
                list_out.append(str_in)
            else:
                str_prefix = str_in.split("_")[0]
                if str_prefix + "_1" in list_in:
                    list_out.append(str_prefix + "_2")
                else:
                    list_out.append(str_prefix + "_1")

        list_out = [(i[0], j) for i, j in zip(list_in, list_out)]

        return list_out

    def replace_multi_mapping_keys(self) -> list[str | None]:
        """Combine mono- and multi-mapping UniProt ID lists into a single list.

        Returns:
        --------
        list[str | None]
            Combined list of UniProt IDs.
        """
        set_key_mono = {
            v.uniprot_id for v in DICT_KINASE.values() if "_" not in v.uniprot_id
        }

        # adjudicate which multi-mapping region is covered
        list_idx_str_multi = [
            (index, row["key_uniprot"], row["AA Start/Stop"])
            for index, row in self.df.iterrows()
            if row["key_uniprot"] not in set_key_mono and row["key_uniprot"] is not None
        ]
        list_idx_str_replace = self.check_multimatch_str(list_idx_str_multi)

        # replace indices for multi-mapping entries in dataframe
        df_copy = self.df.copy()
        for idx, key in list_idx_str_replace:
            df_copy.at[idx, "key_uniprot"] = key

        return df_copy

    @staticmethod
    def return_construct_boundaries(
        str_in: str,
    ) -> tuple[str | None, int | None, str | None, int | None]:
        """Return the start and end indices of the kinase construct from the construct description.

        Parameters:
        -----------
        str_in : str
            "AA Start/Stop" string.

        Returns:
        --------
        tuple[str | None, int | None, str | None, int | None]
            Tuple of (str_start, idx_start, str_end, idx_end).
        """
        if str_in != "Null":
            str_aa_start, str_aa_stop = str_in.split("/")

            str_start, idx_start = str_aa_start[0], int(str_aa_start[1:])
            str_end, idx_end = str_aa_stop[0], int(str_aa_stop[1:])

            return str_start, idx_start, str_end, idx_end
        else:
            return None, None, None, None

    @staticmethod
    def return_missense_mutation(str_in: str) -> tuple[str, int, str]:
        """Return the wild-type residue, codon index, and mutant residue from a missense mutation string.

        Parameters:
        -----------
        str_in : str
            Missense mutation string (e.g., "A123T").

        Returns:
        --------
        tuple[str, int, str]
            Tuple of (str_wt, idx_codon, str_mut).
        """
        str_wt = str_in[0]
        idx_codon = int(str_in[1:-1])
        str_mut = str_in[-1]

        return str_wt, idx_codon, str_mut

    @staticmethod
    def return_deletion_mutation(str_in: str) -> tuple[str, list[int], str]:
        """Return the start residue, list of codon indices, and end residue from a deletion

        Parameters:
        -----------
        str_in : str
            Deletion mutation string (e.g., "A123-125del").

        Returns:
        --------
        tuple[str, list[int], str]
            Tuple of (str_start, list_idx_del, str_end).
        """
        str_in = str_in.replace("del", "")
        list_split = str_in.split("-")

        str_start, idx_start = list_split[0][0], int(list_split[0][1:])
        str_end, idx_end = list_split[1][0], int(list_split[1][1:])

        list_idx_del = list(range(idx_start, idx_end + 1))

        return str_start, list_idx_del, str_end

    def parse_mutation_info(self, str_mut: str) -> tuple[bool, dict[str, list | None]]:
        """Parse mutation information from a mutation string.

        Parameters:
        -----------
        str_mut : str
            Mutation string in the format "A123B" (missense) or "A123-B" (deletion).

        Returns:
        --------
        tuple[bool, dict[str, list | None]]
            Tuple of (is_wild_type, dict of mutations) where dict contains:
                - "missense": list of missense mutations as tuples (wt_aa, position, mut_aa)
                - "deletion": list of deletion mutations as tuples (wt_aa, list of positions, mut_aa or '-')
                If wild-type, returns (True, {"missense": None, "deletion": None}).
                If mutations are present, returns (False, {"missense": [...], "deletion": [...]}).
                Missense mutations are represented as tuples of (WT AA, position, mutant AA).
                Deletion mutations are represented as tuples of (WT start AA, list of positions, WT end AA).
        """
        bool_wt = True if str_mut == "Wild Type" else False

        dict_mut = {"missense": None, "deletion": None}
        if not bool_wt:
            # create list of mutations
            start, end = str_mut.find("("), str_mut.find(")")
            list_muts = [i.strip() for i in str_mut[start + 1 : end].split(",")]
            # categorize mutations into missense and deletions
            list_missense, list_deletion = [], []
            for str_mut_raw in list_muts:
                # deletions
                if str_mut_raw.endswith("del"):
                    tuple_del = self.return_deletion_mutation(str_mut_raw)
                    list_deletion.append(tuple_del)
                # missense mutations
                else:
                    tuple_mut = self.return_missense_mutation(str_mut_raw)
                    list_missense.append(tuple_mut)
            # add to dict if not empty
            if list_missense != []:
                dict_mut["missense"] = list_missense
            if list_deletion != []:
                dict_mut["deletion"] = list_deletion

        return bool_wt, dict_mut
