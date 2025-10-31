import logging

import pandas as pd
from mkt.databases import hgnc
from mkt.databases.aligners import BL2UniProtAligner
from mkt.databases.chembl import ChEMBLMolecule, return_chembl_id
from mkt.databases.datasets.process import (
    DatasetConfig,
    ProcessDataset,
    check_multimatch_str,
    disambiguate_kinase_ids,
)
from mkt.databases.ncbi import ProteinNCBI
from mkt.schema.io_utils import deserialize_kinase_dict
from tqdm import tqdm

logger = logging.getLogger(__name__)

tqdm.pandas()


DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")


# don't currently have a good way to handle these
DICT_DAVIS_DROP = {
    "DiscoverX Gene Symbol": [
        "P.falciparum",  # not human kinase
        "M.tuberculosis",  # not human kinase
        "-phosphorylated",  # no way to handle phosphorylated residues yet
        "-cyclin",  # no way to handle cyclin proteins yet
    ],
    "Construct Description": [
        "(L747-T751del,Sins)",  # not sure what Sins mutation is in EGFR
        "(ITD)",  # internal tandem duplication - complex mutation, need more details
    ],
}
"""Dict[str, list[str]]: terms to drop where key is colname and value is list of terms."""

DICT_DAVIS_MERGE_FIX = {
    "GCN2(Kin.Dom.2,S808G)": {
        "Construct Description": "Mutation (S808G)",  # given name, this seems to be a mutant
    },
    "RIPK5": {
        "Entrez Gene Symbol": "DSTYK",  # updated HGNC gene symbol - just a typo
    },
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

DICT_ID2CHEMBL = {
    "GSK-1838705A": "CHEMBL464552",
    "MLN-120B": "CHEMBL608154",
    "PD-173955": "CHEMBL386051",
    "PP-242": "CHEMBL1241674",
}
"""Dictionary of manual ChEMBL ID mappings for Davis dataset drugs."""


class DavisConfig(DatasetConfig):
    """Configuration for the Davis dataset."""

    name: str = "Davis"
    url_main: str = (
        "https://raw.githubusercontent.com/choderalab/missense-kinase-toolkit/main/data/41587_2011_BFnbt1990_MOESM5_ESM.xls"
    )
    url_supp_drug: str = (
        "https://raw.githubusercontent.com/choderalab/missense-kinase-toolkit/main/data/41587_2011_BFnbt1990_MOESM4_ESM.xls"
    )
    url_supp_kinase: str = (
        "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/DiscoverX_489_Kinase_Assay_Construct_Information.csv"
    )
    col_drug: str = "Drug"
    col_kinase: str = "Target_ID"
    col_y: str = "Y"


class DavisDataset(DavisConfig, ProcessDataset):
    """Davis dataset processing class."""

    def __init__(self, **kwargs):
        """Initialize Davis dataset."""
        super().__init__(**kwargs)

        config_dict = DavisConfig().model_dump()
        config_dict.update(kwargs)

        ProcessDataset.__init__(self, **config_dict)

    def process(self) -> pd.DataFrame:
        """Process the Davis dataset."""
        df = pd.read_excel(self.url_main, sheet_name=0)
        df_drug = pd.read_excel(self.url_supp_drug, sheet_name=0)
        df_kinase = pd.read_csv(self.url_supp_kinase)

        df_drug = self.add_chembl_info(df_drug)

        df_merge = (
            df[["Entrez Gene Symbol", "Kinase"]]
            .merge(
                df_kinase,
                left_on="Kinase",
                right_on="DiscoverX Gene Symbol",
                how="left",
            )
            .drop(columns="Kinase")
        )

        # drop things we cannot currently handle
        df_merge_drop = df_merge.copy()
        for k, v in DICT_DAVIS_DROP.items():
            df_merge_drop = df_merge_drop.loc[
                ~df_merge_drop[k].apply(lambda x: any([i in str(x) for i in v])), :
            ].reset_index(drop=True)

        # apply manual fixes for issues found
        for k1, v1 in DICT_DAVIS_MERGE_FIX.items():
            for k2, v2 in v1.items():
                df_merge_drop.loc[df_merge_drop["DiscoverX Gene Symbol"] == k1, k2] = v2

        # extract kinase dictionary keys and add UniProt full sequences
        list_key = self.extract_dict_kinase_keys(df_merge_drop)
        df_merge_drop["key"] = list_key
        df_merge_drop["uniprot_full"] = df_merge_drop["key"].apply(
            lambda x: DICT_KINASE[x].uniprot.canonical_seq
        )

        # add RefSeq NCBI protein sequences
        list_ncbi = df_merge_drop["Accession Number"].progress_apply(
            lambda x: ProteinNCBI(accession=x).list_seq[0]
        )
        df_merge_drop["ncbi_full"] = list_ncbi

        # only AKTs have no "AA Start/Stop" after drops/fixes - given length, make entire NCBI sequence the construct
        df_merge_drop["AA Start/Stop"] = df_merge_drop.apply(
            lambda row: (
                row["ncbi_full"][0]
                + "1/"
                + row["ncbi_full"][-1]
                + str(len(row["ncbi_full"]))
                if row["AA Start/Stop"] == "Null"
                else row["AA Start/Stop"]
            ),
            axis=1,
        )

        # map NCBI to UniProt indices (and visa versa) using global alignment
        list_map = list(
            map(
                lambda x, y: self.return_ncbi2uniprot_mapping(x, y),
                tqdm(
                    df_merge_drop["ncbi_full"],
                    desc="Mapping NCBI to UniProt indices (and visa versa) using global alignment",
                ),
                df_merge_drop["uniprot_full"],
            )
        )
        df_merge_drop["list_ncbi2uniprot"] = [i[0] for i in list_map]
        df_merge_drop["list_uniprot2ncbi"] = [i[1] for i in list_map]

        df_merge_drop["kd_start"] = df_merge_drop["key"].apply(
            lambda x: DICT_KINASE[x].adjudicate_kd_start()
        )
        df_merge_drop["kd_end"] = df_merge_drop["key"].apply(
            lambda x: DICT_KINASE[x].adjudicate_kd_end()
        )

        df_merge_drop, list_outside_kd = self.replace_mutations_in_sequences(
            df_merge_drop
        )

        # TODO: keep only necessary columns
        # df_keep = df[[self.col_drug, self.col_kinase, self.col_y]]
        # return df_keep

        return df_merge_drop

    def add_source_column(self) -> pd.DataFrame:
        """Add a Davis name column to the DataFrame."""
        df = self.df.copy()
        df["source"] = self.name.title()
        return df

    @staticmethod
    def add_chembl_info(df_in: pd.DataFrame) -> pd.DataFrame:
        """Add ChEMBL preferred IDs and canonical SMILES to the Davis dataset.

        Parameters:
        -----------
        df_in : pd.DataFrame
            Input DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame with ChEMBL SMILES added.
        """
        # get list of drugs with alternative names
        list_davis_drugs = list(
            map(
                lambda x, y: x if str(y) == "nan" else y,
                df_in["Compound Name"],
                df_in["Alternative Name"],
            )
        )

        # query ChEMBL for each drug - have manually adjudicated some changes
        dict_chembl_id = {
            drug: {"source": None, "ids": []} for drug in list_davis_drugs
        }
        for drug in tqdm(list_davis_drugs, desc="Querying drugs in ChEMBL"):
            drug_rev = drug.split(" (")[0]
            chembl_id, source = return_chembl_id(drug_rev)
            dict_chembl_id[drug]["source"] = source
            dict_chembl_id[drug]["ids"].extend(chembl_id)
        dict_chembl_id_rev = {k: v["ids"][0] for k, v in dict_chembl_id.items()}
        dict_chembl_id_rev.update(DICT_ID2CHEMBL)

        # query for ChEMBL ChEMBLMolecule objects
        list_chembl_molec = [
            ChEMBLMolecule(id=v)
            for v in tqdm(dict_chembl_id_rev.values(), desc="Querying ChEBMLMolecule")
        ]
        dict_chembl_molecule = {
            k: {"chembl_id": v, "molecule": mol}
            for (k, v), mol in zip(dict_chembl_id_rev.items(), list_chembl_molec)
        }

        df_in["pref_name"] = [
            v["molecule"].adjudicate_preferred_name(k)
            for k, v in dict_chembl_molecule.items()
        ]
        df_in["smiles"] = [
            v["molecule"].return_smiles() for v in dict_chembl_molecule.values()
        ]

        return df_in

    @staticmethod
    def extract_dict_kinase_keys(df_in: pd.DataFrame) -> list[str]:
        """Disambiguate kinase IDs in the Davis dataset.

        Parameters:
        -----------
        df_in : pd.DataFrame
            Input DataFrame.

        Returns:
        --------
        list[str]
            List of kinase dictionary keys.
        """
        # check for mono and multi mapping
        list_multi_str = disambiguate_kinase_ids(df_in, DICT_KINASE, bool_mono=False)

        # some need to check against an alternate ID (prev or alias) to get proper HGNC gene name
        list_idx_missing = [idx for idx, i in enumerate(list_multi_str) if i == ""]
        dict_hgnc_check = dict.fromkeys(["prev_symbol", "alias_symbol"])
        for k in dict_hgnc_check.keys():
            dict_hgnc_check[k] = (
                df_in.iloc[list_idx_missing]["Entrez Gene Symbol"]
                .apply(
                    lambda x: hgnc.HGNC(
                        input_symbol_or_id=x
                    ).maybe_get_symbol_from_hgnc_search(custom_field=k, custom_term=x)
                )
                .tolist()
            )
        list_missing_combo = [
            i[0] if i != [] else j[0] for i, j in zip(*dict_hgnc_check.values())
        ]
        # check that all updated HGNC names are in DICT_KINASE
        assert all([i in DICT_KINASE for i in list_missing_combo])

        # check for mono mapping only
        list_mono_str = disambiguate_kinase_ids(df_in, DICT_KINASE)

        # adjudicate which multi-mapping region is covered
        list_idx_str_multi = [
            (idx, j)
            for idx, (i, j) in enumerate(zip(list_mono_str, list_multi_str))
            if i != j and j != ""
        ]
        list_str_replace_multi = check_multimatch_str(
            list_idx_str_multi, df_in, DICT_KINASE
        )
        list_idx_multi = [i[0] for i in list_idx_str_multi]

        # replace missing entries in mono with missing and then multi-match fixes
        list_combo_str = list_mono_str.copy()
        list_combo_str = [
            (
                list_missing_combo[list_idx_missing.index(idx)]
                if idx in list_idx_missing
                else i
            )
            for idx, i in enumerate(list_combo_str)
        ]
        list_combo_str = [
            (
                list_str_replace_multi[list_idx_multi.index(idx)]
                if idx in list_idx_multi
                else i
            )
            for idx, i in enumerate(list_combo_str)
        ]

        return list_combo_str

    def return_ncbi2uniprot_mapping(self, str_ncbi: str, str_uniprot: str):
        """Return mapping of NCBI to UniProt indices (and visa versa) using global alignment.

        Parameters:
        -----------
        str_ncbi : str
            NCBI sequence string.
        str_uniprot : str
            UniProt sequence string.

        Returns:
        --------
        list[int], list[int]
            Two lists mapping NCBI to UniProt indices and vice versa;
                index in initial string given by position in list.
        """
        if str_ncbi == str_uniprot:
            list_idx_uniprot2ncbi = list(range(len(str_uniprot)))
            list_idx_ncbi2uniprot = list(range(len(str_ncbi)))
            if self.bool_offset:
                list_idx_uniprot2ncbi = [i + 1 for i in list_idx_uniprot2ncbi]
                list_idx_ncbi2uniprot = [i + 1 for i in list_idx_ncbi2uniprot]
        else:
            aligner = BL2UniProtAligner()
            alignments = aligner.align(str_ncbi, str_uniprot)

            # if True, use 1-based indexing
            if self.bool_offset:
                idx_ncbi, idx_uniprot = 1, 1
            else:
                idx_ncbi, idx_uniprot = 0, 0

            align_ncbi, align_uniprot = alignments[0][0], alignments[0][1]
            list_idx_uniprot2ncbi, list_idx_ncbi2uniprot = [], []
            for char_ncbi, char_uniprot in zip(align_ncbi, align_uniprot):
                # no match in ncbi (but match in uniprot)
                if char_ncbi == "-":
                    list_idx_uniprot2ncbi.append(None)
                    idx_uniprot += 1
                # no match in uniprot (but match in ncbi)
                elif char_uniprot == "-":
                    list_idx_ncbi2uniprot.append(None)
                    idx_ncbi += 1
                # allows for mismatch characters (".") - do not record them
                else:
                    list_idx_ncbi2uniprot.append(idx_uniprot)
                    list_idx_uniprot2ncbi.append(idx_ncbi)

                    idx_ncbi += 1
                    idx_uniprot += 1

            assert len(list_idx_uniprot2ncbi) == len(str_uniprot)
            assert len(list_idx_ncbi2uniprot) == len(str_ncbi)

        return list_idx_ncbi2uniprot, list_idx_uniprot2ncbi

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

    def replace_mutations_in_sequences(
        self, df_in: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[tuple]]:
        """Replace mutations in NCBI and UniProt full sequences in the Davis dataset.

        Parameters:
        -----------
        df_in : pd.DataFrame
            Input DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame with mutations replaced in NCBI and UniProt full sequences.
        list[tuple]
            List of tuples containing information about mutations that fall outside the kinase domain.
        """
        df_out = df_in.copy()
        list_outside_kd = []
        for index, row in df_out.iterrows():

            str_muts = row["Construct Description"]

            if str_muts != "Wild Type":

                start, end = str_muts.find("("), str_muts.find(")")
                list_muts = [i.strip() for i in str_muts[start + 1 : end].split(",")]

                sequences = {
                    "ncbi_full": row["ncbi_full"],
                    "uniprot_full": row["uniprot_full"],
                }
                start_kd, end_kd = row["kd_start"], row["kd_end"]
                for str_mut_raw in list_muts:

                    # missense mutations
                    if not str_mut_raw.endswith("del"):
                        str_wt, idx_codon, str_mut = self.return_missense_mutation(
                            str_mut_raw
                        )

                        # double check
                        assert sequences["ncbi_full"][idx_codon - 1] == str_wt
                        assert sequences["uniprot_full"][idx_codon - 1] == str_wt
                        assert idx_codon in row["list_ncbi2uniprot"]
                        if not pd.isna(start_kd) and not pd.isna(end_kd):
                            try:
                                assert start_kd <= idx_codon
                                assert idx_codon <= end_kd
                            except AssertionError:
                                logger.info(
                                    f"Missense mutation {str_mut_raw} in {row['DiscoverX Gene Symbol']} "
                                    f"falls outside adjudicated kinase domain range {start_kd}-{end_kd}."
                                )
                                list_outside_kd.append(
                                    (
                                        row["DiscoverX Gene Symbol"],
                                        str_mut_raw,
                                        start_kd,
                                        end_kd,
                                    )
                                )

                        for colname in ["ncbi_full", "uniprot_full"]:
                            str_replace = "".join(
                                [
                                    i if idx != idx_codon - 1 else str_mut
                                    for idx, i in enumerate(sequences[colname])
                                ]
                            )
                            assert str_replace[idx_codon - 1] == str_mut
                            df_out.loc[index, colname] = str_replace
                            sequences[colname] = str_replace

                    # deletions
                    else:
                        str_start, list_idx_del, str_end = (
                            self.return_deletion_mutation(str_mut_raw)
                        )
                        list_idx_del_zero = [i - 1 for i in list_idx_del]

                        # double check
                        assert sequences["ncbi_full"][list_idx_del[0] - 1] == str_start
                        assert (
                            sequences["uniprot_full"][list_idx_del[0] - 1] == str_start
                        )
                        assert sequences["ncbi_full"][list_idx_del[-1] - 1] == str_end
                        assert (
                            sequences["uniprot_full"][list_idx_del[-1] - 1] == str_end
                        )
                        assert all(
                            [i in row["list_ncbi2uniprot"] for i in list_idx_del]
                        )
                        if not pd.isna(start_kd) and not pd.isna(end_kd):
                            try:
                                assert start_kd <= list_idx_del[0]
                                assert list_idx_del[-1] <= end_kd
                            except AssertionError:
                                logger.info(
                                    f"Missense mutation {str_mut_raw} in {row['DiscoverX Gene Symbol']} "
                                    f"falls outside adjudicated kinase domain range {start_kd}-{end_kd}."
                                )
                                list_outside_kd.append(
                                    (
                                        row["DiscoverX Gene Symbol"],
                                        str_mut_raw,
                                        start_kd,
                                        end_kd,
                                    )
                                )

                        for colname in ["ncbi_full", "uniprot_full"]:
                            str_replace = "".join(
                                [
                                    "-" if idx in list_idx_del_zero else i
                                    for idx, i in enumerate(sequences[colname])
                                ]
                            )
                            assert all(
                                [
                                    i == "-"
                                    for i, idx in enumerate(str_replace)
                                    if idx in list_idx_del_zero
                                ]
                            )
                            df_out.loc[index, colname] = str_replace
                            sequences[colname] = str_replace

        return df_out, list_outside_kd


# NOT IN USE #

# figured out boundary conditions for RPS6KA4 and RPS6KA5 N-term constructs
# for key in set([i.split("_")[0] for i in davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: "RPS6KA" in x), "key"].unique()]):
#     print(davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: key in x), "key"].values)
#     print(davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: key in x), "DiscoverX Gene Symbol"].values)
#     temp = davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: key in x), "AA Start/Stop"].values
#     print(temp)
#     if len(temp) > 1:
#         if temp[0] != "Null" and temp[1] != "Null":
#             idx_start_11, idx_end_12 = [int(i[1:]) for i in temp[0].split("/")]
#             idx_start_21, idx_end_22 = [int(i[1:]) for i in temp[1].split("/")]
#             print(idx_start_21 - idx_end_12)
#     print(davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: key in x), "kd_end"].values)
#     print()
