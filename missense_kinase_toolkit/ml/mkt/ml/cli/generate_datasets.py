#!/usr/bin/env python

from os import path
from typing import Any

import pandas as pd
from mkt.databases import ncbi
from mkt.databases.io_utils import get_repo_root
from mkt.databases.utils import try_except_return_none_rgetattr
from mkt.schema import io_utils
from nf_rnaseq.variables import DICT_DATABASES
from tqdm import tqdm

# Functions #


def split_if_not_null(str_in, delim="/", val_null="Null"):
    if str_in == val_null:
        return None, None
    else:
        return str_in.split(delim)[0], str_in.split(delim)[1]


def run_id_mapping(
    database: str,
    input_ids: str,
    term_in: str,
    term_out: str,
) -> Any:
    """
    Run ID mapping using the nf-rnaseq API.

    Parameters
    ----------
    database : str
        The database to use for ID mapping.
    input_ids : str
        The input IDs to map.
    term_in : str
        The input term type.
    term_out : str
        The output term type.

    Returns
    -------
    Any
        The result of the ID mapping.

    """
    post_obj = DICT_DATABASES[database]["POST"]["api_object"](
        identifier=input_ids,
        term_in=term_in,
        term_out=term_out,
        url_base=DICT_DATABASES[database]["POST"]["url_base"],
    )
    api_obj = DICT_DATABASES[database]["GET"]["api_object"](
        identifier=input_ids,
        term_in=term_in,
        term_out=term_out,
        url_base=DICT_DATABASES[database]["GET"]["url_base"],
        headers=DICT_DATABASES[database]["GET"]["headers"],
        jobId=post_obj.jobId,
    )
    return api_obj


# Main #

# download data
df_pkis2 = pd.read_csv(
    "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv"
)
df_kinomescan = pd.read_csv(
    "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/DiscoverX_489_Kinase_Assay_Construct_Information.csv"
)

df_pkis_rev = df_pkis2.set_index("Smiles").iloc[:, 6:]
list_pkis_constructs = df_pkis_rev.columns.tolist()

list_acc = (
    df_kinomescan.loc[
        df_kinomescan["DiscoverX Gene Symbol"].isin(list_pkis_constructs),
        "Accession Number",
    ]
    .unique()
    .tolist()
)

dict_seq = {}
for acc in tqdm(list_acc):
    temp = ncbi.ProteinEntrez(acc)
    dict_seq[acc] = temp.fasta_record

df_ncbi = pd.DataFrame(
    {
        "accession": [k for k in dict_seq.keys()],
        "description": [v.description for v in dict_seq.values()],
        "sequence_full": [str(v.seq) for v in dict_seq.values()],
    }
)

df_merge = df_kinomescan.loc[
    df_kinomescan["DiscoverX Gene Symbol"].isin(list_pkis_constructs), :
].merge(
    df_ncbi,
    how="inner",
    left_on="Accession Number",
    right_on="accession",
)
df_merge["AA Start/Stop"] = df_merge["AA Start/Stop"].apply(split_if_not_null)

list_seq_partial = []
for _, row in df_merge.iterrows():
    start, stop = row["AA Start/Stop"][0], row["AA Start/Stop"][1]
    seq = row["sequence_full"]
    if start is not None:
        seq_temp = seq[int(start[1:]) - 1 : int(stop[1:])]
        assert seq_temp[0] == start[0], seq_temp[-1] == stop[0]
        list_seq_partial.append(seq_temp)
    else:
        list_seq_partial.append(None)
df_merge["sequence_partial"] = list_seq_partial

# df_merge.to_csv(path.join(get_repo_root(), "data/pkis2_annotated.csv"), index=False)

# Get UniProt IDs from current accession numbers #

# df_merge = pd.read_csv(path.join(get_repo_root(), "data/pkis2_annotated.csv"))

list_np = df_merge.loc[
    df_merge["Accession Number"].apply(lambda x: x.startswith("NP_")),
    "Accession Number",
].tolist()
database = "UniProtBULK"
term_out = "UniProtKB"

term_in = "RefSeq_Protein"
input_ids = ",".join(list_np)
api_obj_refseq = run_id_mapping(database, input_ids, term_in, term_out)
dict_swissprot = {
    i["from"]: i["to"]["primaryAccession"]
    for i in api_obj.json["results"]
    if i["to"]["entryType"] == "UniProtKB reviewed (Swiss-Prot)"
}

list_other = df_merge.loc[
    df_merge["Accession Number"].apply(lambda x: not x.startswith("NP_")),
    "Accession Number",
].tolist()
term_in = "EMBL-GenBank-DDBJ_CDS"
input_ids = ",".join(
    [i for i in list_other if len(i.split(".")[0]) == 8 and not i.startswith("EA")]
)
api_obj_embl = run_id_mapping(database, input_ids, term_in, term_out)
dict_embl = {
    i["from"]: i["to"]["primaryAccession"]
    for i in api_obj_embl.json["results"]
    if i["to"]["entryType"] == "UniProtKB reviewed (Swiss-Prot)"
}

dict_combo = {**dict_swissprot, **dict_embl}
df_uniprot = pd.DataFrame(dict_combo, index=[1]).T
df_uniprot.columns = ["UniProt_ID"]
df_merge_uniprot = df_merge.merge(
    df_uniprot,
    how="left",
    left_on="Accession Number",
    right_index=True,
)

# 3 UniProt IDs
bool_idx = (df_merge_uniprot["UniProt_ID"].isna()) & (
    df_merge_uniprot["Accession Number"].apply(lambda x: len(x.split(".")[0]) == 6)
)
df_merge_uniprot.loc[bool_idx, "UniProt_ID"] = df_merge_uniprot.loc[
    bool_idx, "Accession Number"
].apply(lambda x: x.split(".")[0])

df_merge_uniprot.to_csv(
    path.join(get_repo_root(), "data/pkis2_annotated.csv"), index=False
)

# set_dup = set(
#     df_merge_uniprot.loc[
#         (
#             (df_merge_uniprot["UniProt_ID"].duplicated()) & \
#             (~df_merge_uniprot["UniProt_ID"].isna())
#         ),
#         "UniProt_ID"
#     ]
# )

# df_merge_uniprot.loc[
#     df_merge_uniprot["UniProt_ID"].isin(set_dup),
#     "DiscoverX Gene Symbol"
# ].tolist()

# Anotate with KinaseInfo #

df_merge = pd.read_csv(path.join(get_repo_root(), "data/pkis2_annotated.csv"))
dict_kinase = io_utils.deserialize_kinase_dict()
dict_kinase_rev = {v.uniprot_id: v for v in dict_kinase.values()}

df_merge_narm = df_merge.loc[~df_merge["UniProt_ID"].isna(), :]
list_to_drop = ["-phosphorylated", "-cyclin", "-autoinhibited"]

df_merge_narm = df_merge_narm.loc[
    ~df_merge_narm["DiscoverX Gene Symbol"].apply(
        lambda x: any([i in x for i in list_to_drop])
    ),
    :,
].reset_index(drop=True)

df_merge_narm.iloc[0]
set_uniprot_dup = set(
    df_merge_narm.loc[df_merge_narm["UniProt_ID"].duplicated(), "UniProt_ID"]
)

df_merge_narm.loc[df_merge_narm["UniProt_ID"].isin(set_uniprot_dup), :].iloc[0]
df_merge_narm.loc[
    df_merge_narm["UniProt_ID"].isin(set_uniprot_dup),
    ["DiscoverX Gene Symbol", "AA Start/Stop"],
]
df_merge_narm.loc[
    df_merge_narm["UniProt_ID"].isin(set_uniprot_dup), "DiscoverX Gene Symbol"
].tolist()

for i in set_uniprot_dup:
    for j in df_merge_narm.loc[
        df_merge_narm["UniProt_ID"] == i, "DiscoverX Gene Symbol"
    ].tolist():
        if "Kin.Dom.1" in j:
            df_merge_narm.loc[
                df_merge_narm["DiscoverX Gene Symbol"] == j, "UniProt_ID"
            ] = (i + "_1")
        elif "Kin.Dom.2" in j:
            df_merge_narm.loc[
                df_merge_narm["DiscoverX Gene Symbol"] == j, "UniProt_ID"
            ] = (i + "_2")
        else:
            print(f"UniProt: {i}\nDiscoverX: {j}\n")

# for k, v in dict_kinase.items():
#     if "_" in k:
#         try:
#             temp = dict_kinase[k].kincore.fasta
#             print(k)
#             print(f"{temp.start}, {temp.end}")
#             print()
#         except:
#             pass

df_merge_narm.loc[
    df_merge_narm["UniProt_ID"].apply(lambda x: "_" in x), "UniProt_ID"
].apply(lambda x: x.split("_")[0]).value_counts()

dict_kinase["RPS6KA4_1"].kincore.fasta
dict_kinase["RPS6KA4_2"].kincore.fasta
for i in set_uniprot_dup:
    dict_kinase[i]


df_merge_narm.loc[
    df_merge_narm["UniProt_ID"].isin(set_uniprot_dup), "sequence_full"
].apply(lambda x: dict_kinase["RPS6KA4_1"].kincore.fasta.seq in x)
dict_kinase["RPS6KA4_1"].kincore.fasta.seq
df_merge["UniProt_ID"] == "O75676"

set_ids = set(df_merge["UniProt_ID"].tolist())
len([k for k in dict_kinase_rev.keys() if k.split("_")[0] in set_ids])

list_seq_partial = df_merge_narm.loc[
    df_merge_narm["UniProt_ID"].apply(lambda x: x not in dict_kinase_rev.keys()),
    "sequence_partial",
].tolist()

for i in list_seq_partial:
    print(
        [
            k
            for k, v in dict_kinase_rev.items()
            if v.kincore is not None and i in v.kincore.fasta.seq
        ]
    )

df_merge_narm_reconciled = df_merge_narm.loc[
    df_merge_narm["UniProt_ID"].apply(lambda x: x in dict_kinase_rev.keys()), :
].reset_index(drop=True)

list_kincore = (
    df_merge_narm_reconciled["UniProt_ID"]
    .apply(
        lambda x: try_except_return_none_rgetattr(
            dict_kinase_rev[x], "kincore.fasta.seq"
        )
    )
    .tolist()
)
df_merge_narm_reconciled["kincore"] = list_kincore

list_klifs = (
    df_merge_narm_reconciled["UniProt_ID"]
    .apply(
        lambda x: try_except_return_none_rgetattr(
            dict_kinase_rev[x], "klifs.pocket_seq"
        )
    )
    .tolist()
)
df_merge_narm_reconciled["klifs"] = list_klifs

list_kincore_group = (
    df_merge_narm_reconciled["UniProt_ID"]
    .apply(
        lambda x: try_except_return_none_rgetattr(
            dict_kinase_rev[x], "kincore.cif.group"
        )
    )
    .tolist()
)
df_merge_narm_reconciled["kincore_group"] = list_kincore_group

df_merge_narm_reconciled.to_csv(
    path.join(get_repo_root(), "data/pkis2_annotated.csv"), index=False
)

df_annot_rev = df_annot.loc[df_annot["klifs"].notnull(), :].reset_index(drop=True)
df_pkis_rev.columns
df_pkis_rev.columns.map(lambda x: x in df_annot_rev["DiscoverX Gene Symbol"].tolist())

df_pkis_rev = df_pkis_rev.iloc[
    :,
    df_pkis_rev.columns.map(
        lambda x: x in df_annot_rev["DiscoverX Gene Symbol"].tolist()
    ),
]
df_pkis_rev = df_pkis_rev.melt(
    var_name="DiscoverX Gene Symbol",
    value_name="percent_displacement",
    ignore_index=False,
).reset_index()

df_combo = df_pkis_rev.merge(
    df_annot_rev[["DiscoverX Gene Symbol", "klifs", "kincore_group"]],
    how="left",
    on="DiscoverX Gene Symbol",
)

df_combo.to_csv(
    "/data1/tanseyw/projects/whitej/missense-kinase-toolkit/data/pkis_data.csv",
    index=False,
)
