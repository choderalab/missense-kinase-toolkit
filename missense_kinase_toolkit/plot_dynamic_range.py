#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib_venn import venn2

from tdc.multi_pred import DTI


def convert_to_percentile(input, orig_max=10000):
    return (input / orig_max) * 100

def convert_from_percentile(input, orig_max=10000, precision=3):
    if precision is not None:
        try:
            return round((input / 100) * orig_max, precision)
        except TypeError:
            return [np.round(i, precision) for i in (input / 100) * orig_max]
    else:
        return (input / 100) * max(orig_max)

def main():
    col_davis_drug = "Drug"
    col_davis_target = "Target_ID"
    col_davis_y = "Y"
    col_davis_y_transformed = "Y_trans"

    data_davis = DTI(name = "DAVIS")
    data_davis.harmonize_affinities("mean")
    df_davis = data_davis.get_data()
    df_davis_pivot = df_davis.pivot(index=col_davis_drug, columns=col_davis_target, values=col_davis_y)
    df_davis[col_davis_y_transformed] = convert_to_percentile(df_davis[col_davis_y])
    temp = convert_from_percentile(df_davis[col_davis_y_transformed], df_davis[col_davis_y])


    df_pkis2 = pd.read_csv("https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv")
    df_pkis2_pivot = df_pkis2.iloc[:, 7:]
    df_pkis2_pivot.index = df_pkis2["Smiles"]
    df_pkis2_melt = df_pkis2_pivot.reset_index().melt(id_vars="Smiles", var_name="Kinase", value_name="Percent Inhibition")
    df_pkis2_melt["1-Percent Inhibition"] = 100 - df_pkis2_melt["Percent Inhibition"]

    na_davis = sum(df_davis["Y"] == df_davis["Y"].max()) / df_davis.shape[0]
    na_pkis2 = sum(df_pkis2_melt["Percent Inhibition"] == 0) / df_pkis2_melt.shape[0]

    fig, ax1 = plt.subplots()
    alpha = 0.25
    sns.histplot(
        data=df_pkis2_melt,
        x="1-Percent Inhibition",
        ax=ax1,
        bins= 100,
        log=True,
        color="blue",
        alpha=alpha,
        label=f"PKIS2 (n={df_pkis2_melt.shape[0]:,})",
    )
    sns.histplot(
        data=df_davis,
        x="Y_trans",
        ax=ax1,
        bins= 100,
        log=True,
        color="green",
        alpha=alpha,
        label=f"Davis (n={df_davis.shape[0]:,})",
    )
    ax1.set_xlabel(r"1-% inhibition (PKIS2)", color="blue")
    ax1.axvline(x=99, color="red", linestyle="--")
    ax1.text(
        x=0.5, 
        y=1.2, 
        s="Comparing dynamic assay ranges", 
        fontsize=12, 
        weight='bold', 
        ha='center', 
        va='bottom', 
        transform=ax1.transAxes
    )
    ax1.text(
        x=0.5,
        y=1.15, 
        s=f"No binding detected: "
        f"{na_davis:.1%} Davis, "
        f"{na_pkis2:.1%} PKIS2", 
        fontsize=8, 
        alpha=0.75, 
        ha='center',
        va='bottom', 
        transform=ax1.transAxes
    )
    ax1.get_xaxis().set_major_formatter(
        lambda x, p: format(x/100, ".0%")
    )
    ax2 = ax1.secondary_xaxis(
        "top", 
        functions=(convert_from_percentile, convert_to_percentile)
    )
    ax2.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2.set_xlabel(r"$K_{d}$ (nM) (Davis)", color="green")
    plt.ylabel(r"$log_{10}$(count)")
    plt.legend(loc="upper left")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig("DiscoverX_Dynamic_Range_Hist.svg")
    plt.clf()

    # set_pkis_kin = set(df_pkis2_pivot.columns.str.upper())
    # set_davis_kin = set(df_davis_pivot.columns.str.upper())

    # set_intersection = set_davis_kin.intersection(set_pkis_kin)
    # set_pkis_only = {i for i in set_pkis_kin if i not in set_davis_kin}
    # set_davis_only = {i for i in set_davis_kin if i not in set_pkis_kin}

    # plot venn diagram of set_intersection, set_pkis_only, set_davis_only
    # plt.figure()
    # venn2(
    #     [set_pkis_kin, set_davis_kin],
    #     set_labels=("PKIS2", "Davis"),
    # )
    # plt.title("Overlapping kinase constructs between PKIS2 and Davis")
    # plt.savefig("Kinase_Overlap.png")

if __name__ == "__main__":
    main()

