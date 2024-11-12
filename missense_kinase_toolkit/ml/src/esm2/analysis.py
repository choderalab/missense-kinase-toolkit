#!/usr/bin/env python

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# os.chdir("/data1/tanseyw/projects/whitej/esm_km_atp/src")
from utils import (
    save_csv2csv, 
    load_csv2dataset, 
    parse_stats_dataframes,
    invert_zscore,
)

path = "/data1/tanseyw/projects/whitej/esm_km_atp/"
list_files = glob.glob(os.path.join(path, "*/fold-*.csv"))
df = pd.read_csv(os.path.join(path, "assets/pkis2_km_atp.csv"))
# log10 transform before z-scoring
labels = df['ATP Conc.(uM)'].apply(np.log10)

list_runs = [
    "5CV-KinCore-esm2_t6_8M_UR50D",
    "5CV-KLIFS_MIN-esm2_t6_8M_UR50D",
    "5CV-KLIFS_FULL-esm2_t6_8M_UR50D",
]

list_logs = [glob.glob(os.path.join(path, run, "*/logs/fold-*.csv")) for run in list_runs]

dict_logs = dict(zip(list_runs, list_logs))

### Load and process training and evaluation loss ###

df_train, df_eval, df_final = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()   
for exp, list_file in dict_logs.items():
    for idx, file in enumerate(list_file):
        df_train_temp, df_eval_temp, df_final_temp = parse_stats_dataframes(file, idx)
        df_train_temp["exp"] = exp
        df_eval_temp["exp"] = exp
        df_final_temp["exp"] = exp
        df_train = pd.concat([df_train, df_train_temp]).reset_index(drop=True)
        df_eval = pd.concat([df_eval, df_eval_temp]).reset_index(drop=True)
        df_final = pd.concat([df_final, df_final_temp]).reset_index(drop=True)

### Load validation and training datasets ###

#TODO: Look at all files
ds_val, ds_train = load_csv2dataset(path, 5, "KLIFS_FULL_data.csv")
df_val = pd.DataFrame()
for idx, ds in enumerate(ds_val):
    df_val_temp = ds.to_pandas()
    df_val_temp["fold"] = idx + 1
    df_val = pd.concat([df_val, df_val_temp]).reset_index(drop=True)

### Plot training loss ###

list_fold = df_train["fold"].unique().tolist()
list_replace = [f"Fold: {i}\n(n = {sum(df_val["fold"] != i)})" for i in list_fold]
df_train["fold_label"] = df_train["fold"].map(dict(zip(list_fold, list_replace)))
g = sns.FacetGrid(df_train, col="fold_label", row="exp", hue="exp", sharey=True, sharex=True)
g.map(sns.lineplot, "step", "loss")
g.add_legend()
g.set_axis_labels("Steps", "Training Loss")
g.set_titles('{col_name}')
plt.savefig(os.path.join(path, "images/train_loss_2024.10.30.png"))

# list_fold = df_train["fold"].unique().tolist()
# list_replace = [f"Fold: {i}\n(n = {sum(df_val["fold"] != i)})" for i in list_fold]
# df_train["fold_label"] = df_train["fold"].map(dict(zip(list_fold, list_replace)))
# g = sns.FacetGrid(df_train, col="fold_label", hue="fold", sharey=False, sharex=False)
# g.map(sns.lineplot, "step", "loss")
# g.set_axis_labels("Steps", "Training Loss")
# g.set_titles('{col_name}')
# plt.savefig(os.path.join(path, "images/train_loss.png"))

### Plot evaluation RMSE ###

list_fold = df_eval["fold"].unique().tolist()
list_replace = [f"Fold: {i}\n(n = {sum(df_val["fold"] == i)})" for i in list_fold]
df_eval["fold_label"] = df_eval["fold"].map(dict(zip(list_fold, list_replace)))
df_eval["log_rmse"] = invert_zscore(df_eval["eval_rmse"], labels)
df_eval["orig_rmse"] = df_eval["log_rmse"].apply(lambda x: 10 ** x)

# list_fold = df_eval["fold"].unique().tolist()
# list_replace = [f"Fold: {i}\n(n = {sum(df_val["fold"] == i)})" for i in list_fold]
# df_eval["fold_label"] = df_eval["fold"].map(dict(zip(list_fold, list_replace)))
# df_eval["orig_rmse"] = invert_zscore(df_eval["eval_rmse"], labels)
# df_eval["orig_rmse"] = df_eval["orig_rmse"].apply(lambda x: 10 ** x)

# Leave RMSE in units of z-score log10(Km, ATP)

sns.set(font_scale=1.5)
df_eval["exp_label"] = df_eval["exp"].map(dict(zip(list_runs, ["KinCore KD", "KLIFS Pocket", "KLIFS Full Region"])))
g = sns.FacetGrid(df_eval, col="fold_label", hue="exp_label", sharey=True, sharex=True)
# for ax in g.axes.flat:
#     ax.axvline(500, color='r', linestyle='dashed', linewidth=1)
g.grid(False)
g.map(sns.lineplot, "step", "log_rmse")
g.set_axis_labels("Steps", "Held-Out RMSE\n" + r"$(log_{10} K_{M, ATP})$")
g.set_titles('{col_name}')
g.add_legend(title="Input sequence")
# g.figsize(8, 6)
plt.savefig(os.path.join(path, "images/eval_rmse_unconverted_2024.10.30.png"))

# g = sns.FacetGrid(df_eval, col="fold_label", hue="fold", sharey=False, sharex=False)
# for ax in g.axes.flat:
#     ax.axvline(500, color='r', linestyle='dashed', linewidth=1)
# g.map(sns.lineplot, "step", "eval_rmse")
# g.set_axis_labels("Steps", "RMSE, Eval (Unconverted)")
# g.set_titles('{col_name}')
# plt.savefig(os.path.join(path, "images/eval_rmse_unconverted.png"))

# Convert RMSE to original scale

g = sns.FacetGrid(df_eval, col="fold_label", hue="fold", sharey=False, sharex=False)
for ax in g.axes.flat:
    ax.axvline(500, color='r', linestyle='dashed', linewidth=1)
g.map(sns.lineplot, "step", "orig_rmse")
g.set_axis_labels("Steps", "RMSE, Eval (Converted)")
g.set_titles('{col_name}')
plt.savefig(os.path.join(path, "images/eval_rmse_converted.png"))

### Plot histogram of labels for validation set ###

list_fold = df_val["fold"].unique().tolist()
list_replace = [f"Fold: {i}\n(n = {sum(df_val["fold"] == i)})" for i in list_fold]
df_val["fold_label"] = df_val["fold"].map(dict(zip(list_fold, list_replace)))
df_val["orig_label"] = invert_zscore(df_val["label"], labels)
df_val["orig_label"] = df_val["orig_label"].apply(lambda x: 10 ** x)

# Leave labels in units of z-score log10(Km, ATP)

g = sns.FacetGrid(df_val, col="fold_label", hue="fold")
g.map(plt.hist, "label")
g.set_axis_labels("z-score, $log_{10}$Km, ATP", "Frequency")
y, x, _ = plt.hist(df_val["label"])
for idx, ax in enumerate(g.axes.flat):
    loc = df_val.loc[df_val["fold"] == idx + 1, "label"].mean()
    ax.axvline(loc, color='r', linestyle='dashed', linewidth=1)
    ax.text(loc + (x.max() - x.min()) * 0.1, y.max() * 0.9, "Mean: " + str(round(loc, 2)), color='r')
g.set_titles('{col_name}')
plt.savefig(os.path.join(path, "images/val_label_hist_zscore.png"), bbox_inches="tight")

# Convert labels to original scale

g = sns.FacetGrid(df_val, col="fold_label", hue="fold")
g.map(plt.hist, "orig_label")
g.set_axis_labels("Km, ATP", "Frequency")
y, x, _ = plt.hist(df_val["orig_label"])
for idx, ax in enumerate(g.axes.flat):
    loc = df_val.loc[df_val["fold"] == idx + 1, "label"].mean()
    ax.axvline(loc, color='r', linestyle='dashed', linewidth=1)
    ax.text(loc + (x.max() - x.min()) * 0.1, y.max() * 0.9, "Mean: " + str(round(loc, 2)), color='r')
g.set_titles('{col_name}')
plt.savefig(os.path.join(path, "images/val_label_hist_orig.png"), bbox_inches="tight")



# import numpy as np
# from utils import calc_zscore
# df = pd.read_csv(os.path.join(path, "assets/pkis2_km_atp.csv"))
# calc_zscore(df["ATP Conc.(uM)"].apply(np.log10))
# df.head()
# df["kd"].apply(len).max()