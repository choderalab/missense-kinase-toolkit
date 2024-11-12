import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, EsmForSequenceClassification

path_model = "/data1/tanseyw/projects/whitej/esm_km_atp/5CV-KLIFS_MIN-esm2_t6_8M_UR50D/full/results/checkpoint-12500"

device = "cpu"

model = EsmForSequenceClassification.from_pretrained(path_model).to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

df_klifs_zehir_muts_alphamissense = pd.read_csv(
    "/data1/tanseyw/projects/whitej/esm_km_atp/assets/klifs_zehir_muts_alphamissense.csv"
)

list_outputs = []
for _, row in df_klifs_zehir_muts_alphamissense.iterrows():
    inputs = tokenizer.encode(row["klifs"], return_tensors="pt").to(device)
    outputs = model.forward(inputs).logits.detach().numpy()[0][0]
    list_outputs.append(outputs)

dict_outputs = dict(zip(df_klifs_zehir_muts_alphamissense["hgnc_name"], list_outputs))

dict_muts = {i: None for i in dict_outputs.keys() if "_" in i}
for key, value in dict_outputs.items():
    if "_" in key:
        wt = key.split("_")[0]
        dict_muts[key] = (dict_outputs[key] - dict_outputs[wt]) / dict_outputs[wt]

df_klifs_zehir_muts_alphamissense["zscore_percent_change"] = (
    df_klifs_zehir_muts_alphamissense["hgnc_name"].apply(
        lambda x: dict_muts[x] * 100 if x in dict_muts.keys() else None
    )
)
df_klifs_zehir_muts_alphamissense["zscore_percent_change_log"] = (
    df_klifs_zehir_muts_alphamissense["zscore_percent_change"].apply(
        lambda x: np.sign(x) * np.log10(np.abs(x))
    )
)


sns.set(font_scale=2)
sns.set_style(style="white")
plt.figure(figsize=(20, 7))
# ax = sns.scatterplot(data = df_klifs_zehir_muts_alphamissense, x = "alphamissense_score", y = "zscore_percent_change", hue = "alphamissense_class")
ax = sns.scatterplot(
    data=df_klifs_zehir_muts_alphamissense,
    x="alphamissense_score",
    y="zscore_percent_change_log",
    hue="alphamissense_class",
)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.yscale('log')
plt.legend(title="Alphamissense Class")
plt.xlabel("Alphamissense Score")
# plt.ylabel(" Predicted Z-score\n% Change vs. Wild-Type")
plt.ylabel(r"$log_{10}$" + " Predicted Z-score\n% Change vs. Wild-Type")
plt.savefig(
    "/data1/tanseyw/projects/whitej/esm_km_atp/images/zscore_percent_change_vs_alphamissense_score_log.png",
    bbox_inches="tight",
)
# plt.savefig("/data1/tanseyw/projects/whitej/esm_km_atp/images/zscore_percent_change_vs_alphamissense_score.png", bbox_inches = "tight")
