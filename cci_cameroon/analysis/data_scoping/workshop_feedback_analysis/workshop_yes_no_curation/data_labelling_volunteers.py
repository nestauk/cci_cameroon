# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import cci_cameroon

project_dir = cci_cameroon.PROJECT_DIR

# %%
data_files_root = str(project_dir) + "/inputs/data/workshop_yes_no"

# %%
full_data = pd.read_excel(f"{data_files_root}/workshop2_comments_french.xlsx")

# %%
full_data.head()

# %%
sample1 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_1.xlsx"
)
sample2 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_2.xlsx"
)
sample3 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_3.xlsx"
)
sample4 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_4.xlsx"
)
sample5 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_5.xlsx"
)
sample6 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_6.xlsx"
)
sample7 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_7.xlsx"
)
sample8 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_8.xlsx"
)
sample9 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_9.xlsx"
)
sample11 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_11.xlsx"
)
sample10 = pd.read_excel(
    f"{data_files_root}/Catégorisation_rumeurs_croyances_et_observations_10.xlsx"
)

# %%
sample1.head()

# %%
print(sample1.columns[:30])
print(sample2.columns[:30])

# %%
full_data.id.unique

# %%
full_data.columns

# %%
columns1 = sample1.columns.str.contains(r"^_\d+")
sample1_df = sample1.iloc[:, columns1].T.reset_index()
columns2 = sample2.columns.str.contains(r"^_\d+")
sample2_df = sample2.iloc[:, columns2].T.reset_index()
columns3 = sample3.columns.str.contains(r"^_\d+")
sample3_df = sample3.iloc[:, columns3].T.reset_index()
columns4 = sample4.columns.str.contains(r"^_\d+")
sample4_df = sample4.iloc[:, columns4].T.reset_index()
columns5 = sample5.columns.str.contains(r"^_\d+")
sample5_df = sample5.iloc[:, columns5].T.reset_index()
columns6 = sample6.columns.str.contains(r"^_\d+")
sample6_df = sample6.iloc[:, columns6].T.reset_index()
columns7 = sample7.columns.str.contains(r"^_\d+")
sample7_df = sample7.iloc[:, columns7].T.reset_index()
columns8 = sample8.columns.str.contains(r"^_\d+")
sample8_df = sample8.iloc[:, columns8].T.reset_index()
columns9 = sample9.columns.str.contains(r"^_\d+")
sample9_df = sample9.iloc[:, columns9].T.reset_index()
columns11 = sample11.columns.str.contains(r"^_\d+")
sample11_df = sample11.iloc[:, columns11].T.reset_index()
columns10 = sample10.columns.str.contains(r"^_\d+")
sample10_df = sample10.iloc[:, columns10].T.reset_index()

# %%
responses = {
    "no": 0,
    "non": 0,
    "No": 0,
    "NO": 0,
    "yes": 1,
    "Yes": 1,
    "Oui": 1,
    "YES": 1,
}

# %%
sample6_df.columns

# %%
sample1_df["id"] = sample1_df["index"].apply(lambda x: x.split("_")[1])
sample2_df["id"] = sample2_df["index"].apply(lambda x: x.split("_")[1])
sample3_df["id"] = sample3_df["index"].apply(lambda x: x.split("_")[1])
sample4_df["id"] = sample4_df["index"].apply(lambda x: x.split("_")[1])
sample5_df["id"] = sample5_df["index"].apply(lambda x: x.split("_")[1])
sample6_df["id"] = sample6_df["index"].apply(lambda x: x.split("_")[1])
sample7_df["id"] = sample7_df["index"].apply(lambda x: x.split("_")[1])
sample8_df["id"] = sample8_df["index"].apply(lambda x: x.split("_")[1])
sample9_df["id"] = sample9_df["index"].apply(lambda x: x.split("_")[1])
sample10_df["id"] = sample10_df["index"].apply(lambda x: x.split("_")[1])
sample11_df["id"] = sample11_df["index"].apply(lambda x: x.split("_")[1])

# %%
sample6.head()

# %%
sample1_df["answer"] = sample1_df[0].map(responses)
sample2_df["answer"] = sample2_df[0].map(responses)
sample3_df["answer"] = sample3_df[0].map(responses)
sample4_df["answer"] = sample4_df[0].map(responses)
sample5_df["answer"] = sample5_df[0].map(responses)
sample6_df["answer"] = sample6_df[0].map(responses)
sample7_df["answer"] = sample7_df[0].map(responses)
sample8_df["answer"] = sample8_df[0].map(responses)
sample9_df["answer"] = sample9_df[0].map(responses)
sample10_df["answer"] = sample10_df[0].map(responses)
sample11_df["answer"] = sample11_df[0].map(responses)

# %%
sample1_df.drop(["index", 0], axis=1, inplace=True)
sample2_df.drop(["index", 0], axis=1, inplace=True)
sample3_df.drop(["index", 0, 1], axis=1, inplace=True)
sample4_df.drop(["index", 0], axis=1, inplace=True)
sample5_df.drop(["index", 0], axis=1, inplace=True)
sample6_df.drop(
    ["index", 0, 1, 2, 3, 4], axis=1, inplace=True
)  # had multiple responses but we need just one
sample7_df.drop(["index", 0], axis=1, inplace=True)
sample8_df.drop(["index", 0], axis=1, inplace=True)
sample9_df.drop(["index", 0], axis=1, inplace=True)
sample10_df.drop(["index", 0], axis=1, inplace=True)
sample11_df.drop(["index", 0], axis=1, inplace=True)

# %%
sample1_df = sample1_df.astype({"id": "float64"})
sample2_df = sample2_df.astype({"id": "float64"})
sample3_df = sample3_df.astype({"id": "float64"})
sample4_df = sample4_df.astype({"id": "float64"})
sample5_df = sample5_df.astype({"id": "float64"})
sample6_df = sample6_df.astype({"id": "float64"})
sample7_df = sample7_df.astype({"id": "float64"})
sample8_df = sample8_df.astype({"id": "float64"})
sample9_df = sample9_df.astype({"id": "float64"})
sample10_df = sample10_df.astype({"id": "float64"})
sample11_df = sample11_df.astype({"id": "float64"})

# %%
result1 = full_data.merge(sample1_df, on="id", how="inner")
result2 = full_data.merge(sample2_df, on="id", how="inner")
result3 = full_data.merge(sample3_df, on="id", how="inner")
result4 = full_data.merge(sample4_df, on="id", how="inner")
result5 = full_data.merge(sample5_df, on="id", how="inner")
result6 = full_data.merge(sample6_df, on="id", how="inner")
result7 = full_data.merge(sample7_df, on="id", how="inner")
result8 = full_data.merge(sample8_df, on="id", how="inner")
result9 = full_data.merge(sample9_df, on="id", how="inner")
result10 = full_data.merge(sample10_df, on="id", how="inner")
result11 = full_data.merge(sample11_df, on="id", how="inner")

# %%
sample10_df.dtypes

# %%
result10.head()

# %%
combined_df = pd.concat(
    [
        result1,
        result2,
        result3,
        result4,
        result5,
        result6,
        result7,
        result8,
        result9,
        result10,
        result11,
    ],
    axis=0,
    ignore_index=True,
)

# %%
combined_df.head()

# %%
combined_df.shape

# %%
combined_df.groupby("id").first().shape

# %%
tally_df = combined_df.groupby("id").sum().reset_index()

# %%
tally_df.answer.unique()

# %%
final_df = full_data.merge(tally_df, on="id", how="inner")

# %%
final_df.shape

# %%
final_df.head()

# %%
# combined_df["retain"] = ["Y" if x in ids else "N" for x in combined_df.id]

# %%
final_df["retain"] = np.where(final_df["answer"] > 1, "Y", "N")

# %%
final_df.shape

# %%
final_df.retain

# %%
final_df[final_df.retain == "Y"].shape[0]

# %%
final_df[final_df.retain == "N"].shape[0]

# %%
final_df.head()

# %%
final_df[final_df.retain == "Y"].code.unique()

# %%
