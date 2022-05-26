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

# %%
project_dir = cci_cameroon.PROJECT_DIR

# %%
data_files_root = str(project_dir) + "/inputs/data/workshop_files"

# %%
# read in the persons files to use the ID field from them
pf1 = pd.read_excel(f"{data_files_root}/assigned_people_files/person1.xlsx")
pf2 = pd.read_excel(f"{data_files_root}/assigned_people_files/person2.xlsx")
pf3 = pd.read_excel(f"{data_files_root}/assigned_people_files/person3.xlsx")
pf4 = pd.read_excel(f"{data_files_root}/assigned_people_files/person4.xlsx")
pf5 = pd.read_excel(f"{data_files_root}/assigned_people_files/person5.xlsx")
pf6 = pd.read_excel(f"{data_files_root}/assigned_people_files/person6.xlsx")
pf7 = pd.read_excel(f"{data_files_root}/assigned_people_files/person7.xlsx")
pf8 = pd.read_excel(f"{data_files_root}/assigned_people_files/person8.xlsx")
pf10 = pd.read_excel(f"{data_files_root}/assigned_people_files/person10.xlsx")
pf9 = pd.read_excel(f"{data_files_root}/assigned_people_files/person9.xlsx")


# %%
# The different response files are loaded
df1 = pd.read_excel(f"{data_files_root}/person_1_labels_Jean Jacques.xlsx")
df2 = pd.read_excel(f"{data_files_root}/person_2_labels FONKOU.xlsx")
df3 = pd.read_excel(f"{data_files_root}/person_3_labels_FoKam Pauline.xlsx")
df4 = pd.read_excel(f"{data_files_root}/person_4_labels_sarah and jacques.xlsx")
df5 = pd.read_excel(f"{data_files_root}/person_5_labels_Harouna Laila.xlsx")
df6 = pd.read_excel(f"{data_files_root}/person_6_labels arthur melingui.xlsx")
df7 = pd.read_excel(f"{data_files_root}/person_7_labels_Ronald.xlsx")
df8 = pd.read_excel(f"{data_files_root}/person_8_labels 3_SELAVY.xlsx")
df9 = pd.read_excel(f"{data_files_root}/person_9_labels_laurel.xlsx")
df10 = pd.read_excel(f"{data_files_root}/person_10_labels_Jane.xlsx")


# %%
# assign the ids to the response dataframes
df1["id"] = pf1["ID"]
df2["id"] = pf2["ID"]
df3["id"] = pf3["ID"]
df4["id"] = pf4["ID"]
df5["id"] = pf5["ID"]
df6["id"] = pf6["ID"]
df7["id"] = pf7["ID"]
df8["id"] = pf8["ID"]
df9["id"] = pf9["ID"]
df10["id"] = pf10["ID"]


# %%
df1.head()

# %%
# drop the completely empty records
df1.dropna(axis=0, how="all", inplace=True)
df2.dropna(axis=0, how="all", inplace=True)
df3.dropna(axis=0, how="all", inplace=True)
df4.dropna(axis=0, how="all", inplace=True)
df5.dropna(axis=0, how="all", inplace=True)
df6.dropna(axis=0, how="all", inplace=True)
df7.dropna(axis=0, how="all", inplace=True)
df8.dropna(axis=0, how="all", inplace=True)
df9.dropna(axis=0, how="all", inplace=True)
df10.dropna(axis=0, how="all", inplace=True)

# %%
# combine the dataframes
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

# %%
df.head()

# %%
df.shape

# %%
df.isnull().sum()

# %%
# rename the columns to shorten
df.columns = [
    "comment",
    "code",
    "is_code_suitable",
    "select_other_codes",
    "new_code",
    "certainty",
    "id",
]

# %%
# uncomment to fill the null values, otherwise leave this out to consider only the first 1200 records as
# most respondents selected only one code from the list.
df[["comment", "code", "is_code_suitable", "certainty", "id"]] = df[
    ["comment", "code", "is_code_suitable", "certainty", "id"]
].ffill()

# %%
# these values should change if you fill nulls above
df.isnull().sum()

# %%
# retain only the first response for each code already assigned to a comment.
df_comments = df[df.comment.notna()].copy()

# %%
df_comments.shape

# %%
df_comments.sort_values("id", inplace=True)

# %%
df_comments.certainty.value_counts()

# %%
print(df.id.value_counts())

# %%
df_comments.certainty.unique()

# %%
df_comments.is_code_suitable.unique()

# %%
certainty = {"Très sûr": 4, "Un peu sûr": 3, "Très incertain": 2, "Pas très sûr": 1}
suitability = {
    "Oui": 1,
    "Non": 0,
    "OUI ": 1,
    "NON ": 0,
    "oui ": 1,
    "non ": 0,
    "oui": 1,
    "non": 0,
    "oui  ": 1,
}

# %%
df_comments.certainty = df_comments.certainty.map(certainty)
df_comments.is_code_suitable = df_comments.is_code_suitable.map(suitability)

# %%
df_comments.certainty

# %%
dumy1 = pd.get_dummies(df_comments["code"])
dumy2 = pd.get_dummies(df_comments.select_other_codes)

# %%
dumy2.drop("Aucun de ces codes n'est approprié", axis=1, inplace=True)

# %%
# multiply the codes by the confindence level
for col in dumy1.columns:
    dumy1[col] = dumy1[col] * df_comments.certainty

# %%
for col in dumy2.columns:
    dumy2[col] = dumy2[col] * df_comments.certainty

# %%
df_combined_dumy = pd.concat([df_comments, dumy1, dumy2], axis=1).copy()

# %%
df_combined_dumy.shape

# %%
results_df = df_combined_dumy.groupby("id").sum().iloc[:, 2:].copy()

# %%
# to hold final processed data
main_df = df_combined_dumy.groupby("id").first().iloc[:, :2]
main_df.shape

# %%
# read out the code with the highest vote
main_df["first_code"] = results_df.idxmax(axis=1)

# %%
# read out code with second maximum vote
# main_df['second_code'] =results_df.columns[results_df.values.argsort(1)[:, -2]]

# %%
main_df.head(10)

# %%
main_df.to_excel(f"{project_dir}/outputs/irfc_staff_labelled_data.xlsx")

# %%

# %%
np.unique((main_df.code == main_df.first_code))

# %%

# %%
