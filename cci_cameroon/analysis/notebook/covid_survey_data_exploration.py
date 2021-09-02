# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cci_cameroon
import logging

# %matplotlib inline

# %%
# to recognize the used characterset
# #!pip install openpyxl

# %%
# read sample data for cameroon collected during covid19 pandemic
project_dir = cci_cameroon.PROJECT_DIR
logging.info(project_dir)
data_df = pd.read_excel(f"{project_dir}/inputs/data/raw_data_cameroon.xlsx", 1)

# %%
data_df.head()

# %%
# compute missing values per column in the dataframe.
percent_missing = data_df.isnull().sum() * 100 / len(data_df)
missing_value_df = pd.DataFrame(
    {"column_name": data_df.columns, "percent_missing": percent_missing}
)

# %%
# visualizing the composition of missing values in the data
missing_value_df.sort_values("percent_missing", inplace=True)
plt.figure(figsize=(15, 10))
plt.ylabel("missing value %", fontsize=16)
plt.xlabel("Fields", fontsize=16)
plt.title("Percentage of missing values by columns", fontsize=20)
missing_value_df["percent_missing"].plot(kind="bar")
plt.xticks([])

# %%
plt.figure(figsize=(15, 10))
missing_value_df["percent_missing"].plot(kind="hist")
plt.title("Distribution of columns with missing values", fontsize=20)
plt.tight_layout()
plt.show()

# %%
# attributes with few missing values(<15%).
plt.figure(figsize=(15, 10))
plt.tick_params(axis="both", labelsize=16)
missing_value_df["percent_missing"].head(25).plot(kind="barh")
plt.title("columns with few/no  missing values ", fontsize=20)
plt.tight_layout()
# plt.savefig(f"{project_dir}/outputs/figures/nepal_descriptive/missing_values_less15.png")
plt.show()

# %%
missing_value_df["percent_missing"].tail(20)

# %%
# drop columns with all null values
data_df.dropna(axis=1, how="all", inplace=True)
127 - data_df.shape[1]

# %%

# %%
data_df.columns[0:60]

# %%
for col in data_df.columns[1:13]:
    plt.figure(figsize=(15, 10))
    data_df[col].value_counts().plot(kind="bar")
    plt.title(col, fontsize=20)
    plt.show()

# %%
data_df.columns[62:]

# %%
data_df["rural_urban"].value_counts().plot(kind="bar")

# %%
137 / 198

# %%
# visualizing communities motive for not willing to take vaccine
for col in data_df.columns[62:73]:
    plt.figure(figsize=(15, 10))
    data_df[col].value_counts().plot(kind="bar")
    plt.title(col, fontsize=20)
    plt.show()

# %%
plt.figure(figsize=(15, 10))
data_df["communities_willing_why_not/not_real"].value_counts().plot(kind="bar")
plt.title("communities_willing_why_not/not_real", fontsize=20)
plt.show()

# %%
data_df["communities_willing_why_not/not_real"].value_counts().plot(kind="bar")
plt.title(
    "Distribution of those who are not willing to get vaccination because they think virus not real"
)

# %%
data_df["communities_willing_why_not/distrust"].value_counts().plot(kind="bar")

# %%
plt.figure(figsize=(15, 10))
data_df["com_vac_concerns/not_risk"].value_counts().plot(kind="bar")
plt.title("Community concern about vaccine not being risky", fontsize=20)
plt.ylabel("frequency")

# %%
plt.figure(figsize=(15, 10))
data_df["com_vac_concerns/dangerous"].value_counts().plot(kind="bar")
plt.title("Community concern - vaccine dangerous", fontsize=20)
plt.ylabel("frequency")

# %%
plt.figure(figsize=(15, 10))
data_df["vaccine_concerns/cost"].value_counts().plot(kind="bar")
plt.title("Community concern - vaccine cost", fontsize=20)
plt.ylabel("frequency")

# %%
plt.figure(figsize=(15, 10))
data_df["vaccine_concerns/home_remedies"].value_counts().plot(kind="bar")
plt.title("Community vaccine concern - home remedies", fontsize=20)
plt.ylabel("frequency")

# %%
data_df.columns[:60]

# %%
data_df.duplicated().sum()

# %%
# Support needs of volunteers
plt.figure(figsize=(15, 10))
data_df["support/allowance"].value_counts().plot(kind="bar")
plt.title("Volunteers support request - provide allowance", fontsize=20)
plt.ylabel("Frequency")

# %%
plt.figure(figsize=(15, 10))
data_df["support/insurance"].value_counts().plot(kind="bar")
plt.title("Volunteers support request - provide insurance", fontsize=20)
plt.ylabel("Frequency")

# %%

# %%
plt.figure(figsize=(15, 10))
data_df["unsafe_why/staff_ignore_measures"].value_counts().plot(kind="bar")
plt.title("Volunteers concerns unsafe - staff ignore measure", fontsize=20)
plt.ylabel("Frequency")

# %%
plt.figure(figsize=(15, 10))
data_df["unsafe_why/lack_info"].value_counts().plot(kind="bar")
plt.title("Volunteers concerns unsafe - lack of information", fontsize=20)
plt.ylabel("Frequency")

# %%

# %%
plt.figure(figsize=(15, 10))
data_df["unsafe_why/lack_WASH"].value_counts().plot(kind="bar")
plt.title("Volunteers concerns unsafe - lack of WASH", fontsize=20)
plt.ylabel("Frequency")

# %%
plt.figure(figsize=(15, 10))
data_df["unsafe_why/mask_quality"].value_counts().plot(kind="bar")
plt.title("Volunteers concerns unsafe - mask quality", fontsize=20)
plt.ylabel("Frequency")

# %%
plt.figure(figsize=(15, 10))
data_df["unsafe_why/aggression"].value_counts().plot(kind="bar")
plt.title("Volunteers concerns - aggression", fontsize=20)
plt.ylabel("Frequency")

# %%
plt.figure(figsize=(15, 10))
data_df["unsafe_why/aggression"].value_counts().plot(kind="bar")
plt.title("Volunteers concerns - aggression", fontsize=20)
plt.ylabel("Frequency")

# %%

# %%

# %%

# %%

# %%
data_df["support_other"].unique()

# %%
data_df["support_PPE"].unique()

# %%
# columns split up to multiple columns with various options - can be safely dropped
split_columns = ["unsafe_why", "vaccine_concerns", "communities_willing_why_not"]

# %%

# %%
data_df.dtypes

# %%
data_df["support_other"].value_counts().plot(kind="barh")

# %%

# %%
