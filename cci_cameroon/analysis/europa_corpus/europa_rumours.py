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
import cci_cameroon
import re
import matplotlib.pyplot as plt

# %%
base_directory = cci_cameroon.PROJECT_DIR

# %%
print(base_directory)

# %%
df1 = pd.read_csv(f"{base_directory}/inputs/eudata/hollande.txt", sep=";")

# %%
df1.shape

# %%
df1.head()

# %%
df1.num_rumor.unique()

# %%
list(df1.content[:20])


# %%
# use this function to remove the urls in the content field
def remove_urls(text):
    return re.sub("http://\S+|https://\S+|\\xa0", "", text)


# %%
contents = [remove_urls(tx) for tx in df1.content]

# %%
contents[90]

# %%
df1.content[90]

# %%
df1["Unnamed: 6"].unique()

# %% [markdown]
# ## Hollande data set
# * The data set is entirely in French and consists of 370 records.
# * Fields in the dataset include name of recorder, recorder's ID,  the tweet, date created and number of retweets.
# * The content does not have tags to show whether a record is a rumour or not. The assumption here is that each records represents a rumour.
# * In the case that the preceding assumption holds, we wouldn't have sample records for text that is not rumour.
# * Pre-processing would involve removing links from the text.
#

# %%
df_matrix = pd.read_csv(f"{base_directory}/inputs/eudata/matrix.csv")

# %%
df_matrix.head()

# %%
df_matrix.isRumor.value_counts()


# %%
df_matrix.isRumor.value_counts().plot(kind="bar")

# %% [markdown]
#

# %%
df_lemon = pd.read_csv(f"{base_directory}/inputs/eudata/lemon.txt", sep=";")

# %%
df_lemon.head()

# %%
list(df_lemon.content[:30])

# %%
467 - 76

# %%
1312 - 11103

# %%
df_lemon.num_rumor.unique()

# %% [markdown]
# # Rumors_disinformation data set

# %%
df2 = pd.read_csv(f"{base_directory}/inputs/eudata/rumors_disinformation2.txt", sep=";")

# %%
df2.head()

# %%
cc = [
    remove_urls(tt) for tt in df2.content[:155]
]  # only 156 records in french and the others are in English.

# %%
df2_fr = df2.iloc[:156, :]

# %%
df2_fr[df2_fr.type_rumor == "DESINFORMATION"].shape

# %%
list(df2_fr.content[0:20])

# %% [markdown]
# * Data set contains both French and English records
# * There are 1612 records in total
# * The first 156 records are in french while the rest are in English
# * Fields present in the dataset are as follows:
#     * type_rumor: this takes takes one of two values  - Rumeur if the text is a rumor, or DESINFORMATION if it is disinformation.
#     * title - the title of the post
#     * date - date post was created
#     * Content which gives  details about the subject
# * Data cleaning would involve removing links and special characters in the text
# * 34 of the 156 records are rumors while 122 records represent disinformation
#

# %%
df2.type_rumor.unique()

# %% [markdown]
# # Rihana concert

# %%
df3 = pd.read_csv(f"{base_directory}/inputs/eudata/RihannaConcert2016Fr.txt")

# %%
df3.x[1]

# %%
df3.columns

# %%
list(df3.iloc[18:40, 1])

# %%

# %%

# %% [markdown]
# # UEFA_Euro 2016

# %%
df4 = pd.read_csv(f"{base_directory}/inputs/eudata/UEFA_Euro_2016_Fr.txt")

# %%
df4.shape

# %%
df4.columns

# %%
df4.head()

# %%
list(df4.x[640:900])

# %% [markdown]
# # Random tweets 1

# %%
df5 = pd.read_csv(f"{base_directory}/inputs/eudata/randomtweets1.txt")

# %%
df5.iloc[0, 1]

# %%
df5.columns

# %%
list(df5.x[:100])

# %% [markdown]
# # Random tweets 2

# %%
df6 = pd.read_csv(f"{base_directory}/inputs/eudata/randomtweets2.txt")

# %%
df6.iloc[73, 1]

# %%
df6.shape

# %%

# %%
list(df6.x[:50])

# %%
dfs = pd.read_csv(f"{base_directory}/inputs/eudata/sample-matrix.csv")

# %%
dfs.head()

# %%
dfs.shape

# %%
dfs.isRumor.unique()

# %%
