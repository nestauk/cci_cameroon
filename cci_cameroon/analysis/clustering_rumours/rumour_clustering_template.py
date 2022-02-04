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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import cci_cameroon
from stop_words import get_stop_words
from langdetect import detect, detect_langs
from googletrans import Translator
from sklearn.metrics import adjusted_rand_score

# %matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering

# %%
# #!pip install stop_words
# #!pip install googletrans==4.0.0-rc1


# %%
project_directory = cci_cameroon.PROJECT_DIR

# %%
data_df = pd.read_excel(
    f"{project_directory}/inputs/data/irfc_staff_labelled_data.xlsx"
)

# %%
data_df.head()

# %%

data_df.comment = [x.lower() for x in data_df.comment]

# %%
data_df["language"] = data_df.comment.apply(lambda x: detect(x))

# %%
pd.DataFrame(data_df.groupby("language").comment.count().sort_values(ascending=True))

# %%
data_df.loc[data_df.language == "en"]

# %%
translator = Translator()

# %%
english_df = data_df[data_df.language == "en"].copy()
spanish_df = data_df[data_df.language == "es"].copy()
data_df = data_df[~data_df.language.isin(["en", "es"])].copy()

# %%
english_df["comment"] = english_df.comment.apply(
    translator.translate, src="en", dest="fr"
).apply(getattr, args=("text",))
spanish_df["comment"] = spanish_df.comment.apply(
    translator.translate, src="es", dest="fr"
).apply(getattr, args=("text",))

# %%
data = pd.concat([data_df, english_df, spanish_df], ignore_index=True)

# %%
data.drop("language", axis=1, inplace=True)

# %%
data.head()

# %%
data.shape

# %%
data["category_id"] = data.first_code.factorize()[0]

# %%
data.groupby("first_code").comment.count().plot(kind="bar")

# %%
data["cluster"] = data.code.factorize()[0]

# %%
comments = data.comment

# %%
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# %%
stop_words = get_stop_words("fr")

# %%
tfidf_vectorizer = TfidfVectorizer(
    min_df=5, encoding="latin-1", ngram_range=(1, 2), stop_words=stop_words
)
comments = tfidf_vectorizer.fit_transform(data.comment).toarray()

# %%
comments

# %%
thresh = 1.2
clusters = hcluster.fclusterdata(comments, thresh, criterion="distance")


# %%
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = hcluster.dendrogram(hcluster.linkage(comments, method="ward"))
plt.axhline(y=6, color="r", linestyle="--")

# %%
# now apply agglomerative clustering to the comments using the determined number of clusters.
cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
cluster.fit_predict(comments)

# %%
data.code.factorize()[0]

# %% [markdown]
# # Concern
# How does one determine the number of clusters in hierarchical clustering and proceed to clustering without human intervention?

# %%
# trying meanshift algorithm
ms = MeanShift()
ms.fit(comments)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

# %%
labels

# %%
# DBSCAN clustering of the comments
distance = [0.2, 0.5, 1, 1.2, 1.5, 1.7, 2, 3, 5]
label_number = []
for epsi in distance:
    db = DBSCAN(eps=epsi, min_samples=50).fit(comments)
    label_number.append(len(np.unique(db.labels_)))
print(label_number)

# %%
labels

# %%

# %%
