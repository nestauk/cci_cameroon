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
import cci_cameroon
from langdetect import detect, detect_langs
from googletrans import Translator
from random import randint

# %matplotlib inline
import faiss
from faiss import normalize_L2
from sentence_transformers import SentenceTransformer
import time
import logging
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import cdlib
from cdlib import algorithms

# %% [markdown]
# # Approach used
#
# * Load and clean the dataset
# * Convert the unlabelled comments into vector of numerical values using a transformer model
# * Using FAISS library, compute pairwise similarity scores for the comments
# * Based on the similarity scores, a connectivity graph
# * Using a community detection algorithm, we generate subgroups/clusters from the graph

# %%
# #!pip install stop_words
# #!pip install googletrans==4.0.0-rc1
# #!pip install networkx
# #!pip install cdlib


# %%
project_directory = cci_cameroon.PROJECT_DIR

# %%
data_df = pd.read_excel(
    f"{project_directory}/inputs/data/irfc_staff_labelled_data.xlsx"
)

# %%
data_df.first_code.unique()

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
data.shape

# %%
data["category_id"] = data.first_code.factorize()[0]

# %%
# checking the distribution of the codes in the data
data.groupby("first_code").comment.count().plot(kind="bar")

# %%
data = data.drop_duplicates(["comment"])
data.reset_index(inplace=True)
data.shape

# %%
# Converting the codes into integers
data["cluster"] = data.code.factorize()[0]

# %%
# using the french_semantic model for word embedding
model = SentenceTransformer("Sahajtomar/french_semantic")
sentence_embeddings = model.encode(data.comment)

# %%
data[
    data.category_id == 2
].shape  # 127 cluster 1(0),137 cluster 2 (1), 136 cluster 3 (2)

# %% [markdown]
# ## We start with a smaller sample of the data (~30 records) to ease inspection of results

# %%
# creating a small subset of the data made up of the different categories
set1 = data[data.category_id == 0]["comment"][:10]
set2 = data[data.category_id == 1]["comment"][:10]
set3 = data[data.category_id == 2]["comment"][:10]
to_use2 = (
    pd.concat([set1, set2, set3]).drop_duplicates().reset_index().drop("index", axis=1)
)
# to_use2 = to_use.reset_index().drop("index",axis=1)

# %%
# create word embeddings using the transformer model
sentence_embeddings2 = model.encode(to_use2.comment)

# %%
comments_df = data[["id", "comment", "cluster"]].copy()


# %%
# generates colors to use for graph
def generate_colors(com):
    colors = []
    for i in range(len(com.communities)):
        colors.append("#%06X" % randint(0, 0xFFFFFF))
    return colors


# method to draw the identified communities
def draw_communities_graph(graph, colors, com):
    color_map = []
    for node in graph:
        for i in range(len(com.communities)):
            if node in com.communities[i]:
                color_map.append(colors[i])
    nx.draw(graph, node_color=color_map, with_labels=True)
    plt.show()


# %%
indp2 = faiss.index_factory(
    sentence_embeddings2.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(sentence_embeddings2)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 3
indp2.train(sentence_embeddings2)
indp2.add(sentence_embeddings2)
distance_matric2, positions2 = indp2.search(
    sentence_embeddings2, sentence_embeddings2.shape[0]
)

# %%
B = kneighbors_graph(
    distance_matric2, neighbors2, mode="connectivity", include_self=False
)
B.toarray()[0]

# %%
# load the graph
G2 = nx.from_numpy_matrix(B, create_using=nx.Graph())
print(G2)
# visualize the graph
nx.draw(G2, with_labels=True)

# %%
# applying a community algorithm to the graph to identify subgroups
coms2 = algorithms.leiden(G2)
draw_communities_graph(G2, generate_colors(coms2), coms2)

# %%
list(to_use2.comment[coms2.communities[0]])  # Observation on hand washing

# %%
list(to_use2.comment[coms2.communities[2]])  # belief that the disease exists

# %%
list(to_use2.comment[coms2.communities[1]])  # belief on wearing of masks

# %%
list(to_use2.comment[coms2.communities[0]])  # Believe masks are not effective

# %%

# %% [markdown]
# # Working with the larger dataset

# %%
indp = faiss.index_factory(
    sentence_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(sentence_embeddings)  # to be used for inner product computation
neighbors = 10

# %%
indp.train(sentence_embeddings)
indp.add(sentence_embeddings)
distance_matric, position = indp.search(
    sentence_embeddings, sentence_embeddings.shape[0]
)  # create similarity matrix for the data

# %%
# create a KNN network with n neighbors using the cosine similarity matrix
A = kneighbors_graph(distance_matric, 5, mode="connectivity", include_self=False)
A.toarray()[0]

# %%
# load the graph
G = nx.from_numpy_matrix(A, create_using=nx.Graph())
print(G)
# visualize the graph
nx.draw(G, with_labels=False)

# %%
len(G.nodes), len(G.edges)

# %%
# using leiden community detection algorithm on the dataset
coms = algorithms.leiden(G)
draw_communities_graph(G, generate_colors(coms), coms)

# %%
len(coms.communities)

# %%
for index in range(len(coms.communities)):
    print(data.comment[coms.communities[index]].index)
    print("######################NEW Group####")

# %%
list(
    data.iloc[[158, 161, 162, 163, 164, 165, 166, 167, 168, 171]].comment
)  # belief about mask wearing by children

# %%
list(
    data.iloc[
        [
            89,
            92,
            94,
            104,
            108,
            110,
            115,
            116,
            117,
            119,
            120,
            126,
            136,
            139,
            140,
            143,
            153,
            155,
            156,
            157,
            159,
            169,
            173,
            180,
            183,
            184,
            185,
            186,
            187,
            189,
            192,
            193,
            197,
            198,
            199,
            208,
            215,
            216,
            218,
            270,
            291,
        ]
    ].comment
)

# %%
list(
    data.iloc[
        [
            0,
            1,
            3,
            4,
            7,
            13,
            18,
            22,
            23,
            24,
            28,
            31,
            33,
            36,
            38,
            64,
            67,
            75,
            79,
            85,
            145,
            328,
            330,
            331,
            341,
            343,
        ]
    ].comment
)

# %%
list(
    data.iloc[
        [
            2,
            27,
            30,
            32,
            35,
            37,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            65,
            66,
            80,
            83,
            84,
            338,
        ]
    ].comment
)

# %%
## Using walktrap community detection algorithm on the data subset

# %%
comsw = algorithms.walktrap(G)
comsw2 = algorithms.walktrap(G2)

# %%
draw_communities_graph(G2, generate_colors(comsw2), comsw2)

# %%
list(to_use2.comment[comsw2.communities[0]])

# %%
list(to_use2.comment[comsw2.communities[1]])

# %%
list(to_use2.comment[comsw2.communities[2]])

# %%
list(to_use2.comment[comsw2.communities[3]])

# %%

# %% [markdown]
# ## General observations
#
# * The higher the number of neigbors in the graph, the bigger the clusters formed. Need to agree on an average number of comments that should come through before algorithm is run in order to choose a suitable value for the number of neighbors.
#
# * Some keywords seem to influence the performance of the model. For example grouping of sentences with the word "existe" and its negation "n'existe pas" suggests the sentences are brought together by the word without much impact of context. Need to investigate more
#

# %% [markdown]
# https://towardsdatascience.com/community-detection-algorithms-9bd8951e7dae

# %%
