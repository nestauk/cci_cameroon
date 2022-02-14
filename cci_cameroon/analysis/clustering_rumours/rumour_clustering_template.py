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
from sklearn.metrics import adjusted_rand_score
from random import randint

# %matplotlib inline
import tensorflow as tf
import faiss
from faiss import normalize_L2
from sentence_transformers import SentenceTransformer
import time
import logging
import torch
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import cdlib
from cdlib import algorithms

# %% [markdown]
# # Approach used
#
# * Load and clean the dataset
# * Convert the unlabelled comments into numerical values using a transformer model
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
data.shape

# %%
data["category_id"] = data.first_code.factorize()[0]

# %%
data.groupby("first_code").comment.count().plot(kind="bar")

# %%
data["cluster"] = data.code.factorize()[0]

# %%
# using the french_semantic model for embedding
model = SentenceTransformer("Sahajtomar/french_semantic")
sentence_embeddings = model.encode(data.comment)

# %%
sentence_embeddings[0]

# %%
data[
    data.category_id == 2
].shape  # 127 cluster 1(0),137 cluster 2 (1), 136 cluster 3 (2)

# %% [markdown]
# # sentence transformer approach
# we use fewer comments in this case

# %%
# creating a small subset of the data made up of the different categories
set1 = data[data.category_id == 0]["comment"][:10]
set2 = data[data.category_id == 1]["comment"][:10]
set3 = data[data.category_id == 2]["comment"][:10]

# %%
to_use = pd.concat([set1, set2, set3])
to_use.drop_duplicates(inplace=True)

# %%
to_use2 = to_use.reset_index().drop("index", axis=1)


# %%
sentence_embeddings2 = model.encode(to_use2.comment)

# %%
comments_df = data[["id", "comment", "cluster"]].copy()

# %%
indp = faiss.index_factory(
    sentence_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)

# %%
faiss.normalize_L2(sentence_embeddings)  # to be used for inner product computation

# %%
neighbors = 10

# %%
indp.train(sentence_embeddings)
indp.add(sentence_embeddings)

# %%
distance_matric, position = indp.search(
    sentence_embeddings, sentence_embeddings.shape[0]
)

# %%
distance_matric

# %%
# create a KNN network with 10 neighbors maximum
A = kneighbors_graph(
    distance_matric, neighbors, mode="connectivity", include_self=False
)
A.toarray()[0]

# %%
# load the graph
G = nx.from_numpy_matrix(A)
print(G)
# visualize the graph
nx.draw(G, with_labels=False)

# %%
len(G.nodes), len(G.edges)


# %%
def edge_to_remove(graph):
    G_dict = nx.edge_betweenness_centrality(graph)
    edge = ()
    # extract the edge with highest edge betweenness centrality score
    for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse=True):
        edge = key
        break
    return edge


# %%
# partition graph into multiple communities
def girvan_newman(graph):
    # find number of connected components
    sg = nx.connected_components(graph)
    sg_count = nx.number_connected_components(graph)
    while sg_count == 1:
        graph.remove_edge(edge_to_remove(graph)[0], edge_to_remove(graph)[1])
        sg = nx.connected_components(graph)
        sg_count = nx.number_connected_components(graph)
    return sg


# %%
# find communities in the graph
c = girvan_newman(G.copy())
# find the nodes forming the communities
node_groups = []
for i in c:
    node_groups.append(list(i))

# %%
data.comment[node_groups[0]][:50]

# %%
# plot the communities
color_map = []
for node in G:
    if node in node_groups[0]:
        color_map.append("blue")
    else:
        color_map.append("green")
nx.draw(G, node_color=color_map, with_labels=True)
plt.show()

# %% [markdown]
# ## Using a subset of the data
#

# %%
indp2 = faiss.index_factory(
    sentence_embeddings2.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(sentence_embeddings2)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 5
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
# #!pip install cdlib

# %% [markdown]
# https://towardsdatascience.com/community-detection-algorithms-9bd8951e7dae

# %%
coms = algorithms.leiden(G)
coms2 = algorithms.leiden(G2)


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
# load the graph
G2 = nx.from_numpy_matrix(B)
print(G2)
# visualize the graph
nx.draw(G2, with_labels=True)

# %%
draw_communities_graph(G2, generate_colors(coms2), coms2)

# %% [markdown]
# ## Looking at sample output of the leiden model

# %%
list(to_use2.comment[coms2.communities[2]])  # belief that the disease exists

# %%
list(to_use2.comment[coms2.communities[3]])

# %%
list(to_use2.comment[coms2.communities[1]])  # belief on hand washing

# %%
list(to_use2.comment[coms2.communities[0]])

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
to_search = model.encode([to_use2.comment[5]])
faiss.normalize_L2(to_search)

# %%
indp2.add(sentence_embeddings2)
dist, post = indp2.search(to_search, 7)

# %%
