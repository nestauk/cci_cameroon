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
from cdlib import evaluation

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
# * We use modularity score to evaluate the strength of the communities formed. The larger the modularity score, the better the communities formed.
#

# %%
project_directory = cci_cameroon.PROJECT_DIR

# %%
data_df = pd.read_excel(
    f"{project_directory}/inputs/data/irfc_staff_labelled_data.xlsx"
)

# %%
data_df.first_code.unique()

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
# ## Using a smaller sample of the data (~30 records) to ease inspection of results

# %%
# creating a small subset of the data made up of the different categories
set1 = data[data.category_id == 0][["comment", "category_id"]][:10]
set2 = data[data.category_id == 1][["comment", "category_id"]][:10]
set3 = data[data.category_id == 2][["comment", "category_id"]][:10]
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
    """draws communities identified in a network. Recieves a graph, colors to use and community object"""
    color_map = []
    for node in graph:
        for i in range(len(com.communities)):
            if node in com.communities[i]:
                color_map.append(colors[i])
    nx.draw(graph, node_color=color_map, with_labels=True)
    plt.show()


def generate_adjacency_matrix(positions, n_neighbors, dimension):
    """Takes the positions obtained from the cosine similarity computation, the number of neighbors to consider,
    the size of the matrix and generates adjacency matrix. Returns an adjacency matrix"""
    adjacency_matrix = np.zeros(
        [dimension, dimension]
    )  # initialize the adjacency matrix to zeros
    # loop through the positions and set neighbors accordingly.
    for row in range(dimension):
        for num in range(n_neighbors):
            # set the neighbors in the adjacency matrix
            adjacency_matrix[row, positions[row][num]] = 1
    return adjacency_matrix


# %%
indp2 = faiss.index_factory(
    sentence_embeddings2.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(sentence_embeddings2)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 10
indp2.train(sentence_embeddings2)
indp2.add(sentence_embeddings2)
distance_matric2, positions2 = indp2.search(
    sentence_embeddings2, sentence_embeddings2.shape[0]
)

# %%
adjacency2 = generate_adjacency_matrix(
    positions2, neighbors2, sentence_embeddings2.shape[0]
)

# %%
np.fill_diagonal(adjacency2, 0)  # takes off self-links

# %%
# load the graph
GG2 = nx.from_numpy_matrix(adjacency2, create_using=nx.Graph())
print(GG2)
# visualize the graph
nx.draw(GG2, with_labels=True)

# %%
# applying a community algorithm to the graph to identify subgroups
coms22 = algorithms.leiden(GG2)
draw_communities_graph(GG2, generate_colors(coms22), coms22)

# %%
evaluation.modularity_density(GG2, coms22)

# %%
coms22.communities

# %%
list(to_use2.comment.iloc[coms22.communities[0]])

# %%
list(
    to_use2.category_id[coms22.communities[0]]
)  # inspecting the ground truth as per labelling exercise

# %%
list(to_use2.comment.iloc[coms22.communities[1]])

# %%
list(
    to_use2.category_id[coms22.communities[1]]
)  # inspecting the ground truth as per labelling exercise

# %%
list(to_use2.comment.iloc[coms22.communities[2]])  # Belief that the disease exists

# %%
list(to_use2.category_id[coms22.communities[2]])

# %%
# evaluating connectivity of the communities formed  using modularity
modularity2 = evaluation.modularity_density(GG2, coms22)

# %%
modularity2.score

# %% [markdown]
# # Working with a larger dataset

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
position.shape

# %%
AA = generate_adjacency_matrix(position, neighbors, sentence_embeddings.shape[0])

# %%
np.fill_diagonal(AA, 0)

# %%
# load the graph
GG = nx.from_numpy_matrix(AA, create_using=nx.Graph())
print(GG)
# visualize the graph
nx.draw(GG, with_labels=False)

# %%
# using leiden community detection algorithm on the dataset
comss = algorithms.leiden(GG)

# %%
draw_communities_graph(GG, generate_colors(comss), comss)

# %%
len(comss.communities)

# %% [markdown]
# ## Sample groups formed by the model

# %%
list(data.iloc[comss.communities[7]].comment)

# %%
list(data.iloc[comss.communities[1]].comment)

# %%
list(data.iloc[comss.communities[2]].comment)

# %%
list(data.iloc[comss.communities[5]].comment)

# %%
# evaluating the model using modularity
modularity = evaluation.modularity_density(GG, comss)

# %%
modularity.score

# %% [markdown]
# ## General observations
#
# * The higher the number of neigbors in the graph, the bigger the clusters formed. Need to agree on an average number of comments that should come through before algorithm is run in order to choose a suitable value for the number of neighbors.
#     * My thoughts: run the model after 300 to 500 comments?
#
# * Some keywords seem to influence the performance of the model. For example grouping of sentences with the word "existe" and its negation "n'existe pas" suggests the sentences are brought together by the word without much impact of context.
#
# * Some subgroups produced by the algorithm could be merged into a larger group without losing information. Need to optimise the algorithm.
#
#
#

# %%
## Using walktrap community detection algorithm on the data subset

# %%
# comsw = algorithms.walktrap(G)
# comsw2 = algorithms.walktrap(G2)

# %%
# draw_communities_graph(G2, generate_colors(comsw2), comsw2)

# %%

# %% [markdown]
#
