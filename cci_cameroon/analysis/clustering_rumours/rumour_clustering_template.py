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
from cci_cameroon.getters.clustering_helper_functions import (
    generate_colors,
    draw_communities_graph,
    generate_adjacency_matrix,
)

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
# File links
w1_file = "multi_label_output_w1.xlsx"
w2_file = "workshop2_comments_french.xlsx"
# Read workshop files
w1 = pd.read_excel(f"{project_directory}/inputs/data/" + w1_file)
w2 = pd.read_excel(f"{project_directory}/inputs/data/" + w2_file)

# %%
# Adding language column
w1["language"] = w1["comment"].apply(lambda x: detect(x))

# %%
# Slicing the data into en, es and remaining
en = w1[w1.language == "en"].copy()
es = w1[w1.language == "es"].copy()
w1 = w1[~w1.language.isin(["en", "es"])].copy()

# %%
# Translating the English and Spanish comments into French
en["comment"] = en.comment.apply(translator.translate, src="en", dest="fr").apply(
    getattr, args=("text",)
)
es["comment"] = es.comment.apply(translator.translate, src="es", dest="fr").apply(
    getattr, args=("text",)
)

# %%
# Merge back together
w1 = pd.concat([w1, en, es], ignore_index=True)

# %%
# Reemove language
w1.drop("language", axis=1, inplace=True)

# %%
# Join the two workshops files together
labelled_data = w1.append(w2, ignore_index=True)

# %%
# Remove white space before and after text
labelled_data.replace(r"^ +| +$", r"", regex=True, inplace=True)

# %%
# Removing 48 duplicate code/comment pairs (from W1)
print("Before duplicate pairs removed: " + str(len(labelled_data)))
labelled_data.drop_duplicates(subset=["code", "comment"], inplace=True)
print("After duplicate pairs removed: " + str(len(labelled_data)))

# %%
# Removing small count codes
to_remove = list(
    labelled_data.code.value_counts()[labelled_data.code.value_counts() < 10].index
)
labelled_data = labelled_data[~labelled_data.code.isin(to_remove)].copy()

# %%
# Dataset for modelling
model_data = labelled_data.copy()
# Create category ID column from the code field (join by _ into one string)
model_data["category_id"] = model_data["code"].str.replace(" ", "_")
id_to_category = dict(model_data[["category_id", "code"]].values)
model_data = (
    model_data.groupby("comment")["category_id"].apply(list).to_frame().reset_index()
)

# %%
model_data = model_data.reset_index()

# %%
model_data.shape

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
# Converting the codes into integers to act as ground truth
data["cluster"] = data.code.factorize()[0]

# %%
# using the french_semantic model for word embedding
model = SentenceTransformer("Sahajtomar/french_semantic")

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
# load the graph
GG2 = nx.from_numpy_matrix(adjacency2, create_using=nx.Graph())
print(GG2)
# visualize the graph
nx.draw(GG2, with_labels=True)

# %%
# applying a community algorithm to the graph to identify subgroups
coms22 = algorithms.leiden(GG2)
draw_communities_graph(
    GG2, generate_colors(len(coms22.communities)), coms22.communities
)

# %%
evaluation.modularity_density(GG2, coms22)

# %%
list(to_use2.comment.iloc[coms22.communities[0]])

# %%
list(
    to_use2.category_id[coms22.communities[0]]
)  # inspecting the ground truth as per labelling exercise

# %%
ax = plt.figure(figsize=(10, 5))
to_use2.category_id[coms22.communities[0]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(to_use2.comment.iloc[coms22.communities[1]])

# %%
list(
    to_use2.category_id[coms22.communities[1]]
)  # inspecting the ground truth as per labelling exercise

# %%
ax = plt.figure(figsize=(10, 5))
to_use2.category_id[coms22.communities[1]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(to_use2.comment.iloc[coms22.communities[2]])  # Belief that the disease exists

# %%
list(to_use2.category_id[coms22.communities[2]])

# %%
ax = plt.figure(figsize=(10, 5))
to_use2.category_id[coms22.communities[2]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
# evaluating connectivity of the communities formed  using modularity
modularity2 = evaluation.modularity_density(GG2, coms22)

# %%
modularity2.score

# %% [markdown]
# # Working with a larger dataset

# %%
sentence_embeddings = model.encode(data.comment)

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
AA = generate_adjacency_matrix(position, neighbors, sentence_embeddings.shape[0])

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
draw_communities_graph(GG, generate_colors(len(comss.communities)), comss.communities)

# %%
len(comss.communities)

# %% [markdown]
# ## Sample groups formed by the model

# %%
list(
    data.iloc[comss.communities[7]].comment
)  # Observation about mask wearing by children

# %%
ax = plt.figure(figsize=(10, 5))
data.category_id[comss.communities[7]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(data.iloc[comss.communities[1]].comment)  # belief that disease exists

# %%
ax = plt.figure(figsize=(10, 5))
data.category_id[comss.communities[1]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(data.iloc[comss.communities[2]].comment)  # belief about wearing face masks

# %%
ax = plt.figure(figsize=(10, 5))
data.category_id[comss.communities[2]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(data.iloc[comss.communities[5]].comment)

# %%
ax = plt.figure(figsize=(10, 5))
data.category_id[comss.communities[5]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
# evaluating the model using modularity
modularity = evaluation.modularity_density(GG, comss)

# %%
modularity.score

# %%

# %%

# %% [markdown]
# # Using data with eight codes

# %%
model_data_embeddings = model.encode(model_data.comment)

# %%
indexf = faiss.index_factory(
    model_data_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(model_data_embeddings)  # to be used for inner product computation
neighbors = 15  # number of neighbors for the graph
indexf.train(model_data_embeddings)
indexf.add(model_data_embeddings)
distance_f, positions_f = indexf.search(
    model_data_embeddings, model_data_embeddings.shape[0]
)

# %%
A_f = generate_adjacency_matrix(positions_f, neighbors, model_data_embeddings.shape[0])

# %%
# load the graph
G_f = nx.from_numpy_matrix(A_f, create_using=nx.Graph())
print(G_f)
# visualize the graph
nx.draw(G_f, with_labels=False)

# %%
coms_f = algorithms.leiden(G_f)

# %%
len(coms_f.communities)

# %%
draw_communities_graph(
    G_f, generate_colors(len(coms_f.communities)), coms_f.communities
)

# %%
list(model_data.iloc[coms_f.communities[0]].comment)

# %%
ax = plt.figure(figsize=(10, 5))
model_data.category_id[coms_f.communities[0]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(model_data.iloc[coms_f.communities[1]].comment)

# %%
ax = plt.figure(figsize=(10, 5))
model_data.category_id[coms_f.communities[1]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(model_data.iloc[coms_f.communities[2]].comment)

# %%
ax = plt.figure(figsize=(10, 5))
model_data.category_id[coms_f.communities[2]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(model_data.iloc[coms_f.communities[3]].comment)

# %%
ax = plt.figure(figsize=(10, 5))
model_data.category_id[coms_f.communities[3]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(model_data.iloc[coms_f.communities[4]].comment)

# %%
ax = plt.figure(figsize=(10, 5))
model_data.category_id[coms_f.communities[4]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")

# %%
list(model_data.iloc[coms_f.communities[5]].comment)

# %%
list(model_data.iloc[coms_f.communities[6]].comment)

# %%
list(model_data.iloc[coms_f.communities[7]].comment)

# %%
list(model_data.iloc[coms_f.communities[8]].comment)

# %%
list(model_data.iloc[coms_f.communities[9]].comment)

# %%
list(model_data.iloc[coms_f.communities[10]].comment)

# %%
list(model_data.iloc[coms_f.communities[11]].comment)

# %%
list(model_data.iloc[coms_f.communities[12]].comment)

# %%

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
# * Findind a suitable metric that could be used to assess the quality of comments in subgroups
#

# %% [markdown]
# ## Other community detection algorithms

# %%
## Using walktrap community detection algorithm on the data subset

# %%
comsw2 = algorithms.walktrap(GG2)

# %%
draw_communities_graph(
    GG2, generate_colors(len(comsw2.communities)), comsw2.communities
)

# %%

# %%

# %%
to_use2.to_excel("to_use2.xlsx", index=False)

# %%
model_data.to_excel("model_data.xlsx", index=False)

# %%
data.to_excel("data.xlsx", index=False)

# %%
# to_use2 = pd.read_excel("to_use2.xlsx")
model_data = pd.read_excel("model_data.xlsx")
# data= pd.read_excel("data.xlsx")

# %%
model_data.head()

# %%
