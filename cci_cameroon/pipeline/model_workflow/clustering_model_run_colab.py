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

# %% [markdown]
# Performing concensus clustering (Google colab version)
# Use this notebook to run the `rumour_clustering_model_development_concesus_clustering` script on google colab.
#
# Steps to take to run:
#
# Upload the following files to Google Colab
# * `clustering_helper_function.py` found in `cci_cameroon/getters` folder
# * `cluster_utils.py` found in `cci_cameroon/pipeline` folder
# *  `not_classified.xlsx` found under `outputs/data` folder
#
# Uncomment the pip install lines in the first cell
# The output of this script is the clusters.xlsx file which contains the clusters of rumours created by the algorthm.

# %%
# #!pip install cdlib
# #!pip install faiss-cpu --no-cache
# #!pip install sentence-transformers
# #!pip install xlsxwriter
# #!pip install leidenalg
# #!pip install umap-learn
# #!pip install cairocffi
# #!pip install igraph

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cdlib import evaluation
from cdlib import algorithms

# %matplotlib inline
import faiss
from faiss import normalize_L2
from sentence_transformers import SentenceTransformer
import networkx as nx
import cdlib
from clustering_helper_functions import (  # use the right location of the file for import to be successful
    generate_colors,
    draw_communities_graph,
    generate_adjacency_matrix,
    generate_adjacency_matrix2,
    generate_edges,
    get_metrics,
    compute_community_silhuoette_scores,
    get_communities_with_threshold,
    generate_community_labels,
)
import cluster_utils
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import community
import matplotlib
import seaborn as sns
from time import time
import xlsxwriter
import igraph as ig
import leidenalg as la
import cairocffi
import umap.umap_ as umap
import seaborn as sns
from time import time
import random

# %%
# load unclassified comments comming in from the classification model
model_data = pd.read_excel("not_classified.xlsx")
column_name = "comment"  # holds the column of interest in the data

# %%
# initialize the french_semantic model for word embedding
model = SentenceTransformer("Sahajtomar/french_semantic")

# %%
# create word embeddings using the transformer model
sentence_embeddings = model.encode(model_data[column_name])

# %%
# Create a 2D embedding using umap
s_reducer = umap.UMAP(n_components=2, random_state=1)
s_embedding = s_reducer.fit_transform(sentence_embeddings)

# %%
sentence_embeddings.shape

# %%
indp2 = faiss.index_factory(
    sentence_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(sentence_embeddings)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 10
indp2.train(sentence_embeddings)
indp2.add(sentence_embeddings)
similarity_distance, sim_positions = indp2.search(
    sentence_embeddings, sentence_embeddings.shape[0]
)

# %%
# applying consensus clustering here
edges, weighted_edges = generate_edges(
    model_data, "comment", sim_positions, similarity_distance
)
adjacency_matrix = generate_adjacency_matrix(
    sim_positions, neighbors2, sentence_embeddings.shape[0]
)  # similarity_distance)

# %%
similarity_distance = np.array(similarity_distance.clip(min=0)).copy()
np.fill_diagonal(similarity_distance, 0)  # remove self-links

# %%
# construct a graph and use it for concensus clustering
W_triu = np.triu(similarity_distance)  # similarity matrix
A_triu = np.triu(adjacency_matrix)  # adjacency_matrix
sources, targets = A_triu.nonzero()
weights = W_triu[A_triu.nonzero()]
edgelist = list(zip(sources.tolist(), targets.tolist()))
g = ig.Graph(edges=edgelist, directed=False)
g.es["weight"] = weights

# %%
adjacency_matrix[0]

# %%
# Create a ConsensusClustering instance
clust = cluster_utils.ConsensusClustering(graph=g, N=500, N_consensus=20, seed=25)

# %%
# Find the consensus partition
consensus_partition2 = clust.consensus_partition

# %%
number_clusters = np.unique(consensus_partition2)

# %%
model_data["consensus"] = consensus_partition2

# %%
retained_clusters = []
for num in range(len(np.unique(consensus_partition2))):
    retained_clusters.append(
        model_data[model_data.consensus == num]["comment"].to_list()
    )
print(retained_clusters)

# %%
# Sanity check by plotting the sorted similarity matrix
plt.figure(figsize=(10, 8))
cluster_utils.plot_sorted_matrix(similarity_distance, consensus_partition2)

# %%
# Sanity check by plotting the sorted co-clustering occurrence matrix
plt.figure(figsize=(10, 8))
cluster_utils.plot_sorted_matrix(clust.COOC, consensus_partition2)

# %%
# plt.figure(figsize=(10,8))
# Use the co-clustering occurrence matrix to estimate node "affinity" to
# their own cluster as well as to other clusters
p = np.array(consensus_partition2)
M = cluster_utils.node_affinity(clust.COOC, p)
cluster_utils.node_affinity_plot(M, p, aspect_ratio=0.03)

# %%
# Use the node affinity matrix to estimate the average cluster affinity to
# itself and to other clusters
plt.figure(figsize=(10, 10))
C = cluster_utils.cluster_affinity_matrix(M, p, symmetric=True, plot=True)

# %% [markdown]
# While clusters are overall well seperated (as judged by the clustering algorithm) note that there are some non-zero affinities between some clusters, e.g., cluster 1 and cluster 4

# %%
# Visualise the data-driven cluster vs. comment correspondence
# Generate integer labels that indicate comment membership
comment_map = dict(
    zip(model_data.comment.unique()[:15], range(len(model_data.comment.unique()[:15])))
)
comment_membership = model_data.comment[:15].apply(lambda x: comment_map[x]).values

# Plot a confusion matrix to show how skills from different sectors are mixing
plt.figure(figsize=(10, 10))
C = cluster_utils.plot_confusion_matrix(
    y_true=consensus_partition2[:15],
    y_pred=comment_membership,
    true_labels=None,
    pred_labels=list(comment_map.keys()),
    normalize_to=0,
)

# %% [markdown]
# ## Visualizing the clusters

# %%
# Create a 2D embedding using umap
s_reducer = umap.UMAP(n_components=2, random_state=1)
s_embedding = s_reducer.fit_transform(sentence_embeddings)

# %%
# preparing data for plotting
model_data["x"] = s_embedding[:, 0]
model_data["y"] = s_embedding[:, 1]

# %%
plot_clusters(model_data, "consensus")

# %%
model_data[model_data.consensus == 0]["comment"].to_list()

# %%
# create a workbook and store the resulting clusters in it. Each cluster in a separate worksheet.
with xlsxwriter.Workbook(f"{project_directory}/outputs/data/clusters.xlsx") as workbook:
    for community in retained_clusters:
        worksheet = workbook.add_worksheet()
        for i in range(len(community)):
            j = i + 1
            ex_col = "A" + str(j)
            worksheet.write(ex_col, community[i])
