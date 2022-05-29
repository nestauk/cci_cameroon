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
from cdlib import evaluation

# %matplotlib inline
import faiss
from faiss import normalize_L2
from sentence_transformers import SentenceTransformer
import networkx as nx
import cdlib
import igraph as ig
import cairocffi
import leidenalg as la
from cdlib import algorithms
from cci_cameroon.pipeline import cluster_utils
from cci_cameroon.pipeline.clustering_helper_functions import (
    generate_colors,
    draw_communities_graph,
    generate_adjacency_matrix,
    generate_adjacency_matrix2,
    generate_edges,
    get_metrics,
    compute_community_silhuoette_scores,
    get_communities_with_threshold,
    generate_community_labels,
    plot_clusters,
)
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import community
import xlsxwriter
import umap.umap_ as umap

# %%
project_directory = cci_cameroon.PROJECT_DIR


# %%
# load unclassified comments comming in from the classification model
model_data = pd.read_excel(f"{project_directory}/outputs/data/not_classified.xlsx")
column_name = "comment"  # holds the column of interest in the data

# %%
model_data

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
import numpy as np

# %%
for num in range(len(np.unique(consensus_partition2))):
    print(num)
number_clusters = np.unique(consensus_partition2)
print(number_clusters)

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

# %%

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

# %%
