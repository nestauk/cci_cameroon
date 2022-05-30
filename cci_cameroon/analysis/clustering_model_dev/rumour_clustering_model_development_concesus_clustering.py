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
from cdlib import evaluation

# %matplotlib inline
import faiss
from faiss import normalize_L2
from sentence_transformers import SentenceTransformer
import networkx as nx
import cdlib
from cdlib import algorithms
from cci_cameroon.getters.clustering_helper_functions import (
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
from cci_cameroon.pipeline import cluster_utils
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import community
import matplotlib
import seaborn as sns
from time import time
import xlsxwriter

# %%
TOKENIZERS_PARALLELISM = False

# %%
import igraph as ig
import leidenalg as la
import cairocffi
import umap.umap_ as umap

# %%
# #!pip install umap-learn

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
# load cleaned data
# read cleaned data from inputs folder
to_use2 = pd.read_excel(f"{project_directory}/inputs/data/to_use2.xlsx")
model_data = pd.read_excel(f"{project_directory}/inputs/data/model_data.xlsx")
data = pd.read_excel(f"{project_directory}/inputs/data/data.xlsx")

# %%
model_data.drop("index", axis=1, inplace=True)

# %%
# using the french_semantic model for word embedding
model = SentenceTransformer("Sahajtomar/french_semantic")
# model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# %%
# we shuffle the data and split it up into equal data chunks. We usBeginning with a chunk
shuffled = model_data.sample(frac=1)
result = np.array_split(shuffled, 25)
df_list = []  # to hold metrics for each group of data chosen
actual_comments_list = []  # holds the lists of retained communities
rejected_groups_list = (
    []
)  # holds the list of All rejected communities with comment count that meet the set threshold
# we start experimenting with 204 data points.
dt = pd.concat([result[0], result[1], result[2], result[3]]).reset_index()
if "level_0" in dt.columns:
    dt.drop("level_0", "index", axis=1, inplace=True)
if "index" in dt.columns:
    dt.drop("index", axis=1, inplace=True)

# %%
retained_comment_lists, rejected_cluster_lists, df_st = get_metrics(
    dt, "comment", model
)

# %%
df_st

# %%
retained_comment_lists[0][1]

# %%
df_list.append(df_st)
actual_comments_list.append(
    retained_comment_lists
)  # stores the groups that are retained
rejected_groups_list.append(rejected_cluster_lists)  # holds rejected clusters.

# %%
for count in range(
    4, len(result) - 15
):  # to work with a maximum of 500 comments (just a reasonable size)
    dt = pd.concat([dt, result[count]]).reset_index().drop("index", axis=1)
    if "level_0" in dt.columns:
        dt.drop(["level_0"], axis=1, inplace=True)
    if "index" in dt.columns:
        dt.drop("index", axis=1, inplace=True)
    c, r, d = get_metrics(dt, "comment", model)
    df_list.append(d)
    actual_comments_list.append(c)
    rejected_groups_list.append(r)

# %%
# We inspect the metrics computed and the quality of the clusters produced. This helps to choose the number of neighbors to take forward
df_all = pd.concat(df_list).reset_index().drop("index", axis=1)
df_all.head(50)

# %%
# In this section, we manually inspect the clusters formed when different number of neighbors are considered.
# actual_comments_list[0][0][4]  ## 1,1,1,.8,1
# actual_comments_list[0][3][0]   ##
# actual_comments_list[0][1][1]
actual_comments_list

# %%
# The rejected clusters for each criterion are inspected
# rejected_groups_list#[0][0][0]# 0(business,exists or real,transmission,not exist),1(transmission -97%,business -2%)

# %% [markdown]
# s_size  neibors AMI  	modularity	modu_density	tot_clusters	silhouette_av	clus_retain comments  clus_qty
#
# 204      15    0.862684	 0.675880	  74.435275	        7             0.251272	      4           107       82%
# 255	     15	   1.000000	 0.697397	  81.028345     	7	          0.245696	      4           140       75.5%
# 255	     10    0.858371	 0.744165	  81.985194	       10	          0.234848	      5           128       70.2%
# 306	     15	   0.968674  0.707146	  98.299726	        9	          0.243298	      5           174       73.6%

# %%
# model_data.drop("index",axis=1,inplace=True)

# %% [markdown]
# ## Observation on growing sizes
#
# * Beginning with 204 comments and incrementing by 51 comments, computed the metrics AMIs, modularity, av silhouette score
# * smallest cluster size 13 comments
# * Largest cluster size 68 comments
#
# s_size  neibors AMI  	modularity	modu_density	tot_clusters	silhouette_av	clus_retain comments  clus_qty
#
# 204      15    0.862684	 0.675880	  74.435275	        7             0.251272	      4           107       82%
# 255	     15	   1.000000	 0.697397	  81.028345     	7	          0.245696	      4           140       75.5%
# 255	     10    0.858371	 0.744165	  81.985194	       10	          0.234848	      5           128       70.2%
# 306	     15	   0.968674  0.707146	  98.299726	        9	          0.243298	      5           174       73.6%
#
# ## Observation on balanced groups with three codes only
# * Using the average silhouette score of clusters as cutoff, three different codes and neighbors =15
#
# * Where the groups are balanced, all retained clusters are at least 95% homogeneous.
#
# * Smaller clusters which can be combined to one larger cluster are formed.
#
# * Some rejected clusters are homogeneous at this cut-off.
#

# %%
# Using sample data of the same size but different codes.
df_n1 = model_data[
    model_data.category_id
    == "['Croyances_sur_le_lavage_des_mains_ou_les_désinfectants_des_mains']"
][:80]
df_n2 = model_data[
    model_data.category_id == '["Croyance_que_l\'épidémie_est_terminée"]'
][:80]
df_n3 = model_data[
    model_data.category_id
    == '["Croyance_que_certaines_personnes_/_institutions_gagnent_de_l\'argent_à_cause_de_la_maladie"]'
][:80]
df_n = pd.concat([df_n1, df_n2, df_n3]).reset_index().drop("index", axis=1)

# %%
comment_lists_n, rejected_groups_n, df_st_n = get_metrics(df_n, "comment", model)

# %%
# do I go for large groups which sometimes would be bad or smaller groups which could
# be merged with >80% homogenuity?
df_st_n

# %%
comment_lists_n[1][0]

# %%
sentence_embeddings400 = model.encode(data.comment)

# %%
index = faiss.index_factory(
    sentence_embeddings400.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
# setting the number of neighbors to consider for graph connectivity
index.train(sentence_embeddings400)
index.add(sentence_embeddings400)
similarity_distance_mat, sim_positions_mat = index.search(
    sentence_embeddings400, 5  # consider only the first n elements
)
weight_mat = generate_adjacency_matrix2(
    sim_positions_mat, 5, sentence_embeddings400.shape[0], similarity_distance_mat
)
G = nx.from_numpy_matrix(weight_mat, create_using=nx.Graph(), parallel_edges=False)
partitions = community.best_partition(G)
comu = algorithms.leiden(G)

# %%
labels_best = []
for n in partitions.items():
    labels_best.append(n[1])

# %%
labels_com = generate_community_labels(
    comu.communities, sentence_embeddings400.shape[0]
)

# %%
# compute adjusted mutual information score for the clusters formed by the two algorithms.
adjusted_mutual_info_score(labels_best, labels_com)

# %%
# The performance of a partition is the ratio of the number of intra-community edges
# plus inter-community non-edges with the total number of potential edges.
nx.algorithms.community.quality.performance(G, comu.communities)

# %%
sil_score = silhouette_samples(sentence_embeddings400, labels_com, metric="cosine")

# %%
get_communities_with_threshold(comu.communities, sil_score, 0.3, data, "comment")

# %%
evaluation.modularity_density(G, comu)

# %%

partitions = community.best_partition(G)

# %%
modularity = community.modularity(partitions, G)
modularity

# %%
ncoms = algorithms.leiden(G)

# %%
len(ncoms.communities)

# %%
evaluation.modularity_density(G, ncoms)

# %% [markdown]
#  **using a weighted graph doesn't cause significant change in the commun-ties formed**

# %%
# embed the comments before spliting to different subgroups
comments_400_embeddings = model.encode(data.comment)

# %%
ind400 = faiss.index_factory(
    comments_400_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(comments_400_embeddings)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 10
ind400.train(comments_400_embeddings)
ind400.add(comments_400_embeddings)
similarity_distance400, sim_positions400 = ind400.search(
    comments_400_embeddings, neighbors2  # consider only the first n elements
)

# %%
edges, weighted_edges = generate_edges(
    data, "comment", sim_positions400, similarity_distance400
)

# %%
nodes = list(set(data.comment))

# %%
NG400 = nx.Graph()
NG400.add_nodes_from(nodes)
NG400.add_edges_from(edges)
NG400.add_weighted_edges_from(weighted_edges)

# %%
Ncom400 = algorithms.leiden(NG400)

# %%
# adjacency matrix for 400 comments for weighted graph
weighted_mat_400 = generate_adjacency_matrix2(
    sim_positions400,
    neighbors2,
    comments_400_embeddings.shape[0],
    similarity_distance400,
)

# %%
GM400 = nx.from_numpy_matrix(
    weighted_mat_400, create_using=nx.Graph(), parallel_edges=False
)

# %%
nx.draw(GM400, with_labels=True)

# %%
# detect communities
comsM400 = algorithms.leiden(GM400)

# %%
len(comsM400.communities)

# %%
data.iloc[comsM400.communities[0]].comment

# %%
# draw the communities formed
draw_communities_graph(
    GM400, generate_colors(len(comsM400.communities)), comsM400.communities
)

# %%
evaluation.modularity_density(GM400, comsM400)

# %%
# generate labels for nodes in the graph to use in computing silhouette scores.
labels2 = generate_community_labels(
    comsM400.communities, comments_400_embeddings.shape[0]
)

# %%
sample_silhouette_scores = silhouette_samples(
    comments_400_embeddings, labels2, metric="cosine"
)

# %%
# sample_silhouette_scores
# print(get_communities_with_threshold(comsM400.communities,sample_silhouette_scores,0.3,data,"comment"))

# %% [markdown]
# ## Silhouette scores for the different communities computed and communities ranked
# * The data is sorted in order of silhoutte score and all clusters presented to be reviewed.

# %%
# add silhouette score column to the dataframe.
sil_scores = compute_community_silhuoette_scores(
    comsM400.communities, sample_silhouette_scores
)
# to_use2["silhouette_score"] = [None]*len(sample_silhouette_scores)
# for c in range(len(sil_scores)):
#    to_use2.silhouette_score.iloc[comsM.communities[c]]=sil_scores[c]


# %%
# sorted data in order of silhouette score and can present sorted clusters to be reviewed
# to_use2.sort_values("silhouette_score",axis=0, ascending=False)

# %%
# generate labels for 400 comments
labels400 = generate_community_labels(
    comsM400.communities, comments_400_embeddings.shape[0]
)

# %%
# compute individual data point silhouette scores
sample_silhouette_scores400 = silhouette_samples(
    comments_400_embeddings, labels400, metric="cosine"
)

# %%
# print(get_communities_with_threshold(comsM400.communities,sample_silhouette_scores400,0.3,data,"comment"))

# %%
# list(data.iloc[comsM400.communities[0]].comment)

# %% [markdown]
# ## Run the clustering on different samples and inspect the formed clusters

# %%
sample_data = model_data.sample(300)

# %%
sample_data.reset_index(inplace=True)

# %%
sample_data_embeddings = model.encode(sample_data.comment)

# %%
indsamp = faiss.index_factory(
    comments_400_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(sample_data_embeddings)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 10
indsamp.train(sample_data_embeddings)
indsamp.add(sample_data_embeddings)
similarity_distance_samp, sim_positions_samp = indsamp.search(
    sample_data_embeddings, neighbors2  # consider only the first n elements
)

# %%
# adjacency matrix for 400 comments for weighted graph
weighted_mat_samp = generate_adjacency_matrix2(
    sim_positions_samp,
    neighbors2,
    sample_data_embeddings.shape[0],
    similarity_distance_samp,
)

# %%
GMsamp = nx.from_numpy_matrix(
    weighted_mat_samp, create_using=nx.Graph(), parallel_edges=False
)

# %%
comsSamp = algorithms.leiden(GMsamp)

# %%
len(comsSamp.communities)

# %%
labels_samp = generate_community_labels(
    comsSamp.communities, sample_data_embeddings.shape[0]
)

# %%
silhouette_scores_samp = silhouette_samples(
    sample_data_embeddings, labels_samp, metric="cosine"
)

# %%
silhouette_score(sample_data_embeddings, labels_samp)

# %%
# print(get_communities_with_threshold(comsSamp.communities,silhouette_scores_samp,0.2,sample_data,"comment"))

# %%
adjacency2 = generate_adjacency_matrix(
    sim_positions_samp, neighbors2, sample_data_embeddings.shape[0]
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
# list(data.comment.iloc[coms22.communities[0]])

# %% [markdown]
# ## Using ground truth, we evaluate the homogeneity of the clusters formed

# %%
# list(data.category_id[coms22.communities[0]]) #inspecting the ground truth as per labelling exercise

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
    sentence_embeddings, neighbors
)  # create similarity matrix for the data

# %%

# %%
# inspect the distribution of the silhouette scores to pick a cutoff for creating edges between nodes in a graph
ax = plt.figure(figsize=(10, 5))
plt.hist(list(matplotlib.cbook.flatten(distance_matric)))
plt.xlabel("Similarity score")
plt.ylabel("frequency")
plt.title("Distribution of similarity score for 1300 comments")
plt.show()

# %% [markdown]
# To generate weighted graph (weights being similarity scores), we Settled at 0.4 as threshold for an edge to be formed. Using this value as cutoff leads to the creation of clusters that are at least 80% homogeneous. A larger value considered leads to many clusters being formed with only one comment.


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

# %% [markdown]
# ## Generating clusters using data with eight different codes obtained from workshop

# %%
model_data_embeddings = model.encode(model_data.comment)

# %%
indexf = faiss.index_factory(
    model_data_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(model_data_embeddings)  # to be used for inner product computation
neighbors = 100  # number of neighbors for the graph
indexf.train(model_data_embeddings)
indexf.add(model_data_embeddings)
distance_f, positions_f = indexf.search(
    model_data_embeddings, model_data_embeddings.shape[0]
)

# %%
A_f = generate_adjacency_matrix2(
    positions_f, neighbors, model_data_embeddings.shape[0], distance_f
)

# %%
W_cluster = np.array(A_f)

# %%
# Number of comments
n_occ = W_cluster.shape[0]

# %%
# Name of this clustering session
session_name = "rumours_concensus_1000"
# Number of nearest neighbours used for the graph construction
nearest_neighbours = [10, 15, 20, 25]
# Ensemble size for the first step
N = 1000
# Ensemble size for the consensus step
N_consensus = 100
# Number of clustering trials for each nearest neighbour value
N_nn = N // len(nearest_neighbours)
# Which clusters to break down from the partition
clusters = "all"  # Either a list of integers, or 'all'
# Path to save the clustering results
fpath = f"{project_directory}/outputs/data/concensus_clusters"

clustering_params = {
    "N": N,
    "N_consensus": N_consensus,
    "N_nn": N_nn,
    "clusters": clusters,
    "fpath": fpath,
    "session_name": session_name,
    "nearest_neighbours": nearest_neighbours,
}

# %%
# Set the random_state variable for reproduciblity
clustering_params["random_state"] = 14523

# %%
# Prepare and save the Level-0 partition file (all in one cluster)
partition_df = pd.DataFrame()
partition_df["id"] = model_data.index.to_list()
partition_df["cluster"] = np.zeros((len(model_data.comment)))
partition_df.to_csv(fpath + session_name + "_clusters_Level0.csv")

# Set the random_state variable for reproduciblity

# %%
# Perform the clustering
cluster_utils.subcluster_nodes(W=W_cluster, l=0, **clustering_params)

# %%
graph = cluster_utils.build_graph(W_cluster, kNN=30)

# %%
# Create a ConsensusClustering instance
clust = cluster_utils.ConsensusClustering(graph=graph, N=100, N_consensus=20, seed=0)
# Find the consensus partition
consensus_partition = clust.consensus_partition
# Check the sizes of the clusters
clust.describe_partition(consensus_partition)

# %%
# Create a ConsensusClustering instance
clust = cluster_utils.ConsensusClustering(graph=graph, N=500, N_consensus=20, seed=0)
# Find the consensus partition
consensus_partition = clust.consensus_partition
# Check the sizes of the clusters
clust.describe_partition(consensus_partition)

# %%
1 - 2 / 61

# %%
1 - 5 / 58

# %%
1 - 3 / 69

# %%
model_data_consensus = model_data.copy()

# %%
model_data_consensus["cluster"] = consensus_partition

# %%
model_data_consensus[model_data_consensus.cluster == 10]["comment"].to_list()

# %% [markdown]
# clusters are overall well seperated (as judged by the clustering algorithm) and there are no non-zero affinities between some clusters

# %%
# Evaluate clustering stability across the ensemble using the adjusted mutual information
ami_avg, ami_matrix = clust.ensemble_AMI(clust.ensemble)

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
    y_true=consensus_partition[:15],
    y_pred=comment_membership,
    true_labels=None,
    pred_labels=list(comment_map.keys()),
    normalize_to=0,
)


# %% [markdown]
# # second approach: Generating the igraph for consensus clustering

# %%
indp2 = faiss.index_factory(
    model_data_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(model_data_embeddings)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 70
indp2.train(model_data_embeddings)
indp2.add(model_data_embeddings)
similarity_distance, sim_positions = indp2.search(
    model_data_embeddings,
    model_data_embeddings.shape[0],  # consider only the first n elements
)

# %%
edges, weighted_edges = generate_edges(
    model_data, "comment", sim_positions, similarity_distance
)
adjacency_matrix = generate_adjacency_matrix(
    sim_positions, neighbors2, model_data_embeddings.shape[0]
)  # similarity_distance)

# %%
similarity_distance = np.array(similarity_distance.clip(min=0)).copy()

# %%
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
# Create a ConsensusClustering instance
clust = cluster_utils.ConsensusClustering(graph=g, N=500, N_consensus=20, seed=25)

# %%
# Find the consensus partition
consensus_partition2 = clust.consensus_partition

# %%
model_data["consensus"] = consensus_partition
model_data["consensus_g"] = consensus_partition2

# %%
model_data.columns

# %%
# Sanity check by plotting the sorted similarity matrix
cluster_utils.plot_sorted_matrix(similarity_distance, consensus_partition2)

# %%
# Sanity check by plotting the sorted co-clustering occurrence matrix
cluster_utils.plot_sorted_matrix(clust.COOC, consensus_partition2)

# %%
# Use the co-clustering occurrence matrix to estimate node "affinity" to
# their own cluster as well as to other clusters
p = np.array(consensus_partition2)
M = cluster_utils.node_affinity(clust.COOC, p)
cluster_utils.node_affinity_plot(M, p, aspect_ratio=0.03)

# %%
M.shape


# %%
def cluster_affinity_matrix2(
    M, cluster_labels, symmetric=True, plot=True, cmap="Blues"
):
    """
    Calculate each cluster's affinity to other clusters based on their constituent
    nodes' affinities to the different clusters.

    Parameters
    ----------
    M (numpy.ndarray):
        Node affinity matrix.
    cluster_labels (list of int):
        Clustering partition with integers denoting cluster labels.
    symmetric (boolean):
        If True, ensures that the cluster affinity matrix is symmetric.
    symmetric (boolean):
        Determines whether the cluster affinity matrix is displayed.

    Returns
    -------
    C (numpy.ndarray):
        Cluster affinity matrix, where elements (k,l) indicates the average
        co-clustering occurrence of cluster k nodes with the nodes of cluster l.
    """

    n_clust = len(np.unique(cluster_labels))
    C = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            if np.isnan(np.mean(M[np.where(cluster_labels == i)[0], j])) == False:
                C[i, j] = np.mean(M[np.where(cluster_labels == i)[0], j])

    if symmetric == True:
        C = 0.5 * C + 0.5 * C.T

    if plot == True:
        plt.imshow(C, cmap=cmap)
        plt.xlabel("cluster", size=20)
        plt.ylabel("cluster", size=20)
        plt.colorbar()
        plt.title("Cluster affinity to other clusters", size=20)
        plt.show()
    plt.savefig(f"{project_directory}/outputs/figures/svg/node_affinity.svg")
    plt.savefig(f"{project_directory}/outputs/figures/png/node_affinity.png")
    return C


# %%
# Use the node affinity matrix to estimate the average cluster affinity to
# itself and to other clusters
plt.figure(figsize=(10, 10))
C = cluster_affinity_matrix2(M, p, symmetric=True, plot=True)

# %%
# Use the node affinity matrix to estimate the average cluster affinity to
# itself and to other clusters
plt.figure(figsize=(10, 10))
C = cluster_affinity_matrix2(M, p, symmetric=True, plot=True)

# %%
# Use the node affinity matrix to estimate the average cluster affinity to
# itself and to other clusters
plt.figure(figsize=(10, 10))
C = cluster_utils.cluster_affinity_matrix(M, p, symmetric=True, plot=True)

# %% [markdown]
# While clusters are overall well seperated (as judged by the clustering algorithm) note that there are some non-zero affinities between some clusters, e.g., cluster 2 and cluster 4

# %%
model_data[model_data.consensus_g == 3]["comment"].to_list()

# %%
# #!pip install umap-learn

# %% [markdown]
# # visualizing the clusters

# %%
# Create a 2D embedding using umap
s_reducer = umap.UMAP(n_components=2, random_state=1)
s_embedding = s_reducer.fit_transform(model_data_embeddings)


# %%
# preparing data for plotting
model_data["x"] = s_embedding[:, 0]
model_data["y"] = s_embedding[:, 1]

# %%
enmax_palette = [
    "#0000FF",
    "#FF6E47",
    "#18A48C",
    "#EB003B",
    "#9A1BB3",
    "#FDB633",
    "#97D9E3",
    "#FF6103",
    "#7B7300",
    "#8B6508",
]
color_codes_wanted = [
    "nesta_blue",
    "nesta_orange",
    "nesta_green",
    "nesta_red",
    "nesta_purple",
    "nesta_yellow",
    "nesta_agua",
    "other",
    "other1",
    "other2",
    "other3",
]
c = lambda x: enmax_palette[color_codes_wanted.index(x)]
# cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)
sns.set_palette(sns.color_palette(enmax_palette))
pal = sns.color_palette(enmax_palette)  # nesta palette
# generate colors for the map
# from random import randint
# colors = []
# for i in range(len(model_data.consensus.unique())):
#    colors.append('#%06X' % randint(0, 0xFFFFFF))
colors = [
    "#32CD32",
    "#3D59AB",
    "#9A1BB3",
    "#050505",
    "#DC143C",
    "#CD9B1D",
    "#FF6E47",
    "#9400D3",
    "#8B7355",
    "#8B6508",
    "#18A48C",
    "#EB003B",
    "#FDB633",
    "#97D9E3",
    "#8B8378",
    "#0000FF",
]


# %%
colors


# %%
def plot_clusters2(df, cluster_column, file_path=None):
    """Plots a scatter plot of the clusters with each cluster having a different color.
    Parameters:
        df: a dataframe which contains a column cluster_column that holds the cluster lables (int). It also contains
        columns x and y which are extracted from the reduced embeddings of the data.
        file_path: directory where image should be saved
    """
    sns.set_style("white", {"axes.facecolor": "1"})
    plt.figure(figsize=(18, 10))
    for cluster in df[cluster_column].unique():
        plt.scatter(
            df[df[cluster_column] == cluster].x,
            df[df[cluster_column] == cluster].y,
            alpha=0.9,
            edgecolors=np.array([255, 255, 255]) / 255,
            color=colors[cluster],
        )
    plt.axis("off")
    # plt.legend(["Belief on wearing face mask","None respect of health measures","Covid19 transmission modes","covid19 transmission modes","mode of transmission","Observation on hand washing","Covid19 does not exist in this area","Covid19 is business","Covid19 is finished","Covid19 is finished"])
    plt.legend(
        [
            "Beliefs about wearing face masks",
            "People not respecting health measures",
            "Belief that Covid exists or is real",
            "Beliefs about hand washing",
            "Covid is terminated",
            "Covid transmission modes",
            "Covid transmission modes",
            "Covid19 does not exist in this area",
            "Belief that institutions make money from Covid",
        ],
        loc=("lower right"),
        fontsize=10,
    )
    plt.title("Clusters formed from the community detection algorithm", size=20)
    if file_path != None:
        name = str(time()) + ".png"
        name2 = str(time()) + ".svg"
        plt.savefig(f"{file_path}/png/{name}")
        plt.savefig(f"{file_path}/svg/{name2}")
    plt.show()


# %%
[
    "Belief on wearing face mask",
    "None respect of health measures",
    "Belief that covid19 exists or is real",
    "4 Belief on hand washing ",
    "3 Covid19 is terminated",
    "8 Covid19 transmission modes",
    "2 covid19 transmission modes",
    "5 Covid19 does not exist in this area",
    "6 Belief that institutions make money from covid19",
]

# %%
print(model_data[model_data.consensus_g == 3]["comment"].to_list())

# %%
model_data_viz = model_data[model_data.consensus_g < 9].copy()

# %%
model_data_viz.consensus_g.unique()

# %%
plt.savefig(f"{project_directory}/outputs/figures/svg/missing_values_by_attribute.svg")
plt.savefig(f"{project_directory}/outputs/figures/png/missing_values_by_attribute.png")

# %%
path = str(project_directory) + "/outputs/figures/"

# %%

# %%
plot_clusters2(model_data_viz, "consensus_g", file_path=path)
plt.savefig("clusters.svg")

# %%
model_data_viz["consensus_label"] = [
    "Beliefs about wearing face masks"
    if x == 1
    else "People not respecting health measures"
    if x == 7
    else "Belief that Covid exists or is real"
    if x == 0
    else "Beliefs about hand washing"
    if x == 4
    else "Covid is terminated"
    if x == 3
    else "Covid transmission modes"
    if x == 8
    else "Covid transmission modes"
    if x == 2
    else "Covid19 does not exist in this area"
    if x == 5
    else "Belief that institutions make money from Covid"
    for x in model_data_viz.consensus_g
]


# %%
model_data_viz[model_data_viz.consensus_g == 0]["consensus_label"]

# %%
model_data_viz.columns

# %%
nn = model_data_viz.groupby(["category_id", "consensus_label"]).size()
# nn= np.asarray(nn).reshape(67,1)


# %%
plot_df = nn.reset_index().pivot(index="category_id", columns="consensus_label")


# %%
plt.figure(figsize=(10, 10))
sns.heatmap(plot_df)

# %%
colors = generate_colors(len(model_data.consensus.unique()))

# %%
import seaborn as sns
from time import time
import random

sns.set_style("white", {"axes.facecolor": "1"})
plt.figure(figsize=(18, 10))
for cluster in model_data.consensus.unique():
    label = model_data[model_data.consensus == cluster].category_id.to_list()[0]
    plt.scatter(
        model_data[model_data.consensus == cluster].x,
        model_data[model_data.consensus == cluster].y,
        alpha=0.9,
        edgecolors=np.array([255, 255, 255]) / 255,
        label=label,
        color=colors[cluster],
    )
plt.axis("off")
# plt.legend(loc=(1.01,0.4))
plt.title(
    "Clusters formed from community detection algorithm using consensus clustering"
)
plt.show()

# %%
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
# fig = (
#    alt.Chart(model_data.reset_index(), width=725, height=725)
#    .mark_circle(size=60)
#    .encode(x="x", y="y", tooltip=["consensus", "consensus_g"], color="cluster:N")
# ).interactive()

# fig


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

# %%
# create a workbook and store the resulting clusters in it. Each cluster in a separate worksheet.
with xlsxwriter.Workbook(f"{project_directory}/inputs/data/clusters.xlsx") as workbook:
    for cluster in model_data.consensus_g.unique():
        community = model_data[model_data.consensus_g == cluster].comment.to_list()
        worksheet = workbook.add_worksheet()
        for i in range(len(community)):
            j = i + 1
            ex_col = "A" + str(j)
            worksheet.write(ex_col, community[i])
