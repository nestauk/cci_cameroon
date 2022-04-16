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
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: cci_cameroon
#     language: python
#     name: cci_cameroon
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cci_cameroon
from langdetect import detect, detect_langs
from googletrans import Translator
from cdlib import evaluation
import logging

# %matplotlib inline
import faiss
from faiss import normalize_L2
from sentence_transformers import SentenceTransformer
import networkx as nx
import cdlib
from cdlib import algorithms
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
)
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import community
import matplotlib

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
# read cleaned data from inputs folder
to_use2 = pd.read_excel(f"{project_directory}/inputs/data/to_use2.xlsx")
model_data = pd.read_excel(f"{project_directory}/inputs/data/model_data.xlsx")
data = pd.read_excel(f"{project_directory}/inputs/data/data.xlsx")

# %%
model_data.shape

# %%
# checking the distribution of the codes in the data
data.groupby("first_code").comment.count().plot(kind="bar")
plt.savefig(
    f"{project_directory}/outputs/figures/clustering/distribution_of_three_codes.png"
)
plt.close()  # comment this out to see the proportions

# %%
# using the french_semantic model for word embedding
model = SentenceTransformer("Sahajtomar/french_semantic")
# model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

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
similarity_distance, sim_positions = indp2.search(
    sentence_embeddings2, neighbors2  # consider only the first n elements
)

# %%
edges, weighted_edges = generate_edges(
    to_use2, "comment", sim_positions, similarity_distance
)

# %%
nodes = list(set(to_use2.comment))

# %%
NG = nx.Graph()
NG.add_nodes_from(nodes)
NG.add_edges_from(edges)
NG.add_weighted_edges_from(weighted_edges)

# %%
# nx.draw(NG)

# %%
sentence_embeddings400 = model.encode(data.comment)

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
dt.drop(["level_0", "index"], axis=1, inplace=True)

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
    dt = pd.concat([dt, result[count]]).reset_index()
    dt.drop(["level_0", "index"], axis=1, inplace=True)
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
# ## We use a balanced sample with fewer codes and inspect the performance of the algorithm

# %%
# Using sample data of the same size but different codes.
df_m1 = model_data[model_data.category_id == "['Croyances_sur_les_masques_de_visage']"][
    :80
]
df_m2 = model_data[
    model_data.category_id == "['Observations_de_non-respect_des_mesures_de_santé']"
][:80]
df_m3 = model_data[
    model_data.category_id == "['Croyances_sur_les_moyens_de_transmission']"
][:80]
df_m = pd.concat([df_m1, df_m2, df_m3]).reset_index().drop("index", axis=1)

# %%
comment_lists_m, rejected_groups_m, df_st_m = get_metrics(df_m, "comment", model)

# %%
df_st_m

# %%
comment_lists_m

# %%
# rejected_groups_m

# %%
# comment_lists_m[0][3]    #1,1,1,1,1 (subgroups on mask and transmission mainly ) 93 comments of 240
# comment_lists_m[1][2] #1,1,1 (subgroups on mask(2) and transmission (1) only) 47 comments of 240
# comment_lists_m[2][1]  #1,0.971     (mask and transmission only)   78 comments of 240
comment_lists_m[3][1]  # 1,0.96        (transmission and mask) 65 comments of 240

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

partitions = community.best_partition(NG)

# %%
modularity = community.modularity(partitions, NG)
modularity

# %%
partitions

# %%
ncoms = algorithms.leiden(NG)

# %%
len(ncoms.communities)

# %%
ncoms.communities[0]

# %%
evaluation.modularity_density(NG, ncoms)

# %%
weight_mat = generate_adjacency_matrix2(
    sim_positions, neighbors2, sentence_embeddings2.shape[0], similarity_distance
)

# %%
GM = nx.from_numpy_matrix(
    weight_mat, create_using=nx.Graph(), parallel_edges=False
)  # parallel_edges set to false for weighted graph

# %%
# Use spring_layout to handle positioning of graph
layout = nx.spring_layout(GM)
# Draw the graph using the layout - with_labels=True if you want node labels.
nx.draw(GM, layout, with_labels=True)

# Get weights of each edge and assign to labels
labels = nx.get_edge_attributes(GM, "weight")

# Draw edge labels using layout and list of labels
nx.draw_networkx_edge_labels(GM, pos=layout, edge_labels=labels)

# Show plot
plt.close()  # comment this out to see the proportions

# %%
# applying a community algorithm to the graph to identify subgroups
comsM = algorithms.leiden(GM)
draw_communities_graph(GM, generate_colors(len(comsM.communities)), comsM.communities)

# %% [markdown]
#  **using a weighted graph doesn't cause significant change in the commun-ties formed**

# %%
evaluation.modularity_density(GM, comsM)

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
plt.savefig(f"{project_directory}/outputs/figures/clustering/graph_400.png")

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
labels2 = generate_community_labels(comsM.communities, sentence_embeddings2.shape[0])

# %%
sample_silhouette_scores = silhouette_samples(
    sentence_embeddings2, labels2, metric="cosine"
)

# %%
sample_silhouette_scores

# %%
print(
    get_communities_with_threshold(
        comsM.communities, sample_silhouette_scores, 0.3, data, "comment"
    )
)

# %% [markdown]
# ## Silhouette scores for the different communities computed and communities ranked
# * The data is sorted in order of silhoutte score and all clusters presented to be reviewed.

# %%
# add silhouette score column to the dataframe.
sil_scores = compute_community_silhuoette_scores(
    comsM.communities, sample_silhouette_scores
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
print(
    get_communities_with_threshold(
        comsM400.communities, sample_silhouette_scores400, 0.3, data, "comment"
    )
)

# %%
list(data.iloc[comsM400.communities[0]].comment)

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
print(
    get_communities_with_threshold(
        comsSamp.communities, silhouette_scores_samp, 0.2, sample_data, "comment"
    )
)

# %%
## inspecting the clusters formed

# %%
list(sample_data.iloc[comsSamp.communities[0]].comment)

# %%
Ncom400.communities[5]

# %%
indp2 = faiss.index_factory(
    sentence_embeddings2.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(sentence_embeddings2)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 10
indp2.train(sentence_embeddings2)
indp2.add(sentence_embeddings2)
distance_matric2, positions2 = indp2.search(sentence_embeddings2, neighbors2)

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

# %% [markdown]
# ## Using ground truth, we evaluate the homogeneity of the clusters formed

# %%
list(
    to_use2.category_id[coms22.communities[0]]
)  # inspecting the ground truth as per labelling exercise

# %%
ax = plt.figure(figsize=(10, 5))
to_use2.category_id[coms22.communities[0]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")
plt.savefig(f"{project_directory}/outputs/figures/code_distribution.png")
plt.close()  # comment this out to see the proportions

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
plt.close()  # comment this out to see the proportions

# %%
list(to_use2.comment.iloc[coms22.communities[2]])  # Belief that the disease exists

# %%
list(to_use2.category_id[coms22.communities[2]])

# %%
ax = plt.figure(figsize=(10, 5))
to_use2.category_id[coms22.communities[2]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")
plt.close()  # comment this out to see the proportions

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
plt.hist(list(matplotlib.cbook.flatten(distance_matric)), color="green")
plt.xlabel("Similarity score")
plt.ylabel("frequency")
plt.title("Distribution of similarity score for 1300 comments")
plt.savefig(
    f"{project_directory}/outputs/figures/clustering/silhouette_scores_distribution.png"
)
plt.close()  # comment this out to see the proportions

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
plt.close()  # comment this out to see the proportions

# %%
# using leiden community detection algorithm on the dataset
comss = algorithms.leiden(GG)

# %%
draw_communities_graph(GG, generate_colors(len(comss.communities)), comss.communities)

# %%
len(comss.communities)

# %% [markdown]
# ## Inspect groups formed when a network is generated with edges that are not weighted.  the model

# %%
list(data.iloc[comss.communities[0]].comment)

# %%
ax = plt.figure(figsize=(10, 5))
plt.title("Distribution of comments based on ground truth")
data.category_id[comss.communities[7]].value_counts().plot(kind="bar")
plt.close()  # comment this out to see the proportions

# %%
list(data.iloc[comss.communities[1]].comment)  # belief that disease exists

# %%
ax = plt.figure(figsize=(10, 5))
data.category_id[comss.communities[1]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")
plt.close()  # comment this out to see the proportions

# %%
list(data.iloc[comss.communities[2]].comment)  # belief about wearing face masks

# %%
ax = plt.figure(figsize=(10, 5))
data.category_id[comss.communities[2]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")
plt.close()  # comment this out to see the proportions

# %%
list(data.iloc[comss.communities[5]].comment)

# %%
ax = plt.figure(figsize=(10, 5))
data.category_id[comss.communities[5]].value_counts().plot(kind="bar")
plt.title("Distribution of comments based on ground truth")
plt.close()

# %%
# evaluating the model using modularity
modularity = evaluation.modularity_density(GG, comss)

# %%
modularity.score

# %% [markdown]
# ## Generating clusters using data with eight different codes obtained from workshop

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
plt.close()  # comment this line out to see the graph produced.

# %%
coms_f = algorithms.leiden(G_f)

# %%
logging.info(len(coms_f.communities))

# %%
draw_communities_graph(
    G_f, generate_colors(len(coms_f.communities)), coms_f.communities
)

# %%
# comments of the various communities can be inspected as in this example by changing the index
logging.info(list(model_data.iloc[coms_f.communities[0]].comment))

# %%
# view comments distribution in a group according to ground truth from labelling exercise
ax = plt.figure(figsize=(10, 5))
model_data.category_id[coms_f.communities[0]].value_counts().plot(
    kind="bar"
).figure.savefig("plot.png")
# plt.title("Distribution of comments based on ground truth")
plt.close()

# %%
list(model_data.iloc[coms_f.communities[1]].comment)

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
# ## Using walktrap community detection algorithms

# %%
## Using walktrap community detection algorithm on the data subset

# %%
comsw2 = algorithms.walktrap(GG2)

# %%
draw_communities_graph(
    GG2, generate_colors(len(comsw2.communities)), comsw2.communities
)
