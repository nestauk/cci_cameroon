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
import xlsxwriter

# %%
project_directory = cci_cameroon.PROJECT_DIR


# %%
# load unclassified comments comming in from the classification model
model_data = pd.read_excel(f"{project_directory}/outputs/data/not_classified.xlsx")
column_name = "comment"  # holds the column of interest in the data

# %%
# initialize the french_semantic model for word embedding
model = SentenceTransformer("Sahajtomar/french_semantic")

# %%
# create word embeddings using the transformer model
sentence_embeddings = model.encode(model_data[column_name])

# %%
indp2 = faiss.index_factory(
    sentence_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
)
faiss.normalize_L2(sentence_embeddings)  # to be used for inner product computation
# setting the number of neighbors to consider for graph connectivity
neighbors2 = 5
indp2.train(sentence_embeddings)
indp2.add(sentence_embeddings)
similarity_distance, sim_positions = indp2.search(
    sentence_embeddings, neighbors2  # consider only the first n elements
)

# %%
weight_mat = generate_adjacency_matrix2(
    sim_positions, neighbors2, sentence_embeddings.shape[0], similarity_distance
)
G = nx.from_numpy_matrix(weight_mat, create_using=nx.Graph(), parallel_edges=False)
comu = algorithms.leiden(G)
labels_best = []
labels_com = generate_community_labels(comu.communities, sentence_embeddings.shape[0])
sil_score = silhouette_samples(sentence_embeddings, labels_com, metric="cosine")
av, _ = compute_community_silhuoette_scores(
    comu.communities, sil_score
)  # the average silhouette_score
reduced_av = av - (0.9 * av)
retained_communities, rejected_communities, statistics = get_communities_with_threshold(
    comu.communities, sil_score, reduced_av, model_data, column_name
)


# %%
retained_communities

# %%
rejected_communities

# %%
# create a workbook and store the resulting clusters in it. Each cluster in a separate worksheet.
with xlsxwriter.Workbook(f"{project_directory}/outputs/data/clusters.xlsx") as workbook:
    for community in retained_communities:
        worksheet = workbook.add_worksheet()
        for i in range(len(community)):
            j = i + 1
            ex_col = "A" + str(j)
            worksheet.write(ex_col, community[i])

# %%
