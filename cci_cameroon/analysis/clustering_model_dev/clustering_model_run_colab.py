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

# %% [markdown]
# ## Clustering model run (Google colab version)
#
# Use this notebook to run the clustering_model_run notebook on google colab.
#
# Steps to take to run:
#
# 1. Upload this notebook to Google Colab
# 2. Upload the following files to google drive (press the upload icon in the 'Files' section to the left of the notebook. The first is found in the `cci_cameroon/pipeline` section of the repository and contains functions needed to run the code. The second is the dataset of unclassified rumours created by running `classification_model_run.py`.
#   - `clustering_helper_function.py`
#   - `not_classfied.xlsx`
# 3. Uncomment the `pip install` lines in the first cell
#
# The output of this script is the `clusters.xlsx` file which contains the clusters of rumours created by the algorthm.

# %%
# #!pip install cdlib
# #!pip install faiss-cpu --no-cache
# #!pip install sentence-transformers
# #!pip install xlsxwriter
# #!pip install leidenalg

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cdlib import evaluation

# %matplotlib inline
import faiss
from faiss import normalize_L2
from sentence_transformers import SentenceTransformer
import networkx as nx
import cdlib
from cdlib import algorithms
from clustering_helper_functions import (
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
with xlsxwriter.Workbook("clusters.xlsx") as workbook:
    for community in retained_communities:
        worksheet = workbook.add_worksheet()
        for i in range(len(community)):
            j = i + 1
            ex_col = "A" + str(j)
            worksheet.write(ex_col, community[i])
