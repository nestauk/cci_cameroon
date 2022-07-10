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

# %% [markdown]
# ## CCI Cameroon model run

# %% [markdown]
# The purpose of this script is to run the classification and the clustering models on test data to demonstrate how they work. There are two parts:
#
# 1. The classification model:
#     This classifies comments into one of eight known categories. Unmatched comments(those not belonging to any of the eight categories) are passed onto the clustering algorithm.
# 2. The clustering algorithm groups the comments into clusters based on similarity among them. These are then saved into an excel sheet for review.
#
# ![overall system](../../../outputs/figures/readme/overall_system.png)
#
#

# %%
# Read in libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
import os
from ast import literal_eval
import pickle
import os.path

# Project modules
import cci_cameroon
from cci_cameroon.pipeline import model_tuning_report as mtr

# for clustering
import matplotlib.pyplot as plt
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
from sklearn.metrics import silhouette_samples, silhouette_score, f1_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import community
import xlsxwriter


# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# File paths for saving
pred_file = f"{project_directory}/outputs/data/all_predictions.xlsx"
no_class_file = f"{project_directory}/outputs/data/not_classified.xlsx"

# %%
# Read test data
X_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/X_test.xlsx", index_col="id"
)["comment"]
# Data 'no response'
no_response_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/no_response_test.xlsx",
    index_col="id",
)["comment"]
y_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_test.xlsx", index_col="id"
)["category_id"]

# %%
y_test = y_test.apply(literal_eval)

# %%
# Load binarizer object fitted on training set
with open(f"{project_directory}/outputs/models/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# %%
# Updating set to include no responses and shuffling
# X_test = X_test.append(no_response_test, ignore_index=False)
# X_test = shuffle(X_test, random_state=1)
X_sample = X_test.reset_index().copy()

# %%
# Get language model
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model
X_test_embeddings_fr = model_fr.encode(list(X_sample.comment[:30]))

# %%
# Loading KNN model (best performing)
knn_model = pickle.load(
    open(f"{project_directory}/outputs/models/final_classification_model.sav", "rb")
)

# %%
# predict on test set
y_pred_knn = knn_model.predict(X_test_embeddings_fr)

# %%
# predictions and not_classified dataframes created
codes = list(mlb.classes_)  # Codes list
predictions, not_classified = mtr.create_pred_dfs(y_pred_knn, codes, X_test[:30])

# %%
# Checks is files exist, if they do append data (removing duplicates). Otherwise save data to new file.
mtr.save_predictions(pred_file, predictions, no_class_file, not_classified)

# %%
y = mlb.transform(y_test)


# %%
f1_score(y[:30], y_pred_knn, average="macro") * 100


# %%
def predict_rumour():
    """For the demo: runs best model knn model on new data and outputs results."""
    new_rumour = input("Please enter a new rumour: ")
    test_text = model_fr.encode([new_rumour])
    preds = mlb.inverse_transform(knn_model.predict(test_text))
    pred_proba = knn_model.predict_proba(test_text)
    if preds[0]:
        for pred in preds:
            for p in list(pred):
                code = p.replace("_", " ")
                print("Thanks for submitting a rumour!")
                print(" ")
                print(
                    "The model predicts your rumour "
                    + '"'
                    + new_rumour
                    + '"'
                    + " is related to: "
                )
                print(code)
    else:
        print(
            'The model could not match your input"',
            new_rumour,
            '" with an existing category.',
        )


# %% [markdown]
# ## Reading sample input from the user and classifying it

# %%
predict_rumour()

# %% [markdown]
# # clustering of new comments

# %%
# load unclassified comments comming in from the classification model
model_data = X_test.copy()
column_name = "comment"  # holds the column of interest in the data

# %%
model_data = model_data.reset_index()

# %%
# create word embeddings using the transformer model
sentence_embeddings = model_fr.encode(model_data[column_name])

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
draw_communities_graph(G, generate_colors(len(comu.communities)), comu.communities)

# %%
retained_communities[0]

# %%
