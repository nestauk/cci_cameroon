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
# ## Classifcation model save

# %% [markdown]
# This script refits the best performing model type and parameters found in the 'classification_model_development' script on the training set and saves the model to disk. This model is then used in the 'model run' folder to run the model on new data (the test set by default).
#
# As well as the training set comprrised of rumours assigned to the eight codes the model is trying to predict, a 'no response' dataset* is also included in the model training. This uses a random set of rumours that don't belong to the eight codes that are assigned as 'no response' (all zeros in the multi-label array). The reason for using this is so the model can be better trained to predict cases that aren't assigned to any of the eight codes.
#
# *See the 'data_splitting' file in the pipeline folder for the creation of this dataset.

# %%
# Read in libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import os
from ast import literal_eval
from sklearn.neighbors import KNeighborsClassifier
import pickle
from pathlib import Path

# Project modules
import cci_cameroon
from cci_cameroon import config

# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Get parameters from config file
knn_nn = config["knn_model_params"]["nearest_neighbours"]
knn_p = config["knn_model_params"]["p"]
knn_weights = config["knn_model_params"]["weights"]

# %%
# Read train/test data
X_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/X_train.xlsx", index_col="id"
)["comment"]
y_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_train.xlsx", index_col="id"
)["category_id"]
# Transform y_train to be readable by binarizer
y_train = y_train.apply(literal_eval)

# Data to use to train 'no response'
no_response_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/no_response_train.xlsx",
    index_col="id",
)["comment"]

# %%
# Transform Y into multilabel format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
# No reponse - all zeros array
y_nr_train = np.array([[0] * 8] * 120)

# Combine training sets with no response training sets
y_train = np.concatenate((y_train, y_nr_train))
X_train = X_train.append(no_response_train, ignore_index=False)

# Shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=1)

# %%
# Get language model
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model
# Encode train transform into word embeddings
X_train_embeddings_fr = model_fr.encode(list(X_train))

# %%
# Fit best performing model (KNN)
knn = KNeighborsClassifier(n_neighbors=knn_nn, p=knn_p, weights=knn_weights)
knn.fit(X_train_embeddings_fr, y_train)

# %%
# Add folder if not already created
Path(f"{project_directory}/outputs/models/").mkdir(parents=True, exist_ok=True)

# %%
# save model and binarizer to disk
with open(f"{project_directory}/outputs/models/mlb.pkl", "wb") as f:
    pickle.dump((mlb), f)

filename = f"{project_directory}/outputs/models/final_classification_model.sav"
pickle.dump(knn, open(filename, "wb"))
