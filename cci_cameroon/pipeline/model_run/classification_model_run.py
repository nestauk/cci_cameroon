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
# ## Classification model run

# %% [markdown]
# The purpose of this script is to run the classification model on new data (defaulted to the test set) and save the results to excel files. The resulting files are:
#
# - all_predictions.xlsx
# - not_classified.xlsx
#
# The classification model is loaded from a saved pretrained model which is created in the 'classfication_model_save' file in pipeline.
#
# If a rumour is classified by the model to one or more of the eight codes it is saved in the 'all_predictions' file. If the rumour cannot be classfied it is saved into the the 'not_classified' file. Both files also save the rumours 'ID' assigned so it can be referenced back to the test set for reporting.
#
# The 'not_classified' file is used as input to run the clustering algorthm in the 'clustering_model_run' file in the same folder.

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

# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# File paths for saving
pred_file = (
    f"{project_directory}/outputs/data/classification_predictions/all_predictions.xlsx"
)
no_class_file = (
    f"{project_directory}/outputs/data/classification_predictions/not_classified.xlsx"
)

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

# %%
# Load binarizer object fitted on training set
with open(f"{project_directory}/outputs/models/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# %%
# Updating set to include no responses and shuffling
X_test = X_test.append(no_response_test, ignore_index=False)
X_test = shuffle(X_test, random_state=1)

# %%
# Get language model
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model
X_test_embeddings_fr = model_fr.encode(list(X_test))

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
predictions, not_classified = mtr.create_pred_dfs(y_pred_knn, codes, X_test)

# %%
# Checks is files exist, if they do append data (removing duplicates). Otherwise save data to new file.
mtr.save_predictions(pred_file, predictions, no_class_file, not_classified)
