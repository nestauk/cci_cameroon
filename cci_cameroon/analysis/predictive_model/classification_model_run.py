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

# %%
# Read in libraries
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import os
from sklearn.metrics import multilabel_confusion_matrix
from ast import literal_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
import os.path

# Project modules
import cci_cameroon
from cci_cameroon.pipeline import process_workshop_data as pwd
from cci_cameroon.pipeline import model_tuning_report as mtr

# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %%
# File paths for saving
pred_file = (
    f"{project_directory}/outputs/data/classification_predictions/all_predictions.xlsx"
)
no_class_file = (
    f"{project_directory}/outputs/data/classification_predictions/not_classified.xlsx"
)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
stop = stopwords.words("french")

# %%
# Read test data
X_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/X_test.xlsx", index_col="id"
)["comment"]
y_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_test.xlsx", index_col="id"
)["category_id"]

# %%
# Data 'no response'
no_response_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/no_response_test.xlsx",
    index_col="id",
)["comment"]

# %%
# No reponse - all zeros
y_nr_test = np.array([[0] * 8] * 30)

# %%
# Transform y_test to be readable by binarizer
y_test = y_test.apply(literal_eval)

# %%
# Transform Y into multilabel format
mlb = MultiLabelBinarizer()
y_test = mlb.transform(y_test)
codes = list(mlb.classes_)  # Codes list

# %%
# Updating set to include no responses and shuffling
y_test = np.concatenate((y_test, y_nr_test))
X_test = X_test.append(no_response_test, ignore_index=False)
X_test, y_test = shuffle(X_test, y_test, random_state=1)

# %%
# Get language model
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model
X_test_embeddings_fr = model_fr.encode(list(X_test))

# %%
# Loading KNN model
knn_model = pickle.load(
    open(f"{project_directory}/outputs/models/final_classification_model.sav", "rb")
)

# %%
# predict on test set
y_pred_knn = knn_model.predict(X_test_embeddings_fr)

# %%
# Add 'no prediction as a class'
y_test = mtr.add_y_class(y_test)
y_pred_knn = mtr.add_y_class(y_pred_knn)
code_cols = [word.replace("_", " ") for word in codes]
code_cols.append("Not classified")

# %%
# Create dfs for all predictions and 'no class predictions'
predictions = pd.DataFrame(y_pred_knn, columns=code_cols)
predictions["comment"] = X_test.reset_index(drop=True)
predictions["id"] = X_test.reset_index()["id"]
not_classified = predictions[predictions["Not classified"] == 1][["id", "comment"]]

# %%
# Checks is files exist, if they do append data (removing duplicates). Otherwise save data to new file.
if os.path.isfile(pred_file):
    predict_df = pd.read_excel(pred_file)
    predict_update = (
        pd.concat([predict_df, predictions]).drop_duplicates().reset_index(drop=True)
    )
    predict_update.to_excel(pred_file, index=False)
else:
    predictions.to_excel(pred_file, index=False)

if os.path.isfile(no_class_file):
    not_class_df = pd.read_excel(no_class_file)
    no_class_update = (
        pd.concat([not_class_df, not_classified])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    no_class_update.to_excel(no_class_file, index=False)
else:
    not_classified.to_excel(no_class_file, index=False)
