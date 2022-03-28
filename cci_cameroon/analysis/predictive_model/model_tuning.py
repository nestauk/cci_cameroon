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
# ## Model tuning

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
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from collections import Counter

# Project modules
import cci_cameroon
from cci_cameroon.pipeline import process_workshop_data as pwd
from cci_cameroon.pipeline import model_tuning_report as mtr

# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Read train/test data
X_train = pd.read_excel(
    f"{project_directory}/outputs/data/data_for_modelling/X_train.xlsx", index_col="id"
)["comment"]
X_test = pd.read_excel(
    f"{project_directory}/outputs/data/data_for_modelling/X_test.xlsx", index_col="id"
)["comment"]
y_train = pd.read_excel(
    f"{project_directory}/outputs/data/data_for_modelling/y_train.xlsx", index_col="id"
)["category_id"]
y_test = pd.read_excel(
    f"{project_directory}/outputs/data/data_for_modelling/y_test.xlsx", index_col="id"
)["category_id"]

# %%
# Data to use to train 'no response'
no_response_train = pd.read_excel(
    f"{project_directory}/outputs/data/data_for_modelling/no_response_train.xlsx",
    index_col="id",
)["comment"]

no_response_test = pd.read_excel(
    f"{project_directory}/outputs/data/data_for_modelling/no_response_test.xlsx",
    index_col="id",
)["comment"]

# %%
y_nr_train = np.array([[0] * 8] * 120)
y_nr_test = np.array([[0] * 8] * 30)

# %%
y_train = y_train.apply(literal_eval)
y_test = y_test.apply(literal_eval)

# %%
# Transform Y into multilabel format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)
codes = list(mlb.classes_)  # Codes list

# %%
y_train = np.concatenate((y_train, y_nr_train))
y_test = np.concatenate((y_test, y_nr_test))

# %%
X_train = X_train.append(no_response_train, ignore_index=False)
X_test = X_test.append(no_response_test, ignore_index=False)

# %%
X_train, y_train = shuffle(X_train, y_train, random_state=1)
X_test, y_test = shuffle(X_test, y_test, random_state=1)

# %%
# Get language models
model = SentenceTransformer(
    "distiluse-base-multilingual-cased-v1"
)  # multi-langauge model that supports French
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model

# %%
# Encode train and test transform into word embeddings
X_train_embeddings = model.encode(list(X_train))
X_test_embeddings = model.encode(list(X_test))
X_train_embeddings_fr = model_fr.encode(list(X_train))
X_test_embeddings_fr = model_fr.encode(list(X_test))

# %%
# Models
knn = KNeighborsClassifier(n_neighbors=7)
rf = RandomForestClassifier(random_state=1)
dt = DecisionTreeClassifier(random_state=1)
nb = MultiOutputClassifier(GaussianNB(), n_jobs=-1)
svm = MultiOutputClassifier(SVC(), n_jobs=-1)

# %%
# Pipelines
pipe_knn = Pipeline(
    steps=[
        ("knn", knn),
    ]
)
pipe_rf = Pipeline(
    steps=[
        ("rf", rf),
    ]
)
pipe_dt = Pipeline(
    steps=[
        ("dt", dt),
    ]
)
pipe_nb = Pipeline(
    steps=[
        ("nb", nb),
    ]
)
pipe_svm = Pipeline(
    steps=[
        ("svm", svm),
    ]
)

# %%
# Parameter grids
param_grid_knn = {
    "knn__n_neighbors": [2, 5, 10, 50],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}

param_grid_rf = {"rf__n_estimators": [10, 50, 100, 200]}

param_grid_dt = {"dt__criterion": ["gini", "entropy"]}

param_grid_nb = {}

param_grid_svm = {
    "svm__estimator__C": [1, 2, 3, 4],
    "svm__estimator__gamma": ["scale", "auto"],
}

# %%
# Perform gridsearch
search_knn = mtr.perform_grid_search(pipe_knn, "f1_micro", param_grid_knn)
search_rf = mtr.perform_grid_search(pipe_rf, "f1_micro", param_grid_rf)
search_dt = mtr.perform_grid_search(pipe_dt, "f1_micro", param_grid_dt)
search_nb = mtr.perform_grid_search(pipe_nb, "f1_micro", param_grid_nb)
search_svm = mtr.perform_grid_search(pipe_svm, "f1_micro", param_grid_svm)

# %%
# %%capture
# Fit to train and get optimum score and parameters
search_knn.fit(X_train_embeddings, y_train)
best_score_knn = search_knn.best_score_
best_params_knn = search_knn.best_params_
search_knn.fit(X_train_embeddings_fr, y_train)
best_score_fr_knn = search_knn.best_score_
best_params_fr_knn = search_knn.best_params_

# %%
# %%capture
# Fit to train and get optimum score and parameters
search_rf.fit(X_train_embeddings, y_train)
best_score_rf = search_rf.best_score_
best_params_rf = search_rf.best_params_
search_rf.fit(X_train_embeddings_fr, y_train)
best_score_fr_rf = search_rf.best_score_
best_params_fr_rf = search_rf.best_params_

# %%
# %%capture
# Fit to train and get optimum score and parameters
search_dt.fit(X_train_embeddings, y_train)
best_score_dt = search_dt.best_score_
best_params_dt = search_dt.best_params_
search_dt.fit(X_train_embeddings_fr, y_train)
best_score_fr_dt = search_dt.best_score_
best_params_fr_dt = search_dt.best_params_

# %%
# %%capture
# Fit to train and get optimum score and parameters
search_svm.fit(X_train_embeddings, y_train)
best_score_svm = search_svm.best_score_
best_params_svm = search_svm.best_params_
search_svm.fit(X_train_embeddings_fr, y_train)
best_score_fr_svm = search_svm.best_score_
best_params_fr_svm = search_svm.best_params_

# %%
# %%capture
# Fit to train and get optimum score and parameters
search_nb.fit(X_train_embeddings, y_train)
best_score_nb = search_nb.best_score_
best_params_nb = search_nb.best_params_
search_nb.fit(X_train_embeddings_fr, y_train)
best_score_fr_nb = search_nb.best_score_
best_params_fr_nb = search_nb.best_params_

# %%
## KNN
print("Best scores from multi-lingual model:")
print(best_score_knn)
print("Optimum parameters:")
print(best_params_knn)
print("Best scores from French language model:")
print(best_score_fr_knn)
print("Optimum parameters:")
print(best_params_fr_knn)

# %%
# Random Forest
print("Best scores from multi-lingual model:")
print(best_score_rf)
print("Optimum parameters:")
print(best_params_rf)
print("Best scores from French language model:")
print(best_score_fr_rf)
print("Optimum parameters:")
print(best_params_fr_rf)

# %%
## Decision Tree
print("Best scores from multi-lingual model:")
print(best_score_dt)
print("Optimum parameters:")
print(best_params_dt)
print("Best scores from French language model:")
print(best_score_fr_dt)
print("Optimum parameters:")
print(best_params_fr_dt)

# %%
## NB
print("Best scores from multi-lingual model:")
print(best_score_nb)
print("Optimum parameters:")
print(best_params_nb)
print("Best scores from French language model:")
print(best_score_fr_nb)
print("Optimum parameters:")
print(best_params_fr_nb)

# %%
## SVM
print("Best scores from multi-lingual model:")
print(best_score_svm)
print("Optimum parameters:")
print(best_params_svm)
print("Best scores from French language model:")
print(best_score_fr_svm)
print("Optimum parameters:")
print(best_params_fr_svm)

# %%
print("SVM best model:")
print(search_svm.best_estimator_)
print(best_params_fr_svm)
print("----")
print("KNN best model:")
print(search_knn.best_estimator_)
print(best_params_fr_knn)
print("----")
print("Random Forest best model:")
print(search_rf.best_estimator_)
print(best_params_fr_rf)
print("----")
print("Decision Tree best model:")
print(search_dt.best_estimator_)
print(best_params_fr_dt)
print("----")
print("Naive Bayes best model:")
print(search_nb.best_estimator_)
print(best_params_fr_nb)
