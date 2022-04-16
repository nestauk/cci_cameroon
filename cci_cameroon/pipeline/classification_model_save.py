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
stop = stopwords.words("french")

# %%
# Read train/test data
X_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/X_train.xlsx", index_col="id"
)["comment"]
y_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_train.xlsx", index_col="id"
)["category_id"]

# %%
# Data to use to train 'no response'
no_response_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/no_response_train.xlsx",
    index_col="id",
)["comment"]

# %%
# No reponse - all zeros
y_nr_train = np.array([[0] * 8] * 120)

# %%
y_train = y_train.apply(literal_eval)

# %%
# Transform Y into multilabel format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)

# %%
y_train = np.concatenate((y_train, y_nr_train))

# %%
X_train = X_train.append(no_response_train, ignore_index=False)

# %%
X_train, y_train = shuffle(X_train, y_train, random_state=1)

# %%
# Get language model
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model
# Encode train transform into word embeddings
X_train_embeddings_fr = model_fr.encode(list(X_train))

# %%
# Fit best performing model
knn = KNeighborsClassifier(n_neighbors=5, p=1, weights="distance")
knn.fit(X_train_embeddings_fr, y_train)

# %%
# save the best model to disk
filename = f"{project_directory}/outputs/models/final_classification_model.sav"
pickle.dump(knn, open(filename, "wb"))
