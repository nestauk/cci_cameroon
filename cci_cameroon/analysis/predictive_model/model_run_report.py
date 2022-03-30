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
# ## Model running and report results

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
X_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/X_test.xlsx", index_col="id"
)["comment"]
y_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_train.xlsx", index_col="id"
)["category_id"]
y_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_test.xlsx", index_col="id"
)["category_id"]

# %%
# Data to use to train 'no response'
no_response_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/no_response_train.xlsx",
    index_col="id",
)["comment"]

no_response_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/no_response_test.xlsx",
    index_col="id",
)["comment"]

# %%
# No reponse - all zeros
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
# Get language model
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model
# Encode train and test transform into word embeddings
X_train_embeddings_fr = model_fr.encode(list(X_train))
X_test_embeddings_fr = model_fr.encode(list(X_test))

# %% [markdown]
# ### Fit and predict with best performing models

# %%
# Best performing models
svm = MultiOutputClassifier(SVC(C=3, gamma="scale"), n_jobs=-1)
knn = KNeighborsClassifier(n_neighbors=5, p=1, weights="distance")
rf = RandomForestClassifier(n_estimators=200, random_state=1)
dt = DecisionTreeClassifier(criterion="entropy", random_state=1)
nb = MultiOutputClassifier(GaussianNB(), n_jobs=-1)

# %%
# Fit and predict on test set
svm.fit(X_train_embeddings_fr, y_train)
y_pred_svm = svm.predict(X_test_embeddings_fr)
knn.fit(X_train_embeddings_fr, y_train)
y_pred_knn = knn.predict(X_test_embeddings_fr)
rf.fit(X_train_embeddings_fr, y_train)
y_pred_rf = rf.predict(X_test_embeddings_fr)
dt.fit(X_train_embeddings_fr, y_train)
y_pred_dt = dt.predict(X_test_embeddings_fr)
nb.fit(X_train_embeddings_fr, y_train)
y_pred_nb = nb.predict(X_test_embeddings_fr)

# %%
# Update test set and predictions to include class for 'no class'
y_test = mtr.add_y_class(y_test)
y_pred_svm = mtr.add_y_class(y_pred_svm)
y_pred_knn = mtr.add_y_class(y_pred_knn)
y_pred_rf = mtr.add_y_class(y_pred_rf)
y_pred_dt = mtr.add_y_class(y_pred_dt)
y_pred_nb = mtr.add_y_class(y_pred_nb)

# %%
codes.append("Not classified as any of the eight codes")

# %%
# Create confusion matrix from the best performing models
cm_svm = multilabel_confusion_matrix(y_test, y_pred_svm)
cm_knn = multilabel_confusion_matrix(y_test, y_pred_knn)
cm_rf = multilabel_confusion_matrix(y_test, y_pred_rf)
cm_dt = multilabel_confusion_matrix(y_test, y_pred_dt)
cm_nb = multilabel_confusion_matrix(y_test, y_pred_nb)

# %%
# %%capture
mtr.save_cm_plots(cm_svm, "svm", codes)
mtr.save_cm_plots(cm_knn, "knn", codes)
mtr.save_cm_plots(cm_rf, "random_forest", codes)
mtr.save_cm_plots(cm_dt, "decision_tree", codes)
mtr.save_cm_plots(cm_nb, "naive_bayes", codes)

# %% [markdown]
# ### Common words in each predicted class
# Using the predictions from the KNN model looking at the most common words in each predicted class and saving the results as bar charts.

# %%
# Remove last class 'no code'
codes = codes[:-1]

# %%
pred_proba = knn.predict_proba(X_train_embeddings_fr)

# %%
counts = mtr.word_counts_class(codes, pred_proba, X_train, stop)

# %%
mtr.common_words_plots(codes, counts)

# %%
