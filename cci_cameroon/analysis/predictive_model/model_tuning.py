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
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import os
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
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

# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR


# %%
def perform_grid_search(pipe, score, parameter_grid):
    """
    Setting parameters for GridSearchCV.
    """
    search = GridSearchCV(
        estimator=pipe,
        param_grid=parameter_grid,
        n_jobs=-1,
        scoring=score,
        cv=10,
        refit=True,
        verbose=3,
    )
    return search


# %%
def save_cm_plots(cm, model_type):
    """
    Save cm plot for each class for chosen model type. Note: model type needs to match folder name in outputs/figures.
    """
    # Loop through codes and save cm plot to outputs/figures sub-folder.
    for i in range(0, len(codes)):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i])
        disp.plot()
        plt.title(codes[i].replace("_", " "), pad=20)
        plt.tight_layout()
        plt.savefig(
            f"{project_directory}/outputs/figures/predictive_models/cm/"
            + model_type
            + "/"
            + codes[i].replace("/", "")
            + "_cm.png",
            bbox_inches="tight",
        )


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
no_response_train = pd.read_excel(
    f"{project_directory}/outputs/data/data_for_modelling/no_response_train.xlsx",
    index_col="id",
)["comment"]

no_response_test = pd.read_excel(
    f"{project_directory}/outputs/data/data_for_modelling/no_response_test.xlsx",
    index_col="id",
)["comment"]

# %%
no_response_train

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
# Combine files and re-shuffle
X_train.head(1)

# %%
from sklearn.utils import shuffle

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
y_train.shape

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
search_knn = perform_grid_search(pipe_knn, "f1_micro", param_grid_knn)
search_rf = perform_grid_search(pipe_rf, "f1_micro", param_grid_rf)
search_dt = perform_grid_search(pipe_dt, "f1_micro", param_grid_dt)
search_nb = perform_grid_search(pipe_nb, "f1_micro", param_grid_nb)
search_svm = perform_grid_search(pipe_svm, "f1_micro", param_grid_svm)

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

# %% [markdown]
# ### Without extra comments

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

# %% [markdown]
# ### With extra comments added

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

# %% [markdown]
# ### Predict on test set

# %%
y_pred_svm = search_svm.best_estimator_.predict(X_test_embeddings_fr)
y_pred_knn = search_knn.best_estimator_.predict(X_test_embeddings_fr)
y_pred_rf = search_rf.best_estimator_.predict(X_test_embeddings_fr)
y_pred_dt = search_dt.best_estimator_.predict(X_test_embeddings_fr)
y_pred_nb = search_nb.best_estimator_.predict(X_test_embeddings_fr)

# %%
y_test_update = []
for item in y_test:
    if item.sum() == 0:
        item = np.append(item, 1)
    else:
        item = np.append(item, 0)
    y_test_update.append(list(item))

y_test_update = np.asarray(y_test_update)

# %%
y_pred_update = []
for item in y_pred_svm:
    if item.sum() == 0:
        item = np.append(item, 1)
    else:
        item = np.append(item, 0)
    y_pred_update.append(list(item))

y_pred_update = np.asarray(y_pred_update)

# %%
y_pred_update

# %%
cm_svm = multilabel_confusion_matrix(y_test_update, y_pred_update)

# %%
cm_svm

# %%
# Create confusion matrix from the best performing models
cm_svm = multilabel_confusion_matrix(y_test, y_pred_svm)
cm_knn = multilabel_confusion_matrix(y_test, y_pred_knn)
cm_rf = multilabel_confusion_matrix(y_test, y_pred_rf)
cm_dt = multilabel_confusion_matrix(y_test, y_pred_dt)
cm_nb = multilabel_confusion_matrix(y_test, y_pred_nb)

# %%
# %%capture
save_cm_plots(cm_svm, "svm")
save_cm_plots(cm_knn, "knn")
save_cm_plots(cm_rf, "random_forest")
save_cm_plots(cm_dt, "decision_tree")
save_cm_plots(cm_nb, "naive_bayes")

# %% [markdown]
# ### Important words / features

# %% [markdown]
# Most common words in each prediction...

# %%
pred_proba = search_knn.best_estimator_.predict_proba(X_train_embeddings_fr)

# %%
code_lists = []
for preds in pred_proba:
    code_list = list(pd.DataFrame(preds)[1])
    code_lists.append(code_list)

preds_df = pd.DataFrame(np.column_stack(code_lists), columns=codes)

# %%
preds_df["comment"] = X_train.reset_index(drop=True)

# %%
preds_df.head(1)

# %%
from nltk.corpus import stopwords
from itertools import chain
from textwrap import wrap

stop = stopwords.words("french")

# %%
counts = []
for code in codes:
    words = chain.from_iterable(
        line.split() for line in preds_df[preds_df[code] >= 0.5]["comment"].str.lower()
    )
    count = Counter(word for word in words if word not in stop)
    counts.append(count)

# %%
print(codes[7].replace("_", " "))

# %%
y = [count for tag, count in counts[7].most_common(20)]
x = [tag for tag, count in counts[7].most_common(20)]

plt.bar(x, y, color="crimson")
title = "Term frequencies: " + codes[7].replace("_", " ")
plt.title("\n".join(wrap(title, 60)), fontsize=14, pad=10)
plt.ylabel("Frequency")
plt.xticks(rotation=90)
for i, (tag, count) in enumerate(counts[7].most_common(20)):
    plt.text(
        i,
        count,
        f" {count} ",
        rotation=90,
        ha="center",
        va="top" if i < 10 else "bottom",
        color="white" if i < 10 else "black",
    )
plt.tight_layout()  # change the whitespace such that all labels fit nicely
plt.show()
