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

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import cci_cameroon
from collections import Counter
from itertools import chain
from textwrap import wrap
import pandas as pd
import os

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
def save_cm_plots(cm, model_type, codes):
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
def word_counts_class(codes, pred_proba, X_train, stop):
    """
    Get count of words per predicted class (after removing stop words).
    """
    code_lists = []
    for preds in pred_proba:
        code_list = list(pd.DataFrame(preds)[1])
        code_lists.append(code_list)
    preds_df = pd.DataFrame(np.column_stack(code_lists), columns=codes)
    preds_df["comment"] = X_train.reset_index(drop=True)
    counts = []
    for code in codes:
        words = chain.from_iterable(
            line.split()
            for line in preds_df[preds_df[code] >= 0.5]["comment"].str.lower()
        )
        count = Counter(word for word in words if word not in stop)
        counts.append(count)
    return counts


# %%
def common_words_plots(codes, counts):
    """
    Plots of most common words per predicted class.
    """
    # Loop through codes and save cm plot to outputs/figures sub-folder.
    for i in range(0, len(codes)):
        y = [count for tag, count in counts[i].most_common(20)]
        x = [tag for tag, count in counts[i].most_common(20)]
        plt.figure()
        plt.bar(x, y, color="crimson")
        title = "Term frequencies: " + codes[i].replace("_", " ")
        plt.title("\n".join(wrap(title, 60)), fontsize=14, pad=10)
        plt.ylabel("Frequency")
        plt.xticks(rotation=90)
        for k, (tag, count) in enumerate(counts[i].most_common(20)):
            plt.text(
                k,
                count,
                f" {count} ",
                rotation=90,
                ha="center",
                va="top" if k < 10 else "bottom",
                color="white" if k < 10 else "black",
            )
        plt.tight_layout()  # change the whitespace such that all labels fit nicely
        plt.savefig(
            f"{project_directory}/outputs/figures/predictive_models/common_words/"
            + codes[i].replace("/", "")
            + "_cw.png",
            bbox_inches="tight",
        )


# %%
def add_y_class(y_set):
    """
    Adding class for case of 'no response' to measure accuracy with confusion matrixes.
    """
    y_set_update = []
    for item in y_set:
        if item.sum() == 0:
            item = np.append(item, 1)
        else:
            item = np.append(item, 0)
        y_set_update.append(list(item))
    y_set_update = np.asarray(y_set_update)
    return y_set_update


# %%
def create_pred_dfs(y_pred_knn, codes, X_test):
    """
    Create dfs for predicted classes and unclassified comments from predicted rumours run on the model.
    """
    # Add 'no prediction as a class'
    # y_test = add_y_class(y_test)
    y_pred_knn = add_y_class(y_pred_knn)
    code_cols = [word.replace("_", " ") for word in codes]
    code_cols.append("Not classified")
    # Create dfs for all predictions and 'no class predictions'
    predictions = pd.DataFrame(y_pred_knn, columns=code_cols)
    predictions["comment"] = X_test.reset_index(drop=True)
    predictions["id"] = X_test.reset_index()["id"]
    not_classified = predictions[predictions["Not classified"] == 1][["id", "comment"]]
    return predictions, not_classified


# %%
def save_predictions(pred_file, predictions, no_class_file, not_classified):
    """
    Function to save prediction outputs from classification model.
    Checks is files exist, if they do append data (removing duplicates). Otherwise save data to new file.
    """
    if os.path.isfile(pred_file):
        predict_df = pd.read_excel(pred_file)
        predict_update = (
            pd.concat([predict_df, predictions])
            .drop_duplicates()
            .reset_index(drop=True)
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


# %%
