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
