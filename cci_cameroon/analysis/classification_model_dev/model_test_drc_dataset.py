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
from textwrap3 import wrap

# Project modules
import cci_cameroon
from cci_cameroon.pipeline import model_tuning_report as mtr
from cci_cameroon.pipeline import process_workshop_data as pwd
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix

# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Update to add to data getters
drc_data = pd.read_excel(
    f"{project_directory}/inputs/data/COVID_19 Community feedback_DATA_ DRC_july2021_dec2021.xlsx",
    sheet_name="FEEDBACK DATA_DONNEES ",
)

# %%
# Names of columns assigned here
feedback_type = "TYPE OF FEEDBACK_TYPE DE RETOUR D'INFORMATION"
code_col = "CODE"
comment_col = "FEEDBACK COMMENT_COMMENTAIRE"

# %%
# Load binarizer object fitted on training set
with open(f"{project_directory}/outputs/models/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# %%
codes = list(mlb.classes_)

# %%
codes_keep = [
    "Belief that some people/institutions are making money because of the disease",
    "Belief that the outbreak has ended",
    "Belief that disease does exist or is real",
    "Belief that the disease does not exist in this region or country",
    "Beliefs about hand washing or hand sanitizers",
    "Beliefs about face masks",
    "Observations of non-compliance with health measures",
]


# %%
def get_df_codes(df):
    codes_df = df[df[feedback_type] == "Rumors_beliefs_observations"].copy()
    codes_df = codes_df[[code_col, comment_col]].copy()
    codes_df.replace(r"^ +| +$", r"", regex=True, inplace=True)
    codes_df.columns = ["code", "comment"]
    codes_df.drop_duplicates(subset=["code", "comment"], inplace=True)
    codes_df = codes_df.loc[codes_df["code"].isin(codes_keep)].copy()
    codes_df.dropna(how="any", inplace=True)
    codes_df = pwd.translate_column(codes_df, "comment")
    codes_df = codes_df.reset_index()
    codes_df = codes_df.rename(columns={"index": "id"})
    codes_df["id"] = codes_df.index
    codes_df.set_index("id", inplace=True)
    return codes_df


# %%
def df_clean(labelled_data):
    """
    Combine and clean x2 files of comments and codes.
    """
    # Remove white space before and after text
    labelled_data.replace(r"^ +| +$", r"", regex=True, inplace=True)
    # Removing 48 duplicate code/comment pairs (from W1)
    labelled_data.drop_duplicates(subset=["code", "comment"], inplace=True)
    return labelled_data


# %%
drc_df = get_df_codes(drc_data)
drc_df = df_clean(drc_df)

# %%
drc_df["code"] = drc_df["code"].map(
    {
        "Belief that some people/institutions are making money because of the disease": "Croyance_que_certaines_personnes_/_institutions_gagnent_de_l'argent_à_cause_de_la_maladie",
        "Belief that the outbreak has ended": "Croyance_que_l'épidémie_est_terminée",
        "Belief that disease does exist or is real": "Croyance_que_la_maladie_existe_ou_est_réelle",
        "Belief that the disease does not exist in this region or country": "Croyance_que_la_maladie_n'existe_pas_dans_cette_région_ou_dans_ce_pays",
        "Beliefs about hand washing or hand sanitizers": "Croyances_sur_le_lavage_des_mains_ou_les_désinfectants_des_mains",
        "Beliefs about face masks": "Croyances_sur_les_masques_de_visage",
        "Observations of non-compliance with health measures": "Observations_de_non-respect_des_mesures_de_santé",
    }
)

# %%
drc_test = pwd.category_id(drc_df)

# %%
drc_test.shape

# %%
# Get language model
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model

# %%
drc_embeddings = model_fr.encode(list(drc_test["comment"]))

# %%
y_drc = drc_test["category_id"]

# %%
y_drc.head(3)

# %%
y_drc = mlb.transform(y_drc)

# %%
y_drc

# %%
# Loading KNN model (best performing)
SVM_model = pickle.load(open(f"{project_directory}/outputs/models/svm_model.sav", "rb"))

# %%
# predict on test set
y_pred_svm = SVM_model.predict(drc_embeddings)

# %%
y_pred_svm

# %%
f1_scores = f1_score(y_drc, y_pred_svm, average="micro")

# %%
f1_scores

# %%
cm_svm = multilabel_confusion_matrix(y_drc, y_pred_svm)

# %%
mtr.save_cm_plots(cm_svm, "drc_svm", codes)

# %%
f1_scores_labels_drc = f1_score(y_drc, y_pred_svm, average=None)

# %%
f1_scores_labels = [
    0.9736842105263158,
    0.8214285714285715,
    0.5806451612903226,
    0.7346938775510204,
    0.9180327868852459,
    0.9310344827586207,
    0.9107142857142857,
    0.7368421052631579,
]

# %%
f1_scores_labels_drc

# %%
# Comparison of train and test f1 scores
data = {
    "Codes": list(codes),
    "f1 DRC": f1_scores_labels_drc,
    "f1 test": f1_scores_labels,
}

# Creates pandas DataFrame.
df = pd.DataFrame(data)
df["Codes"].replace("_", " ", inplace=True, regex=True)

df.sort_values(by="f1 DRC", ascending=False, inplace=True)

# %%
f1_scores_df = pd.melt(df, id_vars="Codes", var_name="Red Cross", value_name="F1 Score")

# %%
f1_scores_df = f1_scores_df[
    f1_scores_df.Codes != "Croyances sur les moyens de transmission"
]

# %%
labels = ["\n".join(wrap(l, 40)) for l in f1_scores_df.Codes]

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
enmax_palette = [
    "#0000FF",
    "#FF6E47",
    "#18A48C",
    "#EB003B",
    "#9A1BB3",
    "#FDB633",
    "#97D9E3",
]
color_codes_wanted = [
    "nesta_blue",
    "nesta_orange",
    "nesta_green",
    "nesta_red",
    "nesta_purple",
    "nesta_yellow",
    "nesta_agua",
]
c = lambda x: enmax_palette[color_codes_wanted.index(x)]
sns.set_palette(sns.color_palette(enmax_palette))
pal = sns.color_palette(enmax_palette)  # nesta palette

# %%
# Plot comparison
plt.figure(figsize=(20, 15))

ax = sns.catplot(
    x="F1 Score",
    y=labels,
    hue="Red Cross",
    data=f1_scores_df,
    kind="bar",
    height=5,
    aspect=2,
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.8))

plt.setp(ax._legend.get_texts(), fontsize=14)
plt.setp(ax._legend.get_title(), fontsize=14)

plt.tight_layout()

# modify individual font size of elements
plt.xlabel("Micro f1 score", fontsize=16, labelpad=20)
plt.ylabel("Codes", fontsize=16)
plt.title("DRC data and test set f1 scores", fontsize=20, y=1.05)
plt.tick_params(axis="both", which="major", labelsize=14)

plt.show()

ax.savefig(
    f"{project_directory}/outputs/figures/predictive_models/results/png/drc_f1.png"
)
ax.savefig(
    f"{project_directory}/outputs/figures/predictive_models/results/svg/drc_f1.svg",
    format="svg",
)

# %%
