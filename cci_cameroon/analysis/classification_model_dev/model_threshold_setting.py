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
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Analysing probability scores to set the threshold

# %%
# Import modules
import pandas as pd
import numpy as np
import cci_cameroon
from googletrans import Translator
from langdetect import detect, detect_langs
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    hamming_loss,
    multilabel_confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math
import pickle
from ast import literal_eval

# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %%
ModelsPerformance = {}


def metricsReport(modelName, test_labels, predictions):
    """Takes in model results and adds metrics to report dictionary."""
    macro_f1 = f1_score(test_labels, predictions, average="macro")
    micro_f1 = f1_score(test_labels, predictions, average="micro")
    hamLoss = hamming_loss(test_labels, predictions)
    ModelsPerformance[modelName] = {
        "Macro f1": macro_f1,
        "Micro f1": micro_f1,
        "Loss": hamLoss,
    }


# %%
def get_probas(pred_proba, codes, thres):
    """Translates pred_proba to % confidence over a set threshold."""
    classified = []
    for pre, code in zip(pred_proba, codes):
        if pre.tolist()[0][1] > thres:
            print(code, round((pre.tolist()[0][1]) * 100, 0), "% confidence")
            classified.append(1)
    return classified


# %%
# File links
w1_file = "multi_label_output_w1.xlsx"
w2_file = "workshop2_comments_french.xlsx"
# Read workshop files
w1 = pd.read_excel(f"{project_directory}/inputs/data/" + w1_file)
w2 = pd.read_excel(f"{project_directory}/inputs/data/" + w2_file)

# %%
pd.set_option("display.max_columns", None)  # or 1000
pd.set_option("display.max_rows", None)  # or 1000
pd.set_option("display.max_colwidth", -1)  # or 199

# %%
# Adding language column
translator = Translator()
w1["language"] = w1["comment"].apply(lambda x: detect(x))

# %%
# Slicing the data into en, es and remaining
en = w1[w1.language == "en"].copy()
es = w1[w1.language == "es"].copy()
w1 = w1[~w1.language.isin(["en", "es"])].copy()

# %%
# Translating the English and Spanish comments into French
en["comment"] = en.comment.apply(translator.translate, src="en", dest="fr").apply(
    getattr, args=("text",)
)
es["comment"] = es.comment.apply(translator.translate, src="es", dest="fr").apply(
    getattr, args=("text",)
)

# %%
# Merge back together
w1 = pd.concat([w1, en, es], ignore_index=True)

# %%
# Reemove language
w1.drop("language", axis=1, inplace=True)

# %%
# Join the two workshops files together
labelled_data = w1.append(w2, ignore_index=True)

# %%
# Remove white space before and after text
labelled_data.replace(r"^ +| +$", r"", regex=True, inplace=True)

# %%
# Removing 48 duplicate code/comment pairs (from W1)
labelled_data.drop_duplicates(subset=["code", "comment"], inplace=True)

# %%
# Removing small count codes
to_remove = list(
    labelled_data.code.value_counts()[labelled_data.code.value_counts() < 10].index
)
labelled_data = labelled_data[~labelled_data.code.isin(to_remove)].copy()

# %%
# Dataset for modelling
model_data = labelled_data.copy()
# Create category ID column from the code field (join by _ into one string)
model_data["category_id"] = model_data["code"].str.replace(" ", "_")
id_to_category = dict(model_data[["category_id", "code"]].values)
model_data = (
    model_data.groupby("comment")["category_id"].apply(list).to_frame().reset_index()
)

# %%
drc_df = pd.read_excel(
    f"{project_directory}/inputs/data/COVID_19 Community feedback_DATA_ DRC_july2021_dec2021xlsx.xlsx",
    sheet_name="FEEDBACK DATA_DONNEES ",
)

# %%
drc_df.columns

# %%
drc_df["TYPE OF FEEDBACK_TYPE DE RETOUR D'INFORMATION"].unique()

# %%
drc_df2 = drc_df[
    drc_df["TYPE OF FEEDBACK_TYPE DE RETOUR D'INFORMATION"]
    == "Rumors_beliefs_observations"
][
    [
        "DISTRICT/STATE/REGION/CITY_DISTRICT/ETAT/REGION/VILLE",
        "SEX_SEXE",
        "AGE RANGE_TRANCHE D'AGE",
        "FEEDBACK COMMENT_COMMENTAIRE",
        "CODE",
    ]
].copy()

# %%
drc_df2.columns = ["area", "sex", "age", "comment", "code"]

# %%
drc_df2.head()

# %%
drc_df2.shape

# %%
drc_df2.code.unique()

# %%
retained_codes = [
    "Belief that some people/institutions are making money because of the disease",
    "Belief that the outbreak has ended",
    "Belief that disease does exist or is real",
    "Belief that the disease does not exist in this region or country",
    "Beliefs about hand washing or hand sanitizers",
    "Beliefs about face masks",
    "Beliefs about ways of transmission",
    "Observations of non-compliance with health measures",
]

# %%
drc_df3 = (
    drc_df2[drc_df2.code.isin(retained_codes)].drop_duplicates("comment").reset_index()
)

# %%
drc_df3.shape

# %%
plt.figure(figsize=(10, 5))
drc_df3.code.value_counts().plot(kind="barh")

# %%
len(drc_df3.comment.unique())

# %%
model_data2 = pd.read_csv(f"{project_directory}/outputs/data/ifrc_comments_areas.csv")

# %%
model_data2 = model_data2[model_data2.code.isin(retained_codes)]

# %%
model_data2.shape

# %%
model_data2 = model_data2.rename(
    columns={"code": "code2", "feedback_comment": "comment"}
)

# %%
model_data2.drop("Unnamed: 0", axis=1, inplace=True)

# %%
X_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/X_test.xlsx", index_col="id"
)
y_test = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_test.xlsx", index_col="id"
)["category_id"]

# %%
y_train = pd.read_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_train.xlsx", index_col="id"
)["category_id"]

# %%
# combine the datasets
X_test["y_test"] = y_test
model_data_result = model_data2.merge(X_test, how="inner", on="comment")

# %%
model_data_result.head()

# %%

# %%
model_data_result.columns

# %%
model_data_result.shape

# %%
model_data_result.area.unique()

# %%
yaounde_df = model_data_result[
    model_data_result.area.isin(
        [
            "YAOUNDE",
            "Yaoundé",
            "YAOUNDE ",
            "MFOUNDI",
            "Obili-Yaoundé",
            "Ekounou",
            "TOKO CENTRE",
            "Yaoundé 2",
            "Mfoundi",
            "Mokolo",
            "Nkol Eton-Yaoundé ",
            "Marché 8ieme-Yaoundé",
            "Etoudi gare routière-Yaoundé ",
            "TORO CENTRE/DEMSA",
            "Biriqueterie-Yaoundé",
            "NLONGKAK",
            "yaoundé",
            "yaounde1",
            "Nsam -Yaoundé",
            "Yaoundé 4",
            "Nkolbisson-Yaoundé",
            "yaounde4",
            "Mendong",
            "Demsa",
            "Yaoundé 7",
            "Yaoundé-quartier Fouda",
            "Nsam II",
            "Poste Centrale-Yaoundé",
            "Yaoundé4 - ekoudoum, ekounou,mvan",
            "POSTE CENTRALE",
            "Yaoundé 6",
            "MVOG NBI-Yaoundé",
            "NSAM",
            "Yaoundé 3",
            "Yaoundé ",
            "POSTE CENTRAL",
        ]
    )
].reset_index()

# %%
douala_df = model_data_result[model_data_result.area == "Douala"].reset_index()
bafoussam_df = model_data_result[
    model_data_result.area.isin(
        ["Bafoussam ", "Foumban", "BAFOUSSAM", "Bafoussam", "Dschang", "Koutaba"]
    )
].reset_index()
bertoua_df = model_data_result[
    model_data_result.area.isin(["Bertoua ", "Bertoua"])
].reset_index()
garoua_df = model_data_result[model_data_result.area.isin(["Garoua"])].reset_index()

# %%

# %%
X_test_yaounde = yaounde_df.comment
y_test_yaounde = yaounde_df.y_test
X_test_douala = douala_df.comment
y_test_douala = douala_df.y_test
X_test_bertoua = bertoua_df.comment
y_test_bertoua = bertoua_df.y_test
X_test_bafoussam = bafoussam_df.comment
y_test_bafoussam = bafoussam_df.y_test
X_test_garoua = garoua_df.comment
y_test_garoua = garoua_df.y_test

# %%
X_test_yaounde

# %%
# Get language model
model_fr = SentenceTransformer("Sahajtomar/french_semantic")  # French language model


# %%
X_test_yaounde_embedding = model_fr.encode(X_test_yaounde)

# %%
X_test_douala_embedding = model_fr.encode(X_test_douala)
X_test_bertoua_embedding = model_fr.encode(X_test_bertoua)
X_test_bafoussam_embedding = model_fr.encode(X_test_bafoussam)
X_test_garoua_embedding = model_fr.encode(X_test_garoua)

# %%
y_train = y_train.apply(literal_eval)
y_test_yaounde = y_test_yaounde.apply(literal_eval)
y_test_douala = y_test_douala.apply(literal_eval)
y_test_bertoua = y_test_bertoua.apply(literal_eval)
y_test_bafoussam = y_test_bafoussam.apply(literal_eval)
y_test_garoua = y_test_garoua.apply(literal_eval)

# %%
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test_yaounde = mlb.transform(y_test_yaounde)

y_test_douala = mlb.transform(y_test_douala)

y_test_bertoua = mlb.transform(y_test_bertoua)

y_test_bafoussam = mlb.transform(y_test_bafoussam)

y_test_garoua = mlb.transform(y_test_garoua)

# %%

# %%

# %%

# %% [markdown]
# # examining model performance across characteristics ends here

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    model_data["comment"], model_data["category_id"], test_size=0.20, random_state=0
)

# %%
X_test2 = X_test.copy()

# %%
y_test_text = y_test.copy()

# %%
# Transform Y into multilabel format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)
codes = list(mlb.classes_)  # Codes list

# %%
# Get multi-langauge model that supports French
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# %%
# Encode train and test transform into word embeddings
X_train_embeddings = model.encode(list(X_train))
X_test_embeddings = model.encode(list(X_test))

# %%
# KNN classifier
knnClf = KNeighborsClassifier(n_neighbors=7)
knnClf.fit(X_train_embeddings, y_train)
knnPredictions = knnClf.predict(X_test_embeddings)

# %%
metricsReport("knnClf_tranformer", y_test, knnPredictions)
ModelsPerformance

# %%
# Create confusion matrix from the best performing model
cm_KNN = multilabel_confusion_matrix(y_test, knnPredictions)
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN[0])
disp.plot()
plt.title(codes[0].replace("_", " "))
plt.show()

# %%
pred_proba = knnClf.predict_proba(X_test_embeddings)

# %%
len(pred_proba)

# %%
codes = list(mlb.classes_)

# %%
preds = mlb.inverse_transform(knnPredictions)
y_test_vals = mlb.inverse_transform(y_test)

# %%
predictions = []
for pred in preds:
    if list(pred):
        for p in list(pred):
            code = p.replace("_", " ")
            predictions.append(code)
    else:
        predictions.append("No prediction")

# %%
code_lists = []
for preds in pred_proba:
    code_list = list(pd.DataFrame(preds)[1])
    code_lists.append(code_list)

preds_df = pd.DataFrame(np.column_stack(code_lists), columns=codes)

# %%
preds_df.shape

# %%
preds_df["total"] = preds_df[preds_df.columns].apply(lambda x: (x >= 0.5).sum(), axis=1)

# %%
preds_df.total.value_counts()

# %%
preds_df["actual"] = list(y_test_text)

# %%
preds_df[["actual1", "actual2"]] = pd.DataFrame(
    preds_df.actual.tolist(), index=preds_df.index
)

# %%
preds_df

# %%
prediction_lists = []

for code in codes:
    x = preds_df[(preds_df.actual1 == code) | (preds_df.actual2 == code)][code]
    prediction_lists.append(x)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from matplotlib import rcParams

# %% [markdown]
# ### Prediction probabilities for true Y labels

# %%
rcParams["figure.figsize"] = 14, 16
fig, axs = plt.subplots(4, 2)

sns.histplot(x=prediction_lists[0], bins=20, ax=axs[0, 0])
sns.histplot(x=prediction_lists[1], bins=20, ax=axs[1, 0])
sns.histplot(x=prediction_lists[2], bins=20, ax=axs[2, 0])
sns.histplot(x=prediction_lists[3], bins=20, ax=axs[3, 0])
sns.histplot(x=prediction_lists[4], bins=20, ax=axs[0, 1])
sns.histplot(x=prediction_lists[5], bins=20, ax=axs[1, 1])
sns.histplot(x=prediction_lists[6], bins=20, ax=axs[2, 1])
sns.histplot(x=prediction_lists[7], bins=20, ax=axs[3, 1])

plt.tight_layout(pad=0)

# %%
import sklearn.metrics as metrics

# %%
test_set_df = pd.DataFrame(y_test, columns=codes)

# %%
test_set_df.shape

# %%
rcParams["figure.figsize"] = 4, 4
((test_set_df.sum() / 255) * 100).plot(kind="bar")


# %%
def get_roc_scores(y_test, pred_proba, num):
    test_item = []
    for item in y_test:
        test_item.append(item[num])
    preds = pred_proba[num][:, 1]
    fpr, tpr, threshold = metrics.roc_curve(test_item, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc, threshold


# %%
fpr0, tpr0, roc_auc0, thresholds0 = get_roc_scores(y_test, pred_proba, 0)
fpr1, tpr1, roc_auc1, thresholds1 = get_roc_scores(y_test, pred_proba, 1)
fpr2, tpr2, roc_auc2, thresholds2 = get_roc_scores(y_test, pred_proba, 2)
fpr3, tpr3, roc_auc3, thresholds3 = get_roc_scores(y_test, pred_proba, 3)
fpr4, tpr4, roc_auc4, thresholds4 = get_roc_scores(y_test, pred_proba, 4)
fpr5, tpr5, roc_auc5, thresholds5 = get_roc_scores(y_test, pred_proba, 5)
fpr6, tpr6, roc_auc6, thresholds6 = get_roc_scores(y_test, pred_proba, 6)
fpr7, tpr7, roc_auc7, thresholds7 = get_roc_scores(y_test, pred_proba, 7)

# %%
import math

# %% [markdown]
# ### ROC curve and Geometric Mean for optimum threshold setting

# %% [markdown]
# g-mean for each threshold true positives and 1 - false positives

# %% [markdown]
# https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/

# %%
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr0 * (1 - fpr0))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print("Best Threshold=%f, G-Mean=%.3f" % (thresholds0[ix], gmeans[ix]))

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("ROC: " + codes[0].replace("_", " "))
plt.plot(fpr0, tpr0, "b", label="AUC = %0.2f" % roc_auc0)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr1 * (1 - fpr1))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print("Best Threshold=%f, G-Mean=%.3f" % (thresholds1[ix], gmeans[ix]))

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("ROC: " + codes[1].replace("_", " "))
plt.plot(fpr1, tpr1, "b", label="AUC = %0.2f" % roc_auc1)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr2 * (1 - fpr2))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print("Best Threshold=%f, G-Mean=%.3f" % (thresholds2[ix], gmeans[ix]))

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("ROC: " + codes[2].replace("_", " "))
plt.plot(fpr2, tpr2, "b", label="AUC = %0.2f" % roc_auc2)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr3 * (1 - fpr3))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print("Best Threshold=%f, G-Mean=%.3f" % (thresholds3[ix], gmeans[ix]))

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("ROC: " + codes[3].replace("_", " "))
plt.plot(fpr3, tpr3, "b", label="AUC = %0.2f" % roc_auc3)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr4 * (1 - fpr4))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print("Best Threshold=%f, G-Mean=%.3f" % (thresholds4[ix], gmeans[ix]))

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("ROC: " + codes[4].replace("_", " "))
plt.plot(fpr4, tpr4, "b", label="AUC = %0.2f" % roc_auc4)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr5 * (1 - fpr5))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print("Best Threshold=%f, G-Mean=%.3f" % (thresholds5[ix], gmeans[ix]))

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("ROC: " + codes[5].replace("_", " "))
plt.plot(fpr5, tpr5, "b", label="AUC = %0.2f" % roc_auc5)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr6 * (1 - fpr6))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print("Best Threshold=%f, G-Mean=%.3f" % (thresholds6[ix], gmeans[ix]))

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("ROC: " + codes[6].replace("_", " "))
plt.plot(fpr6, tpr6, "b", label="AUC = %0.2f" % roc_auc6)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr7 * (1 - fpr7))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print("Best Threshold=%f, G-Mean=%.3f" % (thresholds7[ix], gmeans[ix]))

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("ROC: " + codes[7].replace("_", " "))
plt.plot(fpr7, tpr7, "b", label="AUC = %0.2f" % roc_auc7)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %% [markdown]
# ## Calibration curves

# %% [markdown]
# Mean predicted probability verses fraction of positives at different intervals

# %%
from sklearn.calibration import calibration_curve


# %%
def get_class_preds(y_test, pred_proba, num):
    test_item = []
    for item in y_test:
        test_item.append(item[num])
    preds = pred_proba[num][:, 1]
    prob_true, prob_pred = calibration_curve(test_item, preds, n_bins=5)
    return prob_true, prob_pred


# %%
prob_true0, prob_pred0 = get_class_preds(y_test, pred_proba, 0)
prob_true1, prob_pred1 = get_class_preds(y_test, pred_proba, 1)
prob_true2, prob_pred2 = get_class_preds(y_test, pred_proba, 2)
prob_true3, prob_pred3 = get_class_preds(y_test, pred_proba, 3)
prob_true4, prob_pred4 = get_class_preds(y_test, pred_proba, 4)
prob_true5, prob_pred5 = get_class_preds(y_test, pred_proba, 5)
prob_true6, prob_pred6 = get_class_preds(y_test, pred_proba, 6)
prob_true7, prob_pred7 = get_class_preds(y_test, pred_proba, 7)

# %%
prob_pred0

# %%
rcParams["figure.figsize"] = 15, 10

plt.plot(list(prob_pred0), list(prob_true0), label=codes[0], marker="o")
plt.plot(list(prob_pred1), list(prob_true1), label=codes[1], marker="o")
plt.plot(list(prob_pred2), list(prob_true2), label=codes[2], marker="o")
plt.plot(list(prob_pred3), list(prob_true3), label=codes[3], marker="o")
plt.plot(list(prob_pred4), list(prob_true4), label=codes[4], marker="o")
plt.plot(list(prob_pred5), list(prob_true5), label=codes[5], marker="o")
plt.plot(list(prob_pred6), list(prob_true6), label=codes[6], marker="o")
plt.plot(list(prob_pred7), list(prob_true7), label=codes[7], marker="o")
plt.legend(fontsize=11)
plt.xlabel("Mean predicted probability", fontsize=16)
plt.ylabel("Fraction of positives", fontsize=16)

plt.tight_layout()

# %% [markdown]
# ROC curves + accuracy / F1 measures based on different sensitive characteristics (eg gender, ethnicity) - Pius
# Using the data in the IFRC dataset (eg gender, ethnicity location) build ROC curves based on predictions from different subsets of the data that come from different groups

# %% [markdown]
# ## Bias audi

# %%
# read the saved model
# save the best model to disk
pickle.dump(knn, open(filename, "wb"))

# %%
# load the model from disk
filename = f"{project_directory}/outputs/models/final_classification_model.sav"
loaded_model = pickle.load(open(filename, "rb"))
# result = loaded_model.score(X_test, Y_test)
# print(result)#

# %% [markdown]
# ## ROC curves for Yaounde locality

# %%
metricsReport("knnClf_tranformer", y_test, knnPredictions)
ModelsPerformance

# %%
# plot curves for yaounde data
pred_proba_y = loaded_model.predict_proba(X_test_yaounde_embedding)
y_pred_y = loaded_model.predict(X_test_yaounde_embedding)
metricsReport("knnClf_tranformer", y_test_yaounde, y_pred_y)
ModelsPerformance

# %%
macro_f1_score = []
micro_f1_score = []
macro_f1_score.append(0.8380252070880546)
micro_f1_score.append(0.8486842105263157)

# %%
# plot curves for yaounde data
pred_proba_y = loaded_model.predict_proba(X_test_yaounde_embedding)
fpr0_y, tpr0_y, roc_auc0_y, thresholds0_y = get_roc_scores(
    y_test_yaounde, pred_proba_y, 0
)
fpr1_y, tpr1_y, roc_auc1_y, thresholds1_y = get_roc_scores(
    y_test_yaounde, pred_proba_y, 1
)
fpr2_y, tpr2_y, roc_auc2_y, thresholds2_y = get_roc_scores(
    y_test_yaounde, pred_proba_y, 2
)
fpr3_y, tpr3_y, roc_auc3_y, thresholds3_y = get_roc_scores(
    y_test_yaounde, pred_proba_y, 3
)
fpr4_y, tpr4_y, roc_auc4_y, thresholds4_y = get_roc_scores(
    y_test_yaounde, pred_proba_y, 4
)
fpr5_y, tpr5_y, roc_auc5_y, thresholds5_y = get_roc_scores(
    y_test_yaounde, pred_proba_y, 5
)
fpr6_y, tpr6_y, roc_auc6_y, thresholds6_y = get_roc_scores(
    y_test_yaounde, pred_proba_y, 6
)
fpr7_y, tpr7_y, roc_auc7_y, thresholds7_y = get_roc_scores(
    y_test_yaounde, pred_proba_y, 7
)

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Yaounde ROC: " + codes[0].replace("_", " "))
plt.plot(fpr0_y, tpr0_y, "b", label="AUC = %0.2f" % roc_auc0_y)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Yaounde ROC: " + codes[1].replace("_", " "))
plt.plot(fpr1_y, tpr1_y, "b", label="AUC = %0.2f" % roc_auc1_y)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Yaounde ROC: " + codes[2].replace("_", " "))
plt.plot(fpr2_y, tpr2_y, "b", label="AUC = %0.2f" % roc_auc2_y)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Yaounde ROC: " + codes[3].replace("_", " "))
plt.plot(fpr3_y, tpr3_y, "b", label="AUC = %0.2f" % roc_auc3_y)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Yaounde ROC: " + codes[4].replace("_", " "))
plt.plot(fpr4_y, tpr4_y, "b", label="AUC = %0.2f" % roc_auc4_y)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Yaounde ROC: " + codes[5].replace("_", " "))
plt.plot(fpr5_y, tpr5_y, "b", label="AUC = %0.2f" % roc_auc5_y)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Yaounde ROC: " + codes[6].replace("_", " "))
plt.plot(fpr6_y, tpr6_y, "b", label="AUC = %0.2f" % roc_auc6_y)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Yaounde ROC: " + codes[7].replace("_", " "))
plt.plot(fpr7_y, tpr7_y, "b", label="AUC = %0.2f" % roc_auc7_y)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %% [markdown]
# ## ROC Curves for data from Douala

# %%


# %%
# plot curves for yaounde data
pred_proba_d = loaded_model.predict_proba(X_test_douala_embedding)

# %%
y_pred_d = loaded_model.predict(X_test_douala_embedding)
metricsReport("knnClf_tranformer", y_test_douala, y_pred_d)
ModelsPerformance

# %%
micro_f1_score.append(0.8952380952380952)
macro_f1_score.append(0.9081477732793523)

# %%
fpr0_d, tpr0_d, roc_auc0_d, thresholds0_d = get_roc_scores(
    y_test_douala, pred_proba_d, 0
)
fpr1_d, tpr1_d, roc_auc1_d, thresholds1_d = get_roc_scores(
    y_test_douala, pred_proba_d, 1
)
fpr2_d, tpr2_d, roc_auc2_d, thresholds2_d = get_roc_scores(
    y_test_douala, pred_proba_d, 2
)
fpr3_d, tpr3_d, roc_auc3_d, thresholds3_d = get_roc_scores(
    y_test_douala, pred_proba_d, 3
)
fpr4_d, tpr4_d, roc_auc4_d, thresholds4_d = get_roc_scores(
    y_test_douala, pred_proba_d, 4
)
fpr5_d, tpr5_d, roc_auc5_d, thresholds5_d = get_roc_scores(
    y_test_douala, pred_proba_d, 5
)
fpr6_d, tpr6_d, roc_auc6_d, thresholds6_d = get_roc_scores(
    y_test_douala, pred_proba_d, 6
)
fpr7_d, tpr7_d, roc_auc7_d, thresholds7_d = get_roc_scores(
    y_test_douala, pred_proba_d, 7
)

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Douala ROC: " + codes[0].replace("_", " "))
plt.plot(fpr0_d, tpr0_d, "b", label="AUC = %0.2f" % roc_auc0_d)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Douala ROC: " + codes[1].replace("_", " "))
plt.plot(fpr1_d, tpr1_d, "b", label="AUC = %0.2f" % roc_auc1_d)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Douala ROC: " + codes[2].replace("_", " "))
plt.plot(fpr2_d, tpr2_d, "b", label="AUC = %0.2f" % roc_auc2_d)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Douala ROC: " + codes[3].replace("_", " "))
plt.plot(fpr3_d, tpr3_d, "b", label="AUC = %0.2f" % roc_auc3_d)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Douala ROC: " + codes[4].replace("_", " "))
plt.plot(fpr4_d, tpr4_d, "b", label="AUC = %0.2f" % roc_auc4_d)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Douala ROC: " + codes[5].replace("_", " "))
plt.plot(fpr5_d, tpr5_d, "b", label="AUC = %0.2f" % roc_auc5_d)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Douala ROC: " + codes[6].replace("_", " "))
plt.plot(fpr6_d, tpr6_d, "b", label="AUC = %0.2f" % roc_auc6_d)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
rcParams["figure.figsize"] = 5, 5
plt.title("Douala ROC: " + codes[7].replace("_", " "))
plt.plot(fpr7_d, tpr7_d, "b", label="AUC = %0.2f" % roc_auc7_d)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# %%
# Bafoussam
pred_proba_ba = loaded_model.predict_proba(X_test_bafoussam_embedding)

# %%
# Bafoussam
pred_proba_ba = loaded_model.predict_proba(X_test_bafoussam_embedding)
y_pred_ba = loaded_model.predict(X_test_bafoussam_embedding)
metricsReport("knnClf_tranformer", y_test_bafoussam, y_pred_ba)
ModelsPerformance

# %%
macro_f1_score.append(0.7367115705931495)
micro_f1_score.append(0.853932584269663)

# %%
fpr0_ba, tpr0_ba, roc_auc0_ba, thresholds0_ba = get_roc_scores(
    y_test_bafoussam, pred_proba_ba, 0
)
fpr1_ba, tpr1_ba, roc_auc1_ba, thresholds1_ba = get_roc_scores(
    y_test_bafoussam, pred_proba_ba, 1
)
fpr2_ba, tpr2_ba, roc_auc2_ba, thresholds2_ba = get_roc_scores(
    y_test_bafoussam, pred_proba_ba, 2
)
fpr3_ba, tpr3_ba, roc_auc3_ba, thresholds3_ba = get_roc_scores(
    y_test_bafoussam, pred_proba_ba, 3
)
fpr4_ba, tpr4_ba, roc_auc4_ba, thresholds4_ba = get_roc_scores(
    y_test_bafoussam, pred_proba_ba, 4
)
fpr5_ba, tpr5_ba, roc_auc5_ba, thresholds5_ba = get_roc_scores(
    y_test_bafoussam, pred_proba_ba, 5
)
fpr6_ba, tpr6_ba, roc_auc6_ba, thresholds6_ba = get_roc_scores(
    y_test_bafoussam, pred_proba_ba, 6
)
fpr7_ba, tpr7_ba, roc_auc7_ba, thresholds7_ba = get_roc_scores(
    y_test_bafoussam, pred_proba_ba, 7
)

# %%

# %%
# Bertoua has only one data point and Garoua has only three data points for the test set.
# These are not very useful to measure performance in those areas.

# %%
codes = codes[:8]
codes

# %%
location_df_AUC = pd.DataFrame(
    np.column_stack(
        [
            codes,
            [
                roc_auc0_y,
                roc_auc1_y,
                roc_auc2_y,
                roc_auc3_y,
                roc_auc4_y,
                roc_auc5_y,
                roc_auc6_y,
                roc_auc7_y,
            ],
            [
                roc_auc0_ba,
                roc_auc1_ba,
                roc_auc2_ba,
                roc_auc3_ba,
                roc_auc4_ba,
                roc_auc5_ba,
                roc_auc6_ba,
                roc_auc7_ba,
            ],
            [
                roc_auc0_d,
                roc_auc1_d,
                roc_auc2_d,
                roc_auc3_d,
                roc_auc4_d,
                roc_auc5_d,
                roc_auc6_d,
                roc_auc7_d,
            ],
        ]
    ),
    columns=["codes", "Yaounde AUC", "Bafoussam AUC", "Douala AUC"],
)

# %%
location_df_AUC.to_csv(
    f"{project_directory}/outputs/data/location_AUC.csv", index=False
)

# %%
location_df_AUC.Yaounde = location_df_AUC["Yaounde AUC"].astype("float")
location_df_AUC.Douala = location_df_AUC["Douala AUC"].astype("float")
location_df_AUC.Bafoussam = location_df_AUC["Bafoussam AUC"].astype("float")

# %%
location_df_AUC.head(10)

# %% [markdown]
# ## Process data by gender

# %%
model_data_result.age_range.unique()

# %%
youth_comments = model_data_result[
    model_data_result.age_range == "Youth (13 to 17 years old)"
].reset_index()["comment"]
y_test_youth = model_data_result[
    model_data_result.age_range == "Youth (13 to 17 years old)"
].reset_index()["y_test"]
adult_comments = model_data_result[
    model_data_result.age_range == "Adults (18 - 59 years old)"
].reset_index()["comment"]
y_test_adult = model_data_result[
    model_data_result.age_range == "Adults (18 - 59 years old)"
].reset_index()["y_test"]
children_comments = model_data_result[
    model_data_result.age_range == "Children (under 13 years old)"
].reset_index()["comment"]
y_test_children = model_data_result[
    model_data_result.age_range == "Children (under 13 years old)"
].reset_index()["y_test"]
elderly_comments = model_data_result[
    model_data_result.age_range == "Elderly (60 years and older)"
].reset_index()["comment"]
y_test_elderly = model_data_result[
    model_data_result.age_range == "Elderly (60 years and older)"
].reset_index()["y_test"]

# %%
youth_comments_embedding = model_fr.encode(youth_comments)
adult_comments_embedding = model_fr.encode(adult_comments)
children_comments_embedding = model_fr.encode(children_comments)
elderly_comments_embedding = model_fr.encode(elderly_comments)

# %%
y_test_youth = y_test_youth.apply(literal_eval)
y_test_adult = y_test_adult.apply(literal_eval)
y_test_children = y_test_children.apply(literal_eval)
y_test_elderly = y_test_elderly.apply(literal_eval)

# %%
y_test_youth = mlb.transform(y_test_youth)
y_test_adult = mlb.transform(y_test_adult)
y_test_children = mlb.transform(y_test_children)
y_test_elderly = mlb.transform(y_test_elderly)

# %%
model_data_result.gender.unique()

# %% [markdown]
# # Youth comments only 3, children's comments only 1, elderly comments only 3. Only adults comments were 183. No way to test across age groups

# %%


# %%
model_data_result.gender.unique()

# %%
male_comments = model_data_result[model_data_result.gender == "Male "].reset_index()[
    "comment"
]
female_comments = model_data_result[model_data_result.gender == "Female"].reset_index()[
    "comment"
]
mixed_comments = model_data_result[model_data_result.gender == "Mixed"].reset_index()[
    "comment"
]
dont_know_comments = model_data_result[
    model_data_result.gender == "Don't know"
].reset_index()["comment"]
y_test_male = model_data_result[model_data_result.gender == "Male "].reset_index()[
    "y_test"
]
y_test_female = model_data_result[model_data_result.gender == "Female"].reset_index()[
    "y_test"
]
y_test_mixed = model_data_result[model_data_result.gender == "Mixed"].reset_index()[
    "y_test"
]

y_test_male = y_test_male.apply(literal_eval)
y_test_female = y_test_female.apply(literal_eval)
y_test_mixed = y_test_mixed.apply(literal_eval)

y_test_male = mlb.transform(y_test_male)
y_test_female = mlb.transform(y_test_female)
y_test_mixed = mlb.transform(y_test_mixed)

# %%
male_comments_embedding = model_fr.encode(male_comments)
female_comments_embedding = model_fr.encode(female_comments)
mixed_comments_embedding = model_fr.encode(mixed_comments)

# %%
# Male
pred_proba_male = loaded_model.predict_proba(male_comments_embedding)
y_pred_male = loaded_model.predict(male_comments_embedding)
metricsReport("knnClf_tranformer", y_test_male, y_pred_male)
ModelsPerformance

# %%
# female

pred_proba_female = loaded_model.predict_proba(female_comments_embedding)
y_pred_female = loaded_model.predict(female_comments_embedding)
metricsReport("knnClf_tranformer", y_test_female, y_pred_female)
ModelsPerformance

# %%
# mixed
pred_proba_mixed = loaded_model.predict_proba(mixed_comments_embedding)
y_pred_mixed = loaded_model.predict(mixed_comments_embedding)
metricsReport("knnClf_tranformer", y_test_mixed, y_pred_mixed)
ModelsPerformance

# %%
# Create confusion matrix from the best performing model
cm_KNN_male = multilabel_confusion_matrix(y_test_male, y_pred_male)
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN_male[0])
disp.plot()
plt.title(codes[0].replace("_", " "))
plt.show()

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN_male[1])
disp.plot()
plt.title(codes[1].replace("_", " "))
plt.show()

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN_male[2])
disp.plot()
plt.title(codes[2].replace("_", " "))
plt.show()

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN_male[3])
disp.plot()
plt.title(codes[3].replace("_", " "))
plt.show()

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN_male[4])
disp.plot()
plt.title(codes[4].replace("_", " "))
plt.show()

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN_male[5])
disp.plot()
plt.title(codes[5].replace("_", " "))
plt.show()

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN_male[6])
disp.plot()
plt.title(codes[6].replace("_", " "))
plt.show()

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN_male[7])
disp.plot()
plt.title(codes[7].replace("_", " "))
plt.show()

# %%


# %% [markdown]
# ### Computing the AUC for the different groups
#

# %%
fpr0_male, tpr0_male, roc_auc0_male, _ = get_roc_scores(y_test_male, pred_proba_male, 0)
fpr1_male, tpr1_male, roc_auc1_male, _ = get_roc_scores(y_test_male, pred_proba_male, 1)
fpr2_male, tpr2_male, roc_auc2_male, _ = get_roc_scores(y_test_male, pred_proba_male, 2)
fpr3_male, tpr3_male, roc_auc3_male, _ = get_roc_scores(y_test_male, pred_proba_male, 3)
fpr4_male, tpr4_male, roc_auc4_male, _ = get_roc_scores(y_test_male, pred_proba_male, 4)
fpr5_male, tpr5_male, roc_auc5_male, _ = get_roc_scores(y_test_male, pred_proba_male, 5)
fpr6_male, tpr6_male, roc_auc6_male, _ = get_roc_scores(y_test_male, pred_proba_male, 6)
fpr7_male, tpr7_male, roc_auc7_male, _ = get_roc_scores(y_test_male, pred_proba_male, 7)

# %%
fpr0_female, tpr0_female, roc_auc0_female, _ = get_roc_scores(
    y_test_female, pred_proba_female, 0
)
fpr1_female, tpr1_female, roc_auc1_female, _ = get_roc_scores(
    y_test_female, pred_proba_female, 1
)
fpr2_female, tpr2_female, roc_auc2_female, _ = get_roc_scores(
    y_test_female, pred_proba_female, 2
)
fpr3_female, tpr3_female, roc_auc3_female, _ = get_roc_scores(
    y_test_female, pred_proba_female, 3
)
fpr4_female, tpr4_female, roc_auc4_female, _ = get_roc_scores(
    y_test_female, pred_proba_female, 4
)
fpr5_female, tpr5_female, roc_auc5_female, _ = get_roc_scores(
    y_test_female, pred_proba_female, 5
)
fpr6_female, tpr6_female, roc_auc6_female, _ = get_roc_scores(
    y_test_female, pred_proba_female, 6
)
fpr7_female, tpr7_female, roc_auc7_female, _ = get_roc_scores(
    y_test_female, pred_proba_female, 7
)

# %%
_, _, roc_auc0_mixed, _ = get_roc_scores(y_test_mixed, pred_proba_mixed, 0)
_, _, roc_auc1_mixed, _ = get_roc_scores(y_test_mixed, pred_proba_mixed, 1)
_, _, roc_auc2_mixed, _ = get_roc_scores(y_test_mixed, pred_proba_mixed, 2)
_, _, roc_auc3_mixed, _ = get_roc_scores(y_test_mixed, pred_proba_mixed, 3)
_, _, roc_auc4_mixed, _ = get_roc_scores(y_test_mixed, pred_proba_mixed, 4)
_, _, roc_auc5_mixed, _ = get_roc_scores(y_test_mixed, pred_proba_mixed, 5)
_, _, roc_auc6_mixed, _ = get_roc_scores(y_test_mixed, pred_proba_mixed, 6)
_, _, roc_auc7_mixed, _ = get_roc_scores(y_test_mixed, pred_proba_mixed, 7)

# %%
gender_df = pd.DataFrame(
    np.column_stack(
        [
            codes,
            [
                roc_auc0_male,
                roc_auc1_male,
                roc_auc2_male,
                roc_auc3_male,
                roc_auc4_male,
                roc_auc5_male,
                roc_auc6_male,
                roc_auc7_male,
            ],
            [
                roc_auc0_female,
                roc_auc1_female,
                roc_auc2_female,
                roc_auc3_female,
                roc_auc4_female,
                roc_auc5_female,
                roc_auc6_female,
                roc_auc7_female,
            ],
            [
                roc_auc0_mixed,
                roc_auc1_mixed,
                roc_auc2_mixed,
                roc_auc3_mixed,
                roc_auc4_mixed,
                roc_auc5_mixed,
                roc_auc6_mixed,
                roc_auc7_mixed,
            ],
        ]
    ),
    columns=["Codes", "Male AUC", "Female AUC", "Mixed groups AUC"],
)

# %%
gender_df.head(10)

# %%
gender_df.to_csv(f"{project_directory}/outputs/data/gender_AUC.csv", index=False)

# %%
model_data_result[
    model_data_result.gender == "Mixed"
].reset_index().y_test.value_counts()


# %%
model_data_result[model_data_result.gender == "Male "].reset_index().code.values_count()
model_data_result[model_data_result.gender == "Female"].reset_index()["comment"]
model_data_result[model_data_result.gender == "Mixed"].reset_index()["comment"]
model_data_result[model_data_result.gender == "Don't know"].reset_index()["comment"]


# %%
f1_score(y_test_male, y_pred_male, average=None)

# %%
f1_score(y_test_female, y_pred_female, average=None)

# %%
codes
