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
# ## Classifcation: Model test and report results

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
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
from textwrap3 import wrap

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
svm = MultiOutputClassifier(SVC(C=2, gamma="scale"), n_jobs=-1)
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
# save the models
filename = f"{project_directory}/outputs/models/knn_model.sav"
pickle.dump(knn, open(filename, "wb"))

filename = f"{project_directory}/outputs/models/svm_model.sav"
pickle.dump(svm, open(filename, "wb"))

filename = f"{project_directory}/outputs/models/dt_model.sav"
pickle.dump(dt, open(filename, "wb"))

filename = f"{project_directory}/outputs/models/rf_model.sav"
pickle.dump(rf, open(filename, "wb"))

filename = f"{project_directory}/outputs/models/nb_model.sav"
pickle.dump(nb, open(filename, "wb"))

# %%
# Load models
knn = pickle.load(open(f"{project_directory}/outputs/models/knn_model.sav", "rb"))
svm = pickle.load(open(f"{project_directory}/outputs/models/svm_model.sav", "rb"))
dt = pickle.load(open(f"{project_directory}/outputs/models/dt_model.sav", "rb"))
rf = pickle.load(open(f"{project_directory}/outputs/models/rf_model.sav", "rb"))
nb = pickle.load(open(f"{project_directory}/outputs/models/nb_model.sav", "rb"))

# %%
y_pred_svm = svm.predict(X_test_embeddings_fr)
y_pred_knn = knn.predict(X_test_embeddings_fr)
y_pred_rf = rf.predict(X_test_embeddings_fr)
y_pred_dt = dt.predict(X_test_embeddings_fr)
y_pred_nb = nb.predict(X_test_embeddings_fr)

# %%
from sklearn.metrics import accuracy_score

# %%
accuracy_score(y_test, y_pred_svm)

# %%
f1_score(y_test, y_pred_svm, average="micro")

# %%
f1_svm = f1_score(y_test, y_pred_svm, average=None)

# %%
f1_svm

# %%
f1_svm = f1_score(y_test, y_pred_svm, average="micro")
f1_knn = f1_score(y_test, y_pred_knn, average="micro")
f1_rf = f1_score(y_test, y_pred_rf, average="micro")
f1_dt = f1_score(y_test, y_pred_dt, average="micro")
f1_nb = f1_score(y_test, y_pred_nb, average="micro")
f1_scores = [f1_knn, f1_svm, f1_nb, f1_rf, f1_dt]

# %%
print(
    "KNN: " + str(precision_recall_fscore_support(y_test, y_pred_knn, average="micro"))
)
print("RF: " + str(precision_recall_fscore_support(y_test, y_pred_rf, average="micro")))
print("DT: " + str(precision_recall_fscore_support(y_test, y_pred_dt, average="micro")))
print("NB: " + str(precision_recall_fscore_support(y_test, y_pred_nb, average="micro")))
print(
    "SVM: " + str(precision_recall_fscore_support(y_test, y_pred_svm, average="micro"))
)

# %%
print("KNN: " + str(f1_score(y_test, y_pred_knn, average="micro")))
print("RF: " + str(f1_score(y_test, y_pred_rf, average="micro")))
print("DT: " + str(f1_score(y_test, y_pred_dt, average="micro")))
print("NB: " + str(f1_score(y_test, y_pred_nb, average="micro")))
print("SVM: " + str(f1_score(y_test, y_pred_svm, average="micro")))

# %%
print("KNN: " + str(f1_score(y_test, y_pred_knn, average="macro")))
print("RF: " + str(f1_score(y_test, y_pred_rf, average="macro")))
print("DT: " + str(f1_score(y_test, y_pred_dt, average="macro")))
print("NB: " + str(f1_score(y_test, y_pred_nb, average="macro")))
print("SVM: " + str(f1_score(y_test, y_pred_svm, average="macro")))

# %%
# Comparison of train and test f1 scores
data = {
    "Models": ["KNN", "Random Forest", "Decision Tree", "Gaussian Naive Bayes", "SVM"],
    "f1 train": [
        0.8254718922034187,
        0.6435466340825209,
        0.5930789490687917,
        0.7349059625159102,
        0.8462465464030492,
    ],
    "f1 test": [
        0.8356435643564355,
        0.7000000000000002,
        0.6272189349112426,
        0.7145242070116863,
        0.8607068607068606,
    ],
}

# Creates pandas DataFrame.
df = pd.DataFrame(data)

# %%
f1_scores = pd.melt(
    df, id_vars="Models", var_name="Evaluation type", value_name="Micro F1 Score"
)

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
labels = ["\n".join(wrap(l, 15)) for l in f1_scores.Models]

# %%
labels

# %%
# Plot comparison
ax = sns.catplot(
    x=labels,
    y="Micro F1 Score",
    hue="Evaluation type",
    data=f1_scores,
    kind="bar",
    height=5,
    aspect=1,
    order=["SVM", "KNN", "Gaussian Naive\nBayes", "Random Forest", "Decision Tree"],
)
plt.xticks(rotation=90)

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.8))

plt.setp(ax._legend.get_texts(), fontsize=14)
plt.setp(ax._legend.get_title(), fontsize=14)

plt.tight_layout()

# modify individual font size of elements
plt.xlabel("Models", fontsize=16, labelpad=20)
plt.ylabel("Micro f1 score", fontsize=16)
plt.title("Training and test set f1 scores", fontsize=20, y=1.05)
plt.tick_params(axis="both", which="major", labelsize=14)

ax.savefig(
    f"{project_directory}/outputs/figures/predictive_models/results/png/training_test_f1.png"
)
ax.savefig(
    f"{project_directory}/outputs/figures/predictive_models/results/svg/training_test_f1.svg",
    format="svg",
)

# %%
f1_scores_labels = list(f1_score(y_test, y_pred_svm, average=None))

# %%
# Comparison of train and test f1 scores
data = {"Codes": list(codes), "F1 scores": f1_scores_labels}

# Creates pandas DataFrame.
df = pd.DataFrame(data)
df["Codes"].replace("_", " ", inplace=True, regex=True)

df.sort_values(by="F1 scores", ascending=False, inplace=True)

# %%
from textwrap import wrap

# %%
labels = ["\n".join(wrap(l, 40)) for l in df.Codes]

# %%
labels

# %%
from matplotlib import pyplot as plt
import seaborn as sns

# %%
df.head(1)

# %%
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="F1 scores", y=labels, data=df, color="#0000FF")

plt.xlabel("f1 scores", fontsize=16)
plt.ylabel("", fontsize=1)

plt.title("f1 scores per code", fontsize=20, y=1.05)
plt.tick_params(axis="both", which="major", labelsize=14)

plt.tight_layout()

ax.figure.savefig(
    f"{project_directory}/outputs/figures/predictive_models/results/png/f1_scores_code.png"
)
ax.figure.savefig(
    f"{project_directory}/outputs/figures/predictive_models/results/svg/f1_scores_code.svg",
    format="svg",
)

# %%
# Get predictions lists
preds = mlb.inverse_transform(y_pred_svm)
y_test_vals = mlb.inverse_transform(y_test)

# %%
# Comparison of train and test f1 scores
data = {"comments": X_test, "preds": preds, "actual": y_test_vals}

# Creates pandas DataFrame.
df = pd.DataFrame(data)

# %%
df.actual.value_counts()

# %%
pd.set_option("display.max_colwidth", None)

# %%
df[df.actual == ("Croyance_que_la_maladie_existe_ou_est_réelle",)]

# %%
y_test_values = []
for val in y_test_vals:
    if list(val):
        for p in list(val):
            code = p.replace("_", " ")
            y_test_values.append(code)
    else:
        y_test_values.append("No label")

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

# %% [markdown]
# ### Model accuracy workshop CRC

# %%
# Read in updated file
acc_workshop_file1 = pd.read_excel(
    f"{project_directory}/inputs/data/accuracy_workshop/workshop_test_data_review.xlsx",
    sheet_name="comments_update",
)

# %%
acc_workshop_file1.head(1)

# %%
acc_workshop_file1["category_id"] = acc_workshop_file1["code"].str.replace(" ", "_")

# %%
acc_workshop_file = (
    acc_workshop_file1.groupby(["comment"])["category_id"]
    .apply(list)
    .to_frame()
    .reset_index()
)
y_test_workp_acc = mlb.transform(acc_workshop_file["category_id"])

# %%
acc_workshop_file1.head(1)

# %%
acc_workshop_file.shape

# %%
y_test_workp_acc[0]

# %%
worksp_X_test = acc_workshop_file["comment"]
worksp_y_test = y_test_workp_acc

# %%
X_worksp_embeddings = model_fr.encode(list(acc_workshop_file["comment"]))

# %%
wrksp_predictions = svm.predict(X_worksp_embeddings)

# %%
f1_scores_labels_crc = list(f1_score(worksp_y_test, wrksp_predictions, average=None))

# %%
f1_score(worksp_y_test, wrksp_predictions, average="micro")

# %%
f1_scores_labels

# %%
f1_scores_labels_crc

# %%
# Comparison of train and test f1 scores
data = {
    "Codes": list(codes),
    "f1 CRC": f1_scores_labels_crc,
    "f1 test": f1_scores_labels,
}

# Creates pandas DataFrame.
df = pd.DataFrame(data)
df["Codes"].replace("_", " ", inplace=True, regex=True)

df.sort_values(by="f1 CRC", ascending=False, inplace=True)

# %%
f1_scores_df = pd.melt(df, id_vars="Codes", var_name="Red Cross", value_name="F1 Score")

# %%
labels = ["\n".join(wrap(l, 40)) for l in f1_scores_df.Codes]

# %%
labels

# %%
f1_scores_df

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


# modify individual font size of elements
plt.xlabel("Micro f1 score", fontsize=16, labelpad=20)
plt.ylabel("Codes", fontsize=16)
plt.title("CRC new data and test set f1 scores", fontsize=20, y=1.05)
plt.tick_params(axis="both", which="major", labelsize=14)

plt.show()

ax.savefig(
    f"{project_directory}/outputs/figures/predictive_models/results/png/f1_crc.png"
)
ax.savefig(
    f"{project_directory}/outputs/figures/predictive_models/results/svg/f1_crc.svg",
    format="svg",
)

# %%
from matplotlib import pyplot as plt
import seaborn as sns

# %%
df.head(1)

# %%
plt.figure(figsize=(8, 6))
ax = sns.barplot(x="F1 DRC", y=labels, data=df, color="#0000FF")

plt.xlabel("F1 scores", fontsize=16)
plt.ylabel("", fontsize=1)

plt.title("F1 Scores per code - DRC set", fontsize=20, y=1.05)
plt.tick_params(axis="both", which="major", labelsize=14)

# %% [markdown]
# ### Temp Nepal

# %%
# Basic (shelter)
f1_binary = [
    0.7808219178,
    0.9473684211,
    0.8402366864,
    0.8,
    0.84375,
    0.8012820513,
    0.8267477204,
    0.9726027397,
    0.9795918367,
    0.8764044944,
    0.7476635514,
]

f1_macro = [
    0.5941146626,
    0.4736842105,
    0.8170649081,
    0.8032786885,
    0.8325892857,
    0.7930021368,
    0.8082078085,
    0.4863013699,
    0.4897959184,
    0.4382022472,
    0.5598782873,
]

basic_items = [
    "plastic tarpaulin",
    "blanket",
    "sari",
    "male dhoti",
    "shouting cloth jeans",
    "printed cloth",
    "terry cloth",
    "utensil set",
    "water bucket",
    "nylon rope",
    "sack packing bag",
]

# %%
# Non-Basic (dignity)
f1_binary = [
    0.9455184534,
    0.9795918367,
    0.9795918367,
    0.9455184534,
    0.9527145359,
    0.9247311828,
    0.9342806394,
    0.6818181818,
    0.8167330677,
    0.9031078611,
    0.7235023041,
]

f1_macro = [
    0.4727592267,
    0.4897959184,
    0.4897959184,
    0.4727592267,
    0.5108400266,
    0.4623655914,
    0.4671403197,
    0.7159090909,
    0.4389787788,
    0.4515539305,
    0.5003053689,
]

non_basic_items = [
    "cotton towel",
    "bathing soap",
    "laundry soap",
    "tooth brush and paste",
    "sanitary pad",
    "ladies underwear",
    "torch light",
    "whistle blow",
    "nail cutter",
    "hand sanitizer",
    "liquid chlorine",
]

# %%
# Comparison of train and test f1 scores
data = {
    "Shelter items": basic_items,
    "F1 binary": f1_binary,
    "F1 macro": f1_macro,
}

# Creates pandas DataFrame.
df = pd.DataFrame(data)

df.sort_values(by="F1 binary", ascending=False, inplace=True)

# %%
f1s_shelter = pd.melt(
    df, id_vars="Shelter items", var_name="F1 types", value_name="F1 Score"
)

# %%
# Plot comparison
plt.figure(figsize=(20, 15))

ax = sns.catplot(
    x="F1 Score",
    y="Shelter items",
    hue="F1 types",
    data=f1s_shelter,
    kind="bar",
    height=5,
    aspect=2,
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.8))

plt.setp(ax._legend.get_texts(), fontsize=14)
plt.setp(ax._legend.get_title(), fontsize=14)


# modify individual font size of elements
plt.xlabel("F1 scores", fontsize=16, labelpad=20)
plt.ylabel("Shelter items", fontsize=16)
plt.title("F1 binary and macro scores for Shelter items", fontsize=20, y=1.05)
plt.tick_params(axis="both", which="major", labelsize=14)

plt.show()

# %%
