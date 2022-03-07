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
# ## Classification Model Accuracy Workshop

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
from textwrap import wrap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
def predict_rumour(thres):
    """For the demo: runs knn model on new data and outputs results."""
    unclassified = []
    codes = list(mlb.classes_)
    new_rumour = input("Please enter a new rumour ")
    test_text = model.encode([new_rumour])
    preds = mlb.inverse_transform(knnClf.predict(test_text))
    pred_proba = knnClf.predict_proba(test_text)
    print("Probability for each class above: " + str(thres))
    classified = get_probas(pred_proba, codes, thres)
    if sum(classified) < 1:
        print(
            "The model could not find a suitable code. Will will look into your rumour and get back to you."
        )
        unclassified.append({"comment": new_rumour, "prediction_scores": pred_proba})
        print(unclassified)
        # [] Save the results somewhere....
    else:
        print(" ")
        for pred in preds:
            for p in list(pred):
                code = p.replace("_", " ")
                print("Thanks for submitting a rumour!")
                print(" ")
                print(
                    "The model predicts your rumour "
                    + '"'
                    + new_rumour
                    + '"'
                    + " is related to: "
                )
                print(code)
                print(" ")
                correct = input("Is this correct? (please answer Yes or No) ")
                if correct.lower() == "yes":
                    print(" ")
                    print("Thanks for the letting us know")
                    print(" ")
                    print("Here is some information about " + code)
                    print(" ")
                else:
                    print(
                        "Thanks for letting us know. We will look into this and get back to you."
                    )
    return pred_proba


# %% [markdown]
# ### Preproc / cleaning

# %%
# File links
w1_file = "multi_label_output_w1.xlsx"
w2_file = "workshop2_comments_french.xlsx"
# w2_file = "retained_comments_w2.xlsx"
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
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    model_data["comment"], model_data["category_id"], test_size=0.20, random_state=0
)

# %%
y_test.head(2)

# %%
# Transform Y into multilabel format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)
codes = list(mlb.classes_)  # Codes list

# %%
y_test

# %%
# Get multi-langauge model that supports French
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# %%
# Encode train and test transform into word embeddings
X_train_embeddings = model.encode(list(X_train))
X_test_embeddings = model.encode(list(X_test))

# %% [markdown]
# ### Modelling

# %%
# KNN classifier
knnClf = KNeighborsClassifier(n_neighbors=7)
knnClf.fit(X_train_embeddings, y_train)
knnPredictions = knnClf.predict(X_test_embeddings)

# %% [markdown]
# ### Outputs for workhop

# %% [markdown]
# Notes:
#
# - Will be a very small test set 6-18 rumours per code (for all 8 codes)
# - Predictions per rumour? (examples or all?)
# - Overall prediction table? Easier to interpret version of a CM?

# %% [markdown]
# Reduced set

# %%
# Read in updated file
acc_workshop_file1 = pd.read_excel(
    f"{project_directory}/inputs/data/accuracy_workshop/workshop_test_data_review.xlsx",
    sheet_name="comments_update",
)

# %%
acc_workshop_file1["category_id"] = acc_workshop_file1["code"].str.replace(" ", "_")

# %%
acc_workshop_file1.head(5)

# %%
acc_workshop_file = (
    acc_workshop_file1.groupby(["comment"])["category_id"]
    .apply(list)
    .to_frame()
    .reset_index()
)
y_test_workp_acc = mlb.transform(acc_workshop_file["category_id"])

# %%
acc_workshop_file

# %%
y_test_workp_acc[6]

# %%
worksp_X_test = acc_workshop_file["comment"]
worksp_y_test = y_test_workp_acc

# %%
X_worksp_embeddings = model.encode(list(acc_workshop_file["comment"]))

# %%
wrksp_predictions = knnClf.predict(X_worksp_embeddings)
wrksp_pred_proba = knnClf.predict_proba(X_worksp_embeddings)

# %%
# Create confusion matrix from the best performing model
cm_wrkp = multilabel_confusion_matrix(worksp_y_test, wrksp_predictions)

# %%
plt.rcParams.update({"font.size": 10})

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[0])
disp.plot()
plt.title(codes[0].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[1])
disp.plot()
plt.title(codes[1].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[2])
disp.plot()
plt.title(codes[2].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[3])
disp.plot()
plt.title(codes[3].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[4])
disp.plot()
plt.title(codes[4].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[5])
disp.plot()
plt.title(codes[5].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[6])
disp.plot()
plt.title(codes[6].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[7])
disp.plot()
plt.title(codes[7].replace("_", " "))
plt.show()

# %%
lists = (
    list(cm_wrkp[0][1]),
    list(cm_wrkp[1][1]),
    list(cm_wrkp[2][1]),
    list(cm_wrkp[3][1]),
    list(cm_wrkp[4][1]),
    list(cm_wrkp[5][1]),
    list(cm_wrkp[6][1]),
    list(cm_wrkp[7][1]),
)

# %%
codes = list(mlb.classes_)
code_labels = [code.replace("_", " ") for code in codes]
code_labels = ["\n".join(wrap(l, 40)) for l in code_labels]

# %%
code_pred_totals = pd.DataFrame(
    lists, index=code_labels, columns=["Incorrect", "Correct"]
)

# %%
code_pred_totals.Correct.sum()

# %%
code_pred_totals.Incorrect.sum()

# %%
44 / (44 + 21)

# %%
plt.rcParams.update({"font.size": 12})

code_pred_totals.plot(kind="barh", figsize=(10, 8))
plt.xlabel("Count", fontsize=14)
plt.title("Model predictions per code - updated reduced set", fontsize=20)

# %%
code_lists = []
for preds in wrksp_pred_proba:
    code_list = list(pd.DataFrame(preds)[1])
    code_lists.append(code_list)

preds_df = pd.DataFrame(np.column_stack(code_lists), columns=codes)

# %%
preds = mlb.inverse_transform(wrksp_predictions)
worksp_y_test_vals = mlb.inverse_transform(worksp_y_test)

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
actuals = []
for pred in worksp_y_test_vals:
    if list(pred):
        for p in list(pred):
            code = p.replace("_", " ")
            actuals.append(code)
    else:
        actuals.append("No prediction")

# %%
preds_df["prediction"] = predictions
preds_df["prediction score"] = preds_df.iloc[:, :8].max(axis=1)
preds_df.loc[preds_df["prediction score"] < 0.5, "prediction score"] = 0

# %%
preds_df["actual"] = actuals
preds_df["comment"] = worksp_X_test

# %%
preds_df[
    (preds_df.actual == "Croyance que l'épidémie est terminée")
    & (preds_df.prediction != "Croyance que l'épidémie est terminée")
][["prediction", "comment"]]

# %%
predict_actual = pd.DataFrame(
    {"comment": worksp_X_test, "actual": worksp_y_test_vals, "predicted": predictions}
)

# %%

# %%
a = pd.DataFrame(acc_workshop_file1.code.value_counts().reset_index())
b = pd.DataFrame(predict_actual.predicted.value_counts().reset_index())

# %%
a.columns = ["index", "code assigned"]

# %%
merged = pd.merge(a, b, on="index", how="outer")
merged.set_index("index", inplace=True)

# %%
merged

# %%
merged.predicted.sum()

# %%
merged["code assigned"].sum()

# %%
plt.rcParams.update({"font.size": 22})

merged.plot(kind="barh", figsize=(15, 12))
plt.xlabel("Count", fontsize=22)
plt.ylabel(" ", fontsize=22)
plt.title(
    "Code predicted by model vs the code suggested in workshop - reduced set",
    fontsize=25,
)
plt.show()

# %%
predict_actual[predict_actual.predicted == "No prediction"]

# %% [markdown]
# Full comments / codes

# %%
# Read in workshop_test_data file
# acc_workshop_file1 = pd.read_excel(f"{project_directory}/inputs/data/accuracy_workshop/workshop_test_data.xlsx")

# %%
acc_workshop_file1["category_id"] = acc_workshop_file1["code"].str.replace(" ", "_")

# %%
acc_workshop_file1.head(5)

# %%
acc_workshop_file = (
    acc_workshop_file1.groupby(["comment"])["category_id"]
    .apply(list)
    .to_frame()
    .reset_index()
)
y_test_workp_acc = mlb.transform(acc_workshop_file["category_id"])

# %%
array([0, 0, 0, 0, 0, 1, 0, 0])

# %%
y_test_workp_acc[1]

# %%
worksp_X_test = acc_workshop_file["comment"]
worksp_y_test = y_test_workp_acc

# %%
X_worksp_embeddings = model.encode(list(acc_workshop_file["comment"]))

# %%
wrksp_predictions = knnClf.predict(X_worksp_embeddings)
wrksp_pred_proba = knnClf.predict_proba(X_worksp_embeddings)

# %%
# Create confusion matrix from the best performing model
cm_wrkp = multilabel_confusion_matrix(worksp_y_test, wrksp_predictions)

# %%
plt.rcParams.update({"font.size": 10})

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[0])
disp.plot()
plt.title(codes[0].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[1])
disp.plot()
plt.title(codes[1].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[2])
disp.plot()
plt.title(codes[2].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[3])
disp.plot()
plt.title(codes[3].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[4])
disp.plot()
plt.title(codes[4].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[5])
disp.plot()
plt.title(codes[5].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[6])
disp.plot()
plt.title(codes[6].replace("_", " "))
plt.show()

# %%
# Look at the results for the first code
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wrkp[7])
disp.plot()
plt.title(codes[7].replace("_", " "))
plt.show()

# %%
lists = (
    list(cm_wrkp[0][1]),
    list(cm_wrkp[1][1]),
    list(cm_wrkp[2][1]),
    list(cm_wrkp[3][1]),
    list(cm_wrkp[4][1]),
    list(cm_wrkp[5][1]),
    list(cm_wrkp[6][1]),
    list(cm_wrkp[7][1]),
)

# %%
codes = list(mlb.classes_)
code_labels = [code.replace("_", " ") for code in codes]
code_labels = ["\n".join(wrap(l, 40)) for l in code_labels]

# %%
code_pred_totals = pd.DataFrame(
    lists, index=code_labels, columns=["Incorrect", "Correct"]
)

# %%
code_pred_totals.Correct.sum()

# %%
code_pred_totals.Incorrect.sum()

# %%
42 / (42 + 29)

# %%
plt.rcParams.update({"font.size": 12})

code_pred_totals.plot(kind="barh", figsize=(10, 8))
plt.xlabel("Count", fontsize=14)
plt.title("Model predictions per code", fontsize=20)

# %%
code_lists = []
for preds in wrksp_pred_proba:
    code_list = list(pd.DataFrame(preds)[1])
    code_lists.append(code_list)

preds_df = pd.DataFrame(np.column_stack(code_lists), columns=codes)

# %%
preds = mlb.inverse_transform(wrksp_predictions)
worksp_y_test_vals = mlb.inverse_transform(worksp_y_test)

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
actuals = []
for pred in worksp_y_test_vals:
    if list(pred):
        for p in list(pred):
            code = p.replace("_", " ")
            actuals.append(code)
    else:
        actuals.append("No prediction")

# %%
preds_df["prediction"] = predictions
preds_df["prediction score"] = preds_df.iloc[:, :8].max(axis=1)
preds_df.loc[preds_df["prediction score"] < 0.5, "prediction score"] = 0

# %%
preds_df["actual"] = actuals
preds_df["comment"] = worksp_X_test

# %%
preds_df[
    (preds_df.actual == "Croyance que l'épidémie est terminée")
    & (preds_df.prediction != "Croyance que l'épidémie est terminée")
][["prediction", "comment"]]

# %%
predict_actual = pd.DataFrame(
    {"comment": worksp_X_test, "actual": worksp_y_test_vals, "predicted": predictions}
)

# %%

# %%
a = pd.DataFrame(acc_workshop_file1.code.value_counts().reset_index())
b = pd.DataFrame(predict_actual.predicted.value_counts().reset_index())

# %%
a.columns = ["index", "code assigned"]

# %%
merged = pd.merge(a, b, on="index", how="outer")
merged.set_index("index", inplace=True)

# %%
merged

# %%
merged.predicted.sum()

# %%
merged["code assigned"].sum()

# %%
plt.rcParams.update({"font.size": 22})

merged.plot(kind="barh", figsize=(15, 12))
plt.xlabel("Count", fontsize=22)
plt.ylabel(" ", fontsize=22)
plt.title("Code predicted by model vs the code suggested in workshop", fontsize=25)
plt.show()

# %%
predict_actual[predict_actual.predicted == "No prediction"]

# %%
