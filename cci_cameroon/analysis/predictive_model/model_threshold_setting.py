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

import seaborn as sns
from matplotlib import rcParams
import sklearn.metrics as metrics
from sklearn.calibration import calibration_curve

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
model_data["category_id"]

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    model_data["comment"], model_data["category_id"], test_size=0.20, random_state=0
)

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

# %%
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

# %%
