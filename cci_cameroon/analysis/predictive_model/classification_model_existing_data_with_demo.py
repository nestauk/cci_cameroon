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
# ### Classifying rumours

# %% [markdown]
# #### Steps:
#
# 1. Takes in the 8 codes and their comments processed from W1 & W2
# 2. Performs some cleaning (removing duplicate pairs, translating text - only for W1)
# 3. Transforms comments into vectors to input into model
#     - TFIDF and sentence transformers
# 4. Runs x3 model types on both versions to produce x6 models
# 5. Takes the best performing model and creates demo version

# %%
# Import modules
import pandas as pd
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


# %% [markdown]
# The below uses actual outputs from W1 and the existing labels for W2. Will need to update based on real data / format.

# %%
# File links
w1_file = "multi_label_output_w1.xlsx"
w2_file = "workshop2_comments_french.xlsx"
# Read workshop files
w1 = pd.read_excel(f"{project_directory}/inputs/data/" + w1_file)
w2 = pd.read_excel(f"{project_directory}/inputs/data/" + w2_file)

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
print("Before duplicate pairs removed: " + str(len(labelled_data)))
labelled_data.drop_duplicates(subset=["code", "comment"], inplace=True)
print("After duplicate pairs removed: " + str(len(labelled_data)))

# %% [markdown]
# There are very few codes classified that aren't in the 8 codes chosen for labelling

# %%
# Looking at count of comments per code
labelled_data.code.value_counts()

# %%
# Removing small count codes
to_remove = list(
    labelled_data.code.value_counts()[labelled_data.code.value_counts() < 10].index
)
labelled_data = labelled_data[~labelled_data.code.isin(to_remove)].copy()

# %%
# Three cases where a comment is labelled with two different codes (from w1 questions)
labelled_data.id.value_counts().head(5)

# %% [markdown]
# The total cases of multiple code assignments is caused by both the outputs of the multiple assinments in workshop 1 and cases where a comment is assigned to different codes in the IFRC dataset.

# %%
# 15 cases where a comment is labelled with two different codes
labelled_data.comment.value_counts().rename_axis("comment").reset_index(name="count")[
    "count"
].value_counts().plot(kind="bar")
plt.title("Count of labels per comment")

# %%
# Dataset for modelling
model_data = labelled_data.copy()
# Create category ID column from the code field (join by _ into one string)
model_data["category_id"] = model_data["code"].str.replace(" ", "_")
id_to_category = dict(model_data[["category_id", "code"]].values)
model_data = (
    model_data.groupby("comment")["category_id"].apply(list).to_frame().reset_index()
)

# %% [markdown]
# ### Class balance

# %%
# Display counts per classes
labelled_data.code.value_counts().plot.bar()

# %% [markdown]
# ## Classification
#
# ### Starting with a basic model pipeline
# 1. TF-IDF vector
# 2. KNN model

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    model_data["comment"], model_data["category_id"], test_size=0.20, random_state=0
)

# %%
# Transform Y into multilabel format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)
codes = list(mlb.classes_)  # Codes list

# %%
# French stop words
stop_words = get_stop_words("fr")
# Create tfidf vector, fit with train and apply to train and test
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# %%
# KNN classifier
knnClf = KNeighborsClassifier()
knnClf.fit(X_train_vector, y_train)
knnPredictions = knnClf.predict(X_test_vector)

# %%
# Decision tree classifier
dtClassifier = DecisionTreeClassifier()
dtClassifier.fit(X_train_vector, y_train)
dtPreds = dtClassifier.predict(X_test_vector)

# %%
# Random forest classifier
rfClassifier = RandomForestClassifier(n_jobs=-1)
rfClassifier.fit(X_train_vector, y_train)
rfPreds = rfClassifier.predict(X_test_vector)

# %%
# Report results from the tfidf models
metricsReport("knnClf_tfidf", y_test, knnPredictions)
metricsReport("dtClassifier_tfidf", y_test, dtPreds)
metricsReport("rfClassifier_tfidf", y_test, rfPreds)

# %% [markdown]
# ### Using sentence transformers

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
# Decision tree classifier
dtClassifier = DecisionTreeClassifier()
dtClassifier.fit(X_train_embeddings, y_train)
dtPreds = dtClassifier.predict(X_test_embeddings)

# %%
# Random forest classifier
rfClassifier = RandomForestClassifier(n_jobs=-1)
rfClassifier.fit(X_train_embeddings, y_train)
rfPreds = rfClassifier.predict(X_test_embeddings)

# %%
# Collect and report all results
metricsReport("knnClf_tranformer", y_test, knnPredictions)
metricsReport("dtClassifier_transformer", y_test, dtPreds)
metricsReport("rfClassifier_transformer", y_test, rfPreds)
print("Results from all models")
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
# Look at the results for the second code
print(codes[1].replace("_", " "))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN[1])
disp.plot()
plt.show()

# %% [markdown]
# ### Live view of the model

# %% [markdown]
# Using best performing model - KNN with transformer. Set the threshold to see probabilities for each class (eg 0.8 will show probabilities >= 80%)

# %%
predict_rumour(0.8)

# %% [markdown]
# ### Issues to solve /things to still do:
# - Update with reduced dataset from workshop this could:
#     - Improve the model as it is cleaning the dataset (removing incorrect comments in codes)
#     - Badly affect the model as the number of comments per class greatly reduce
#         - What do we do in this case?
# - It doesn't currently handle multi-class cases well
# - Tuning of model using training set
# - Looking at what features / words / phrases are strong predictors for a class
