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
)
from sklearn.neighbors import KNeighborsClassifier

# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %% [markdown]
# The below uses actual outputs from W1 and the existing labels for W2. Will need to update based on real data / format.

# %%
# File links
w1_file = "multi_label_output_w1.xlsx"
w2_file = "workshop2_comments_french.xlsx"

w1 = pd.read_excel(f"{project_directory}/outputs/data/" + w1_file)

w2 = pd.read_excel(f"{project_directory}/outputs/data/" + w2_file)

# %%
w1.shape

# %%
translator = Translator()
w1["language"] = w1["comment"].apply(lambda x: detect(x))

# %%
pd.DataFrame(w1.groupby("language").comment.count().sort_values(ascending=False)).head(
    15
)

# %%
en = w1[w1.language == "en"].copy()
es = w1[w1.language == "es"].copy()
w1 = w1[~w1.language.isin(["en", "es"])].copy()

# %%
en["comment"] = en.comment.apply(translator.translate, src="en", dest="fr").apply(
    getattr, args=("text",)
)
es["comment"] = es.comment.apply(translator.translate, src="es", dest="fr").apply(
    getattr, args=("text",)
)

# %%
w1 = pd.concat([w1, en, es], ignore_index=True)

# %%
w1.drop("language", axis=1, inplace=True)

# %%
labelled_data = w1.append(w2, ignore_index=True)

# %%
labelled_data.head(5)

# %%
# Remove white space before and after text
labelled_data.replace(r"^ +| +$", r"", regex=True, inplace=True)

# %%
# Removing 48 duplicate code/comment pairs (from W1)
print("Before duplicate pairs removed: " + str(len(labelled_data)))
labelled_data.drop_duplicates(subset=["code", "comment"], inplace=True)
print("After duplicate pairs removed: " + str(len(labelled_data)))

# %%
labelled_data.head(1)

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

# %% [markdown]
# Will need to assess again when all the real data is through as to whether there is enough cases to warrent a multi-label model.

# %%
# 15 cases where a comment is labelled with two different codes
labelled_data.comment.value_counts().rename_axis("comment").reset_index(
    name="count"
).head(20)

# %%
model_data = labelled_data.copy()
model_data["category_id"] = model_data["code"].factorize()[0]
model_data = (
    model_data.groupby("comment")["category_id"].apply(list).to_frame().reset_index()
)

# %%
model_data.head(15)

# %% [markdown]
# ### Class balance

# %%
labelled_data.code.value_counts().plot.bar()

# %% [markdown]
# ## Classification
#
# ### Starting with a basic model pipeline
# 1. TF-IDF vector
# 2. KNN model

# %%
X_train, X_test, y_train, y_test = train_test_split(
    model_data["comment"], model_data["category_id"], test_size=0.33, random_state=0
)

# %%
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)

# %%
y_train

# %%
list(mlb.classes_)

# %%
stop_words = get_stop_words("fr")

# %%
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# %%
knnClf = KNeighborsClassifier()
knnClf.fit(X_train, y_train)
knnPredictions = knnClf.predict(X_test)

# %%
ModelsPerformance = {}


def metricsReport(modelName, test_labels, predictions):
    macro_f1 = f1_score(test_labels, predictions, average="macro")
    micro_f1 = f1_score(test_labels, predictions, average="micro")
    hamLoss = hamming_loss(test_labels, predictions)
    ModelsPerformance[modelName] = {
        "Macro f1": macro_f1,
        "Micro f1": micro_f1,
        "Loss": hamLoss,
    }


# %%
metricsReport(knnClf, y_test, knnPredictions)

# %%
ModelsPerformance
