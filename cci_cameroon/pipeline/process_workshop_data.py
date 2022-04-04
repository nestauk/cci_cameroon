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
from googletrans import Translator
from langdetect import detect, detect_langs

# %%
translator = Translator()


def translate_column(df, col):
    """
    Translate english and spanish text to french in df column.
    """
    # Adding language column
    df["language"] = df[col].apply(lambda x: detect(x))
    # Slicing the data into en, es and remaining
    en = df[df.language == "en"].copy()
    es = df[df.language == "es"].copy()
    df_rem = df[~df.language.isin(["en", "es"])].copy()
    # Translating the English and Spanish comments into French
    en[col] = en.comment.apply(translator.translate, src="en", dest="fr").apply(
        getattr, args=("text",)
    )
    es[col] = es.comment.apply(translator.translate, src="es", dest="fr").apply(
        getattr, args=("text",)
    )
    # Merge back together
    df_comb = pd.concat([df_rem, en, es], ignore_index=True)
    # Reemove language
    df_comb.drop("language", axis=1, inplace=True)
    return df_comb


# %%
def combine_files_clean(w1, w2):
    """
    Combine and clean x2 files of comments and codes.
    """
    # Join the two workshops files together
    labelled_data = w1.append(w2, ignore_index=True)
    # Remove white space before and after text
    labelled_data.replace(r"^ +| +$", r"", regex=True, inplace=True)
    # Removing 48 duplicate code/comment pairs (from W1)
    labelled_data.drop_duplicates(subset=["code", "comment"], inplace=True)
    # Removing small count codes
    to_remove = list(
        labelled_data.code.value_counts()[labelled_data.code.value_counts() < 10].index
    )
    model_data = labelled_data[~labelled_data.code.isin(to_remove)].copy()
    return model_data


# %%
def category_id(labelled_data):
    # Create category ID column from the code field (join by _ into one string)
    labelled_data["category_id"] = labelled_data["code"].str.replace(" ", "_")
    group_labelled_data = (
        labelled_data.groupby("comment")["category_id"]
        .apply(list)
        .to_frame()
        .reset_index()
    )
    # group_labelled_data['category_id'] = group_labelled_data['category_id'].str[0]
    return group_labelled_data
