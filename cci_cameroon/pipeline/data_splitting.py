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
# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

# Project modules
import cci_cameroon
from cci_cameroon.getters import get_data as gd
from cci_cameroon.pipeline import process_workshop_data as pwd

# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %%
w1 = gd.workshop_1
w2 = gd.workshop_2

# %%
# Update to add to data getters
ifrc_data = pd.read_excel(
    f"{project_directory}/inputs/data/COVID_19 Community feedback_Cameroon.xlsx",
    sheet_name="FEEDBACK DATA_DONNEES ",
)

# %%
codes_remove = [
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
def get_other_codes(df):
    other_codes = df[
        df["TYPE OF FEEDBACK_TYPE DE RETOUR D'INFORMATION"]
        == "Rumors_beliefs_observations"
    ].copy()
    other_codes = other_codes[["CODE", "FEEDBACK COMMENT_COMMENTAIRE"]].copy()
    other_codes.replace(r"^ +| +$", r"", regex=True, inplace=True)
    other_codes.columns = ["code", "comment"]
    other_codes.drop_duplicates(subset=["code", "comment"], inplace=True)
    other_codes = other_codes.loc[~other_codes["code"].isin(codes_remove)].copy()
    other_codes.dropna(how="any", inplace=True)
    other_codes_sample = other_codes.sample(n=150, random_state=1)
    other_codes_sample = pwd.translate_column(other_codes_sample, "comment")
    other_codes_sample["code"] = "no code"
    other_codes_sample = other_codes_sample.reset_index()
    other_codes_sample = other_codes_sample.rename(columns={"index": "id"})
    other_codes_sample["id"] = other_codes_sample.index + 3887
    other_codes_sample.set_index("id", inplace=True)
    return other_codes_sample


# %%
other_codes = get_other_codes(ifrc_data)

# %%
# other_codes.reset_index('')

# %%
# Sort to ensure the order is consistent each time before splitting
w1.sort_values(by="id", inplace=True)
w2.sort_values(by="id", inplace=True)

# %%
w1 = pwd.translate_column(w1, "comment")
model_data = pwd.combine_files_clean(w1, w2)
model_data = pwd.category_id(model_data)

# %%
model_data = shuffle(model_data, random_state=1)

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    model_data["comment"], model_data["category_id"], test_size=0.20, random_state=1
)

# %%
# Split data
other_train, other_test = train_test_split(
    other_codes["comment"], test_size=0.20, random_state=1
)

# %%
X_train

# %%
other_train

# %%
len(other_test)

# %%
X_train

# %%
print(X_train.shape)
print(X_test.shape)

# %%
# Write into csv files
X_train.to_excel(
    f"{project_directory}/inputs/data/data_for_modelling/X_train.xlsx",
    index=True,
    index_label="id",
)

X_test.to_excel(
    f"{project_directory}/inputs/data/data_for_modelling/X_test.xlsx",
    index=True,
    index_label="id",
)

y_train.to_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_train.xlsx",
    index=True,
    index_label="id",
)

y_test.to_excel(
    f"{project_directory}/inputs/data/data_for_modelling/y_test.xlsx",
    index=True,
    index_label="id",
)

# %%
other_train.to_excel(
    f"{project_directory}/inputs/data/data_for_modelling/no_response_train.xlsx",
    index=True,
    index_label="id",
)

other_test.to_excel(
    f"{project_directory}/inputs/data/data_for_modelling/no_response_test.xlsx",
    index=True,
    index_label="id",
)

# %%
other_train

# %%
