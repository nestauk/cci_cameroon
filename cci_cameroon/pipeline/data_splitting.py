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
print(X_train.shape)
print(X_test.shape)

# %%
# Write into csv files
X_train.to_excel(
    f"{project_directory}/outputs/data/data_for_modelling/X_train.xlsx",
    index=True,
    index_label="id",
)

X_test.to_excel(
    f"{project_directory}/outputs/data/data_for_modelling/X_test.xlsx",
    index=True,
    index_label="id",
)

y_train.to_excel(
    f"{project_directory}/outputs/data/data_for_modelling/y_train.xlsx",
    index=True,
    index_label="id",
)

y_test.to_excel(
    f"{project_directory}/outputs/data/data_for_modelling/y_test.xlsx",
    index=True,
    index_label="id",
)

# %%
