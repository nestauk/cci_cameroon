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

# Project modules
import cci_cameroon

# %%
# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %%
# File links
w1_file = "multi_label_output_w1.xlsx"
w2_file = "workshop2_comments_french.xlsx"
# Read workshop files
workshop_1 = pd.read_excel(f"{project_directory}/inputs/data/" + w1_file)
workshop_2 = pd.read_excel(f"{project_directory}/inputs/data/" + w2_file)

# %%
