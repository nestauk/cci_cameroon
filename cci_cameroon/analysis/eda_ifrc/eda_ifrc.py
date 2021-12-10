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
import pandas as pd
import numpy as np
import cci_cameroon
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %%
ifrc_data = pd.read_excel(
    f"{project_directory}/inputs/data/COVID_19 Community feedback_Cameroon.xlsx",
    sheet_name="FEEDBACK DATA_DONNEES",
)

# %%
ifrc_data.head(1)  # Check results

# %% [markdown]
# A mix of objects and float columns

# %%
ifrc_data.info()


# %% [markdown]
# ### Missing data

# %%
def percent_missing(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame(
        {"column_name": df.columns, "percent_missing": percent_missing}
    )
    missing_value_df.sort_values("percent_missing", inplace=True)
    return missing_value_df


# %%
percent_missing_df = percent_missing(ifrc_data)

# %%
percent_missing_df.plot(kind="barh", figsize=(7, 5))
plt.title("Percent missing in df - all columns")

# %%
# Nine columns completely missing
completely_missing = list(
    percent_missing_df[percent_missing_df["percent_missing"] == 100]["column_name"]
)
len(completely_missing)

# %%
# Remove completely missing
ifrc_data.drop(completely_missing, inplace=True, axis=1)

# %%
# Two columns > 97% missing
high_missing = list(
    percent_missing_df[percent_missing_df["percent_missing"] > 90].head(2)[
        "column_name"
    ]
)

# %%
high_missing

# %%
# Remove > 97% missing
ifrc_data.drop(high_missing, inplace=True, axis=1)

# %%
percent_missing_df = percent_missing(ifrc_data)

# %%
percent_missing_df.plot(kind="barh", figsize=(7, 5))
plt.title("Percent missing in df - remaining columns")

# %% [markdown]
# No columns are identical in missing behaviour but category and code have a section of inputs that match in missingness.

# %%
msno.matrix(ifrc_data)

# %% [markdown]
# ### Unique values

# %%
fig, ax = plt.subplots(dpi=100)
ifrc_data.nunique().sort_values().plot(kind="barh")
plt.title("Number of unique values in df columns")
fig.savefig("figs/unique_values_df.png", bbox_inches="tight")

# %%
# COUNTRY_PAYS has only 1 value (Cameroon)
ifrc_data.nunique()[ifrc_data.nunique() == 1]

# %%
# Remove COUNTRY_PAYS
ifrc_data.drop(["COUNTRY_PAYS"], inplace=True, axis=1)

# %% [markdown]
# Need to find out more about the 'UNIQUE_ID' column. Does it represent the person interviewed (if so there are descrepencies where the age info is different for one case).

# %%
fig, ax = plt.subplots(dpi=100)
ifrc_data["UNIQUE_ID"].value_counts().reset_index()["UNIQUE_ID"].value_counts().tail(
    8
).plot(kind="bar")
plt.title("Frequency of value counts of UNIQUE_ID (greater than 1)")
fig.savefig("figs/frequency_uniqueID.png", bbox_inches="tight")

# %%
ifrc_data["UNIQUE_ID"].value_counts().head(10).plot(kind="bar")
plt.title("Most common 10 values - UNIQUE_ID")

# %% [markdown]
# ### Re-name columns

# %%
ifrc_data.columns

# %%
ifrc_data.columns = [
    "id",
    "data",
    "area",
    "feedback_channel",
    "gender",
    "age_range",
    "other_diversity_factors",
    "feedback_comment",
    "type of feedback",
    "frequency",
    "action_taken",
    "category",
    "code",
]

# %% [markdown]
# ### Demographic questions

# %% [markdown]
# #### Area

# %%
fig, ax = plt.subplots(dpi=100)

print("Number of unique values: " + str(ifrc_data["area"].nunique()))
ifrc_data["area"].value_counts().sort_values().tail(20).plot(
    kind="barh", figsize=(5, 10)
)
plt.title("Top 20 values in the area column")

fig.savefig("figs/top_20_area.png", bbox_inches="tight")

# %%
percent_missing_df[
    percent_missing_df["column_name"]
    == "DISTRICT/STATE/REGION/CITY_DISTRICT/ETAT/REGION/VILLE"
]["percent_missing"]

# %% [markdown]
# #### Gender

# %%
fig, ax = plt.subplots(dpi=100)

print("Number of unique values: " + str(ifrc_data["gender"].nunique()))
ifrc_data["gender"].value_counts(normalize=True).mul(100).sort_values().plot(
    kind="bar", figsize=(5, 5)
)
plt.title("Percentage per gender")

fig.savefig("figs/percent_gender.png", bbox_inches="tight")

# %%
percent_missing_df[percent_missing_df["column_name"] == "SEX_SEXE"]["percent_missing"]

# %% [markdown]
# #### Age range

# %%
fig, ax = plt.subplots(dpi=100)

print("Number of unique values: " + str(ifrc_data["age_range"].nunique()))
ifrc_data["age_range"].value_counts(normalize=True).mul(100).sort_values().plot(
    kind="bar", figsize=(5, 3)
)
plt.title("Percentage per age_range")

fig.savefig("figs/percent_age.png", bbox_inches="tight")

# %%
percent_missing_df[percent_missing_df["column_name"] == "AGE RANGE_TRANCHE D'AGE"][
    "percent_missing"
]

# %% [markdown]
# #### Other diversity factors

# %%
fig, ax = plt.subplots(dpi=100)

print("Number of unique values: " + str(ifrc_data["other_diversity_factors"].nunique()))
ifrc_data["other_diversity_factors"].value_counts(normalize=True).mul(
    100
).sort_values().plot(kind="bar", figsize=(5, 3))
plt.title("Percentage per other_diversity_factors")

fig.savefig("figs/percent_disability.png", bbox_inches="tight")

# %%
# 70% 'None'
ifrc_data["other_diversity_factors"].value_counts(normalize=True).mul(100).head(1)

# %%
percent_missing_df[
    percent_missing_df["column_name"]
    == "OTHER DIVERSITY FACTORS_AUTRES FACTEURS DE DIVERSITE"
]["percent_missing"]

# %% [markdown]
# ### Feedback channel

# %%
ifrc_data["feedback_channel"].value_counts(normalize=True).mul(100).head(5)

# %%
fig, ax = plt.subplots(dpi=100)

print("Number of unique values: " + str(ifrc_data["feedback_channel"].nunique()))
ifrc_data["feedback_channel"].value_counts(normalize=True).mul(100).sort_values().plot(
    kind="bar", figsize=(9, 5)
)
plt.title("Percentage per feedback channel")

fig.savefig("figs/feedback_channel.png", bbox_inches="tight")

# %%
feedback_channel = (
    ifrc_data.groupby(["feedback_channel", "gender"]).size().reset_index(name="count")
)
feedback_channel = feedback_channel.pivot(
    index="feedback_channel", columns="gender", values="count"
).fillna(0)
feedback_channel["Total"] = (
    feedback_channel["Don't know"]
    + feedback_channel["Female"]
    + feedback_channel["Male "]
    + feedback_channel["Mixed"]
)
feedback_channel[list(feedback_channel.columns[0:4])] = (
    feedback_channel[list(feedback_channel.columns[0:4])].div(
        feedback_channel["Total"].values, axis=0
    )
    * 100
)

# %%
fig, ax = plt.subplots(dpi=100)

feedback_channel["Mixed"].sort_values().plot(kind="bar")
plt.title("Percent responses being mixed gender groups per feedback channel")

fig.savefig("figs/mixed_gender_feedback_channel.png", bbox_inches="tight")

# %% [markdown]
# ### Number of times / frequency

# %% [markdown]
# - 'Collect feedback' and 'respond to the question' found as French text in the values

# %%
ifrc_data["frequency"] = pd.to_numeric(ifrc_data["frequency"], errors="coerce")
ifrc_data["frequency"].fillna(0, inplace=True)

# %%
plt.hist(ifrc_data["frequency"], density=True, bins=60)

# %% [markdown]
# ### Action taken

# %%
fig, ax = plt.subplots(dpi=100)

print("Number of unique values: " + str(ifrc_data["action_taken"].nunique()))
ifrc_data["action_taken"].value_counts(normalize=True).mul(100).head(10).plot(
    kind="barh", figsize=(9, 5)
)
plt.title("Top 10 values for 'actions taken' field (%)")

fig.savefig("figs/actions_taken.png", bbox_inches="tight")

# %% [markdown]
# ### Type of feedback / category / code

# %% [markdown]
# #### Type of feedback

# %%
print("Number of unique values: " + str(ifrc_data["type of feedback"].nunique()))
print(ifrc_data["type of feedback"].value_counts(normalize=True).mul(100))
print(ifrc_data["type of feedback"].value_counts())

# %% [markdown]
# #### Category

# %%
print("Number of unique values: " + str(ifrc_data["category"].nunique()))

# %%
feedback_type_cat = (
    ifrc_data.groupby(["type of feedback", "category"]).size().reset_index(name="count")
)
types_of_feedback = list(feedback_type_cat["type of feedback"].unique())

# %%
fig, ax = plt.subplots(dpi=100)

for f_type in types_of_feedback:
    feedback_type_cat[feedback_type_cat["type of feedback"] == f_type].set_index(
        "category"
    )["count"].sort_values().plot(kind="barh", figsize=(9, 5))
    plt.title("Categories under " + f_type)
    fig.savefig("figs/categories_in_" + f_type + ".png", bbox_inches="tight")

# %%
feedback_type_cat[
    feedback_type_cat["type of feedback"] == "Rumors_beliefs_observations"
].set_index("category")["count"].sort_values()

# %%
feedback_type_cat[
    feedback_type_cat["type of feedback"] == "Rumors_beliefs_observations"
].set_index("category")["count"].sort_values().plot(
    kind="barh", figsize=(10, 9), color="#993344"
)
plt.title("Categories under rumours, beliefs and observations")

# %% [markdown]
# #### Code

# %%
print("Number of unique values: " + str(ifrc_data["code"].nunique()))

# %%
feedback_type_code = (
    ifrc_data.groupby(["type of feedback", "code"]).size().reset_index(name="count")
)

# %%
rumours_code = feedback_type_code[
    feedback_type_code["type of feedback"] == "Rumors_beliefs_observations"
].copy()
rumours_code.sort_values(by="count", inplace=True)

# %%
rumours_10 = rumours_code.tail(10)

# %%
fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
barh = ax.barh(rumours_10["code"], rumours_10["count"])
plt.title("Top 10: Codes for rumours, belief and observations")

fig.savefig("figs/codes_rumours.png", bbox_inches="tight")

# %%
feedback__cat_code = (
    ifrc_data.groupby(["type of feedback", "category", "code"])
    .size()
    .reset_index(name="count")
)
rumours = feedback__cat_code[
    feedback__cat_code["type of feedback"] == "Rumors_beliefs_observations"
].copy()

# %%
rumours.sort_values(by="count", ascending=False).head(5)

# %%
categories = list(rumours["category"].unique())

# %%
ax = sns.catplot(
    x="code",
    y="count",
    data=rumours[rumours["category"] == categories[2]],
    order=list(
        rumours[rumours["category"] == categories[2]].sort_values(
            by="count", ascending=False
        )["code"]
    ),
    kind="bar",
    color="g",
)

ax.fig.set_figwidth(12)
ax.fig.set_figheight(5)

plt.xticks(rotation=90)
plt.title(categories[2])

# %%
ax = sns.catplot(
    x="code",
    y="count",
    data=rumours[rumours["category"] == "Beliefs_about_the_disease_outbreak"],
    order=list(
        rumours[
            rumours["category"] == "Beliefs_about_the_disease_outbreak"
        ].sort_values(by="count", ascending=False)["code"]
    ),
    kind="bar",
    color="b",
)

ax.fig.set_figwidth(20)
ax.fig.set_figheight(5)

plt.xticks(rotation=90)
plt.title("Beliefs_about_the_disease_outbreak")

# %%
ax = sns.catplot(
    x="code",
    y="count",
    data=rumours[
        rumours["category"] == "Beliefs_about_behaviors_that_protect_people_prevention"
    ],
    order=list(
        rumours[
            rumours["category"]
            == "Beliefs_about_behaviors_that_protect_people_prevention"
        ].sort_values(by="count", ascending=False)["code"]
    ),
    kind="bar",
    color="r",
)

ax.fig.set_figwidth(15)
ax.fig.set_figheight(5)

plt.xticks(rotation=90)
plt.title("Beliefs_about_behaviors_that_protect_people_prevention")

# %% [markdown]
# ### Feedback comments

# %%
ifrc_data.head(3)

# %%
affected_by_disease = ifrc_data[
    ifrc_data["code"] == "Belief about who is or is not affected by the disease"
]

# %%
pd.options.display.max_colwidth = 500

# %%
list(affected_by_disease["feedback_comment"].head(10))

# %%
english_translation = [
    "COVID-19 is a disease reserved for the elderly', 'COVID-19 only affects fair-skinned people",
    "covid-19 does not catch black people', 'COVID-19 is a disease for strangers",
    "covid-19 does not affect children of God', 'covid-19 does not kill young people",
    "covid-19 only infects old people",
]

# %%
communication_info = ifrc_data[
    ifrc_data["code"]
    == "Observation or belief about communication or information about the disease"
]

# %%
list(communication_info["feedback_comment"].head(10))

# %%
english_translation = [
    "I heard about COVID-19 on television",
    "women are more open to awareness",
    "Main source of information: the media",
    "we hear about covid-19 on tv and radio",
    "We heard about COVID-19 on TV and from neighbors",
]

# %% [markdown]
# ### Language

# %%
from langdetect import detect, detect_langs

# %%
rum_bel_ob = ifrc_data[
    ifrc_data["type of feedback"] == "Rumors_beliefs_observations"
].copy()

# %%
rum_bel_ob["language"] = rum_bel_ob["feedback_comment"].apply(lambda x: detect(x))

# %%
sum(
    rum_bel_ob.groupby("language").feedback_comment.count().sort_values(ascending=False)
)

# %%
pd.DataFrame(
    rum_bel_ob.groupby("language").feedback_comment.count().sort_values(ascending=True)
).plot(kind="barh")
plt.title("Language frequency for rumour_belief_observation feedback comments")

# %% [markdown]
# ### Outstanding questions
#
# 1. Does the unique ID column represent the person being interviewed or the person conducting the survey?
#     1. If its the first, how do we explain different age values in one id and high number for 5672 (over 1,000)
# 2. Number of times (how does this work if each row is a unique item?
#
#
# ### Notes
#
# - The rules for categorisation don't seem clear in some cases. Assumption was that a code was a more granular form of a category but in some cases there are the same code across different categories.
#
#
# ### To consider
# - Are we thinking about all elements in the rumours_beliefs_observations field - even if its more of an opinion / comment rather than a rumour?

# %%
