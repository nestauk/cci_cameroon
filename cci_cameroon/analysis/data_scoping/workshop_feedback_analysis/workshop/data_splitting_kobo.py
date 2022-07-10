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
from sklearn.utils import shuffle
import numpy as np
import cci_cameroon
from langdetect import detect, detect_langs
from googletrans import Translator

# Set directory
project_directory = cci_cameroon.PROJECT_DIR

# %%
ifrc_data = pd.read_excel(
    f"{project_directory}/inputs/data/COVID_19 Community feedback_Cameroon.xlsx",
    sheet_name="FEEDBACK DATA_DONNEES ",
)

# %%
workshop1 = pd.read_excel("comments_label_workshop.xlsx")

# %%
cameroon_rumours = ifrc_data[
    (
        ifrc_data["TYPE OF FEEDBACK_TYPE DE RETOUR D'INFORMATION"]
        == "Rumors_beliefs_observations"
    )
    & (ifrc_data["COUNTRY_PAYS"] == "Cameroon")
][["FEEDBACK COMMENT_COMMENTAIRE", "CODE"]].copy()

# %%
cameroon_rumours.columns = ["comment", "code"]

# %%
cameroon_rumours = cameroon_rumours[cameroon_rumours["code"].notna()].copy()

# %%
workshop1.drop("ID", axis=1, inplace=True)

# %%
cameroon_rumours.replace(r"^ +| +$", r"", regex=True, inplace=True)
workshop1.replace(r"^ +| +$", r"", regex=True, inplace=True)

# %%
cameroon_rumours.drop_duplicates(inplace=True)
workshop1.drop_duplicates(inplace=True)

# %%
workshop1.shape

# %%
cameroon_rumours.shape

# %%
workshop2 = (
    pd.merge(
        cameroon_rumours, workshop1, on=["code", "comment"], indicator=True, how="outer"
    )
    .query('_merge=="left_only"')
    .drop("_merge", axis=1)
)

# %%
# Starting index at 401 (workshop 1 went to 400 so we can merge later)
workshop2.reset_index(inplace=True, drop=True)
workshop2.index += 401

# %%
workshop2 = workshop2.rename_axis("id").reset_index()

# %%
workshop1.head(1)

# %%
workshop2.tail(2)

# %%
# Saving file so we can retrive the index numbers later (will need to merge with the workshop 1 indexed file)
workshop2.to_excel(f"{project_directory}/outputs/data/codes_comments_index.xlsx")

# %%
workshop2.shape

# %%
workshop2.tail(2)

# %%
translator = Translator()
workshop2["language"] = workshop2["comment"].apply(lambda x: detect(x))

# %%
workshop2.head(5)

# %%
pd.DataFrame(
    workshop2.groupby("language").comment.count().sort_values(ascending=False)
).head(5)

# %%
en = workshop2[workshop2.language == "en"].copy()
es = workshop2[workshop2.language == "es"].copy()
ca = workshop2[workshop2.language == "ca"].copy()
it = workshop2[workshop2.language == "it"].copy()

# %%
# workshop2 = workshop2[~workshop2.language.isin(['en', 'es', 'ca', 'it'])].copy()
fr = workshop2[workshop2.language == "fr"].copy()

# %%
en["comment"] = en.comment.apply(translator.translate, src="en", dest="fr").apply(
    getattr, args=("text",)
)
es["comment"] = es.comment.apply(translator.translate, src="es", dest="fr").apply(
    getattr, args=("text",)
)
ca["comment"] = ca.comment.apply(translator.translate, src="ca", dest="fr").apply(
    getattr, args=("text",)
)
it["comment"] = it.comment.apply(translator.translate, src="it", dest="fr").apply(
    getattr, args=("text",)
)

# %%
workshop2 = pd.concat([en, fr, es, ca, it], ignore_index=True)

# %%
workshop2.shape

# %%
workshop2.drop("language", axis=1, inplace=True)

# %%
workshop2.code.value_counts().head(13)

# %%
# Code is >=120
# Code is not too broad

# %%
codes = [
    "Beliefs about ways of transmission",
    "Belief that some people/institutions are making money because of the disease",
    "Belief that the outbreak has ended",
    "Observations of non-compliance with health measures",
    "Belief that the disease does not exist in this region or country",
]

# %%
workshop2 = workshop2[workshop2["code"].isin(codes)].copy()

# %%
# Removing a few more comments to get the size down to divide by 11
to_remove1 = list(workshop2[workshop2.code == "Beliefs about ways of transmission"].id)[
    0:12
]

# %%
workshop2 = workshop2[~workshop2["id"].isin(to_remove1)]

# %%
workshop2.shape

# %%
workshop2.to_excel("temp_workshop2_french.xlsx")

# %%
workshop2 = pd.read_excel("temp_workshop2_french.xlsx")

# %%
workshop2.head(10)

# %%
workshop2 = shuffle(workshop2)

# %%
workshop2["comment"] = "Commenter: " + workshop2["comment"].astype(str)
workshop2["code"] = "###### Code: " + workshop2["code"].astype(str)

# %%
workshop2

# %%
workshop2.shape


# %%
def split_dataframe(df, chunk_size):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
    return chunks


# %%
split_dataframe(workshop2, 85)[0].shape

# %%
dfs = split_dataframe(workshop2, 85)

# %%
df1 = dfs[0]
df2 = dfs[1]
df3 = dfs[2]
df4 = dfs[3]
df5 = dfs[4]
df6 = dfs[5]
df7 = dfs[6]
df8 = dfs[7]
df9 = dfs[8]
df10 = dfs[9]
df11 = dfs[10]

# %%
p1 = pd.concat([df1, df4, df8])
p2 = pd.concat([df1, df5, df8])
p3 = pd.concat([df1, df5, df9])
p4 = pd.concat([df2, df5, df9])
p5 = pd.concat([df2, df6, df9])
p6 = pd.concat([df2, df6, df10])
p7 = pd.concat([df3, df6, df10])
p8 = pd.concat([df3, df7, df10])
p9 = pd.concat([df3, df7, df11])
p10 = pd.concat([df4, df7, df11])
p11 = pd.concat([df4, df8, df11])

# %%
p1 = shuffle(p1)
p2 = shuffle(p2)
p3 = shuffle(p3)
p4 = shuffle(p4)
p5 = shuffle(p5)
p6 = shuffle(p6)
p7 = shuffle(p7)
p8 = shuffle(p8)
p9 = shuffle(p9)
p10 = shuffle(p10)
p11 = shuffle(p11)

# %%
p1.shape


# %%
# Empty rows to fit the format of the XLS form sheet (so can directly paste into)
def add_empty_rows(df):
    s = pd.Series(np.nan, df.columns)
    f = lambda d: d.append([s, s, s], ignore_index=True)
    grp = np.arange(len(df)) // 10
    df_updated = df.groupby(grp, group_keys=False).apply(f).reset_index(drop=True)
    return df_updated


# %%
p1 = add_empty_rows(p1)
p2 = add_empty_rows(p2)
p3 = add_empty_rows(p3)
p4 = add_empty_rows(p4)
p5 = add_empty_rows(p5)
p6 = add_empty_rows(p6)
p7 = add_empty_rows(p7)
p8 = add_empty_rows(p8)
p9 = add_empty_rows(p9)
p10 = add_empty_rows(p10)
p11 = add_empty_rows(p11)

# %%
p1.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person1.xlsx",
    index=False,
)
p2.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person2.xlsx",
    index=False,
)
p3.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person3.xlsx",
    index=False,
)
p4.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person4.xlsx",
    index=False,
)
p5.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person5.xlsx",
    index=False,
)
p6.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person6.xlsx",
    index=False,
)
p7.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person7.xlsx",
    index=False,
)
p8.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person8.xlsx",
    index=False,
)
p9.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person9.xlsx",
    index=False,
)
p10.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person10.xlsx",
    index=False,
)
p11.to_excel(
    f"{project_directory}/outputs/data/workshop2_files/comments_codes/person11.xlsx",
    index=False,
)

# %%
