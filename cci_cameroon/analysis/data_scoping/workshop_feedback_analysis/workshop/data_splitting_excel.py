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

# %%
df = pd.read_excel("comments_label_workshop.xlsx")

# %%
df.head(5)

# %%
df = shuffle(df)

# %%
df.head(5)


# %%
def split_dataframe(df, chunk_size):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
    return chunks


# %%
split_dataframe(df, 40)[0].shape

# %%
df1 = split_dataframe(df, 40)[0]
df2 = split_dataframe(df, 40)[1]
df3 = split_dataframe(df, 40)[2]
df4 = split_dataframe(df, 40)[3]
df5 = split_dataframe(df, 40)[4]
df6 = split_dataframe(df, 40)[5]
df7 = split_dataframe(df, 40)[6]
df8 = split_dataframe(df, 40)[7]
df9 = split_dataframe(df, 40)[8]
df10 = split_dataframe(df, 40)[9]

# %%
p1 = pd.concat([df1, df4, df7])
p2 = pd.concat([df1, df4, df8])
p3 = pd.concat([df1, df5, df8])
p4 = pd.concat([df2, df5, df8])
p5 = pd.concat([df2, df5, df9])
p6 = pd.concat([df2, df6, df9])
p7 = pd.concat([df3, df6, df9])
p8 = pd.concat([df3, df6, df10])
p9 = pd.concat([df3, df7, df10])
p10 = pd.concat([df4, df7, df10])

# %%
p9.tail(1)

# %%
p10.tail(1)


# %%
def empty_rows(df):
    n = 2
    new_index = pd.RangeIndex(len(df) * (n + 1))
    new_df = pd.DataFrame(np.nan, index=new_index, columns=df.columns)
    ids = np.arange(len(df)) * (n + 1)
    new_df.loc[ids] = df.values
    return new_df


# %%
p1 = empty_rows(p1)
p2 = empty_rows(p2)
p3 = empty_rows(p3)
p4 = empty_rows(p4)
p5 = empty_rows(p5)
p6 = empty_rows(p6)
p7 = empty_rows(p7)
p8 = empty_rows(p8)
p9 = empty_rows(p9)
p10 = empty_rows(p10)

# %%
p2.head(10)

# %%
p1.to_excel("files/person1.xlsx", index=False)
p2.to_excel("files/person2.xlsx", index=False)
p3.to_excel("files/person3.xlsx", index=False)
p4.to_excel("files/person4.xlsx", index=False)
p5.to_excel("files/person5.xlsx", index=False)
p6.to_excel("files/person6.xlsx", index=False)
p7.to_excel("files/person7.xlsx", index=False)
p8.to_excel("files/person8.xlsx", index=False)
p9.to_excel("files/person9.xlsx", index=False)
p10.to_excel("files/person10.xlsx", index=False)

# %%
p9.tail(10)

# %%
p10.tail(10)

# %%
p8.tail(10)

# %%
