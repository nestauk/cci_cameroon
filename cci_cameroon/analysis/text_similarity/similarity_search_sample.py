# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import os
import scipy
import tensorflow as tf
import faiss
from sentence_transformers import SentenceTransformer
import requests
from io import StringIO
import time

# %%
# install sentence tranformers library using command below
# conda install -c conda-forge sentence-transformers


# %%
# read in dataset from online
urls = [
    "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.train.tsv",
    "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.test.tsv",
    "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/OnWN.test.tsv",
    "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2013/OnWN.test.tsv",
    "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/OnWN.test.tsv",
    "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/images.test.tsv",
    "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2015/images.test.tsv",
]
# we loop through each url and create our sentences data
sentences = []
for url in urls:
    res = requests.get(url)
    # extract to dataframe
    data = pd.read_csv(StringIO(res.text), sep="\t", header=None, error_bad_lines=False)
    # add to columns 1 and 2 to sentences list
    sentences.extend(data[1].tolist())
    sentences.extend(data[2].tolist())

# %%
data.columns

# %%
sentences[:10]

# %%
# remove duplicates and NaN
sentences = [word for word in list(set(sentences)) if type(word) is str]

# %%
# initialize sentence transformer model - using sentence-BERT. Sentence embedding takes time
model = SentenceTransformer("bert-base-nli-mean-tokens")
# create sentence embeddings
sentence_embeddings = model.encode(sentences)
sentence_embeddings.shape

# %%
dimension = sentence_embeddings.shape[1]

# %%
# initialize the index with the dimension of the dataset
index = faiss.IndexFlatL2(dimension)

# %%
sentence_embeddings.shape

# %%
index.is_trained

# %%
# we load the embeddings to the index
index.add(sentence_embeddings)
index.ntotal

# %%
sentences[:5]


# %%
def readInput():
    txt = input("Enter text and hit enter : ")  # read user's input
    return str(txt)


# %%
user_input = readInput()

# %%
k = 2  # we wish to return the index of the two closest strings to our query
xq = model.encode([user_input])

# %%
# we now perform a search for similarity
# %time
distance, position = index.search(xq, k)  # search
print(position)
print(distance)

# %%
sentences[202]

# %%
sentences[9133]

# %% [markdown]
# # Drawback of  IndexFlatL2
# It performs and exhaustive search, comparing the query string to all the vectors before returning the closest match.
# The computation time quickly grows with the size of the dataset.
# * We can partition the index to improve on the search time, although this would provide an approximate match as opposed
# to the exhaustive search.

# %%
num_cells = 50  # Number of cells the index should be divided into
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, num_cells)

# %%
index.is_trained

# %%
index.train(sentence_embeddings)
index.is_trained  # check if index is now trained

# %%
index.add(sentence_embeddings)
index.ntotal  # number of embeddings indexed

# %%
# we now perform a search for similarity using the input text read earlier on.
# %time
distance, position = index.search(xq, k)  # search
print(position)
print(distance)

# %% [markdown]
# # By compressing the vectors, we further improve the computation time

# %%
m = 8  # number of centroid IDs in final compressed vectors
bits = 8  # number of bits in each centroid

quantizer = faiss.IndexFlatL2(dimension)  # we keep the same L2 distance flat index
index = faiss.IndexIVFPQ(quantizer, dimension, num_cells, m, bits)

# %%
index.train(sentence_embeddings)
index.add(sentence_embeddings)

# %%
index.nprobe = 10  # nprobe specifies the number of neigboring cells to be used

# %%
# we now perform a search for similarity using the input text read earlier on.
# %time
distance, position = index.search(xq, k)  # search
print(position)
print(distance)

# %%
sentences[position[0][0]]

# %%
sentences[position[0][1]]

# %% [markdown]
# ## For the given dataset, we notice that the algorithm returns the same set of indexex. However, when compression of the vectors happens, the indexes are returned in a reversed order. This shows that accuracy is compromised. The least distance between the vectors increases from 90 to 130.

# %%

# %%

# %% [markdown]
# ## https://www.pinecone.io/learn/faiss-tutorial/

# %%

# %%
