# -*- coding: utf-8 -*-
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
from sentence_transformers import SentenceTransformer, util
import requests
from io import StringIO
import time

# %%
# install sentence tranformers library using command below
# conda install -c conda-forge sentence-transformers


# %%
# read in sample text dataset from online
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

# %% [markdown]
# # Similarity search - things to consider when choosing an approach
#
# 1. Symetric -> the query is identical to the text stored in length and form. Candidate transformer distilBERT
# 2. Assymetric -> the query differs from the text stored. Text stored is usually larger than the query. This is the category into which our task falls.
#
# Questions for team:
#
# 1. do we consider this problem a symetric or assymetric one? I guess assymetric because we do not have control over the size of rumours created?
#
# 2. With concerns on having rumours in the database in both English and French, do we train separate models to handle this?
#
# 3. Should we be more concerned with great match of the query or with computation time? This will influence our choice of [FAISS](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) index to use
#
#
#

# %%
data.columns

# %%
sentences[:10]

# %%
# remove duplicates and NaN
sentences = [word for word in list(set(sentences)) if type(word) is str]

# %%
# initialize sentence transformer model - using BERT. Sentence embedding takes time
model = SentenceTransformer(
    "bert-base-nli-mean-tokens"
)  # loading simple pre-trained transformer model for learning purposes
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
print("Similarity:", util.dot_score(xq, sentence_embeddings)[0][0])

# %%
print("Similarity:", util.dot_score(xq, sentence_embeddings)[0][1])

# %%
sentences[202]

# %%
sentences[9133]

# %% [markdown]
# # Output using the Flat index with L2
# * Algorithm runs for 5µs and returns top 2 nearest neighbors to the input text.
# * First text has a similarity index of 57.92
# * Second output has similarity index of 4.78
# * Clearly, the sentence with a higher similarity is closer to the input query

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

# %%
print("Similarity:", util.dot_score(xq[0], sentence_embeddings)[0][202])

# %%
print("Similarity:", util.dot_score(xq[0], sentence_embeddings)[0][9133])

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

# %%
print("Similarity:", util.dot_score(xq, sentence_embeddings)[0][1])

# %% [markdown]
# ## For the given dataset, we notice that the algorithm returns the same set of indexes. However, when compression of the vectors happens, the indexes are returned in a reversed order. This shows that accuracy is compromised. The least distance between the vectors increases from 90 to 130.

# %%
import pickle

# %%
# using the distilbert model for embedding
model2 = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

# %%
# produce sentence vectors
embeddings = model2.encode(sentences)

# %%
# create a faiss index for the search using the embenddings
index_dis = faiss.IndexFlatL2(embeddings.shape[1])

# %%
index_dis.add(embeddings)  # add the vector to the index created

# %%
# search the index for our given input xq
# %time
distance, position = index_dis.search(xq, k)  # search
print(position)
print(distance)

# %%
sentences[position[0][0]]

# %%
sentences[position[0][1]]

# %%
print("Similarity:", util.dot_score(xq[0], embeddings)[0][5323])

# %%
print("Similarity:", util.dot_score(xq[0], sentence_embeddings)[0][6622])

# %% [markdown]
#
#
