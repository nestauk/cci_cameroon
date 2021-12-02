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
import logging
import re

# %matplotlib inline
import torch
import regex
import omegaconf
import numpy as np

# %%
# install sentence tranformers library using command below
# conda install -c conda-forge sentence-transformers
# pip install hydra-core omegaconf
import cci_cameroon

base_dir = cci_cameroon.PROJECT_DIR


# %% [markdown]
# # Function that fetches data from remote URLs

# %%
def fetch_data(url_list):
    """function fetches data to use from a remote source(s).
    @param: list of urls
    return: list of sentences. A version that returns a dataframe that holds the rumours and the labels would
    developed for use."""
    # we loop through each url and create our sentences data
    sentences = []
    for url in urls:
        res = requests.get(url)
        # extract to dataframe
        data = pd.read_csv(
            StringIO(res.text), sep="\t", header=None, error_bad_lines=False
        )
        # add to columns 1 and 2 to sentences list
        sentences.extend(data[1].tolist())
        sentences.extend(data[2].tolist())
    return sentences


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
sentences = fetch_data(urls)

# %% [markdown]
# # Similarity search - things to consider when choosing an approach
#
# 1. Symmetric -> the query is identical to the text stored in length and form. Candidate transformer distilBERT
# 2. Assymmetric -> the query differs from the text stored. Text stored is usually larger than the query. This is the category into which our task falls I guess.
#
# Questions for team:
#
# 1. do we consider this problem a symmetric or assymmetric one? I guess assymmetric because we do not have control over the size of rumours created?
#
# 2. With concerns on having rumours in the database in both English and French, do we train separate models to handle this?
#
# 3. Should we be more concerned with great match of the query or with computation time? This will influence our choice of [FAISS](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) index to use
#
#
# N.B The FAISS library does not compute similarity score. It only returns the vector distance. However, one can normalze the vectors and compute their dot product using the appropraite index since the similarity

# %%
sentences[:10]

# %% [markdown]
# ## Removing word duplicates and NAN from the collected data

# %%
# preprocessing the sentences fetched by removing duplicates and NaN
sentences = [word for word in list(set(sentences)) if type(word) is str]

# %% [markdown]
# ## initialize sentence transformer model - using BERT. Sentence embedding takes time
# The bert-base-nli-mean-tokens pre-trained model initialized here and used to create sentence embeddings.

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

# %% [markdown]
# ## Faiss flat index
# We create a faiss flat index to search the vector embeddings for a match with a new string.
# The Euclidean distance is used to measure similarity between strings. The shorter the distance,
# the greater the similarity.

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
sentences[:15]


# %% [markdown]
# ## readInput
# a function to read in sample string used in the match search.

# %%
# This function allows a user to input a search string for testing
def readInput():
    txt = input("Enter text and hit enter : ")  # read user's input
    return str(txt)


# %%
# read sample input
user_input = readInput()

# %%
k = 5  # we wish to return the index of the two closest strings to our query
xq = model.encode([user_input])

# %% [markdown]
# ## Performing match search using faiss index
# This function searches the top k matching vectors in the corpus and returns the distances between the vectors and the position of the strings in the original sentence collection/matrix

# %%
# we now perform a search for similarity
# %time
distance, position = index.search(xq, k)  # search
print(position)
print(distance)

# %%
# get the text
for i in position[0]:
    print(sentences[i])
print(distance[0])

# %%
sentences[position[0][0]]

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
index2 = faiss.IndexIVFFlat(quantizer, dimension, num_cells)

# %%
index2.is_trained

# %%
index2.train(sentence_embeddings)
index2.is_trained  # check if index is now trained

# %%
index2.add(sentence_embeddings)
index2.ntotal  # number of embeddings indexed

# %%
# we now perform a search for similarity using the input text read earlier on.
# %time
distance2, position2 = index2.search(xq, k)  # search
print(position2)
print(distance2)

# %%
# get the text
for i in position2[0]:
    print(sentences[i])
print(distance2[0])

# %% [markdown]
# ## By compressing the vectors, we further improve the computation time
# Compression is implemented in the following cell by specifying the number of centroids and bits in each centroid

# %%
m = 8  # number of centroid IDs in final compressed vectors
bits = 8  # number of bits in each centroid
quantizer = faiss.IndexFlatL2(dimension)  # we keep the same L2 distance flat index
index3 = faiss.IndexIVFPQ(quantizer, dimension, num_cells, m, bits)

# %%
index3.train(sentence_embeddings)
index3.add(sentence_embeddings)

# %%
index3.nprobe = 1  # nprobe specifies the number of neigboring cells to be used

# %%
# we now perform a search for similarity using the input text read earlier on.
# %time
distance3, position3 = index3.search(xq, k)  # search
# get the text
for i in position3[0]:
    print(sentences[i])
print(distance3[0])

# %%
sentences[position[0][0]]

# %%
sentences[position[0][1]]

# %%
print("Similarity:", util.dot_score(xq, sentence_embeddings)[0][position[0][1]])

# %% [markdown]
# ## Output summary
# For the given dataset, we notice that the algorithm returns the same set of indexes. However, when compression of the vectors happens, the indexes are returned in a reversed order. This shows that accuracy is compromised. The least distance between the vectors increases from 90 to 130.

# %%
# pip install hydra-core omegaconf

# %% [markdown]
# # Using the distilbert model for embedding

# %%
# using the distilbert model for embedding
model2 = SentenceTransformer(
    "distilbert-base-nli-stsb-mean-tokens"
)  # best performance for all distillbert pretrained models.

# %%
# check GPU availability and set cuda if present.
if torch.cuda.is_available():
    model2 = model2.to(torch.device("cuda"))
print(model2.device)

# %%
# produce sentence vectors
embeddings = model2.encode(sentences)

# %%
# Step 1: Change data type
embeddings2 = np.array([embedding for embedding in embeddings]).astype("float32")

# %%
print(embeddings[:10])

# %%
print(embeddings2[:10])

# %%
# create a faiss index for the search using the embenddings
index_dis = faiss.IndexFlatL2(embeddings.shape[1])

# %%
index_dis.add(embeddings)  # add the vector to the index created

# %%
# search the index for our given input xq
# %time
distance4, position4 = index_dis.search(xq, k)  # search
# get the text
for i in position4[0]:
    print(sentences[i])
print(distance4[0])

# %%
sentences[position4[0][0]]

# %%
num_cells = 50  # Number of cells the index should be divided into
quantizer = faiss.IndexFlatL2(dimension)
index_dis2 = faiss.IndexIVFFlat(quantizer, dimension, num_cells)

# %%
index_dis2.train(embeddings)
index_dis2.add(embeddings)

# %%
# search the index for our given input xq
# %time
distance5, position5 = index_dis2.search(xq, k)  # search
# get the text
for i in position5[0]:
    print(sentences[i])
print(distance5[0])

# %%
print("Similarity:", util.dot_score(xq, embeddings)[0][5168])
# We use cosine-similarity and torch.topk to find the highest 5 scores
cos_scores = util.pytorch_cos_sim(xq, embeddings)[0]
top_results = torch.topk(cos_scores, k=k)

# %%
top_results

# %%
print("Similarity:", util.dot_score(xq[0], sentence_embeddings)[0][1308])

# %% [markdown]
# # Using CamemBERT model

# %%
import torch

# workaround to 403 Http limit exceeded, include the following line of code before using torch.hub
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
camembert = torch.hub.load("pytorch/fairseq", "camembert")
camembert.eval()  # disable dropout (or leave in train mode to finetune)

# %%
print(torch.__version__)

# %% [markdown]
# ## 1. https://www.pinecone.io/learn/faiss-tutorial/
# 2. Choosing index https://towardsdatascience.com/billion-scale-semantic-similarity-search-with-faiss-sbert-c845614962e2
# 3. Points to consider when choosing an index: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
# 4. distilBert impl mis-info covid  https://github.com/kstathou/vector_engine/blob/master/notebooks/001_vector_search.ipynb
#
# 5. Roberta https://pytorch.org/hub/pytorch_fairseq_roberta/
#
# 6. Camembert is French implementation of Roberta https://github.com/pytorch/fairseq/tree/main/examples/camembert
# 7. Paper with multilingual text embeddings - frence : https://jep-taln2020.loria.fr/wp-content/uploads/JEP-TALN-RECITAL-2020_paper_209.pdf MUSE and M-BERT are chosen.
#
#
#

# %%
# #!pip install bitarray
# #!pip install hydra-core omegaconf
# #!pip install fairseq
# !pip install --no-cache-dir sentencepiece

# %%
import torch

# workaround to 403 Http limit exceeded, include the following line of code before using torch.hub
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
xlmr = torch.hub.load("pytorch/fairseq:main", "xlmr.large")
xlmr.eval()  # disable dropout (or leave in train mode to finetune)

# %%
print(xlmr)

# %%
print(embeddings11)

# %% [markdown]
# # Using a pre-trained model for French text : french_semantic
# This model is trained using text in french language. We evaluate its performance on french text below.
# The difference in the models lies mostly at the level of embedding the text.

# %%
# using the french_semantic model for embedding
model_fr = SentenceTransformer("Sahajtomar/french_semantic")

# %%
sentences1 = [
    "J'aime mon téléphone",
    "Mon téléphone n'est pas bon.",
    "Votre téléphone portable est superbe.",
]
sentences2 = [
    "Est-ce qu'il neige demain?",
    "Récemment, de nombreux ouragans ont frappé les États-Unis",
    "Le réchauffement climatique est réel",
]
embeddings11 = model_fr.encode(sentences1, convert_to_tensor=True)
embeddings22 = model_fr.encode(sentences2, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(embeddings11, embeddings22)
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print(cosine_scores[i][j])
"""
"""

# %%
print(cosine_scores)

# %%
sample1 = ["J'aime le téléphone que j'ai!"]
sample1_emb = model_fr.encode(sample1, convert_to_tensor=True)
cosine_scores1 = util.pytorch_cos_sim(sample1_emb, embeddings11)

# %%
print(cosine_scores1)

# %%
# Trying to use faiss index for the search
emb11 = model_fr.encode(sentences1)
emb11.shape

# %%
fr_index1 = faiss.IndexFlatL2(emb11.shape[1])

# %%
print(emb11)

# %%
fr_index1.add(emb11)

# %%
sam_emb = model_fr.encode(["J'aime le téléphone que j'ai!"])

# %%
# using Faiss index to compute vector distances
d, p = fr_index1.search(sam_emb, 2)

# %%
print(
    d, p
)  # the first sentence has a great match and the eucledian distance is smaller - 27.04

# %%
# another test case
samp2 = ["Avoir un téléphone peux attirer les soucis!"]
samp2_emb = model_fr.encode(samp2, convert_to_tensor=True)

# %%
cosine_scoresb = util.pytorch_cos_sim(samp2_emb, embeddings11)
print(cosine_scoresb)

# %%
# Using the FAISS flat index to compute the distance.
samp2_emb = model_fr.encode(samp2)
d, p = fr_index1.search(samp2_emb, 2)
print(d, p)

# %%

# %% [markdown]
# # The LaBSE pre-trained model
# It is a multilingual model pre-trained in 109 language. We use it for similarity match search of french text

# %%
from transformers import BertModel, BertTokenizerFast

# %%
tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()

# %%
fr_sentences = [
    "la chaleur tue le coronavirus",
    "le virus se trans,et d'un contact à un autre part les mains",
    " le soleil tue le virus",
    "LE Nord ne peut pas avoir le virus. il fait trop chaud",
    "Le virus se transmet à travers les gouttelettes qui sortent du nez et de la bouche",
    "Aucun groupe dans notre communauté n'est responsable de la propagation du virus",
]
fr_search = ["Le virus ne résiste pas à la chaleur"]
fr_inputs = tokenizer(fr_sentences, return_tensors="pt", padding=True)
fr_search_input = tokenizer(fr_search, return_tensors="pt", padding=True)
with torch.no_grad():
    fr_outputs = model(**fr_inputs)
    fr_search_output = model(**fr_search_input)
# extract the embeddings
fr_embeddings_out = fr_outputs.pooler_output
fr_embedding_search = fr_search_output.pooler_output

# %%
# To calculate similarity between them, we compute the L2 norms of the embeddings
import torch.nn.functional as F


def similarity(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )


# %%
print(similarity(fr_embedding_search, fr_embeddings_out))

# %%
cosine_scoresb2 = util.pytorch_cos_sim(fr_embedding_search, fr_embeddings_out)
print(cosine_scoresb2)

# %% [markdown]
#

# %%
# compute cosine similarity using the sentences used for french_semantic model
# names of variables are : sentences1 , sample1
sentences1_inputs = tokenizer(sentences1, return_tensors="pt", padding=True)
sample1_input = tokenizer(sample1, return_tensors="pt", padding=True)
with torch.no_grad():
    sentences1_outputs = model(**sentences1_inputs)
    sample1_output = model(**sample1_input)
# extract the embeddings
sentences1_embeddings_out = sentences1_outputs.pooler_output
sample1_embedding_search = sample1_output.pooler_output

# %%
print(similarity(sample1_embedding_search, sentences1_embeddings_out))

# %%
