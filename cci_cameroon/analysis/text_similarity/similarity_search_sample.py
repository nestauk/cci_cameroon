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
import cci_cameroon
from cci_cameroon.getters.external_data import (
    fetch_data,
    similarity,
    compute_embedding_cosign_score,
)

# %%
# !pip list

# %% [markdown]
# # Semantic Textual Similarity matching
# This notebook contains an implementation of sample transformer models for both English and French text. Text Embeddings are generated using different pre-trained models and their performance on matching sample strings is evaluated. The cosine similarity score or the Euclidean distance between the created vectors is used for the evaluation.

# %%
# install sentence tranformers library using command below
# conda install -c conda-forge sentence-transformers
# pip install hydra-core omegaconf
# #!pip install bitarray
# #!pip install --no-cache-dir sentencepiece
base_dir = cci_cameroon.PROJECT_DIR


# %%
urls = []

# %%
# read in sample text dataset from online.
with open(f"{base_dir}/inputs/data/english_url.txt") as f:
    urls = [u.strip("\n") for u in f]
sentences = fetch_data(urls)

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

# %% [markdown]
# # Using French models
# We create a sample corpus of French sentences that would be used across the different pre-trained models below.

# %%
sentences_french = [
    "C'est une personne heureuse",
    "J'aime mon téléphone",
    "Mon téléphone n'est pas bon.",
    "Elle est heureuse apres avoir eu le BAC",
    "Votre téléphone portable est superbe.",
    "Récemment, de nombreux ouragans ont frappé les États-Unis",
    "Aujourd'hui est une journée ensoleillée",
]
sample_french1 = ["J'aime le téléphone que j'ai!"]
sample_french2 = ["C'est une personne très heureuse"]

# %%
import pickle

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
# workaround to 403 Http limit exceeded, include the following line of code before using torch.hub
# torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

# %% [markdown]
# # FlauBert pre-trained model

# %%
model_flaubert = SentenceTransformer("hugorosen/flaubert_base_uncased-xnli-sts")

# %%
flaubert_file = "flaubert_model.pkl"
pickle.dump(model_flaubert, open(flaubert_file, "wb"))

# some time later...

# load the model from disk
# loaded_model = pickle.load(open(flaubert_file 'rb'))

# %%
scores_flaubert1 = compute_embedding_cosign_score(
    model_flaubert, sample_french1, sentences_french
)
scores_flaubert2 = compute_embedding_cosign_score(
    model_flaubert, sample_french2, sentences_french
)
print(scores_flaubert1)
print(scores_flaubert2)

# %% [markdown]
# # Using a pre-trained model for French text : french_semantic
# This model is trained using text in french language. We evaluate its performance on french text below.
# The difference in the models lies mostly at the level of embedding the text.

# %%
# using the french_semantic model for embedding
model_fr = SentenceTransformer("Sahajtomar/french_semantic")

# %%
fr_file = "fr_semantic_model.pkl"
pickle.dump(model_fr, open(fr_file, "wb"))

# some time later...

# load the model from disk
# loaded_model = pickle.load(open(ffr_file 'rb'))

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
cosine_scores = compute_embedding_cosign_score(
    model_fr, sentences1, sentences2, tensor=True
)
# printing the cosign values
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print(cosine_scores[i][j])

# %%
print(cosine_scores)

# %%
sample1 = ["J'aime le téléphone que j'ai!"]
cosine_scores1 = compute_embedding_cosign_score(
    model_fr, sample1, sentences1, tensor=True
)

# %%
print(cosine_scores1)

# %% [markdown]
# ### Using the french_semantic model on the general sample data created above to compare its performance with other models

# %%
embeddings_fr_semantic = model_fr.encode(sentences_french, convert_to_tensor=True)
embeddings_sample1_fr_semantich = model_fr.encode(
    sample_french1, convert_to_tensor=True
)
embeddings_sample2_fr_semantic = model_fr.encode(sample_french2, convert_to_tensor=True)


# %%
print(
    compute_embedding_cosign_score(
        model_fr, sample_french1, sentences_french, tensor=True
    )
)
print(
    compute_embedding_cosign_score(
        model_fr, sample_french2, sentences_french, tensor=True
    )
)

# %% [markdown]
# ## Using Faiss index to compute the distances for french text

# %%
# use a sample faiss index for the search
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

# %% [markdown]
# # The LaBSE pre-trained model
# It is a multilingual model pre-trained in 109 language. We use it for similarity match search of french text

# %%
# using Hugging face
from transformers import BertModel, BertTokenizerFast

# %%
tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()

# %%
labse_file = "labse_model.pkl"
pickle.dump(model, open(labse_file, "wb"))

# some time later...

# load the model from disk
# labse_model = pickle.load(open(labse_file, 'rb'))

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
print(similarity(fr_embedding_search, fr_embeddings_out))

# %%
cosine_scoresb2 = util.pytorch_cos_sim(fr_embedding_search, fr_embeddings_out)
print(cosine_scoresb2)

# %%
# compute cosine similarity using the sentences used for french_semantic model
# names of variables are : sentences1 , sample1
sentences1_fr_inputs = tokenizer(sentences_french, return_tensors="pt", padding=True)
sample1_fr_input = tokenizer(sample_french1, return_tensors="pt", padding=True)
sample2_fr_input = tokenizer(sample_french2, return_tensors="pt", padding=True)
with torch.no_grad():
    sentences1_fr_outputs = model(**sentences1_fr_inputs)
    sample1_fr_output = model(**sample1_fr_input)
    sample2_fr_output = model(**sample2_fr_input)
# extract the embeddings
sentences1_fr_embeddings_out = sentences1_fr_outputs.pooler_output
sample1_fr_embedding_search = sample1_fr_output.pooler_output
sample2_fr_embedding_search = sample2_fr_output.pooler_output

# %%
print(similarity(sample1_fr_embedding_search, sentences1_fr_embeddings_out))
print(similarity(sample2_fr_embedding_search, sentences1_fr_embeddings_out))

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
