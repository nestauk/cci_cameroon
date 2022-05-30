# Clustering algorithm

## Introduction

This document summarises the approach used to prototype a tool that forms clusters from comments collected by the Cameroon Red Cross in the context of COVID-19 pandemic. The project is in two parts: one that uses supervised learning to classify comments into various categories; and a second that clusters comments that could not be assigned to any of the existing categories at a level of confidence. The goal of the exercise is to facilitate the process of assigning codes/labels to the comments and providing appropriate responses to communities in case of identified disinformation.

In the sections that follow, we describe the source of data and how the comments were processed. Attention is then focused on using Natural Language Processing NLP techniques to prepare and cluster the data. Figure 1 shows the key stages of the process.

![Clustering steps](../../../outputs/figures/readme/clustering_overall.png)

Figure 1: Key stages of the process

## Capturing the meanings of the comments

To cluster sentences together, the algorithm used needs to capture their meaning. Numerical representations (embeddings) of the comments are first created using a transformer model. A transformer model is a type of neural network which is pretrained on large volumes of text to learn the relationships between different words in a particular language. These models would later transfer the knowledge learned to generate numerical representations of other texts in the same language. In our case we employed the french semantic transformer model. We selected the french semantic transformer model because its embeddings produced better results than others tested. The code that evaluates the performance of transformer models on French text in found in the notebook similarity_search_sample.py

## Calculating similarity

After generating numerical representations of the comments using the transformer model, we proceeded to compute similarity scores for n neighbors of each data point. To compute the similarities faster, we used Facebook Artificial Intelligence Similarity Search FAISS library. Based on the CRC's rate of data collection, we expect the algorithm to be run each time with less than a thousand or so new comments. This data size led us to use a flat FAISS index for accuracy. This also guarantees a quick search given the data size. We then calculated the cosine similarity score between the embedding of each comment and the top 10 similar comments in the pool. The output of this step is an adjacency matrix whose inputs are cosine scores. The adjacency matrix is generated using util functions found in the notebook clustering_helper_functions.py under the pipeline directory.

## Generating a graph network

Having identified the 10 most similar comments to each comment, a network is generated using the adjacency matrix. The nodes in the network are the individual data points in the dataset. An edge exists between node A and node B if data point B is among the top 10 similar data points to data point A. To ensure that distinct comments stand alone, we set a minimum threshold for the similarity score of 0.3. Thus an edge is formed between node A and node B in the graph if the cosine similarity score between the embedding of sentence A and that of sentence B is at least 0.3. The i-graph module is used to generate the connectivity graph.

In the next step, the generated graph is fed into a community detection algorithm to group the data points into different clusters. Clustering algorithm like K-Means is not used because the number of suitable clusters is not known a priori. With growing data sizes, we expect the optimal number of clusters to change. The community detection algorithm used is leiden from the leidenalg library. Leiden algorithm is used because it's built on laivan algorithms and is faster. A helper function that colors the different communities formed from the graph is found in the cluster_helper_functions.py notebook under pipeline.

## Consensus clustering

To evaluate the robustness of the community detection algorithm, we used consensus clustering. In this approach, a repeated runs of the algorithm are made and an ensemble of clusters produced. Pairwise Adjusted Mutual information score is calculated to determine the proportion of times data points are assigned to the same cluster for the different runs.

## Set up and run the model

- clone the project and cd into the cci_cameroon directory
- run the command `make install` to create virtual environment and install dependencies
- run cd to `analysis/clustering_model_dev` while in the directory
  - run the command `python3 rumour_clustering_model_development_concesus_clustering.py`
  - run the command `python3 rumour_clustering_model_development.py` to explore clustering approach using networkx
  - run the command `python3 rumour_clustering_test.py` to test the clustering approach using networkx library

Each of the
Once run, the model takes in a dataset and performs the following:

1. Creates numberical representations (embeddings) of the sentences using a transformer model
2. Searches for the five most similar comments to each data point using a FAISS index
3. Generates an adjacency matrix using the cosine similarity scores where the nodes are the different data points and the edges the cosine similarity scores
4. Use the matrix to generate a graph network
5. The network is then fed into a community detection model which produces the groups.
6. An ensemble of clusters is produced from repeated runs of the model and used to compute Adjusted Mutual Information Score for determining the algorithm's robustness

## Output

The final output of the model are the created groups. These are saved in excel file clusters.xslx under the outputs/data directory with each group saved in a different worksheet.
