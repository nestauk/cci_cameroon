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
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import networkx as nx
import faiss
from faiss import normalize_L2
import community
from cdlib import algorithms
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from cdlib import evaluation


# %%
# generates colors to use for graph
def generate_colors(num):
    """Function generates a set of colors for the communities generated. Takes in the number
    of communities in the graph"""
    colors = []
    for i in range(num):
        colors.append("#%06X" % randint(0, 0xFFFFFF))
    return colors


# method to draw the identified communities
def draw_communities_graph(graph, colors, communities):
    """draws communities identified in a network. Recieves a graph, colors to use and list of communities"""
    color_map = []
    for node in graph:
        for i in range(len(communities)):
            if node in communities[i]:
                color_map.append(colors[i])
    nx.draw(graph, node_color=color_map, with_labels=True)
    plt.show()


def generate_adjacency_matrix(positions, n_neighbors, dimension):
    """Takes the positions obtained from the cosine similarity computation, the number of neighbors to consider,
    the size of the matrix and generates adjacency matrix. Returns an adjacency matrix"""
    adjacency_matrix = np.zeros(
        [dimension, dimension]
    )  # initialize the adjacency matrix to zeros
    # loop through the positions and set neighbors accordingly.
    for row in range(dimension):
        for num in range(n_neighbors):
            # set the neighbors in the adjacency matrix
            adjacency_matrix[row, positions[row][num]] = 1
    np.fill_diagonal(adjacency_matrix, 0)  # takes off self-connections
    return adjacency_matrix


# new section
def generate_edges(df, column_name, top_n_matrix, weight_matrix):
    """This method generates the edges to be used in a graph. Takes in a dataframe which holds the data,
    column_name which specifies the column of interest(comments column) and top_n_matrix which is a matrix
    containing the positions in the dataframe of the top n similar comments to each comment.
    returns the edges and weighted edges."""
    n = top_n_matrix.shape[1]  # this is the number of similar neighbors considered
    comments = []
    top_neighbors = []
    weights = []
    for ind in range(df.shape[0]):
        current_comment = list(
            [df.iloc[ind][column_name]] * (top_n_matrix.shape[1] - 1)
        )  # this replicates the comment n times
        comments.extend(current_comment)
        neighbors = list(
            df.iloc[top_n_matrix[ind][1:]][column_name]
        )  # this ensures that self-edges are not formed
        top_neighbors.extend(neighbors)
        weights.extend(weight_matrix[ind][1:])
    edges = list(zip(comments, top_neighbors))
    weighted_edges = list(zip(comments, top_neighbors, weights))
    return edges, weighted_edges


def generate_adjacency_matrix2(positions, n_neighbors, dimension, weights):
    """Takes the positions obtained from the cosine similarity computation, the number of neighbors to consider,
    the size of the matrix and generates adjacency matrix. Returns an adjacency matrix"""
    adjacency_matrix = np.zeros(
        [dimension, dimension]
    )  # initialize the adjacency matrix to zeros
    # loop through the positions and set neighbors accordingly.
    for row in range(dimension):
        for num in range(n_neighbors):
            # set the neighbors in the adjacency matrix only if the similarity score is positive
            if weights[row][num] >= 0.3:
                adjacency_matrix[row, positions[row][num]] = weights[row][num]
    np.fill_diagonal(adjacency_matrix, 0)  # takes off self-connections
    return adjacency_matrix


def generate_community_labels(community_list, dimension):
    """Takes a list of communities and generates labels for each data point.
    dimension is the number of nodes in the graph. Returns a list of labels"""
    group_labels = []  # labels to be assigned to the different communities
    for i in range(len(community_list)):
        group_labels.append("Group_" + str(i))
    labels2 = [None] * dimension
    for i in range(len(community_list)):
        for j in community_list[i]:
            labels2[j] = group_labels[i]
    return labels2


def compute_community_silhuoette_scores(communities, sample_silhouette_scores):
    """Computes the average silhouette score for each community.returns a list of the values computed"""
    avg_scores = []
    for i in range(len(communities)):
        avg_scores.append(
            np.mean([sample_silhouette_scores[m] for m in communities[i]])
        )
    return np.mean(avg_scores), avg_scores


def get_communities_with_threshold(
    community_list, silhouette_score_list, thresh, data, col_name
):
    """Identifies communities that meet set criteria. Takes a list of communities and sample silhouette score for
    the data. Returns a dictionary of lists with each list containing community member count and average silhouette
    score for a community"""
    retained_communities_stats = {}
    retained_communities = []
    communities_not_retained_some = (
        []
    )  # some communities not retained but have the minimum number of comments set
    for i in range(len(community_list)):
        if (
            len(community_list[i])
            > (len(silhouette_score_list) / len(community_list)) * 0.3
        ):
            score = np.mean([silhouette_score_list[m] for m in community_list[i]])
            # if score > thresh:Setting a silhuoette threshold doesn't play a great role in determining the cluster quality with varying neighbors
            details = [len(community_list[i]), score]
            retained_communities_stats[i] = details
            retained_communities.append(list(data.iloc[community_list[i]][col_name]))
        elif len(community_list[i]) > 0:
            communities_not_retained_some.append(
                list(data.iloc[community_list[i]][col_name])
            )
    return (
        retained_communities,
        communities_not_retained_some,
        retained_communities_stats,
    )


def get_metrics(data, column_name, transformer_model):
    """Takes in a data frame, column of interest and transformer model. Computes various metrices for different
    number of neighbors. Returns a dataframe containing the statistics of interes"""
    sent_embeddings = transformer_model.encode(data[column_name])
    neighbor_list = [5, 10, 15, 20, 25, 30, 35]
    modularity_list = []
    average_silhouette_score_list = []
    number_of_clusters_list = []  # holds the number of clusters produced
    retained_clusters_list = []  # holds list of comments in each of the clusters
    modularity_density_list = []
    adjusted_mutual_score_list = []
    number_of_comments_list = []
    number_of_retained_clusters_list = (
        []
    )  # holds the number of clusters which meet minimum criteria and are retained
    some_rejected_clusters_list = (
        []
    )  # holds ALL rejected clusters which meet the minimum number of comments set
    for n in neighbor_list:
        index = faiss.index_factory(
            sent_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
        )
        # setting the number of neighbors to consider for graph connectivity
        index.train(sent_embeddings)
        index.add(sent_embeddings)
        similarity_distance_mat, sim_positions_mat = index.search(
            sent_embeddings, n  # consider only the first n elements
        )
        weight_mat = generate_adjacency_matrix2(
            sim_positions_mat, n, sent_embeddings.shape[0], similarity_distance_mat
        )
        G = nx.from_numpy_matrix(
            weight_mat, create_using=nx.Graph(), parallel_edges=False
        )
        partitions = community.best_partition(G)
        comu = algorithms.leiden(G)
        labels_best = []
        for nod in partitions.items():
            labels_best.append(nod[1])
        labels_com = generate_community_labels(
            comu.communities, sent_embeddings.shape[0]
        )
        sil_score = silhouette_samples(sent_embeddings, labels_com, metric="cosine")
        av, _ = compute_community_silhuoette_scores(
            comu.communities, sil_score
        )  # the average silhouette_score
        reduced_av = av - (0.4 * av)
        (
            retained_communities,
            rejected_communities,
            statistics,
        ) = get_communities_with_threshold(
            comu.communities, sil_score, reduced_av, data, "comment"
        )
        # save computed values for the current number of neighbors
        adjusted_mutual_score_list.append(
            adjusted_mutual_info_score(labels_best, labels_com)
        )  # store the computed value
        average_silhouette_score_list.append(av)
        number_of_clusters_list.append(
            len(comu.communities)
        )  # number of communities formed
        retained_clusters_list.append(
            retained_communities
        )  # this holds a list of the actual comments in the groups for inspection
        some_rejected_clusters_list.append(rejected_communities)
        modularity_list.append(community.modularity(partitions, G))
        modularity_density_list.append(evaluation.modularity_density(G, comu).score)
        number_of_comments_list.append(sent_embeddings.shape[0])
        number_of_retained_clusters_list.append(
            len(retained_communities)
        )  # holds the number of clusters that meets threshold conditions
    df_stats = pd.DataFrame(
        {
            "no_comments": number_of_comments_list,
            "n_neighbors": neighbor_list,
            "AMI": adjusted_mutual_score_list,
            "modularity": modularity_list,
            "modularity_density": modularity_density_list,
            "total_clusters": number_of_clusters_list,
            "silhouette_av": average_silhouette_score_list,
            "clusters_retained_count": number_of_retained_clusters_list,
        }
    )
    return retained_clusters_list, some_rejected_clusters_list, df_stats


# %%

# %%
