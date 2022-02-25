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


# %%
