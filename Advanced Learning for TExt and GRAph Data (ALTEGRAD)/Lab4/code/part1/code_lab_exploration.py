"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

path_to_file = "../datasets/CA-HepTh.txt"

G = nx.read_edgelist(path_to_file, delimiter = "\t", comments = "#")
print(f"The graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

############## Task 2

connected_components = list(nx.connected_components(G))

print(f"The graph has {len(connected_components)} connected components.")

if len(connected_components) > 1:
    largest_connected_component = max(connected_components, key = len)
    largest_connected_component_G = G.subgraph(largest_connected_component)

    print(f"The largest connected component has {largest_connected_component_G.number_of_nodes()} nodes and {largest_connected_component_G.number_of_edges()} edges.")

    print(f"The proportion of nodes in the largest connected component is {largest_connected_component_G.number_of_nodes() / G.number_of_nodes():.3f}.")
    print(f"The proportion of edges in the largest connected component is {largest_connected_component_G.number_of_edges() / G.number_of_edges():.3f}.")
else:
    print("The graph is already connected.")