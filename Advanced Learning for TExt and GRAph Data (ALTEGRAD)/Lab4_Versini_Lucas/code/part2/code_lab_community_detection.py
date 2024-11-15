"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    n = len(G.nodes)
    A = nx.to_numpy_array(G)
    D_inv = np.diag(1 / (A @ np.ones(n)))
    L = np.eye(n) - D_inv @ A

    d = 2 # Not specified in the task. 2 to speed up the computation. 0 <= d <= L.shape[0] - 2.
    vals, vecs = eigs(L, k = d, which = "SR")
    vals, vecs = vals.real, vecs.real

    kmeans = KMeans(n_clusters = k, random_state = 0).fit(vecs[:, vals.argsort()])

    labels = kmeans.labels_
    clustering = {node: label for node, label in zip(G.nodes, labels)}
    ##################    
    
    return clustering


############## Task 4

##################
path_to_file = "../datasets/CA-HepTh.txt"

G = nx.read_edgelist(path_to_file, delimiter = "\t", comments = "#")

connected_components = list(nx.connected_components(G))
largest_connected_component = max(connected_components, key = len)
largest_connected_component_G = G.subgraph(largest_connected_component)

k = 50
clustering = spectral_clustering(largest_connected_component_G, k)
##################

############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    mod = 0.
    m = G.number_of_edges()

    for c in set(clustering.values()):
        nodes_in_c = [node for node, label in clustering.items() if label == c]
        l_c = G.subgraph(nodes_in_c).number_of_edges()
        d_c = sum([G.degree(node) for node in nodes_in_c])
        mod += l_c / m - (d_c / (2 * m))**2
    ##################
    
    return mod



############## Task 6

##################
print(f"Modularity of the spectral clustering (using d = 2): {modularity(G, clustering):.4f}.")

random_clusters = [randint(0, k - 1) for _ in range(G.number_of_nodes())]
# Or use numpy instead:
# random_clusters = np.random.randint(0, k, G.number_of_nodes())
random_clustering = {node: random_clusters[i] for i, node in enumerate(G.nodes())}
print(f"Modularity of the random clustering: {modularity(G, random_clustering):.4f}.")
##################







