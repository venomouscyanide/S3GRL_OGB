# https://github.com/chuanqichen/cs224w/blob/aeebce6810221bf04a9a14d8d4369be76691b608/ddi/gnn_augmented_node2vec_random.py#L151
import ast
import pickle

import torch

CLUSTER_FILENAME = "./features/clustering.txt"
PAGERANK_FILENAME = "./features/pagerank.txt"
DEGREE_FILENAME = "./features/degree.pkl"
CENTRALITY_FILENAME = "./features/centrality.pkl"


def get_features(n_nodes):
    with open(PAGERANK_FILENAME, "r") as f:
        contents = f.read()
        pagerank_dict = ast.literal_eval(contents)
    pagerank_vals = torch.FloatTensor(list(pagerank_dict.values())).reshape((n_nodes, 1))

    with open(CLUSTER_FILENAME, "r") as f:
        contents = f.read()
        clustering_dict = ast.literal_eval(contents)
    cluster_vals = torch.FloatTensor(list(clustering_dict.values())).reshape((n_nodes, 1))

    with open(DEGREE_FILENAME, "rb") as f:
        degree_dict = pickle.load(f)
    degree_vals = torch.FloatTensor(list(clustering_dict.values())).reshape((n_nodes, 1))

    with open(CENTRALITY_FILENAME, "rb") as f:
        centrality_dict = pickle.load(f)
    centrality_vals = torch.FloatTensor(list(clustering_dict.values())).reshape((n_nodes, 1))

    ones = torch.ones((n_nodes, 1))
    features = torch.cat((ones, pagerank_vals, cluster_vals, centrality_vals, degree_vals), 1)
    return features
