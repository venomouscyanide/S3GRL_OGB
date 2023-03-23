import ast
import pickle

import numpy as np
import torch
from networkx import degree_centrality
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, to_networkx
from tqdm import tqdm

CLUSTER_FILENAME = "./features/clustering.txt"
PAGERANK_FILENAME = "./features/pagerank.txt"
DEGREE_FILENAME = "./features/degree.pkl"
CENTRALITY_FILENAME = "./features/centrality.pkl"


def get_features(n_nodes, data):
    # Adapted from: https://github.com/chuanqichen/cs224w/blob/aeebce6810221bf04a9a14d8d4369be76691b608/ddi/gnn_augmented_node2vec_random.py#L151
    with open(PAGERANK_FILENAME, "r") as f:
        contents = f.read()
        pagerank_dict = ast.literal_eval(contents)
    pagerank_vals = torch.FloatTensor(list(pagerank_dict.values())).reshape((n_nodes, 1))

    with open(CLUSTER_FILENAME, "r") as f:
        contents = f.read()
        clustering_dict = ast.literal_eval(contents)
    cluster_vals = torch.FloatTensor(list(clustering_dict.values())).reshape((n_nodes, 1))

    # degree_vals = degree(data.edge_index[1], data.num_nodes, dtype=torch.long).reshape(shape=(n_nodes, 1))
    # degree_vals = torch.FloatTensor(list(clustering_dict.values())).reshape((n_nodes, 1))
    degree_vals = torch.FloatTensor(list(degree_centrality(to_networkx(data)).values())).reshape((n_nodes, 1))

    with open(CENTRALITY_FILENAME, "rb") as f:
        centrality_dict = pickle.load(f)
    centrality_vals = torch.FloatTensor(list(centrality_dict['betweenness_centrality'].values())).reshape((n_nodes, 1))

    ones = torch.ones((n_nodes, 1))
    features = torch.cat((ones, pagerank_vals, cluster_vals, centrality_vals, degree_vals), 1)
    return features


def resource_allocation(adj_matrix, link_list, batch_size=32768):
    '''
    0:cn neighbor
    1:aa
    2:ra
    '''
    A = adj_matrix
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    w1 = A.sum(axis=0) / A.sum(axis=0)
    temp = np.log(A.sum(axis=0))
    temp = 1 / temp
    temp[np.isinf(temp)] = 1
    D_log = A.multiply(temp).tocsr()
    D = A.multiply(w).tocsr()
    D_common = A.multiply(w1).tocsr()

    link_index = link_list.t()  # (2,:)
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
    ra = []
    cn = []
    aa = []

    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx]
        ra.append(np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten())
        aa.append(np.array(np.sum(A[src].multiply(D_log[dst]), 1)).flatten())
        cn.append(np.array(np.sum(A[src].multiply(D_common[dst]), 1)).flatten())
        # break

    cn = np.concatenate(cn, 0)
    ra = np.concatenate(ra, 0)
    aa = np.concatenate(aa, 0)
    return torch.FloatTensor(cn), torch.FloatTensor(ra), torch.FloatTensor(aa)
