import random

import torch
from scipy.sparse import dok_matrix
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import SIGN
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor, from_scipy, spspmm

from tqdm import tqdm

import scipy.sparse as ssp
import numpy as np

from scipy.sparse import vstack


class TunedSIGN(SIGN):
    """
    Custom SIGN class for PoS and SoP
    """

    def __call__(self, data, sign_k):
        data = super().__call__(data)
        if sign_k == -1:
            for idx in range(1, self.K):
                data.pop(f'x{idx}')
        return data

    def SoP_data_creation(self, sop_data_list):
        original_data = sop_data_list[0]

        for index, data in enumerate(sop_data_list, start=1):
            assert data.edge_index is not None
            row, col = data.edge_index
            adj_t = SparseTensor(row=col, col=row, value=torch.tensor(data.edge_weight),
                                 sparse_sizes=(data.num_nodes, data.num_nodes))

            assert data.x is not None

            original_data[f'x{index}'] = (adj_t @ data.x)

        # the following keys are useless in SIGN-esque training
        del original_data['node_id']
        del original_data['num_nodes']
        del original_data['edge_index']
        del original_data['edge_weight']

        return original_data


class OptimizedSignOperations:
    @staticmethod
    def get_SoP_plus_prepped_ds(powers_of_A, link_index, A, x, y, verbose=False, ratio_per_hop=1, sign_kwargs=None):
        # TODO; no support for labeling, no support for >1 sign_k values
        # print("SoP Plus Optimized Flow.")
        # optimized SoP Plus flow, everything is created on the CPU, then in train() sent to GPU on a batch basis
        if len(powers_of_A) > 1:
            raise NotImplementedError

        list_of_training_edges = link_index.t().tolist()
        num_training_egs = len(list_of_training_edges)

        all_data = []
        power_of_a = powers_of_A[0]

        xs = []
        ys = []
        start_index = []
        end_index = []
        all_subgraphs = []
        start = 0

        if verbose:
            print("Stacking all links' rows")
        lil_matrix = power_of_a.to_scipy().tolil()

        for link_number in tqdm(range(0, num_training_egs * 2, 2), disable=not verbose, ncols=70):
            src, dst = list_of_training_edges[int(link_number / 2)]
            interim_src = lil_matrix[src]
            interim_src[0, dst] = 0
            interim_dst = lil_matrix[dst]
            interim_dst[0, src] = 0

            interim_src_tensor = torch.tensor(interim_src.todense(), dtype=torch.bool)[0]
            interim_dst_tensor = torch.tensor(interim_dst.todense(), dtype=torch.bool)[0]

            strat = sign_kwargs['k_node_set_strategy']
            if strat == "intersection":
                interim = torch.logical_and(interim_src_tensor, interim_dst_tensor)
            elif strat == "union":
                interim = torch.logical_or(interim_src_tensor, interim_dst_tensor)
            else:
                raise NotImplementedError(f"Strat {strat} not implemented")

            strat_indices = (interim == True).nonzero(as_tuple=True)[0].tolist()
            if ratio_per_hop != 1:
                strat_indices = random.sample(strat_indices,
                                              int(ratio_per_hop * len(strat_indices)))

            # cn = power_of_a[intersection_indices]
            all_indices = strat_indices + [src, dst]
            subgraph = lil_matrix[all_indices]
            all_subgraphs.append(subgraph)
            start_index.append(start)
            next = start + len(all_indices)
            end_index.append(next - 1)
            start = next

            xs.append(x[[all_indices]])
            ys.append(y)
        if verbose:
            print("Vstacking individual links")
        all_subgraphs = vstack(all_subgraphs)
        if verbose:
            print("Multiplying in one-shot")
        x1 = all_subgraphs @ x

        x1 = torch.from_numpy(x1)
        if verbose:
            print("Finishing with Data object creation")

        for link_number in tqdm(range(0, num_training_egs), disable=not verbose, ncols=70):
            data = Data(
                x=xs[link_number], y=ys[link_number],
            )
            setattr(data, f"x1", x1[start_index[link_number]: end_index[link_number] + 1])
            all_data.append(data)

        return all_data

    @staticmethod
    def get_SoP_prepped_ds(powers_of_A, link_index, A, x, y, verbose=False):
        # print("SoP Optimized Flow.")
        # optimized SoP flow, everything is created on the CPU, then in train() sent to GPU on a batch basis

        sop_data_list = []

        a_global_list = []
        g_global_list = []
        normalized_powers_of_A = powers_of_A
        g_h_global_list = []

        list_of_training_edges = link_index.t().tolist()
        num_training_egs = len(list_of_training_edges)

        if verbose:
            print("Setting up A Global List")
        for index, power_of_a in enumerate(normalized_powers_of_A, start=0):
            if verbose:
                print(f"Constructing A[{index}]")
            a_global_list.append(
                dok_matrix((num_training_egs * 2, A.shape[0]), dtype=np.float32)
            )
            power_of_a_scipy_lil = power_of_a.to_scipy().tolil()
            list_of_lilmtrx = []
            for link_number in tqdm(range(0, num_training_egs * 2, 2), disable=not verbose, ncols=70):
                src, dst = list_of_training_edges[int(link_number / 2)]
                interim_src = power_of_a_scipy_lil.getrow(src)
                interim_src[0, dst] = 0
                interim_dst = power_of_a_scipy_lil.getrow(dst)
                interim_dst[0, src] = 0
                list_of_lilmtrx.append(interim_src)
                list_of_lilmtrx.append(interim_dst)

            to_update = a_global_list[index]
            if verbose:
                print("Converting to DOK")
            for overall_row, item in tqdm(enumerate(list_of_lilmtrx), disable=not verbose, ncols=70):
                data = item.data
                rows = item.rows

                to_update[overall_row, rows[0]] = data[0]

            idx, values = from_scipy(a_global_list[index])
            a_global_list[index] = torch.sparse_coo_tensor(idx, values, size=[num_training_egs * 2, A.shape[0]],
                                                           dtype=torch.float32)
        if verbose:
            print("Setting up G Global List")
        original_x = x.detach()
        x = x.to_sparse()
        for operator_id in tqdm(range(len(normalized_powers_of_A)), disable=not verbose, ncols=70):
            mult_index, mult_value = spspmm(a_global_list[operator_id].coalesce().indices(),
                                            a_global_list[operator_id].coalesce().values(), x.indices(),
                                            x.values(), a_global_list[0].size()[0], a_global_list[0].size()[1],
                                            x.size()[1])
            g_global_list.append(torch.sparse_coo_tensor(mult_index, mult_value, size=[a_global_list[0].size()[0],
                                                                                       x.size()[-1]]).to_dense())
        if verbose:
            print("Setting up G H Global List")
        for index, src_dst_x in tqdm(enumerate(g_global_list, start=0), disable=not verbose, ncols=70):
            g_h_global_list.append(torch.empty(size=[num_training_egs * 2, g_global_list[index].shape[-1] + 1]))
            if verbose:
                print(f"Setting up G H Global [{index}]")
            for link_number in range(0, num_training_egs * 2, 2):
                src, dst = list_of_training_edges[int(link_number / 2)]
                h_src = normalized_powers_of_A[index][src, src].to_dense()
                h_dst = normalized_powers_of_A[index][dst, dst].to_dense()
                g_h_global_list[index][link_number] = torch.hstack(
                    [h_src[0], g_global_list[index][link_number]])
                g_h_global_list[index][link_number + 1] = torch.hstack(
                    [h_dst[0], g_global_list[index][link_number + 1]])
        if verbose:
            print("Finishing Prep with creation of data")
        x = original_x
        for link_number in tqdm(range(0, num_training_egs * 2, 2), disable=not verbose, ncols=70):
            src, dst = list_of_training_edges[int(link_number / 2)]
            data = Data(
                x=torch.hstack(
                    [torch.tensor([[1], [1]]),
                     torch.vstack([x[src], x[dst]]),
                     ]),
                y=y,
            )

            for global_index, all_i_operators in enumerate(g_h_global_list):
                src_features = g_h_global_list[global_index][link_number]
                dst_features = g_h_global_list[global_index][link_number + 1]
                subgraph_features = torch.vstack([src_features, dst_features])

                data[f'x{global_index + 1}'] = subgraph_features
            sop_data_list.append(data)
        return sop_data_list

    @staticmethod
    def get_PoS_prepped_ds(link_index, num_hops, A, ratio_per_hop, max_nodes_per_hop, directed, A_csc, x, y,
                           sign_kwargs, rw_kwargs, verbose=False, node_label='zo'):
        # optimized PoS flow
        if verbose:
            print("PoS Optimized Flow.")
        from utils import k_hop_subgraph
        pos_data_list = []
        # print("Start with PoS data prep")

        K = sign_kwargs['sign_k']

        for src, dst in tqdm(link_index.t().tolist(), disable=not verbose, ncols=70):
            tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                                 max_nodes_per_hop, node_features=x, y=y,
                                 directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)
            csr_subgraph = tmp[1]
            csr_shape = csr_subgraph.shape[0]
            num_nodes = len(tmp[0])

            u, v, value = ssp.find(csr_subgraph)
            u, v, value = torch.LongTensor(u), torch.LongTensor(v), torch.LongTensor(value)

            edge_index = torch.vstack([u, v])
            if directed:
                edge_index, value = to_undirected(edge_index, num_nodes=num_nodes, edge_attr=value)
            edge_index, value = gcn_norm(edge_index, edge_weight=value.to(torch.float), add_self_loops=True)

            subgraph_features = tmp[3]
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[-1], value=value,
                                 sparse_sizes=(csr_shape, csr_shape))
            subgraph = adj_t

            from utils import py_g_drnl_node_labeling
            if node_label == 'drnl':
                label = py_g_drnl_node_labeling(edge_index, 0, 1, num_nodes=num_nodes).reshape((num_nodes, 1))
            elif node_label == 'zo':
                label = torch.tensor([[1]] + [[1]] + [[0]] * (csr_shape - 2))
            else:
                raise NotImplementedError("Check label scheme")

            assert subgraph_features is not None

            powers_of_a = [subgraph]
            for _ in range(K - 1):
                powers_of_a.append(subgraph @ powers_of_a[-1])

            # source, target is always 0, 1
            selected_rows = [0, 1]
            for index, power_of_a in enumerate(powers_of_a):
                powers_of_a[index] = power_of_a[selected_rows]

            x_a = label
            x_b = subgraph_features
            subg_x = torch.hstack([x_a, x_b])

            trimmed_x = subg_x[[0, 1]]
            data = Data(x=trimmed_x, y=y)

            for index, power_of_a in enumerate(powers_of_a, start=1):
                data[f'x{index}'] = power_of_a @ subg_x

            pos_data_list.append(data)

        return pos_data_list

    @staticmethod
    def get_PoS_Plus_prepped_ds(link_index, num_hops, A, ratio_per_hop, max_nodes_per_hop, directed, A_csc, x, y,
                                sign_kwargs, rw_kwargs, verbose=False, node_label='zo'):
        # optimized PoS Plus flow
        if verbose:
            print("PoS Plus Optimized Flow.")
        from utils import k_hop_subgraph, neighbors
        pos_data_list = []
        if verbose:
            print("Start with PoS Plus data prep")

        K = sign_kwargs['sign_k']

        for src, dst in tqdm(link_index.t().tolist(), disable=not verbose, ncols=70):
            tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                                 max_nodes_per_hop, node_features=x, y=y,
                                 directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)
            csr_subgraph = tmp[1]
            csr_shape = csr_subgraph.shape[0]

            u, v, value = ssp.find(csr_subgraph)
            u, v, value = torch.LongTensor(u), torch.LongTensor(v), torch.LongTensor(value)
            num_nodes = len(tmp[0])

            edge_index = torch.vstack([u, v])
            if directed:
                edge_index, value = to_undirected(edge_index, num_nodes=num_nodes, edge_attr=value)
            edge_index, value = gcn_norm(edge_index, edge_weight=value.to(torch.float), add_self_loops=True,
                                         improved=True)
            subgraph_features = tmp[3]
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[-1], value=value,
                                 sparse_sizes=(csr_shape, csr_shape))
            subgraph = adj_t
            from utils import py_g_drnl_node_labeling
            if node_label == 'drnl':
                label = py_g_drnl_node_labeling(edge_index, 0, 1, num_nodes=num_nodes).reshape((num_nodes, 1))
            elif node_label == 'zo':
                label = torch.tensor([[1]] + [[1]] + [[0]] * (csr_shape - 2))
            else:
                raise NotImplementedError("Check label scheme")

            assert subgraph_features is not None
            powers_of_a = [subgraph]

            for _ in range(K - 1):
                powers_of_a.append(subgraph @ powers_of_a[-1])

            # source, target is always 0, 1
            strat = sign_kwargs['k_node_set_strategy']
            if not directed:
                if strat == 'union':
                    one_hop_nodes = neighbors({0}, csr_subgraph).union(neighbors({1}, csr_subgraph))
                    one_hop_nodes.discard(0)
                    one_hop_nodes.discard(1)
                elif strat == 'intersection':
                    one_hop_nodes = neighbors({0}, csr_subgraph).intersection(neighbors({1}, csr_subgraph))
                else:
                    raise NotImplementedError(f"check strat {strat}")
            else:
                csc_subgraph = csr_subgraph.tocsc()
                neighbors_src = neighbors({0}, csr_subgraph).union(neighbors({0}, csc_subgraph, False))
                neighbors_dst = neighbors({1}, csr_subgraph).union(neighbors({1}, csc_subgraph, False))

                if strat == 'union':
                    one_hop_nodes = neighbors_src.union(neighbors_dst)
                    one_hop_nodes.discard(0)
                    one_hop_nodes.discard(1)
                elif strat == 'intersection':
                    one_hop_nodes = neighbors_src.intersection(neighbors_dst)
                else:
                    raise NotImplementedError(f"check strat {strat}")
            strat_hop_nodes = one_hop_nodes

            selected_rows = [0, 1] + list(strat_hop_nodes)
            for index, power_of_a in enumerate(powers_of_a):
                powers_of_a[index] = power_of_a[selected_rows]

            if strat == 'union' or strat == 'intersection':
                x_a = label
                x_b = subgraph_features
                subg_x = torch.hstack([x_a, x_b])
            else:
                raise NotImplementedError(f"check strategy {strat}")

            trimmed_x = subg_x[selected_rows]
            data = Data(x=trimmed_x, y=y)
            subg_x = torch.hstack([x_a, x_b])

            for index, power_of_a in enumerate(powers_of_a, start=1):
                data[f'x{index}'] = power_of_a @ subg_x

            pos_data_list.append(data)

        return pos_data_list
