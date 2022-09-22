import torch
from torch_geometric.transforms import SIGN
from torch_sparse import SparseTensor


class TunedSIGN(SIGN):
    """
    Helps pop the keys that are not consumed during training
    """

    def __call__(self, data, sign_k):
        data = super().__call__(data)
        if sign_k == -1:
            for idx in range(1, self.K):
                data.pop(f'x{idx}')
        return data

    def beagle_data_creation(self, powers_of_A):
        original_data = powers_of_A[0]

        for index, data in enumerate(powers_of_A, start=1):
            assert data.edge_index is not None
            row, col = data.edge_index
            adj_t = SparseTensor(row=col, col=row,
                                 sparse_sizes=(data.num_nodes, data.num_nodes))

            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

            assert data.x is not None
            # xs = [data.x]
            #
            # xs += [adj_t @ data.x]
            original_data[f'x{index}'] = adj_t @ data.x

        return original_data
