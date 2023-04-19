import torch_geometric
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


def viz_karate():
    dataset = torch_geometric.datasets.KarateClub()
    graph = to_networkx(dataset[0])

    f = plt.figure(1, figsize=(10, 10))
    nx.draw(graph, with_labels=True, pos=nx.spring_layout(graph, seed=42), arrows=False, node_size=600, font_size=15,
            node_color='grey')
    plt.show()  # check if same as in the doc visually
    f.savefig("karate.pdf", bbox_inches='tight')


if __name__ == '__main__':
    viz_karate()
