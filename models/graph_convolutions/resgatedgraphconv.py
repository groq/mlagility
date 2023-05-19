# labels: test_group::mlagility name::resgatedgraphconv author::graph_convolutions task::Graph_Machine_Learning
"""
The residual gated graph convolutional operator from the `"Residual Gated Graph ConvNets"
<https://arxiv.org/abs/1711.07553>`_ paper
"""

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ResGatedGraphConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = ResGatedGraphConv(dataset.num_features, dataset.num_classes)
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
