# labels: test_group::mlagility name::tagconv author::graph_convolutions task::Graph_Machine_Learning
"""
The topology adaptive graph convolutional networks operator from the
`"Topology Adaptive Graph Convolutional Networks"
<https://arxiv.org/abs/1710.10370>`_ paper
"""
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import TAGConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = TAGConv(dataset.num_features, dataset.num_classes)
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
