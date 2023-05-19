# labels: test_group::mlagility name::feastconv author::graph_convolutions task::Graph_Machine_Learning
"""
The (translation-invariant) feature-steered convolutional operator from the
`"FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis"
<https://arxiv.org/abs/1706.05206>`_ paper
"""

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import FeaStConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = FeaStConv(dataset.num_features, dataset.num_classes)
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
