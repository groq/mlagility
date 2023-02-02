# labels: test_group::mlagility name::armaconv author::graph_convolutions
"""
The ARMA graph convolutional operator from the `"Graph Neural Networks with Convolutional
ARMA Filters"
<https://arxiv.org/abs/1901.01343>`_ paper
"""
from mlagility.parser import parse
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ARMAConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2

# Parsing command-line arguments
out_channels = parse(["out_channels"])


model = ARMAConv(dataset.num_features, out_channels)
inputs = {
    "x": torch.ones(data.num_nodes, data.num_features, dtype=torch.float),
    "edge_index": torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long),
}

# Call model
model(**inputs)
