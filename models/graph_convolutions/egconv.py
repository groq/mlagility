# labels: test_group::mlagility name::egconv author::graph_convolutions task::Graph_Machine_Learning
"""
Efficient Graph Convolution from the `"Adaptive Filters and
Aggregator Fusion for Efficient Graph Convolutions"
<https://arxiv.org/abs/2104.01481>`_ paper.
"""

from mlagility.parser import parse
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import EGConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2

# Parsing command-line arguments
out_channels = parse(["out_channels"])[0]


model = EGConv(dataset.num_features, out_channels)
# out_channels must be divisible by the number of heads
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
