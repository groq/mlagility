# labels: test_group::mlagility name::dnaconv author::graph_convolutions task::Graph_Machine_Learning
"""
The dynamic neighborhood aggregation operator from the
`"Just Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural Networks"
<https://arxiv.org/abs/1904.04849>`_ paper
"""

from mlagility.parser import parse
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DNAConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2

# Parsing command-line arguments
num_layers, in_channels = parse(["num_layers", "in_channels"])


model = DNAConv(in_channels)
# The input node features of shape :obj:`[num_nodes, num_layers, channels]`
x = torch.ones(data.num_nodes, num_layers, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
