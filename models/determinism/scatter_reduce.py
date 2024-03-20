# labels: name::scatter_reduce_prod test_group::nondeterministic_layers
"""
https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
"""
import torch

torch.manual_seed(0)

class ScatterReduceProd(torch.nn.Module):
    def __init__(self):
        super(ScatterReduceProd, self).__init__()
    def forward(self, input, index, src):
        return input.scatter_reduce(0, index, src, reduce="prod")
    
model = ScatterReduceProd()

src = torch.rand(6, dtype=torch.float64)
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.rand(4, dtype=torch.float64)

inputs = {
    "input": input,
    "index": index,
    "src": src,
}

model(**inputs)