# labels: name::scatter_reduce test_group::determinism
"""
https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
"""
import torch

torch.manual_seed(0)

class ScatterReduceProd(torch.nn.Module):
    def __init__(self):
        super(ScatterReduceProd, self).__init__()
    def forward(self, input, index, src):
        output = torch.scatter_reduce(input, 0, index, src, reduce="prod")
        return output
    
model = ScatterReduceProd()

inputs = {
    "input": torch.randn(4, dtype=torch.float64),
    "index": torch.tensor([0, 1, 0, 1, 2, 1]),
    "src": torch.randn(6, dtype=torch.float64),
}

model(**inputs)

class ScatterReduceSum(torch.nn.Module):
    def __init__(self):
        super(ScatterReduceSum, self).__init__()
    def forward(self, input, index, src):
        output = torch.scatter_reduce(input, 0, index, src, reduce="sum")
        return output
    
model = ScatterReduceSum()

inputs = {
    "input": torch.randn(4, dtype=torch.float64),
    "index": torch.tensor([0, 1, 0, 1, 2, 1]),
    "src": torch.randn(6, dtype=torch.float64),
}

model(**inputs)