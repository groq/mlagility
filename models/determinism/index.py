# labels: name::index test_group::determinism
"""
https://pytorch.org/docs/stable/generated/torch.index_add.html#torch.index_add
https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html#torch.Tensor.index_copy_
"""
import torch

torch.manual_seed(0)

class IndexAdd(torch.nn.Module):
    def __init__(self):
        super(IndexAdd, self).__init__()
    def forward(self, x, dim, index, t):
        return x.index_add_(dim, index, t)
    
model = IndexAdd()
inputs = {
    "x": torch.ones(5,3),
    "t": torch.tensor([[1,2,3],[4,5,6],[7,8,9]]),
    "index": torch.tensor([0,4,2]),
}

model(**inputs)

class IndexCopy(torch.nn.Module):
    def __init__(self):
        super(IndexCopy, self).__init__()
    def forward(self, x, dim, index, t):
        return x.index_copy_(0, index, t)
    
model = IndexCopy()
model(**inputs)