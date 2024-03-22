# labels: name::index_put test_group::determinism
"""
https://pytorch.org/docs/stable/generated/torch.Tensor.index_put.html#torch.Tensor.index_put
"""
import torch

torch.manual_seed(0)

class IndexPut(torch.nn.Module):
    def __init__(self):
        super(IndexPut, self).__init__()
    def forward(self, x, indices, values):
        return x.index_put_(indices, values)
    
model = IndexPut()
inputs = {
    "x": torch.randn(10, dtype=torch.float64),
    "indices": (torch.randint(low=0, high=10, size=(20,)),),
    "values": torch.randn(20, dtype=torch.float64),
}

model(**inputs)


class Put(torch.nn.Module):
    def __init__(self):
        super(Put, self).__init__()
    def forward(self, x, index, source):
        return x.put(index, source, accumulate=True)
    
model = Put()
inputs = {
    "x": torch.tensor([[4,3,5],[6,7,8]]),
    "index": torch.tensor([1,3]),
    "source": torch.tensor([9,10])
}

model(**inputs)