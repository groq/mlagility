# labels: name::maxunpool test_group::determinism
"""
https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool1d.html#torch.nn.MaxUnpool1d
"""
# note input indices must have repeats for UnPool to be non-deterministic
# TODO: verify the above is true in these examples
import torch

torch.manual_seed(0)

class MaxUnPool1d(torch.nn.Module):
    def __init__(self, size):
        super(MaxUnPool1d, self).__init__()
        self.pool = torch.nn.MaxPool1d(size, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool1d(size, stride=2)
    def forward(self, x):
        y, indices = self.pool(x)
        return self.unpool(y, indices)
    
model = MaxUnPool1d(2)
inputs = {"x": torch.tensor([[[1.,2.,3.,4.,5.,6.,7.,8.]]])}

model(**inputs)


class MaxUnPool2d(torch.nn.Module):
    def __init__(self, size):
        super(MaxUnPool2d, self).__init__()
        self.pool = torch.nn.MaxPool2d(size, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(size, stride=2)
    def forward(self, x):
        y, indices = self.pool(x)
        return self.unpool(y, indices)
    
model = MaxUnPool2d(2)
inputs = {"x": torch.tensor([[[1.,2.,3.,4.],
                              [5.,6.,7.,8.],
                              [9.,10.,11.,12.],
                              [13.,14.,15.,16.]]])}

model(**inputs)


class MaxUnPool3d(torch.nn.Module):
    def __init__(self, size):
        super(MaxUnPool3d, self).__init__()
        self.pool = torch.nn.MaxPool3d(size, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool3d(size, stride=2)
    def forward(self, x):
        y, indices = self.pool(x)
        return self.unpool(y, indices)
    
model = MaxUnPool3d(3)
inputs = {"x": torch.randn(20,16,51,33,15)}

model(**inputs)