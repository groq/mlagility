# labels: name::matmuls test_group::determinism
import torch

torch.manual_seed(0)

class BmmModel(torch.nn.Module):
    def __init__(self):
        super(BmmModel, self).__init__()
    
    def forward(self, x, y):
        output = torch.bmm(x,y)
        return output
    
model = BmmModel()
inputs = {"x": torch.randn(10,6,100),
          "y": torch.randn(10,100,8)}

model(**inputs)


class MmModel(torch.nn.Module):
    def __init__(self):
        super(MmModel, self).__init__()
        
    def forward(self, x, y):
        output = torch.mm(x,y)
        return output
    
model = MmModel()
inputs = {"x": torch.randn(100,200),
          "y": torch.randn(200,50)}

model(**inputs)

class MvModel(torch.nn.Module):
    def __init__(self):
        super(MvModel, self).__init__()
        
    def forward(self, x, y):
        output = torch.mv(x,y)
        return output
    
model = MvModel()
inputs = {"x": torch.randn(100,200),
          "y": torch.randn(200)}

model(**inputs)