# labels: name::conv1d test_group::determinism
"""
https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d
"""
import torch

torch.manual_seed(0)

class Conv1dModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1dModel, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        output = self.conv(x)
        return output
    
in_channels = 16
out_channels = 33
kernel_size = 3

model = Conv1dModel(in_channels, out_channels, kernel_size)
inputs = {"x": torch.randn(20,16,50)}

model(**inputs)



class Conv2dModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2dModel, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        output = self.conv(x)
        return output
    
model = Conv2dModel(in_channels, out_channels, kernel_size)
inputs = {"x": torch.randn(20,16,50,50)}

model(**inputs)


class Conv3dModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv3dModel, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        output = self.conv(x)
        return output
    
model = Conv3dModel(in_channels, out_channels, kernel_size)
inputs = {"x": torch.randn(20,16,50,50,50)}

model(**inputs)