import torch
from mlagility import benchit

torch.manual_seed(0)

# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate model and generate inputs
input_size = 1000
output_size = 500
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

benchit(pytorch_model, inputs, device="nvidia")
benchit(pytorch_model, inputs, device="x86")
