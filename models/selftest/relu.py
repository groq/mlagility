# labels: name::relu author::selftest
import torch

torch.manual_seed(0)


class ReLUTestModel(torch.nn.Module):
    def __init__(self):
        super(ReLUTestModel, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(x)
        return output


dim1 = 10
dim2 = 10

# Model and input configurations
model = ReLUTestModel()
inputs = {"x": torch.rand(dim1, dim2)}


# Call model
model(**inputs)
