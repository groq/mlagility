"""
The purpose of this script is to demonstrate what happens
with benchit if the target script throws an exception.

You can try it out with:

benchit negative.py

And you should see benchit direct you to a file that
contains the stack trace for the exception.
"""


import torch

torch.manual_seed(0)

# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        a = 1 / 0
        return output


# Instantiate model and generate inputs
input_size = 10
output_size = 5
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

pytorch_outputs = pytorch_model(**inputs)

# Print results
print(f"pytorch_outputs: {pytorch_outputs}")
