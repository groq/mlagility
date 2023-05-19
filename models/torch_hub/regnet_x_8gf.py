# labels: test_group::mlagility name::regnet_x_8gf author::torch_hub task::Computer_Vision
from mlagility.parser import parse
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, num_channels, width, height = parse(
    ["batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "pytorch/vision:v0.13.1",
    "regnet_x_8gf",
    weights=None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
