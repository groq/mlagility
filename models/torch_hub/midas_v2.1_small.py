# labels: test_group::mlagility name::midas_v2.1_small author::torch_hub task::Computer_Vision
"""https://pytorch.org/hub/intelisl_midas_v2/"""
from mlagility.parser import parse
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

inputs = {"x": torch.ones(batch_size, num_channels, height, width, dtype=torch.float)}


# Call model
model(**inputs)
