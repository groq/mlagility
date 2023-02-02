# labels: test_group::mlagility name::hardnet68 author::torch_hub_vision
"""
https://github.com/pytorch/hub/blob/master/pytorch_vision_hardnet.md
"""

import mlagility
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, num_channels, width, height = mlagility.parse(
    ["batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "PingoLH/Pytorch-HarDNet",
    "hardnet68",
    weights=None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
