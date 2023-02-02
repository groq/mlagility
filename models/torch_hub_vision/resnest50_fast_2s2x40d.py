# labels: test_group::mlagility name::resnest50_fast_2s2x40d author::torch_hub_vision
"""
https://github.com/pytorch/hub/blob/master/pytorch_vision_resnest.md
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
    "zhanghang1989/ResNeSt",
    "resnest50_fast_2s2x40d",
    weights=None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
