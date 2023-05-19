# labels: test_group::mlagility name::resnet50_ibn_a author::torch_hub task::Computer_Vision
"""
https://github.com/pytorch/hub/blob/master/pytorch_vision_ibnnet.md
"""

from mlagility.parser import parse
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, num_channels, width, height = parse(
    ["batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "XingangPan/IBN-Net",
    "resnet50_ibn_a",
    weights=None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
