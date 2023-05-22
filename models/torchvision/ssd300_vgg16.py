# labels: test_group::mlagility name::ssd300_vgg16 author::torchvision task::Computer_Vision
"""
https://pytorch.org/vision/stable/models/ssd.html
"""

from mlagility.parser import parse
import torch
from torchvision.models.detection import ssd300_vgg16


torch.manual_seed(0)

# Parsing command-line arguments
batch_size, num_channels, width, height = parse(
    ["batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = ssd300_vgg16()
model.eval()
inputs = {"images": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
