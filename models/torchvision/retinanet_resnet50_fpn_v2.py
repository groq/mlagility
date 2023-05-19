# labels: test_group::mlagility name::retinanet_resnet50_fpn_v2 author::torchvision task::Computer_Vision
"""
https://pytorch.org/vision/stable/models/retinanet.html
"""

from mlagility.parser import parse
import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2


torch.manual_seed(0)

# Parsing command-line arguments
batch_size, num_channels, width, height = parse(
    ["batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = retinanet_resnet50_fpn_v2()
model.eval()
inputs = {"images": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
