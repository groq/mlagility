# labels: test_group::mlagility name::fasterrcnn_mobilenet_v3_large_fpn author::torchvision task::Computer_Vision
"""
https://pytorch.org/vision/stable/models/faster_rcnn.html
"""

from mlagility.parser import parse
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn


torch.manual_seed(0)

# Parsing command-line arguments
batch_size, num_channels, width, height = parse(
    ["batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = fasterrcnn_mobilenet_v3_large_fpn()
model.eval()
inputs = {"images": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
