# labels: test_group::mlagility name::detr_for_object_detection author::transformers task::Computer_Vision
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# This version of DETR performs object detection,
# while the default model outputs raw hidden states.

# Model and input configurations
model = transformers.DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
inputs = {
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
    "pixel_mask": torch.ones(batch_size, height, width, dtype=torch.int64),
}


# Call model
model(**inputs)
