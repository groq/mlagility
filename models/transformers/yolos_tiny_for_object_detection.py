# labels: test_group::mlagility name::yolos_tiny_for_object_detection author::huggingface_pytorch task::Computer_Vision
"""https://huggingface.co/hustvl/yolos-tiny"""
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
model = transformers.YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

inputs = {
    "pixel_values": torch.ones(
        [batch_size, num_channels, height, width], dtype=torch.float
    )
}


# Call model
model(**inputs)
