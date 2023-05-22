# labels: test_group::mlagility name::deit_tiny_for_image_classification author::transformers task::Computer_Vision
"""https://huggingface.co/facebook/deit-tiny-patch16-224"""
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
model = transformers.ViTForImageClassification.from_pretrained(
    "facebook/deit-tiny-patch16-224"
)
inputs = {
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
}


# Call model
model(**inputs)
